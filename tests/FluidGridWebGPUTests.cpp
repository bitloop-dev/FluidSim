#include "FluidGrid.h"
#include "FluidGridGPU.h"
#include "FluidTestSetup.h"
#include "FluidGridEditTests.h"
#include "PressureMultigridGPU.h"
#include "UniformPressureReduction.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
using fluid::FluidGrid;
using fluid::FluidGridGPU;
using fluid::test::initializeTestMaterials;
using fluid::test::uploadTestMaterials;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

void pressureReductionsMatchHost()
{
    using namespace fluid::webgpu;
    Compute compute;
    UniformPressureReduction reduction(compute, 1031 * 1031);
    auto result = compute.field(1, 1, 4);
    auto check = [&](int w, int h, bool fractional)
    {
        auto input = compute.field(w, h), other = compute.field(w, h);
        std::vector<float> a(w * h), b(w * h);
        for (int i = 0; i < w * h; ++i)
        {
            a[i] = fractional ? float(std::sin(i * .17) * .13 + .02) : float((i * 17 + 3) % 11 - 5);
            b[i] = fractional ? float(std::cos(i * .11) * .07 - .01) : float((i * 5 + 1) % 7 - 3);
        }
        a.back() = 13; b.back() = -7; // An odd final cell must participate in both sums and products.
        compute.write(input, a); compute.write(other, b);
        for (int mode = 0; mode < 3; ++mode)
        {
            std::array<double, 4> expected{};
            for (int i = 0; i < w * h; ++i)
            {
                if (mode != 1) expected[0] += a[i];
                if (mode == 0) expected[1] = std::max(expected[1], double(std::abs(a[i])));
                if (mode == 1) expected[0] += double(a[i]) * b[i];
                if (mode == 2) expected[2] += double(a[i]) * b[i];
            }
            reduction.reduce(input, result, mode, &other);
            const auto actual = compute.read(result);
            for (int channel = 0; channel < 4; ++channel)
            {
                const double tolerance = fractional ? 2.e-5 * std::max(1., std::abs(expected[channel])) : 0.;
                require(std::abs(actual[channel] - expected[channel]) <= tolerance,
                        "Uniform pressure reduction differs from host sum/dot/max");
            }
        }
        std::cout << "reductions " << w << 'x' << h << " fractional=" << fractional << '\n';
    };
    for (const auto& [w, h] : {std::pair{1, 1}, {1, 129}, {129, 1}, {17, 19}, {511, 513}, {512, 512}, {1031, 1031}})
        check(w, h, false);
    check(37, 23, true);

    auto input = compute.field(1031, 1031), other = compute.field(1031, 1031);
    std::vector<float> a(1031 * 1031, 1.f), b(a.size(), 1.f);
    for (int mode = 0; mode < 3; ++mode)
    {
        a.back() = mode == 1 ? 1.f : std::numeric_limits<float>::quiet_NaN();
        b.back() = std::numeric_limits<float>::infinity();
        compute.write(input, a); compute.write(other, b);
        reduction.reduce(input, result, mode, &other);
        require(compute.read(result)[3] == 1.f, "Uniform pressure reduction lost a non-finite flag");
    }
    // Poisoned partials from a larger reduction must not enter a subsequent small result.
    auto small = compute.field(3, 1);
    compute.write(small, std::vector<float>{1.f, -2.f, 4.f});
    reduction.reduce(small, result, 0);
    require(compute.read(result) == std::vector<float>({3.f, 4.f, 0.f, 0.f}),
            "Uniform pressure reduction retained stale partial statistics");
}

FluidGridGPU::State pool(int width, int height)
{
    FluidGridGPU::State state;
    state.liquid_fraction.resize(size_t(width) * height);
    for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
    {
        const double surface = x + height * (0.5 + 0.04 * std::cos(3.14159265358979323846 * (x * 3 + 0.5) / width));
        state.liquid_fraction[y * width + x] = float(std::clamp(y + 1.0 - surface, 0.0, 1.0));
    }
    return state;
}

void pressureBalancesGravity()
{
    for (const auto& [w, h] : { std::pair{15, 11}, std::pair{1, 13}, std::pair{13, 1} })
    {
        FluidGridGPU grid(w, h, 4);
        FluidGridGPU::State state;
        state.liquid_fraction.assign(w * h, 1);
        uploadTestMaterials(grid, state);
        grid.applyGravity(1.0 / 120);
        auto result = grid.project(1.0 / 120, 2000, 1e-4);
        std::cout << "hydrostatic " << w << 'x' << h << " iterations=" << result.iterations << " divergence=" << result.max_divergence << '\n';
        require(result.converged, "GPU hydrostatic projection failed");
        state = grid.download();
        for (float v : state.v) require(std::abs(v) < 0.005, "GPU hydrostatic velocity failed to settle");
    }
}

void projectionMatchesCPU()
{
    for (const auto& [w, h] : { std::pair{17, 13}, std::pair{128, 128} })
    {
        FluidGrid cpu(w, h, 4);
        auto state = pool(w, h);
        cpu.liquid_fraction.assign(state.liquid_fraction.begin(), state.liquid_fraction.end());
        initializeTestMaterials(cpu);
        cpu.applyGravity(1.0 / 120);
        const auto reference = cpu.project(1.0 / 120, 2000, 1e-8);
        require(reference.converged, "CPU projection reference failed");
        for (auto preconditioner : {FluidGridGPU::PressurePreconditioner::Diagonal, FluidGridGPU::PressurePreconditioner::Multigrid})
        {
            FluidGridGPU gpu(w, h, 4);
            gpu.pressure_preconditioner = preconditioner;
            uploadTestMaterials(gpu, state);
            gpu.applyGravity(1.0 / 120);
            const auto result = gpu.project(1.0 / 120, 2000, 1e-4);
            const auto output = gpu.download();
            double error = 0;
            for (size_t i = 0; i < output.u.size(); ++i) error = std::max(error, std::abs(output.u[i] - cpu.u[i]));
            for (size_t i = 0; i < output.v.size(); ++i) error = std::max(error, std::abs(output.v[i] - cpu.v[i]));
            std::cout << "mixed projection " << w << 'x' << h << " iterations=" << result.iterations
                      << " divergence=" << result.max_divergence << " max_velocity_error=" << error << '\n';
            require(result.converged, "mixed-density GPU projection failed");
            require(error < 0.005, "GPU pressure correction differs from the CPU reference");
        }
    }
}

void zeroTimeAndFailedProjectionAreHandled()
{
    FluidGridGPU grid(8, 8, 1);
    uploadTestMaterials(grid, pool(8, 8));
    const auto before = grid.download();
    require(!grid.step(0).converged, "zero-time step unexpectedly advanced");
    const auto after = grid.download();
    require(before.liquid_fraction == after.liquid_fraction && before.u == after.u && before.v == after.v, "zero-time step changed state");
    grid.applyGravity(0.01);
    require(!grid.project(0.01, 0).converged, "zero-iteration projection unexpectedly converged");
}

void multigridPreservesPCGContract()
{
    using namespace fluid::webgpu;
    Compute runner;
    for (auto [w, h] : {std::pair{1, 1}, {1, 129}, {131, 1}, {2, 67}, {13, 9}, {65, 37}, {128, 128}, {512, 512}, {513, 257}})
    {
        PressureMultigridGPU multigrid(runner, w, h);
        auto coefficients=runner.field(w,h,4), rhs=runner.field(w,h), stats=runner.field(1,1,4);
        std::vector<float> weights(size_t(w) * h * 4), a(w * h), b(w * h), combined(w * h);
        auto density = [](int x, int y) { return (x / 3 + y / 4) % 2 == 0 ? 0.1f : 100.0f; };
        for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            float* face = weights.data() + (y * w + x) * 4;
            const float center = density(x, y);
            if (x > 0) face[0] = 2.0f / (center + density(x - 1, y));
            if (x + 1 < w) face[1] = 2.0f / (center + density(x + 1, y));
            if (y > 0) face[2] = 2.0f / (center + density(x, y - 1));
            if (y + 1 < h) face[3] = 2.0f / (center + density(x, y + 1));
            const int i = y * w + x;
            a[i] = float(std::sin(i * 0.31) + std::cos(i * 0.073));
            b[i] = float(std::cos(i * 0.17) - std::sin(i * 0.019));
        }
        runner.write(coefficients,weights);
        multigrid.rebuild(coefficients);
        auto center = [](auto& values)
        {
            const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            for (auto& value : values) value = typename std::decay_t<decltype(values)>::value_type(value - mean);
        };
        center(a); center(b);
        for (size_t i = 0; i < a.size(); ++i) combined[i] = 1.3f * a[i] - 0.7f * b[i];
        auto apply = [&](const std::vector<float>& input)
        {
            runner.write(rhs,input);
            const float reduction[] = {float(std::accumulate(input.begin(), input.end(), 0.0)), 0, 0, 0};
            runner.write(stats,reduction);
            const auto pixels = runner.read(multigrid.apply(rhs, stats));
            std::vector<double> result(input.size());
            for (size_t i = 0; i < result.size(); ++i) result[i] = pixels[i];
            center(result);
            return result;
        };
        const auto ma = apply(a), mb = apply(b), mc = apply(combined), repeat = apply(a);
        require(ma == repeat, "GPU multigrid retained an old correction");
        double error = 0, scale = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            error = std::max(error, std::abs(mc[i] - (1.3 * ma[i] - 0.7 * mb[i])));
            scale = std::max(scale, std::abs(mc[i]));
        }
        require(error < 5e-4 * std::max(1.0, scale), "GPU multigrid is not linear within float accuracy");
        const double a_mb = std::inner_product(a.begin(), a.end(), mb.begin(), 0.0);
        const double b_ma = std::inner_product(b.begin(), b.end(), ma.begin(), 0.0);
        const double energy_a = std::inner_product(a.begin(), a.end(), ma.begin(), 0.0);
        const double energy_b = std::inner_product(b.begin(), b.end(), mb.begin(), 0.0);
        require(std::abs(a_mb - b_ma) < 5e-4 * std::max(1.0, std::sqrt(energy_a * energy_b)), "GPU multigrid is not symmetric within float accuracy");
        require(w * h == 1 ? energy_a == 0 : energy_a > 0 && energy_b > 0, "GPU multigrid is not positive");
        std::fill(a.begin(), a.end(), 7.0f);
        const auto constant = apply(a);
        double constant_error = 0.0;
        for (double value : constant) constant_error = std::max(constant_error, std::abs(value));
        std::cout << "multigrid " << w << 'x' << h << " linearity=" << error / std::max(1.0, scale)
                  << " constant_error=" << constant_error << " scale=" << scale << '\n';
        require(constant_error <= 5e-6 * std::max(1.0, scale), "GPU multigrid did not remove its nullspace within float accuracy");
    }
}

void largeMixedPressureConvergesWithinDefaultBudget()
{
    FluidGridGPU grid(512, 512, 4);
    auto state = pool(512, 512);
    uploadTestMaterials(grid, state);
    grid.applyGravity(1.0 / 120);
    const auto result = grid.project(1.0 / 120);
    require(result.converged, "512x512 GPU multigrid exhausted the default pressure budget");
    uploadTestMaterials(grid, state);
    grid.applyGravity(1.0 / 120);
    const auto limited = grid.project(1.0 / 120, 1, 1e-6);
    require(!limited.converged && limited.iterations == 1, "GPU pressure exceeded or misreported a one-iteration budget");
}

void cachedVelocityStatsRemainCurrent()
{
    FluidGridGPU cached(48, 32, 4), refreshed(48, 32, 4);
    const auto initial = pool(48, 32);
    uploadTestMaterials(cached, initial);
    uploadTestMaterials(refreshed, initial);
    for (int tick = 0; tick < 20; ++tick)
    {
        // A zero force leaves velocities unchanged while forcing the uncached CFL path.
        refreshed.applyGravity(0.0);
        const double dt = tick < 10 ? 1.0 / 120.0 : 0.25;
        require(cached.step(dt, 2000, 1e-3).converged && refreshed.step(dt, 2000, 1e-3).converged, "CFL cache comparison failed to converge");
        const auto a = cached.download(), b = refreshed.download();
        require(a.liquid_fraction == b.liquid_fraction && a.u == b.u && a.v == b.v && a.pressure == b.pressure,
                "cached velocity statistics changed transport or pressure");
    }
    const auto restart = cached.download();
    uploadTestMaterials(cached, restart);
    uploadTestMaterials(refreshed, restart);
    refreshed.applyGravity(0.0);
    cached.advect(0.1);
    refreshed.advect(0.1);
    const auto a = cached.download(), b = refreshed.download();
    require(a.liquid_fraction == b.liquid_fraction && a.u == b.u && a.v == b.v, "upload retained stale velocity statistics");
}

void commonStateAndColorReadback()
{
    FluidGridGPU grid(9, 7, 2.5);
    FluidGridGPU::State initial;
    initial.liquid_fraction.resize(63);
    initial.liquid_fraction[2 * 9 + 3] = 0.375f;
    uploadTestMaterials(grid, initial);
    const auto before = grid.download();
    require(before.liquid_fraction == initial.liquid_fraction && before.u.size() == 70 && before.v.size() == 72 && before.pressure.size() == 63,
            "WebGPU state transfer has incorrect field sizes or contents");
    auto checkColor = [&](float expected)
    {
        const auto rgba = grid.colorPixels();
        const float* pixel = rgba.data() + 4 * (2 * 9 + 3);
        require(rgba.size() == 63 * 4 && pixel[0] == 0 && pixel[1] == 0 && pixel[2] == expected && pixel[3] == 1,
                "WebGPU color readback is stale or incorrect");
    };
    checkColor(0.375f);
    require(fluid::test::sameEditState(before, grid.download()), "color readback mutated physics state");
    auto edit = grid.beginEdit();
    edit.setCell(3, 2, fluid::Phase::Liquid, grid.liquid_mixture.components[0].element); edit.commit(); checkColor(1);
    edit.setCell(3, 2, fluid::Phase::Gas, grid.gas_mixture.components[0].element); edit.commit(); checkColor(0);
    require(grid.step(1.0 / 120, 2000, 1e-3).converged, "WebGPU common physics call failed");
    grid.destroy(); grid.upload({}); checkColor(0);
    const auto reset = grid.download();
    for (const auto* field : {&reset.liquid_fraction, &reset.u, &reset.v, &reset.pressure})
        for (float value : *field) require(value == 0, "empty upload did not reset state");
    for (const auto* phase : {&reset.liquid_mass, &reset.gas_mass})
        for (const auto& field : *phase) for (float value : field) require(value == 0, "empty upload did not reset components");
    grid.liquid_mixture.components.clear(); grid.gas_mixture.components.clear();
    grid.upload({}); checkColor(0); grid.destroy();
}

void transportMatchesCPU()
{
    for (const auto velocity : { bl::DVec2{1, 0.75}, bl::DVec2{-1, 0.75}, bl::DVec2{1, -0.75}, bl::DVec2{-1, -0.75} })
    {
        FluidGrid cpu(48, 48, 1);
        FluidGridGPU gpu(48, 48, 1);
        FluidGridGPU::State state;
        state.liquid_fraction.resize(48 * 48);
        state.u.assign(49 * 48, float(velocity.x));
        state.v.assign(48 * 49, float(velocity.y));
        for (int y = 18; y < 30; ++y)
        for (int x = 18; x < 30; ++x)
            state.liquid_fraction[y * 48 + x] = float(std::clamp(6.0 - std::hypot(x - 23.5, y - 23.5), 0.0, 1.0));
        cpu.liquid_fraction.assign(state.liquid_fraction.begin(), state.liquid_fraction.end());
        cpu.u.assign(state.u.begin(), state.u.end());
        cpu.v.assign(state.v.begin(), state.v.end());
        initializeTestMaterials(cpu);
        const double initial = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.0);
        uploadTestMaterials(gpu, state);
        cpu.advect(1);
        gpu.advect(1);
        state = gpu.download();
        double fraction_error = 0, velocity_error = 0;
        for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
            fraction_error = std::max(fraction_error, std::abs(state.liquid_fraction[i] - cpu.liquid_fraction[i]));
        for (int y = 8; y < 40; ++y)
        for (int x = 8; x < 40; ++x)
        {
            velocity_error = std::max(velocity_error, std::abs(state.u[y * 49 + x] - velocity.x));
            velocity_error = std::max(velocity_error, std::abs(state.v[y * 48 + x] - velocity.y));
        }
        const double drift = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.0) - initial;
        std::cout << "diagonal " << velocity.x << ',' << velocity.y << " fraction_error=" << fraction_error
                  << " velocity_error=" << velocity_error << " volume_drift=" << drift << '\n';
        require(fraction_error < 2e-6, "GPU geometric transport differs from CPU");
        require(velocity_error < 2e-4, "GPU density changes altered uniform motion");
        require(std::abs(drift) < initial * 1e-6, "GPU uniform transport lost liquid");
    }
}

void transportPreservesPureMaterial()
{
    constexpr int size = 19;
    FluidGridGPU grid(size, size, 4);
    FluidGridGPU::State state;
    state.liquid_fraction.assign(size * size, 1.0f);
    state.u.resize((size + 1) * size);
    state.v.resize(size * (size + 1));
    // Differences of a streamfunction give a closed, circulating flow without a pressure solve.
    auto stream = [](int x, int y) {
        return x == 0 || y == 0 || x == size || y == size ? 0.0f : float(30 * std::sin(x * 3.141592653589793 / size) * std::sin(y * 3.141592653589793 / size));
    };
    for (int y = 0; y < size; ++y)
    for (int x = 0; x <= size; ++x) state.u[grid.uIndex(x, y)] = stream(x, y + 1) - stream(x, y);
    for (int y = 0; y <= size; ++y)
    for (int x = 0; x < size; ++x) state.v[grid.vIndex(x, y)] = stream(x, y) - stream(x + 1, y);
    uploadTestMaterials(grid, state);
    grid.advect(0.137);
    for (float value : grid.download().liquid_fraction)
        require(value == 1.0f, "GPU transport accumulated roundoff in a pure material");
}

void transportKeepsStraightInterfacesSharp()
{
    FluidGridGPU gpu(40, 8, 1);
    FluidGridGPU::State state;
    state.liquid_fraction.resize(40 * 8);
    state.u.assign(41 * 8, 1);
    for (int y = 0; y < 8; ++y)
    for (int x = 8; x < 16; ++x) state.liquid_fraction[y * 40 + x] = 1;
    uploadTestMaterials(gpu, state);
    gpu.advect(8);
    state = gpu.download();
    for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 40; ++x)
        require(std::abs(state.liquid_fraction[y * 40 + x] - (x >= 16 && x < 24 ? 1.0 : 0.0)) < 1e-6,
                "GPU whole-cell translation blurred a straight interface");
}

void transportKeepsShallowInterfacesStraight()
{
    double largest_error = 0.0;
    FluidGridGPU grid(96, 96, 1.0);
    for (bool vertical : {false, true})
    for (bool reverse_phase : {false, true})
    for (double slope : {0.0, -0.1, -0.05, 0.05, 0.1})
    for (double speed : {-1.0, 1.0})
    {
        FluidGridGPU::State state;
        state.liquid_fraction.resize(96 * 96);
        auto& flow = vertical ? state.v : state.u;
        flow.assign(96 * 97, float(speed));

        auto exactFraction = [&](double along, double depth)
        {
            // Integrate the clipped liquid height across a cell under a known straight line.
            auto integral = [](double h) { return h <= 0.0 ? 0.0 : h >= 1.0 ? h - 0.5 : 0.5 * h * h; };
            const double a = depth + 1.0 - (48.25 + slope * (along - 48.0));
            const double b = a - slope;
            const double fraction = slope == 0.0 ? std::clamp(a, 0.0, 1.0) : (integral(a) - integral(b)) / slope;
            return reverse_phase ? 1.0 - fraction : fraction;
        };
        for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
            state.liquid_fraction[y * grid.width + x] = float(vertical ? exactFraction(y, x) : exactFraction(x, y));
        uploadTestMaterials(grid, state);

        for (int step = 0; step < 20; ++step)
            grid.advect(0.25);
        state = grid.download();

        // Inspect the interior, where uniform translation is unaffected by the closed walls.
        for (int y = 24; y < 72; ++y)
        for (int x = 24; x < 72; ++x)
        {
            const double expected = vertical ? exactFraction(y - 5.0 * speed, x) : exactFraction(x - 5.0 * speed, y);
            largest_error = std::max(largest_error, std::abs(state.liquid_fraction[y * grid.width + x] - expected));
        }
    }
    std::cout << "GPU shallow-interface fraction error=" << largest_error << '\n';
    require(largest_error < 1e-5, "GPU uniform transport corrugated a shallow straight interface");
}

void nonuniformTransportMatchesCPU()
{
    const int w = 32, h = 24;
    auto state = pool(w, h);
    state.u.resize((w + 1) * h);
    state.v.resize(w * (h + 1));
    // differences of one stream function give a closed, divergence-free swirling flow
    auto stream = [&](int x, int y) { return 80.0 * std::sin(3.141592653589793 * x / w) * std::sin(3.141592653589793 * y / h); };
    for (int y = 0; y < h; ++y)
    for (int x = 0; x <= w; ++x) state.u[y * (w + 1) + x] = float(stream(x, y + 1) - stream(x, y));
    for (int y = 0; y <= h; ++y)
    for (int x = 0; x < w; ++x) state.v[y * w + x] = float(stream(x, y) - stream(x + 1, y));
    FluidGrid cpu(w, h, 1);
    cpu.liquid_fraction.assign(state.liquid_fraction.begin(), state.liquid_fraction.end());
    cpu.u.assign(state.u.begin(), state.u.end());
    cpu.v.assign(state.v.begin(), state.v.end());
    initializeTestMaterials(cpu);
    FluidGridGPU gpu(w, h, 1);
    uploadTestMaterials(gpu, state);
    cpu.advect(0.1);
    gpu.advect(0.1);
    state = gpu.download();
    double fraction_error = 0, velocity_error = 0;
    for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
        fraction_error = std::max(fraction_error, std::abs(state.liquid_fraction[i] - cpu.liquid_fraction[i]));
    for (size_t i = 0; i < state.u.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.u[i] - cpu.u[i]));
    for (size_t i = 0; i < state.v.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.v[i] - cpu.v[i]));
    std::cout << "nonuniform transport fraction_error=" << fraction_error << " velocity_error=" << velocity_error << '\n';
    require(fraction_error < 2e-6 && velocity_error < 0.005, "GPU nonuniform momentum transport differs from CPU");
}

void fullStepsMatchCPU()
{
    FluidGrid cpu(32, 24, 4);
    FluidGridGPU gpu(32, 24, 4);
    auto state = pool(32, 24);
    cpu.liquid_fraction.assign(state.liquid_fraction.begin(), state.liquid_fraction.end());
    initializeTestMaterials(cpu);
    uploadTestMaterials(gpu, state);
    double fraction_error = 0, velocity_error = 0;
    for (int tick = 0; tick < 20; ++tick)
    {
        require(cpu.step(1.0 / 120, 2000, 1e-8).converged, "CPU step reference failed");
        require(gpu.step(1.0 / 120, 2000, 1e-3).converged, "GPU comparison step failed");
        state = gpu.download();
        for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
            fraction_error = std::max(fraction_error, std::abs(state.liquid_fraction[i] - cpu.liquid_fraction[i]));
        for (size_t i = 0; i < state.u.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.u[i] - cpu.u[i]));
        for (size_t i = 0; i < state.v.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.v[i] - cpu.v[i]));
    }
    std::cout << "20 full steps fraction_error=" << fraction_error << " velocity_error=" << velocity_error << '\n';
    require(fraction_error < 2e-4 && velocity_error < 0.01, "GPU full steps differ from CPU reference");
}

void stageTimingPreservesState()
{
    FluidGridGPU timed(32, 24, 4), plain(32, 24, 4);
    const auto initial = pool(32, 24);
    uploadTestMaterials(timed, initial);
    uploadTestMaterials(plain, initial);
    timed.gas_damping_rate = plain.gas_damping_rate = .1;
    FluidGridGPU::StepTimings timings;
    for (int tick = 0; tick < 8; ++tick)
    {
        const auto a = timed.step(1. / 120, 2000, 1e-3, &timings);
        const auto b = plain.step(1. / 120, 2000, 1e-3);
        require(a.converged && b.converged && a.iterations == b.iterations && a.max_divergence == b.max_divergence,
                "Stage timing changed uniform projection");
        const auto observed = timed.download(), expected = plain.download();
        require(observed.liquid_fraction == expected.liquid_fraction && observed.liquid_mass == expected.liquid_mass &&
                observed.gas_mass == expected.gas_mass && observed.u == expected.u && observed.v == expected.v &&
                observed.pressure == expected.pressure, "Stage timing changed uniform state");
    }
    timed.step(0., 2000, 1e-3, &timings);
    require(timings.transport_ms == 0 && timings.forces_ms == 0 && timings.pressure_ms == 0,
            "Zero-time step retained earlier stage timings");
    std::cout << "stage timing: 8 timed/untimed uniform steps are identical\n";
}

void tinyFractionsRemainFinite()
{
    for (float amount : { 1e-8f, 1e-20f, 1e-35f })
    {
        FluidGridGPU gpu(8, 8, 1);
        FluidGridGPU::State state;
        state.liquid_fraction.resize(64);
        state.liquid_fraction[3 * 8 + 3] = amount;
        state.liquid_fraction[3 * 8 + 4] = amount * 0.5f;
        state.liquid_fraction[4 * 8 + 3] = amount * 0.25f;
        state.u.assign(9 * 8, 1);
        state.v.assign(8 * 9, 0.75f);
        uploadTestMaterials(gpu, state);
        gpu.advect(0.25);
        state = gpu.download();
        for (const auto* field : { &state.liquid_fraction, &state.u, &state.v })
            for (float value : *field) require(std::isfinite(value), "tiny liquid fractions produced non-finite transport");
    }
    // this near-empty remnant reproduced a float 0/0 during the sloshing scene
    FluidGridGPU gpu(8, 8, 4);
    FluidGridGPU::State state;
    state.liquid_fraction.resize(64);
    state.liquid_fraction[3 * 8 + 3] = 1.654361225e-24f;
    state.liquid_fraction[3 * 8 + 4] = 1.051768366e-22f;
    state.liquid_fraction[4 * 8 + 3] = 6.335743373e-20f;
    state.u.assign(9 * 8, -28.16068459f);
    state.v.assign(8 * 9, 0.01431973279f);
    uploadTestMaterials(gpu, state);
    gpu.advect(1.0 / 120);
    state = gpu.download();
    for (const auto* field : { &state.liquid_fraction, &state.u, &state.v })
        for (float value : *field) require(std::isfinite(value), "near-empty sloshing cell produced non-finite transport");
}

// Vary ingredient proportions without adding extra liquid/gas interfaces.
template<class Grid>
typename Grid::State componentPool(Grid& grid, int liquid_count, int gas_count, bool varied_density = true)
{
    using Scalar = typename Grid::Scalar;
    const auto water = grid.elements.add({"Water", {0.1f, 0.75f, 0.95f}});
    const auto red = grid.elements.add({"Red", {0.95f, 0.25f, 0.15f}});
    const auto air = grid.elements.add({"Air", {0.015f, 0.025f, 0.06f}});
    const auto cloud = grid.elements.add({"Cloud", {0, 1, 1}});
    for (int k = 0; k < liquid_count; ++k)
        grid.liquid_mixture.components.push_back({k % 2 ? red : water, varied_density && k == 1 ? 200.0 : 100.0});
    for (int k = 0; k < gas_count; ++k)
        grid.gas_mixture.components.push_back({k == 0 ? air : k == 1 ? cloud : water, varied_density && k > 1 ? 0.2 : 0.1});
    typename Grid::State state;
    auto fractions = pool(grid.width, grid.height).liquid_fraction;
    if (liquid_count == 0) std::fill(fractions.begin(), fractions.end(), 0.0f);
    if (gas_count == 0) std::fill(fractions.begin(), fractions.end(), 1.0f);
    state.liquid_fraction.assign(fractions.begin(), fractions.end());
    const double volume = grid.cell_size * grid.cell_size;
    auto seed = [&](const auto& mixture, auto& fields, bool liquid)
    {
        const size_t count = mixture.components.size();
        fields.assign(count, std::vector<Scalar>(fractions.size()));
        for (size_t i = 0; i < fractions.size(); ++i)
        {
            std::vector<double> weights(count);
            double sum = 0;
            for (size_t k = 0; k < count; ++k)
            {
                // Include an identically empty ingredient when the fixture has room.
                weights[k] = varied_density && count > 2 && k + 1 == count ? 0.0 :
                    1.0 + 0.8 * std::sin(0.17 * (i % grid.width) + 0.11 * (i / grid.width) + k * 1.7);
                sum += weights[k];
            }
            for (size_t k = 0; k < count; ++k)
                fields[k][i] = Scalar(mixture.components[k].density * volume *
                    (liquid ? fractions[i] : 1.0 - fractions[i]) * weights[k] / sum);
        }
    };
    seed(grid.liquid_mixture, state.liquid_mass, true);
    seed(grid.gas_mixture, state.gas_mass, false);
    return state;
}

struct ComponentErrors { double volume = 0, drift = 0, negative = 0; };
ComponentErrors checkComponents(const FluidGridGPU& grid, const FluidGridGPU::State& state,
    const FluidGridGPU::State& initial, double volume_tolerance = 5e-6)
{
    ComponentErrors error;
    auto check = [&](const auto& mixture, const auto& fields, const auto& original, bool liquid)
    {
        std::vector<double> fractions(state.liquid_fraction.size());
        for (size_t k = 0; k < fields.size(); ++k)
        {
            const double scale = mixture.components[k].density * grid.cell_size * grid.cell_size;
            const double before = std::accumulate(original[k].begin(), original[k].end(), 0.0);
            const double after = std::accumulate(fields[k].begin(), fields[k].end(), 0.0);
            if (before == 0) require(after == 0, "an empty component gained mass");
            else error.drift = std::max(error.drift, std::abs(after - before) / before);
            for (size_t i = 0; i < fractions.size(); ++i)
            {
                require(std::isfinite(fields[k][i]), "non-finite component mass");
                const double fraction = fields[k][i] / scale;
                fractions[i] += fraction;
                error.negative = std::max(error.negative, -fraction);
            }
        }
        for (size_t i = 0; i < fractions.size(); ++i)
            error.volume = std::max(error.volume, std::abs(fractions[i] - (liquid ? state.liquid_fraction[i] : 1.0 - state.liquid_fraction[i])));
    };
    check(grid.liquid_mixture, state.liquid_mass, initial.liquid_mass, true);
    check(grid.gas_mixture, state.gas_mass, initial.gas_mass, false);
    // Check local volume accuracy separately from conservation of each ingredient's total mass.
    require(error.volume < volume_tolerance && error.drift < 1e-4 && error.negative < 2e-5, "component mass or phase volume invariant failed");
    return error;
}

void componentsMatchCPU()
{
    for (auto [liquids, gases] : {std::pair{1, 0}, std::pair{0, 1}, std::pair{2, 2}, std::pair{3, 4}, std::pair{9, 9}})
    {
        FluidGridGPU gpu(31, 23, 2.5);
        FluidGrid cpu(31, 23, 2.5);
        auto initial = componentPool(gpu, liquids, gases);
        auto reference = componentPool(cpu, liquids, gases);
        // Start from exactly the same representable masses, so the comparison isolates arithmetic.
        for (size_t k = 0; k < initial.liquid_mass.size(); ++k)
            reference.liquid_mass[k].assign(initial.liquid_mass[k].begin(), initial.liquid_mass[k].end());
        for (size_t k = 0; k < initial.gas_mass.size(); ++k)
            reference.gas_mass[k].assign(initial.gas_mass[k].begin(), initial.gas_mass[k].end());
        gpu.upload(initial);
        cpu.upload(reference);
        const auto roundtrip = gpu.download();
        require(roundtrip.liquid_mass == initial.liquid_mass && roundtrip.gas_mass == initial.gas_mass, "packed component upload/download changed mass");
        double mass_error = 0, velocity_error = 0;
        for (int tick = 0; tick < 20; ++tick)
        {
            require(cpu.step(1.0 / 120, 2000, 1e-8).converged, "CPU component reference failed");
            require(gpu.step(1.0 / 120, 2000, 1e-3).converged, "GPU component projection failed");
            const auto state = gpu.download();
            checkComponents(gpu, state, initial);
            auto compare = [&](const auto& a, const auto& b, const auto& mixture) {
                for (size_t k = 0; k < a.size(); ++k)
                    for (size_t i = 0; i < a[k].size(); ++i)
                        mass_error = std::max(mass_error, std::abs(a[k][i] - b[k][i]) / (mixture.components[k].density * 6.25));
            };
            compare(state.liquid_mass, cpu.liquid_mass, cpu.liquid_mixture);
            compare(state.gas_mass, cpu.gas_mass, cpu.gas_mixture);
            for (size_t i = 0; i < state.u.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.u[i] - cpu.u[i]));
            for (size_t i = 0; i < state.v.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.v[i] - cpu.v[i]));
        }
        std::cout << "components=" << liquids + gases << " mass_fraction_error=" << mass_error << " velocity_error=" << velocity_error << '\n';
        require(mass_error < 2e-4 && velocity_error < 0.01, "GPU components differ from CPU reference");
        const auto pixels=gpu.colorPixels();
        const auto current=gpu.download();
        for (size_t i=0; i<current.liquid_fraction.size(); ++i)
        {
            std::array<double,3> expected={};
            for (auto phase : {std::pair{&gpu.liquid_mixture,&current.liquid_mass},std::pair{&gpu.gas_mixture,&current.gas_mass}})
                for (size_t k=0; k<phase.first->components.size(); ++k)
                {
                    const auto& component=phase.first->components[k];
                    for (int c=0; c<3; ++c) expected[c]+=(*phase.second)[k][i]*gpu.elements[component.element].color[c]/(component.density*gpu.cell_size*gpu.cell_size);
                }
            for (int c=0; c<3; ++c) require(std::abs(pixels[4*i+c]-expected[c])<5e-6,"GPU catalogue colors differ from masses");
            require(pixels[4*i+3]==1,"GPU catalogue alpha differs");
        }

    }
}

void vanishingSliversCannotExportExcessMass()
{
    // This develops tiny liquid remnants that previously produced negative density at tick 120.
    FluidGridGPU grid(512, 512, 4);
    const auto initial = componentPool(grid, 2, 2, false);
    grid.upload(initial);
    for (int tick = 0; tick < 140; ++tick)
        require(grid.step(1.0 / 120, 2000, 1e-3).converged, "vanishing donor sliver destabilized component transport");
    checkComponents(grid, grid.download(), initial);
}

void closedPool(int steps, int size, double tolerance, int count)
{
    FluidGridGPU grid(size, size, 4);
    auto state = componentPool(grid, count / 2, count - count / 2);
    grid.upload(state);
    const auto initial = state;
    ComponentErrors component_errors;
    const double initial_volume = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.0);
    // use the MAC faces' dual-cell masses for kinetic energy, and the box bottom as zero potential
    auto energy = [&](const FluidGridGPU::State& sample)
    {
        auto density = [&](int x, int y) {
            double mass = 0; for (const auto* phase : { &sample.liquid_mass, &sample.gas_mass }) for (const auto& component : *phase) mass += component[y * size + x];
            return mass / (grid.cell_size * grid.cell_size);
        };
        double total = 0;
        for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
        {
            total += density(x, y) * grid.gravity_y * grid.cell_size * (size - y - 0.5);
            if (x > 0)
            {
                double u = sample.u[y * (size + 1) + x];
                total += 0.25 * (density(x - 1, y) + density(x, y)) * u * u;
            }
            if (y > 0)
            {
                double v = sample.v[y * size + x];
                total += 0.25 * (density(x, y - 1) + density(x, y)) * v * v;
            }
        }
        return total * grid.cell_size * grid.cell_size;
    };
    const double initial_energy = energy(grid.download());
    double max_energy_ratio = 1;
    double largest_drift = 0, largest_divergence = 0, min_fraction = 1, max_fraction = 0, max_speed = 0;
    long long iterations = 0;
    double solve_seconds = 0;
    const auto started = std::chrono::steady_clock::now();
    for (int tick = 1; tick <= steps; ++tick)
    {
        const auto solve_start = std::chrono::steady_clock::now();
        const auto result = grid.step(1.0 / 120, 2000, tolerance);
        grid.synchronize();
        solve_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
        iterations += result.iterations;
        largest_divergence = std::max(largest_divergence, result.max_divergence);
        if (!result.converged)
        {
            std::cout << "pool failure tick=" << tick << " iterations=" << result.iterations << " divergence=" << result.max_divergence << '\n';
            const auto failed = grid.download();
            for (const auto& [name, field] : { std::pair{"liquid", &failed.liquid_fraction}, std::pair{"u", &failed.u},
                                             std::pair{"v", &failed.v}, std::pair{"pressure", &failed.pressure} })
                for (size_t i = 0; i < field->size(); ++i)
                    if (!std::isfinite((*field)[i])) { std::cout << "first non-finite " << name << " index=" << i << '\n'; break; }
        }
        require(result.converged, "GPU pool projection failed");
        state = grid.download();
        // Many CFL splits accumulate float mismatch between mass and independent phase geometry.
        // Long-run quality budget: 0.2% of one cell, with the same stricter global mass check.
        const auto errors = checkComponents(grid, state, initial, 2e-3);
        component_errors.volume = std::max(component_errors.volume, errors.volume);
        component_errors.drift = std::max(component_errors.drift, errors.drift);
        component_errors.negative = std::max(component_errors.negative, errors.negative);
        const double volume = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.0);
        largest_drift = std::max(largest_drift, std::abs(volume - initial_volume) / initial_volume);
        for (float f : state.liquid_fraction)
        {
            min_fraction = std::min(min_fraction, double(f));
            max_fraction = std::max(max_fraction, double(f));
            require(std::isfinite(f) && f >= -2e-5 && f <= 1.0 + 2e-5, "GPU fraction left [0,1]");
        }
        for (const auto* field : { &state.u, &state.v, &state.pressure })
            for (float value : *field) require(std::isfinite(value), "GPU pool has non-finite state");
        for (const auto* field : { &state.u, &state.v })
            for (float value : *field) max_speed = std::max(max_speed, double(std::abs(value)));
        for (int y = 0; y < size; ++y)
            require(state.u[y * (size + 1)] == 0 && state.u[y * (size + 1) + size] == 0, "GPU side wall leaked");
        for (int x = 0; x < size; ++x)
            require(state.v[x] == 0 && state.v[size * size + x] == 0, "GPU top or bottom wall leaked");
        require(largest_drift < 1e-4, "GPU pool lost more than 0.01 percent of its liquid");
        const double energy_ratio = energy(state) / initial_energy;
        max_energy_ratio = std::max(max_energy_ratio, energy_ratio);
        require(max_energy_ratio < 1.02, "GPU pool energy grew more than two percent above its initial value");
        if (tick <= 3 || tick % 120 == 0 || tick == steps)
        {
            double center_x = 0, center_y = 0;
            for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
            {
                center_x += state.liquid_fraction[y * size + x] * (x + 0.5) * grid.cell_size;
                center_y += state.liquid_fraction[y * size + x] * (y + 0.5) * grid.cell_size;
            }
            std::cout << "pool tick=" << tick << " time=" << tick / 120.0 << " max_divergence=" << largest_divergence
                      << " fraction=[" << min_fraction << ',' << max_fraction << "] max_relative_drift=" << largest_drift
                      << " max_speed=" << max_speed << " mean_iterations=" << double(iterations) / tick
                      << " centroid=" << center_x / volume << ',' << center_y / volume
                      << " energy_ratio=" << energy_ratio << " max_energy_ratio=" << max_energy_ratio
                      << " component_volume_error=" << component_errors.volume << " component_drift=" << component_errors.drift << " negative_mass_fraction=" << component_errors.negative
                      << " physics_seconds=" << solve_seconds << " elapsed=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n';
        }
    }
}
}

int main(int argc,char** argv)
{
    try
    {
        std::cout<<std::unitbuf; std::cerr<<std::unitbuf;
        if (argc==2 && std::string(argv[1])=="--reductions")
        {
            pressureReductionsMatchHost();
            std::cout<<"PASS WebGPU reductions\n"; return 0;
        }
        if (argc>=3 && std::string(argv[1])=="--pool-steps")
        {
            closedPool(std::stoi(argv[2]),argc>4?std::stoi(argv[4]):64,argc>3?std::stod(argv[3]):1e-3,argc>5?std::stoi(argv[5]):4);
            std::cout<<"PASS WebGPU pool\n"; return 0;
        }
        if (argc==2 && std::string(argv[1])=="--smoke")
        {
            pressureBalancesGravity(); zeroTimeAndFailedProjectionAreHandled();
            stageTimingPreservesState();
            std::cout<<"PASS WebGPU smoke\n"; return 0;
        }
        if (argc==2 && std::string(argv[1])=="--multigrid")
        {
            multigridPreservesPCGContract();
            std::cout<<"PASS WebGPU multigrid\n"; return 0;
        }
        if (argc==2 && std::string(argv[1])=="--projection")
        {
            projectionMatchesCPU(); largeMixedPressureConvergesWithinDefaultBudget();
            std::cout<<"PASS WebGPU projection\n"; return 0;
        }
        if (argc==2 && std::string(argv[1])=="--transport")
        {
            tinyFractionsRemainFinite(); transportMatchesCPU(); transportPreservesPureMaterial();
            transportKeepsStraightInterfacesSharp(); transportKeepsShallowInterfacesStraight();
            nonuniformTransportMatchesCPU(); fullStepsMatchCPU(); componentsMatchCPU();
            std::cout<<"PASS WebGPU transport\n"; return 0;
        }
        pressureReductionsMatchHost();
        pressureBalancesGravity();
        multigridPreservesPCGContract();
        largeMixedPressureConvergesWithinDefaultBudget();
        cachedVelocityStatsRemainCurrent();
        commonStateAndColorReadback();
        fluid::test::materialEditingIsSparseAndReusable<FluidGridGPU>();
        fluid::test::bulkMaterialEditsPreserveEveryCell<FluidGridGPU>();
        fluid::test::repeatedMaterialEditsPreservePhaseVolumes<FluidGridGPU>();
        projectionMatchesCPU();
        tinyFractionsRemainFinite();
        transportMatchesCPU();
        transportPreservesPureMaterial();
        transportKeepsStraightInterfacesSharp();
        transportKeepsShallowInterfacesStraight();
        nonuniformTransportMatchesCPU();
        fullStepsMatchCPU();
        stageTimingPreservesState();
        componentsMatchCPU();
        vanishingSliversCannotExportExcessMass();
        zeroTimeAndFailedProjectionAreHandled();
        std::cout<<"PASS WebGPU regressions\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr<<"FAIL "<<error.what()<<'\n'; return 1; }
}
