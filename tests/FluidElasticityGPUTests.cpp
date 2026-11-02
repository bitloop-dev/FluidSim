#include "FluidFieldInternal.h"
#include "FluidGrid.h"
#include "UniformCPU/FluidGridCPU.h"
#include "UniformWebGPU/ElasticityGPU.h"
#include "UniformWebGPU/FluidGridGPU.h"
#include <chrono>
#include <cstring>
#include <cmath>
#include <iostream>
#include <numeric>

using fluid::FluidGridCPU;
using fluid::FluidGridGPU;
using namespace fluid::webgpu;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void configure(FluidGridCPU& grid, double modulus)
{
    fluid::SubstanceCatalogue catalogue;
    const auto solid = catalogue.add({"Elastic", {0.2f, 0.8f, 0.6f}});
    const auto gas = catalogue.add({"Gas", {0, 0, 0}});
    grid.configureMaterials(std::move(catalogue), {{{solid, 10, 0, modulus}}}, {{{gas, 0.01}}});
    grid.gravity_y = 0;
}

void configure(FluidGridGPU& grid, const FluidGridCPU& source, const fluid::FluidWebGPUContext& context)
{
    grid.setWebGPUContext(context);
    grid.substances = source.substances;
    grid.liquid_mixture = source.liquid_mixture;
    grid.gas_mixture = source.gas_mixture;
    grid.gravity_y = 0;
}

FluidGridCPU::State block(int n, double dx, double stretch = 1.15, bool history = true)
{
    FluidGridCPU::State state;
    state.liquid_fraction.resize(n * n);
    state.liquid_mass.assign(1, std::vector<double>(n * n));
    state.gas_mass.assign(1, std::vector<double>(n * n));
    if (history)
    {
        state.elastic_log_xx.assign(n * n, 2 * std::log(stretch));
        state.elastic_log_xy.assign(n * n, 0);
        state.elastic_log_yy.assign(n * n, -2 * std::log(stretch));
    }
    const double side = 0.3 * n;
    auto overlap = [](int x, double side, double center)
    {
        return std::max(0.0, std::min(x + 1.0, center + side / 2) - std::max(double(x), center - side / 2));
    };
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            const int i = y * n + x;
            const double f = overlap(x, side * stretch, n * 0.5) * overlap(y, side / stretch, n * 0.5);
            state.liquid_fraction[i] = f;
            state.liquid_mass[0][i] = 10 * f * dx * dx;
            state.gas_mass[0][i] = 0.01 * (1 - f) * dx * dx;
        }
    }
    return state;
}

template <class A, class B> double difference(const A& a, const B& b)
{
    require(a.size() == b.size(), "state size mismatch");
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i)
        error = std::max(error, std::abs(double(a[i]) - double(b[i])));
    return error;
}

void evolving(const fluid::FluidWebGPUContext& context)
{
    constexpr int n = 40;
    FluidGridCPU cpu(n, n, 1);
    configure(cpu, 100);
    FluidGridGPU gpu(n, n, 1);
    configure(gpu, cpu, context);
    const auto initial = fluid::convertFluidState<float>(block(n, 1));
    cpu.upload(fluid::convertFluidState<double>(initial));
    gpu.upload(initial);
    for (int i = 0; i < 100; ++i)
    {
        require(cpu.step(0.01, 500, 1e-5).converged, "CPU evolving projection failed");
        require(gpu.step(0.01, 500, 1e-5).converged, "GPU evolving projection failed");
    }
    const auto state = gpu.download();
    const double fraction_error = difference(state.liquid_fraction, cpu.liquid_fraction);
    const double velocity_error = std::max(difference(state.u, cpu.u), difference(state.v, cpu.v));
    std::cout << "evolving fraction=" << fraction_error << " velocity=" << velocity_error << '\n';
    require(fraction_error < 1e-4 && velocity_error < 1e-4, "GPU elastic trajectory differs from CPU");

    fluid::FluidCheckpoint checkpoint;
    gpu.saveCheckpoint(checkpoint);
    require(checkpoint.metadata.at("version") == 10, "elastic checkpoint version");
    FluidGridGPU restored(n, n, 1);
    restored.setWebGPUContext(context);
    restored.loadCheckpoint(checkpoint);
    require(restored.liquid_mixture.components[0].shear_modulus == 100, "checkpoint lost stiffness");
    for (int i = 0; i < 20; ++i)
    {
        require(gpu.step(0.01, 500, 1e-5).converged, "checkpoint source step failed");
        require(restored.step(0.01, 500, 1e-5).converged, "checkpoint restored step failed");
    }
    const auto a = gpu.download();
    const auto b = restored.download();
    require(a.u == b.u && a.v == b.v && a.elastic_log_xx == b.elastic_log_xx && a.elastic_log_xy == b.elastic_log_xy &&
                a.elastic_log_yy == b.elastic_log_yy,
        "native checkpoint changed elastic continuation");

    restored.upload(a);
    require(gpu.step(0.01, 500, 1e-5).converged && restored.step(0.01, 500, 1e-5).converged,
        "portable elastic continuation failed");
    const auto portable = restored.download();
    const auto uninterrupted = gpu.download();
    require(difference(portable.u, uninterrupted.u) < 1e-4 &&
                difference(portable.elastic_log_xx, uninterrupted.elastic_log_xx) < 1e-4,
        "portable state lost elastic history");

    gpu.liquid_mixture.components[0].shear_modulus = 0;
    gpu.upload(fluid::convertFluidState<float>(block(n, 1, 1, false)));
    require(gpu.download().elastic_log_xx.empty(), "zero stiffness retained elastic history");
    require(gpu.step(0.01, 500, 1e-5).converged, "zero stiffness restart failed");

    gpu.liquid_mixture.components[0].shear_modulus = 10000;
    gpu.liquid_mixture.components[0].dynamic_viscosity = 2;
    gpu.upload(initial);
    const auto stiff = gpu.step(0.03, 1000, 1e-5);
    require(stiff.converged && stiff.physical_steps > 1 && stiff.viscosity_iterations > 0,
        "elastic stability subdivisions or viscosity coupling failed");

    fluid::FluidGrid facade(n, n, 1);
    facade.setDevice(fluid::FluidDevice::GPU);
    facade.configureMaterials(cpu.substances, cpu.liquid_mixture, cpu.gas_mixture);
    facade.upload(block(n, 1));
    require(facade.step(0.01, 500, 1e-5).converged, "GPU facade elasticity failed");
}

void oscillation(const fluid::FluidWebGPUContext& context)
{
    constexpr int n = 32;
    constexpr double dx = 1.5;
    FluidGridCPU materials(n, n, dx);
    configure(materials, 100);
    FluidGridGPU gpu(n, n, dx);
    configure(gpu, materials, context);
    const auto initial = fluid::convertFluidState<float>(block(n, dx));
    gpu.upload(initial);
    const double mass = std::accumulate(initial.liquid_mass[0].begin(), initial.liquid_mass[0].end(), 0.0);
    double minimum_aspect = 2;
    double maximum_volume_error = 0;
    for (int step = 1; step <= 600; ++step)
    {
        require(gpu.step(0.01, 1000, 1e-5).converged, "GPU recovery cycle projection failed");
        if (step % 10 != 0)
            continue;

        const auto state = gpu.download();
        double xx = 0;
        double yy = 0;
        double volume = 0;
        double squared_error = 0;
        for (int y = 1; y + 1 < n; ++y)
        {
            for (int x = 1; x + 1 < n; ++x)
            {
                const int i = y * n + x;
                const double f = state.liquid_fraction[i];
                require(f >= 0 && f <= 1, "GPU recovery fraction is unbounded");
                require(std::abs(state.liquid_mass[0][i] / (10 * dx * dx) - f) < 2e-5,
                    "GPU recovery mass/phase inconsistency");
                xx += f * std::pow(x + 0.5 - n * 0.5, 2);
                yy += f * std::pow(y + 0.5 - n * 0.5, 2);
                if (f <= 1e-12)
                    continue;

                const double error = std::exp(0.5 * (state.elastic_log_xx[i] + state.elastic_log_yy[i])) - 1;
                maximum_volume_error = std::max(maximum_volume_error, std::abs(error));
                squared_error += f * error * error;
                volume += f;
            }
        }
        require(maximum_volume_error < 0.15 && std::sqrt(squared_error / volume) < 0.025,
            "GPU first-cycle reference-volume error exceeded experimental budget");
        minimum_aspect = std::min(minimum_aspect, std::sqrt(xx / yy));
        const double current_mass = std::accumulate(state.liquid_mass[0].begin(), state.liquid_mass[0].end(), 0.0);
        require(std::abs(current_mass / mass - 1) < 2e-5, "GPU recovery mass drift");
    }
    require(minimum_aspect < 0.9, "GPU block did not recover through its relaxed shape");
    std::cout << "recovery cycle minimum aspect=" << minimum_aspect << " Jmax=" << maximum_volume_error << '\n';
}

void pressurePrecision(const fluid::FluidWebGPUContext& context)
{
    constexpr int n = 32;
    FluidGridCPU materials(n, n, 1);
    configure(materials, 0);
    FluidGridGPU grid(n, n, 1);
    configure(grid, materials, context);
    auto state = fluid::convertFluidState<float>(block(n, 1, 1, false));
    std::fill(state.liquid_fraction.begin(), state.liquid_fraction.end(), 1.0f);
    std::fill(state.liquid_mass[0].begin(), state.liquid_mass[0].end(), 10.0f);
    std::fill(state.gas_mass[0].begin(), state.gas_mass[0].end(), 0.0f);
    state.u.resize((n + 1) * n);
    state.v.resize(n * (n + 1));
    const double k = std::acos(-1.0) / n;
    for (int y = 0; y < n; ++y)
    {
        for (int x = 1; x < n; ++x)
            state.u[y * (n + 1) + x] = float(3000 * std::sin(k * x) * std::cos(k * (y + 0.5)));
    }
    for (int y = 1; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
            state.v[y * n + x] = float(-3000 * std::cos(k * (x + 0.5)) * std::sin(k * y));
    }
    grid.upload(state);
    const auto result = grid.project(0.01, -1, 1e-9);
    require(result.accepted() && result.precision_limited && !result.converged && result.iterations < 100,
        "pressure did not stop honestly at float velocity precision");

    // Independently inspect the stored floats in double precision, with a local ulp budget.
    const auto projected = grid.download();
    auto ulp = [](float value)
    {
        const float magnitude = std::abs(value);
        return double(std::nextafter(magnitude, std::numeric_limits<float>::infinity())) - magnitude;
    };
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            const float left = projected.u[y * (n + 1) + x];
            const float right = projected.u[y * (n + 1) + x + 1];
            const float top = projected.v[y * n + x];
            const float bottom = projected.v[(y + 1) * n + x];
            const double div = (double(right) - left) + (double(bottom) - top);
            require(std::abs(div) <= 1e-9 + ulp(left) + ulp(right) + ulp(top) + ulp(bottom),
                "pressure accepted divergence beyond local rounding precision");
        }
    }

    // Fast remote cells must not excuse a large residual in a slow cell.
    state.u.assign(state.u.size(), 0);
    state.v.assign(state.v.size(), 0);
    state.u[10 * (n + 1) + 10] = 1;
    state.u[24 * (n + 1) + 25] = 1048576;
    state.u[25 * (n + 1) + 25] = -1048576;
    state.v[25 * n + 24] = -1048576;
    state.v[25 * n + 25] = 1048576;
    grid.upload(state);
    require(!grid.project(0.01, 0, 1e-9).accepted(), "roundoff allowance hid unresolved physical divergence");
    std::cout << "pressure precision iterations=" << result.iterations << " divergence=" << result.max_divergence
              << '\n';
}

void gravityContact(const fluid::FluidWebGPUContext& context)
{
    constexpr int n = 256;
    FluidGridCPU materials(n, n, 1);
    configure(materials, 1000);
    FluidGridGPU grid(n, n, 1);
    configure(grid, materials, context);
    grid.gravity_y = 1;
    const auto initial = fluid::convertFluidState<float>(block(n, 1));
    grid.upload(initial);
    const double initial_mass = std::accumulate(initial.liquid_mass[0].begin(), initial.liquid_mass[0].end(), 0.0);
    double lowest_center = 0;
    double rebound_center = n;
    int maximum_steps = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 1; i <= 3000; ++i)
    {
        const auto result = grid.step(.01, 256, 1e-5);
        require(result.accepted(), "gravity contact pressure failed");
        maximum_steps = std::max(maximum_steps, result.physical_steps);
        if (i % 100 != 0)
            continue;
        const auto state = grid.download();
        double mass = 0;
        double center = 0;
        for (int y = 0; y < n; ++y)
        {
            for (int x = 0; x < n; ++x)
            {
                const int cell = y * n + x;
                const double f = state.liquid_fraction[cell];
                const double m = state.liquid_mass[0][cell];
                require(f >= 0 && f <= 1 && std::abs(m / 10 - f) < 2e-5, "gravity contact phase inconsistency");
                require(std::isfinite(state.elastic_log_xx[cell]) && std::isfinite(state.elastic_log_xy[cell]) &&
                            std::isfinite(state.elastic_log_yy[cell]),
                    "gravity contact lost finite strain");
                mass += m;
                center += m * (y + .5);
            }
        }
        require(std::abs(mass / initial_mass - 1) < 2e-5, "gravity contact lost material mass");
        center /= mass;
        lowest_center = std::max(lowest_center, center);
        if (i > 1800)
            rebound_center = std::min(rebound_center, center);
    }
    require(lowest_center > .9 * n && rebound_center < .86 * n, "falling elastic block did not deform and rebound");
    std::cout << "gravity1 contact/rebound centers=" << lowest_center << '/' << rebound_center
              << " max physical steps=" << maximum_steps
              << " wall seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << '\n';
}

void legacyCheckpoint(const fluid::FluidWebGPUContext& context)
{
    constexpr int n = 32;
    FluidGridCPU source(n, n, 1);
    configure(source, 100);
    FluidGridGPU grid(n, n, 1);
    configure(grid, source, context);
    grid.upload(fluid::convertFluidState<float>(block(n, 1)));
    fluid::FluidCheckpoint checkpoint;
    grid.saveCheckpoint(checkpoint);

    // Version 9 stored displacement and extension validity in its final three-channel field.
    checkpoint.metadata["version"] = 9;
    std::vector<float> displacement(3 * n * n);
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            const int i = y * n + x;
            displacement[3 * i] = float(-.1 * (x + .5));
            displacement[3 * i + 1] = float(.2 * (y + .5));
            displacement[3 * i + 2] = 0;
        }
    }
    const size_t bytes = displacement.size() * sizeof(float);
    std::memcpy(checkpoint.data.data() + checkpoint.data.size() - bytes, displacement.data(), bytes);
    grid.loadCheckpoint(checkpoint);
    const auto state = grid.download();
    for (int i = 0; i < n * n; ++i)
    {
        if (state.liquid_fraction[i] <= 0)
            continue;
        require(std::abs(state.elastic_log_xx[i] + 2 * std::log(.9)) < 2e-6 &&
                    std::abs(state.elastic_log_yy[i] + 2 * std::log(1.2)) < 2e-6 &&
                    std::abs(state.elastic_log_xy[i]) < 2e-6,
            "legacy checkpoint conversion lost deformation or volume history");
    }
    require(grid.step(.01, 500, 1e-5).accepted(), "converted legacy checkpoint could not advance");

    // A checkpoint immediately after upload has no rebuilt exterior support yet.
    // Gas reference values were arbitrary in the old API; they cannot enter the material gradient.
    auto relaxed = fluid::convertFluidState<float>(block(n, 1, 1));
    auto save_unextended = [&]
    {
        grid.upload(relaxed);
        grid.saveCheckpoint(checkpoint);
        checkpoint.metadata["version"] = 9;
        for (int y = 0; y < n; ++y)
        {
            for (int x = 0; x < n; ++x)
            {
                const int i = y * n + x;
                displacement[3 * i] = relaxed.liquid_fraction[i] > 1e-12 ? 0 : float(-x - .5);
                displacement[3 * i + 1] = relaxed.liquid_fraction[i] > 1e-12 ? 0 : float(-y - .5);
                displacement[3 * i + 2] = 0;
            }
        }
        std::memcpy(checkpoint.data.data() + checkpoint.data.size() - bytes, displacement.data(), bytes);
    };
    save_unextended();
    grid.loadCheckpoint(checkpoint);
    const auto restored = grid.download();
    for (int i = 0; i < n * n; ++i)
    {
        require(restored.elastic_log_xx[i] == 0 && restored.elastic_log_xy[i] == 0 && restored.elastic_log_yy[i] == 0,
            "legacy unextended exterior changed relaxed material");
    }

    relaxed.liquid_fraction[0] = 1e-14f;
    relaxed.liquid_mass[0][0] = 1e-13f;
    save_unextended();
    bool rejected = false;
    try
    {
        grid.loadCheckpoint(checkpoint);
    }
    catch (const std::runtime_error& error)
    {
        rejected = std::string(error.what()).find("without defined reference history") != std::string::npos;
    }
    require(rejected, "legacy conversion silently invented strain for undefined history");
    std::cout << "version-9 reference checkpoint conversion passed\n";
}

double wave(const fluid::FluidWebGPUContext& context, int n)
{
    FluidGridCPU materials(n, n, 1);
    configure(materials, 100);
    FluidGridGPU grid(n, n, 1);
    configure(grid, materials, context);
    auto initial = fluid::convertFluidState<float>(block(n, 1, 1, false));
    std::fill(initial.liquid_fraction.begin(), initial.liquid_fraction.end(), 1.0f);
    std::fill(initial.liquid_mass[0].begin(), initial.liquid_mass[0].end(), 10.0f);
    std::fill(initial.gas_mass[0].begin(), initial.gas_mass[0].end(), 0.0f);
    initial.u.resize((n + 1) * n);
    initial.v.resize(n * (n + 1));
    const double pi = std::acos(-1.0);
    const double k = pi / n;
    constexpr double amplitude = 0.001;
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x <= n; ++x)
            initial.u[y * (n + 1) + x] = float(amplitude * std::sin(k * x) * std::cos(k * (y + 0.5)));
    }
    for (int y = 0; y <= n; ++y)
    {
        for (int x = 0; x < n; ++x)
            initial.v[y * n + x] = float(-amplitude * std::cos(k * (x + 0.5)) * std::sin(k * y));
    }
    grid.upload(initial);
    const double frequency = std::sqrt(20.0) * k;
    const double period = 2 * pi / frequency;
    const double norm = std::inner_product(initial.u.begin(), initial.u.end(), initial.u.begin(), 0.0) +
                        std::inner_product(initial.v.begin(), initial.v.end(), initial.v.begin(), 0.0);
    double error = 0;
    for (int i = 1; i <= 1800; ++i)
    {
        require(grid.step(period / 1800, 500, 1e-7).converged, "GPU wave projection failed");
        if (i % 10 != 0)
            continue;
        const auto state = grid.download();
        const double projection = (std::inner_product(initial.u.begin(), initial.u.end(), state.u.begin(), 0.0) +
                                      std::inner_product(initial.v.begin(), initial.v.end(), state.v.begin(), 0.0)) /
                                  norm;
        error = std::max(error, std::abs(projection - std::cos(frequency * period * i / 1800)));
    }
    std::cout << "wave n=" << n << " error=" << error << '\n';
    require(error < 0.035, "GPU wave amplitude/frequency inaccurate");
    return error;
}
}

int main()
{
    try
    {
        fluid::FluidWebGPUContext context;
        evolving(context);
        oscillation(context);
        pressurePrecision(context);
        const double coarse = wave(context, 20);
        const double fine = wave(context, 40);
        require(fine < 0.6 * coarse, "GPU wave did not converge under refinement");
        gravityContact(context);
        legacyCheckpoint(context);
        std::cout << "PASS GPU elasticity\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL GPU elasticity: " << error.what() << '\n';
        return 1;
    }
}
