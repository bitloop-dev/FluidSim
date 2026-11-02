#include "UniformWebGPU/FluidGridGPU.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace fluid;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void conservativeCompression(int count)
{
    constexpr int w = 32;
    constexpr int h = 8;
    constexpr double dt = 0.02;
    FluidGridGPU grid(w, h, 1.0);

    grid.gravity_y = 0.0;
    grid.setPressureRecoveryTime(0.1);

    FluidGridGPU::State state;
    state.liquid_fraction.assign(w * h, 0.3f);
    state.u.resize((w + 1) * h);
    state.v.assign(w * (h + 1), 0.0f);

    for (int y = 1; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
            state.v[y * w + x] = 2.0f;
    }
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x <= w; ++x)
            state.u[y * (w + 1) + x] = x == w ? 0.0f : float(std::sin(2.0 * 3.141592653589793 * x / w));
    }

    for (int k = 0; k < count; ++k)
    {
        const auto id = grid.substances.add({"Component", {0.2f, 0.4f, 0.6f}});
        const double density = k + 1.0;
        if (k % 2 == 0)
        {
            grid.liquid_mixture.components.push_back({id, density});
            state.liquid_mass.emplace_back(w * h, float(0.3 * density / ((count + 1) / 2)));
        }
        else
        {
            grid.gas_mixture.components.push_back({id, density});
            state.gas_mass.emplace_back(w * h, float(0.7 * density / (count / 2)));
        }
    }

    grid.upload(state);
    grid.advect(dt);
    const auto after = grid.download();

    for (int phase = 0; phase < 2; ++phase)
    {
        const auto& before_mass = phase == 0 ? state.liquid_mass : state.gas_mass;
        const auto& after_mass = phase == 0 ? after.liquid_mass : after.gas_mass;
        for (size_t k = 0; k < before_mass.size(); ++k)
        {
            const double before = std::accumulate(before_mass[k].begin(), before_mass[k].end(), 0.0);
            const double result = std::accumulate(after_mass[k].begin(), after_mass[k].end(), 0.0);

            require(std::abs(result / before - 1.0) < 2e-6, "Conservative compression changed component mass");
        }
    }

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const int i = y * w + x;
            double q = 0.0;
            for (size_t k = 0; k < after.liquid_mass.size(); ++k)
                q += after.liquid_mass[k][i] / grid.liquid_mixture.components[k].density;
            for (size_t k = 0; k < after.gas_mass.size(); ++k)
                q += after.gas_mass[k][i] / grid.gas_mixture.components[k].density;

            const double x_factor = 1.0 - dt * (state.u[y * (w + 1) + x + 1] - state.u[y * (w + 1) + x]);
            const double expected = x_factor * (1.0 - dt * (state.v[(y + 1) * w + x] - state.v[y * w + x]));

            require(std::abs(q - expected) < 2e-6, "Compression does not match conservative continuity");
            require(after.liquid_fraction[i] >= 0.0f && after.liquid_fraction[i] <= 1.0f, "Compressed geometry invalid");
            if (y == h / 2)
                require(std::abs(after.v[y * w + x] - 2.0f) < 2e-6, "Compression changed uniform transverse velocity");
        }
    }

    std::cout << "PASS conservative compression with " << count << " components\n";
}

void temporalRecovery(int frames, bool short_wave, bool contrast = false)
{
    constexpr int w = 64;
    constexpr int h = 8;
    constexpr double dt = 0.02 / 6.0;
    FluidGridGPU grid(w, h, 1.0);

    grid.gravity_y = 0.0;
    grid.setPressureRecoveryTime(frames * 0.02);
    const auto water = grid.substances.add({"Water", {0, 0, 1}});
    const double liquid_density = contrast ? 100.0 : 10.0;
    grid.liquid_mixture.components = {{water, liquid_density}};
    FluidGridGPU::State state;
    state.liquid_fraction.assign(w * h, 1.0f);
    state.liquid_mass.assign(1, std::vector<float>(w * h));
    if (contrast)
    {
        const auto gas = grid.substances.add({"Gas", {0, 1, 1}});
        grid.gas_mixture.components = {{gas, 0.1}};
        state.gas_mass.assign(1, std::vector<float>(w * h));
    }

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const double error = 0.02 * std::cos((short_wave ? 31.0 : 1.0) * 3.141592653589793 * (x + 0.5) / w);
            const double f = !contrast || x < w / 2 ? 1.0 : 0.0;
            state.liquid_fraction[y * w + x] = float(f);
            state.liquid_mass[0][y * w + x] = float(liquid_density * f * (1.0 + error));
            if (contrast)
                state.gas_mass[0][y * w + x] = float(0.1 * (1.0 - f) * (1.0 + error));
        }
    }

    grid.upload(state);
    const double initial_mass = std::accumulate(state.liquid_mass[0].begin(), state.liquid_mass[0].end(), 0.0);
    const double initial_gas = contrast ? std::accumulate(state.gas_mass[0].begin(), state.gas_mass[0].end(), 0.0) : 0.0;
    double initial_rms = 0.02 / std::sqrt(2.0);

    for (int tick = 1; tick <= frames * 6 * 3; ++tick)
    {
        const auto result = grid.step(dt, 2000, 0.001);

        require(result.accepted(), "Recovery projection broke down");

        const auto current = grid.download();
        double squares = 0.0;
        double max_error = 0.0;

        for (size_t i = 0; i < current.liquid_fraction.size(); ++i)
        {
            const float mass = current.liquid_mass[0][i];
            const float gas = contrast ? current.gas_mass[0][i] : 0.0f;

            require(
                std::isfinite(mass) && mass >= -1e-6 && std::isfinite(gas) && gas >= -1e-6, "Recovery mass invalid");

            const double e = mass / liquid_density + gas / 0.1 - 1.0;
            squares += e * e;
            max_error = std::max(max_error, std::abs(e));
        }

        const double mass = std::accumulate(current.liquid_mass[0].begin(), current.liquid_mass[0].end(), 0.0);

        require(std::abs(mass / initial_mass - 1.0) < 1e-5, "Recovery lost mass");
        if (contrast)
        {
            const double gas = std::accumulate(current.gas_mass[0].begin(), current.gas_mass[0].end(), 0.0);

            require(std::abs(gas / initial_gas - 1.0) < 1e-5, "Recovery lost gas");
        }

        const double ratio = std::sqrt(squares / (w * h)) / initial_rms;
        if (tick % 6 == 0 || (contrast && tick < 12))
        {
            std::cout << "recovery N=" << frames << " short=" << short_wave << " contrast=" << contrast
                      << " frame=" << tick / 6.0 << " error_ratio=" << ratio << " max_error=" << max_error
                      << " divergence=" << result.max_divergence << std::endl;
        }

        require(max_error < 0.5, "Recovery amplified volume error beyond 50 percent");
        if (tick == frames * 6)
            require(ratio < 0.3, "Recovery missed the requested settling horizon");
        if (tick == frames * 6 * 3)
            require(ratio < 0.05, "Recovery left a persistent volume disturbance");
    }
}

void checkpointRecovery()
{
    constexpr int w = 24;
    constexpr int h = 12;
    FluidGridGPU original(w, h, 1.0), resumed(w, h, 1.0);

    original.gravity_y = 0.0;
    original.setPressureRecoveryTime(0.1);
    const auto water = original.substances.add({"Water", {0, 0, 1}});
    original.liquid_mixture.components = {{water, 10.0}};
    FluidGridGPU::State state;
    state.liquid_fraction.assign(w * h, 1.0f);
    state.liquid_mass.assign(1, std::vector<float>(w * h));

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
            state.liquid_mass[0][y * w + x] = float(10.0 + 0.1 * std::cos(3.141592653589793 * (x + 0.5) / w));
    }

    original.upload(state);

    for (int i = 0; i < 11; ++i)
        require(original.step(0.003, 2000, 0.001).accepted(), "Checkpoint warmup failed");

    FluidCheckpoint checkpoint;
    original.saveCheckpoint(checkpoint);

    require(checkpoint.metadata.at("version") == 5, "Recovery checkpoint needs its own model version");

    auto invalid = checkpoint;
    invalid.metadata["preconditioner"] = int(FluidGridGPU::PressurePreconditioner::Diagonal);
    bool rejected = false;
    try
    {
        original.loadCheckpoint(invalid);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    FluidCheckpoint unchanged;
    original.saveCheckpoint(unchanged);

    require(rejected && unchanged.metadata == checkpoint.metadata && unchanged.data == checkpoint.data,
        "Invalid recovery checkpoint changed live state");

    resumed.loadCheckpoint(checkpoint);
    require(resumed.getPressureRecoveryTime() == 0.1, "Checkpoint lost recovery horizon");

    for (int i = 0; i < 30; ++i)
    {
        const auto a = original.step(0.003, 2000, 0.001);
        const auto b = resumed.step(0.003, 2000, 0.001);

        require(a.accepted() && b.accepted() && a.iterations == b.iterations, "Recovery cadence did not resume");

        FluidCheckpoint left, right;
        original.saveCheckpoint(left);
        resumed.saveCheckpoint(right);

        require(left.data == right.data, "Recovery checkpoint continuation differs");
    }

    std::cout << "PASS recovery checkpoint continues identical state and cadence\n";
}

void hydrostaticRecovery()
{
    constexpr int w = 33;
    constexpr int h = 25; // Also exercise partial 4x4 health-reduction tiles.
    FluidGridGPU grid(w, h, 1.0);

    grid.gravity_y = 10.0;
    grid.setSurfaceTension(100.0);
    grid.setPressureRecoveryTime(0.2);
    const auto water = grid.substances.add({"Water", {0, 0, 1}});
    const auto gas = grid.substances.add({"Gas", {0, 1, 1}});
    grid.liquid_mixture.components = {{water, 100.0}};
    grid.gas_mixture.components = {{gas, 0.1}};
    FluidGridGPU::State state;
    state.liquid_fraction.resize(w * h);
    state.liquid_mass.assign(1, std::vector<float>(w * h));
    state.gas_mass.assign(1, std::vector<float>(w * h));

    for (int i = 0; i < w * h; ++i)
    {
        const float f = i / w < h / 2 ? 1.0f : 0.0f;
        state.liquid_fraction[i] = f;
        state.liquid_mass[0][i] = 100.0f * f;
        state.gas_mass[0][i] = 0.1f * (1.0f - f);
    }

    grid.upload(state);
    double peak_speed = 0.0;
    double peak_error = 0.0;

    for (int i = 0; i < 300; ++i)
    {
        const auto result = grid.step(0.003, 2000, 0.001);

        require(result.accepted(), "Hydrostatic recovery broke down");
        peak_error = std::max(peak_error, result.max_volume_error);
        const auto current = grid.download();

        for (const auto* field : {&current.u, &current.v})
        {
            for (float value : *field)
                peak_speed = std::max(peak_speed, std::abs(double(value)));
        }
    }

    std::cout << "hydrostatic peak_speed=" << peak_speed << " volume_error=" << peak_error << '\n';

    require(peak_speed < 0.01 && peak_error < 0.001, "Recovery lost hydrostatic support");

    // An edit replaces material at reference volume; the derived occupancy must be invalidated.
    auto edit = grid.beginEdit();
    edit.setCell(3, 3, Phase::Gas, gas);
    edit.commit();
    std::vector<float> colors;
    grid.writeColorPixels(colors, true);

    require(colors[(3 * w + 3) * 4 + 3] == 0.0f, "Recovery edit retained stale phase geometry");
    require(grid.step(0.003, 2000, 0.001).accepted(), "Recovery failed after a material edit");
}

void expensiveFiniteTransport()
{
    constexpr int w = 8;
    constexpr int h = 8;
    constexpr double pi = 3.14159265358979323846;
    FluidGridGPU grid(w, h, 1.0);

    grid.gravity_y = 0.0;
    grid.setPressureRecoveryTime(0.1);
    const auto gas = grid.substances.add({"Gas", {0.0f, 0.0f, 0.0f}});
    grid.gas_mixture = {{{gas, 1.0}}};
    FluidGridGPU::State state;
    state.liquid_fraction.assign(w * h, 0.0f);
    state.gas_mass.assign(1, std::vector<float>(w * h, 1.0f));
    state.u.assign((w + 1) * h, 0.0f);
    state.v.assign(w * (h + 1), 0.0f);

    auto psi = [&](int x, int y)
    {
        return 10000.0 * std::sin(pi * x / w) * std::sin(pi * y / h);
    };

    for (int y = 0; y < h; ++y)
    {
        for (int x = 1; x < w; ++x)
            state.u[y * (w + 1) + x] = float(psi(x, y + 1) - psi(x, y));
    }
    for (int y = 1; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
            state.v[y * w + x] = float(psi(x, y) - psi(x + 1, y));
    }

    grid.upload(state);

    // This finite, closed-box circulation needs more than the former256 subdivisions.
    grid.advect(0.05);
    const auto after = grid.download();

    for (const auto* field : {&after.u, &after.v, &after.gas_mass[0]})
    {
        for (float value : *field)
            require(std::isfinite(value), "Expensive transport produced nonfinite state");
    }

    require(std::abs(std::accumulate(after.gas_mass[0].begin(), after.gas_mass[0].end(), 0.0) - w * h) < 0.001,
        "Expensive transport lost mass");

    for (int y = 0; y < h; ++y)
    {
        require(
            after.u[y * (w + 1)] == 0.0f && after.u[y * (w + 1) + w] == 0.0f, "Expensive transport opened side walls");
    }
    for (int x = 0; x < w; ++x)
        require(after.v[x] == 0.0f && after.v[h * w + x] == 0.0f, "Expensive transport opened top/bottom walls");

    for (auto& v : state.u)
        v *= 0.0001f;
    for (auto& v : state.v)
        v *= 0.0001f;
    state.u[4 * (w + 1) + 4] += 3.0f;

    for (bool relaxed : {false, true})
    {
        grid.destroy();
        grid.setPressureRecoveryTime(relaxed ? 0.1 : 0.0);
        grid.upload(state);

        require(grid.project(0.01, -1, 0.001).converged, "Unbounded projection failed to correct divergence");
        require(grid.step(0.01, -1, 0.001).accepted(), "Unbounded pressure failed at initialization");
        require(grid.step(0.01, -1, 0.001).accepted(), "Unbounded pressure changed the recovery work schedule");
    }

    std::cout << "PASS expensive finite transport and unbounded accurate/relaxed pressure\n";
}
} // namespace

int main(int, char**)
{
    try
    {
        conservativeCompression(4);
        conservativeCompression(20);
        expensiveFiniteTransport();

        for (int frames : {1, 2, 5, 10})
        {
            temporalRecovery(frames, false);
            temporalRecovery(frames, true);
        }

        temporalRecovery(5, false, true);
        temporalRecovery(10, false, true);
        checkpointRecovery();
        hydrostaticRecovery();

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
