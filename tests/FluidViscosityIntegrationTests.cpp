#include "FluidGrid.h"
#include "FluidFieldInternal.h"
#include "UniformWebGPU/FluidGridGPU.h"
#include "UniformWebGPU/ViscositySolverGPU.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace fluid;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

bool same(const FluidGridGPU::State& a, const FluidGridGPU::State& b)
{
    return a.u == b.u && a.v == b.v && a.pressure == b.pressure && a.liquid_fraction == b.liquid_fraction &&
           a.liquid_mass == b.liquid_mass && a.gas_mass == b.gas_mass;
}

void initialize(FluidGridGPU& grid, int liquids, int gases)
{
    for (int i = 0; i < liquids + gases; ++i)
    {
        const auto id = grid.substances.add({"ingredient", {1, 0, 0}});
        if (i < liquids)
            grid.liquid_mixture.components.push_back({id, 10.0 + i, 20.0 + 100.0 * i});
        else
            grid.gas_mixture.components.push_back({id, 0.1 + 0.01 * (i - liquids), 0.0});
    }

    grid.gravity_y = 0.0;
    const int w = grid.width;
    const int h = grid.height;
    const float area = float(grid.cell_size * grid.cell_size);
    FluidGridGPU::State state;
    state.liquid_fraction.resize(w * h);
    state.liquid_mass.assign(liquids, std::vector<float>(w * h));
    state.gas_mass.assign(gases, std::vector<float>(w * h));

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const int i = y * w + x;
            const float f = float(y + 1) / float(h + 1);
            state.liquid_fraction[i] = f;

            for (int k = 0; k < liquids; ++k)
                state.liquid_mass[k][i] = area * f * float(grid.liquid_mixture.components[k].density) / liquids;
            for (int k = 0; k < gases; ++k)
                state.gas_mass[k][i] = area * (1 - f) * float(grid.gas_mixture.components[k].density) / gases;
        }
    }

    auto psi = [=](int x, int y)
    {
        if (x == 0 || x == w || y == 0 || y == h)
            return 0.0;

        return 2.0 * std::sin(3.141592653589793 * x / w) * std::sin(3.141592653589793 * y / h);
    };

    state.u.resize((w + 1) * h);
    state.v.resize(w * (h + 1));

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x <= w; ++x)
            state.u[y * (w + 1) + x] = float(psi(x, y + 1) - psi(x, y));
    }
    for (int y = 0; y <= h; ++y)
    {
        for (int x = 0; x < w; ++x)
            state.v[y * w + x] = float(psi(x, y) - psi(x + 1, y));
    }

    grid.upload(state);
}

void blending(int liquids, int gases)
{
    FluidGridGPU grid(24, 18, 0.7);
    initialize(grid, liquids, gases);
    grid.setComponentViscosity(Phase::Gas, gases - 1, 5.0);
    const auto initial = grid.download();
    std::vector<float> material(2 * grid.width * grid.height);

    for (size_t i = 0; i < initial.liquid_fraction.size(); ++i)
    {
        double mass = 0.0;
        double volume = 0.0;
        double mu_volume = 0.0;

        for (int phase = 0; phase < 2; ++phase)
        {
            const auto& values = phase == 0 ? initial.liquid_mass : initial.gas_mass;
            const auto& mixture = phase == 0 ? grid.liquid_mixture : grid.gas_mixture;
            for (size_t k = 0; k < values.size(); ++k)
            {
                mass += values[k][i];
                const double v = values[k][i] / mixture.components[k].density;
                volume += v;
                mu_volume += v * mixture.components[k].dynamic_viscosity;
            }
        }

        material[2 * i] = float(mass / (grid.cell_size * grid.cell_size));
        material[2 * i + 1] = float(mu_volume / volume);
    }

    auto& compute = detail::FluidFieldAccess::commands(grid.getWebGPUContext());
    auto u = compute.field(grid.width + 1, grid.height);
    auto v = compute.field(grid.width, grid.height + 1);
    auto coefficients = compute.field(grid.width, grid.height, 2);

    compute.write(u, initial.u);
    compute.write(v, initial.v);
    compute.write(coefficients, material);
    webgpu::ViscositySolverGPU reference(compute, grid.width, grid.height);
    reference.solve(u, v, coefficients, 0.01, grid.cell_size);
    const auto expected_u = compute.read(u);
    const auto expected_v = compute.read(v);

    require(grid.applyViscosity(0.01) > 0, "Material viscosity did not solve");

    const auto result = grid.download();
    double error = 0.0;

    for (size_t i = 0; i < result.u.size(); ++i)
        error = std::max(error, std::abs(double(result.u[i] - expected_u[i])));
    for (size_t i = 0; i < result.v.size(); ++i)
        error = std::max(error, std::abs(double(result.v[i] - expected_v[i])));

    require(error < 2e-5, "GPU material blend differs from independent volume-weighted coefficients");
    require(result.liquid_fraction == initial.liquid_fraction && result.liquid_mass == initial.liquid_mass &&
                result.gas_mass == initial.gas_mass,
        "Viscosity modified composition or phase geometry");
    require(grid.applyViscosity(0.0) == 0 && same(result, grid.download()), "Zero timestep changed state");

    for (size_t k = 0; k < grid.liquid_mixture.components.size(); ++k)
        grid.setComponentViscosity(Phase::Liquid, k, 0.0);
    for (size_t k = 0; k < grid.gas_mixture.components.size(); ++k)
        grid.setComponentViscosity(Phase::Gas, k, 0.0);

    require(grid.applyViscosity(0.01) == 0 && same(result, grid.download()), "Disabled viscosity changed state");
    grid.setComponentViscosity(Phase::Gas, gases - 1, 5.0);

    require(grid.applyViscosity(0.01) > 0, "Gas-only viscosity did not activate the solver");

    const auto gas_only = grid.download();

    require(gas_only.liquid_fraction == result.liquid_fraction && gas_only.liquid_mass == result.liquid_mass &&
                gas_only.gas_mass == result.gas_mass,
        "Gas-only viscosity changed composition");
    std::cout << "PASS " << liquids + gases
              << "-ingredient volume blending, unchanged mass/geometry, disabled path; error=" << error << '\n';
}

void continuation(double recovery)
{
    FluidGridGPU grid(24, 18, 1.0);
    grid.setPressureRecoveryTime(recovery);
    initialize(grid, 2, 2);

    for (int i = 0; i < 12; ++i)
        require(grid.step(0.004, 2000, 0.001).accepted(), "Viscous coupled step failed");
    FluidCheckpoint saved;
    grid.saveCheckpoint(saved);

    require(saved.metadata.at("version") == 6, "Viscous checkpoint did not identify new format");

    FluidGridGPU restored(24, 18, 1.0);
    restored.loadCheckpoint(saved);

    require(restored.liquid_mixture.components[1].dynamic_viscosity == 120.0, "Checkpoint lost material viscosity");

    for (int i = 0; i < 12; ++i)
    {
        const auto a = grid.step(0.004, 2000, 0.001);
        const auto b = restored.step(0.004, 2000, 0.001);

        require(a.accepted() && b.accepted() && a.viscosity_iterations == b.viscosity_iterations,
            "Checkpoint changed viscosity solver work");
    }

    require(same(grid.download(), restored.download()), "Viscous checkpoint continuation differs");

    const auto before = restored.download();
    saved.metadata["liquid_mixture"][0]["dynamic_viscosity"] = -1.0;
    bool rejected = false;
    try
    {
        restored.loadCheckpoint(saved);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }

    require(rejected && same(before, restored.download()), "Invalid saved viscosity changed live state");
    std::cout << "PASS viscous checkpoint continuation and rejected input; recovery=" << recovery << '\n';
}

void unsupported()
{
    FluidGrid grid(8, 8, 1.0);
    bool rejected = false;
    try
    {
        grid.setComponentViscosity(Phase::Liquid, 0, 1.0);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(rejected, "CPU silently accepted positive viscosity");

    SubstanceCatalogue catalogue;
    const auto id = catalogue.add({"liquid", {1, 0, 0}});
    FluidMixture mixture{{{id, 10.0, 100.0}}};

    rejected = false;
    try
    {
        grid.configureMaterials(catalogue, mixture, {});
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(rejected, "CPU silently accepted viscous materials");
    grid.setDevice(FluidDevice::GPU);
    grid.configureMaterials(catalogue, mixture, {});
    rejected = false;
    try
    {
        grid.setDevice(FluidDevice::CPU);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(rejected && grid.getDevice() == FluidDevice::GPU, "Backend change discarded viscosity");
    std::cout << "PASS unsupported backend rejection\n";
}
}

int main()
{
    try
    {
        std::cout << std::unitbuf;
        blending(2, 2);
        blending(17, 3);
        continuation(0.0);
        continuation(0.02);
        unsupported();

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
