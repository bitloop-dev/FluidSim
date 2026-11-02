#include "FluidGrid.h"
#include "UniformWebGPU/FluidGridGPU.h"
#include <cmath>
#include <iostream>
#include <numeric>

namespace
{
using fluid::FluidGridGPU;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Operation> void rejects(Operation&& operation, const char* message)
{
    bool rejected = false;
    try
    {
        operation();
    }
    catch (const std::exception&)
    {
        rejected = true;
    }

    require(rejected, message);
}

bool same(const FluidGridGPU::State& a, const FluidGridGPU::State& b)
{
    return a.liquid_fraction == b.liquid_fraction && a.liquid_mass == b.liquid_mass && a.gas_mass == b.gas_mass &&
           a.u == b.u && a.v == b.v && a.pressure == b.pressure && a.elastic_log_xx == b.elastic_log_xx &&
           a.elastic_log_xy == b.elastic_log_xy && a.elastic_log_yy == b.elastic_log_yy;
}

void configure(FluidGridGPU& grid, const fluid::FluidWebGPUContext& context)
{
    grid.setWebGPUContext(context);
    const auto solid = grid.substances.add({"Elastic material", {0.2f, 0.8f, 0.6f}});
    const auto gas = grid.substances.add({"Gas", {0.0f, 0.0f, 0.0f}});
    grid.liquid_mixture = {{{solid, 10.0, 2.0, 20000.0}}};
    grid.gas_mixture = {{{gas, 0.01}}};
    grid.gravity_y = 1.0;
}

FluidGridGPU::State initialState(int width, int height)
{
    const size_t cells = size_t(width) * height;
    FluidGridGPU::State state;
    state.liquid_fraction.resize(cells);
    state.liquid_mass.assign(1, std::vector<float>(cells));
    state.gas_mass.assign(1, std::vector<float>(cells));
    state.elastic_log_xx.assign(cells, float(2.0 * std::log(1.05)));
    state.elastic_log_xy.assign(cells, 0.0f);
    state.elastic_log_yy.assign(cells, float(-2.0 * std::log(1.05)));
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const size_t i = size_t(y) * width + x;
            const float fraction =
                float(std::max(0.0, std::min(x + 1.0, width * 0.7) - std::max(double(x), width * 0.3)) *
                      std::max(0.0, std::min(y + 1.0, height * 0.7) - std::max(double(y), height * 0.3)));
            state.liquid_fraction[i] = fraction;
            state.liquid_mass[0][i] = 10.0f * fraction;
            state.gas_mass[0][i] = 0.01f * (1.0f - fraction);
        }
    }

    return state;
}

void integration(const fluid::FluidWebGPUContext& context)
{
    constexpr int width = 24;
    constexpr int height = 20;
    constexpr double dt = 0.02;
    const auto initial = initialState(width, height);
    FluidGridGPU grid(width, height, 1.0);
    FluidGridGPU control(width, height, 1.0);
    configure(grid, context);
    configure(control, context);
    grid.upload(initial);
    control.upload(initial);
    require(!grid.getImplicitElasticity(), "Explicit mode is not the default");

    grid.setImplicitElasticity(true);
    grid.setImplicitElasticity(false);
    for (int step = 0; step < 3; ++step)
    {
        const auto a = grid.step(dt, 1000, 1e-5);
        const auto b = control.step(dt, 1000, 1e-5);
        require(a.accepted() && b.accepted() && a.iterations == b.iterations && a.physical_steps == b.physical_steps &&
                    a.elasticity_iterations == 0,
            "Disabled mode changed explicit work");
    }

    fluid::FluidCheckpoint explicit_saved;
    fluid::FluidCheckpoint control_saved;
    grid.saveCheckpoint(explicit_saved);
    control.saveCheckpoint(control_saved);
    require(explicit_saved.metadata == control_saved.metadata && explicit_saved.data == control_saved.data &&
                explicit_saved.metadata.at("version") == 10,
        "Disabled mode changed the explicit checkpoint");

    const auto before_toggle = grid.download();
    grid.setImplicitElasticity(true);
    require(same(before_toggle, grid.download()), "Live implicit toggle changed physical state/history");
    rejects([&]
    {
        grid.setSurfaceTension(1.0);
    }, "Implicit mode accepted surface tension");
    rejects([&]
    {
        grid.setPressureRecoveryTime(0.1);
    }, "Implicit mode accepted pressure recovery");
    require(grid.getSurfaceTension() == 0.0 && grid.getPressureRecoveryTime() == 0.0,
        "Rejected implicit combination changed settings");

    for (int step = 0; step < 4; ++step)
    {
        const auto result = grid.step(dt, 1000, 1e-5);
        require(result.accepted() && result.physical_steps == 1 && result.elasticity_iterations > 0 &&
                    result.iterations == 0 && result.viscosity_iterations > 0,
            "Implicit complete step did not use the coupled solve/viscosity split");
    }

    fluid::FluidCheckpoint saved;
    grid.saveCheckpoint(saved);
    require(saved.metadata.at("version") == 11 && saved.metadata.at("implicit_elasticity") == true,
        "Checkpoint lost implicit mode");
    FluidGridGPU restored(width, height, 1.0);
    restored.setWebGPUContext(context);
    restored.loadCheckpoint(saved);
    require(restored.getImplicitElasticity(), "Checkpoint did not restore implicit mode");
    for (int step = 0; step < 4; ++step)
    {
        const auto a = grid.step(dt, 1000, 1e-5);
        const auto b = restored.step(dt, 1000, 1e-5);
        require(a.accepted() && b.accepted() && a.elasticity_iterations == b.elasticity_iterations &&
                    same(grid.download(), restored.download()),
            "Checkpoint changed implicit continuation");
    }

    const auto before_disable = grid.download();
    grid.setImplicitElasticity(false);
    restored.setImplicitElasticity(false);
    require(same(before_disable, grid.download()), "Live explicit toggle changed physical state/history");
    require(grid.step(dt, 1000, 1e-5).accepted() && restored.step(dt, 1000, 1e-5).accepted() &&
                same(grid.download(), restored.download()),
        "Explicit continuation after implicit mode lost pressure/history");

    restored.setImplicitElasticity(true);
    restored.loadCheckpoint(explicit_saved);
    require(!restored.getImplicitElasticity(), "Older explicit checkpoint inherited implicit mode");
    auto invalid = saved;
    invalid.metadata["surface_tension"] = 1.0;
    const auto before_reject = restored.download();
    rejects([&]
    {
        restored.loadCheckpoint(invalid);
    }, "Invalid implicit checkpoint accepted");
    require(same(before_reject, restored.download()) && !restored.getImplicitElasticity(),
        "Invalid implicit checkpoint changed live state");

    // A failed coupled solve must report failure and leave the constitutive update uncommitted.
    grid.upload(initial);
    grid.setImplicitElasticity(true);
    const auto zero_budget = grid.step(dt, 0, 1e-8);
    require(!zero_budget.accepted() && zero_budget.elasticity_iterations == 0,
        "Zero coupled iteration budget was not respected");
    grid.upload(initial);
    const auto before_failure = grid.download();
    const auto failed = grid.step(dt, 1, 1e-8);
    require(
        !failed.accepted() && failed.elasticity_iterations == 1, "Coupled iteration failure was hidden by projection");
    const auto failure_state = grid.download();
    require(failure_state.elastic_log_xx == before_failure.elastic_log_xx &&
                failure_state.elastic_log_xy == before_failure.elastic_log_xy &&
                failure_state.elastic_log_yy == before_failure.elastic_log_yy,
        "Failed solve committed deformation");
    std::cout << "PASS native toggle, disabled parity, viscosity, checkpoint continuation and failed solve\n";
}

void commonApi(const fluid::FluidWebGPUContext& context)
{
    fluid::FluidGrid common(24, 20, 1.0);
    rejects([&]
    {
        common.setImplicitElasticity(true);
    }, "CPU accepted implicit elasticity");
    common.setDevice(fluid::FluidDevice::GPU);
    common.setAdaptiveMode(fluid::AdaptiveMode::MAX_2);
    rejects([&]
    {
        common.setImplicitElasticity(true);
    }, "Adaptive WebGPU accepted implicit elasticity");
    common.setAdaptiveMode(fluid::AdaptiveMode::DISABLED);

    FluidGridGPU native(24, 20, 1.0);
    configure(native, context);
    common.setWebGPUContext(context);
    common.configureMaterials(native.substances, native.liquid_mixture, native.gas_mixture);
    common.setGravityY(native.gravity_y);
    common.setSurfaceTension(1.0);
    rejects([&]
    {
        common.setImplicitElasticity(true);
    }, "Implicit toggle accepted existing surface tension");
    common.setSurfaceTension(0.0);
    common.setImplicitElasticity(true);
    native.setImplicitElasticity(true);
    const auto initial = initialState(24, 20);
    common.upload(fluid::convertFluidState<double>(initial));
    native.upload(initial);
    for (int step = 0; step < 3; ++step)
    {
        const auto a = common.step(0.02, 1000, 1e-5);
        const auto b = native.step(0.02, 1000, 1e-5);
        require(a.accepted() && b.accepted() && a.elasticity_iterations == b.elasticity_iterations &&
                    same(fluid::convertFluidState<float>(common.download()), native.download()),
            "Common API changed native implicit stepping");
    }

    common.destroy();
    common.setComponentViscosity(fluid::Phase::Liquid, 0, 0.0);
    rejects([&]
    {
        common.setDevice(fluid::FluidDevice::CPU);
    }, "Backend change discarded implicit mode");
    require(common.getDevice() == fluid::FluidDevice::GPU && common.getImplicitElasticity(),
        "Rejected backend selection changed implicit mode");
    std::cout << "PASS common/native parity and unsupported backend/combination rejection\n";
}
}

int main()
{
    try
    {
        fluid::FluidWebGPUContext context;
        integration(context);
        commonApi(context);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL implicit elasticity integration: " << error.what() << '\n';
        return 1;
    }
}
