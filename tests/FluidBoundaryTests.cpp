#include "UniformWebGPU/FluidGridGPU.h"
#include "FluidGrid.h"
#include <cmath>
#include <iostream>
#include <numeric>

using namespace fluid;

namespace
{
void require(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}

void materials(FluidGridGPU& grid, double viscosity = 0.0)
{
    for (int i = 0; i < 10; ++i)
    {
        const auto id = grid.substances.add({"material", {1, 0, 0}});
        (i < 5 ? grid.liquid_mixture : grid.gas_mixture).components.push_back({id, i < 5 ? 14.0 : 0.1, viscosity});
    }

    grid.gravity_y = 0.0;
}

BoundaryFlows through(const FluidGridGPU& grid, BoundarySide side, Phase phase, double speed)
{
    const int count = int(side) < 2 ? grid.height : grid.width;
    return {{{{side, 0, count}, phase, phase == Phase::Liquid ? 4u : 9u, speed}},
        {{{BoundarySide(int(side) ^ 1), 0, count}, speed}}};
}

FluidGridGPU::State filled(const FluidGridGPU& grid, Phase phase, BoundarySide side, float speed)
{
    FluidGridGPU::State state;
    const int n = grid.width * grid.height;
    state.liquid_fraction.assign(n, phase == Phase::Liquid ? 1.0f : 0.0f);
    state.liquid_mass.assign(5, std::vector<float>(n));
    state.gas_mass.assign(5, std::vector<float>(n));
    auto& masses = phase == Phase::Liquid ? state.liquid_mass : state.gas_mass;
    masses[4].assign(n, float((phase == Phase::Liquid ? 14.0 : 0.1) * grid.cell_size * grid.cell_size));
    const float signed_speed = (int(side) % 2) ? -speed : speed;
    state.u.assign((grid.width + 1) * grid.height, int(side) < 2 ? signed_speed : 0.0f);
    state.v.assign(grid.width * (grid.height + 1), int(side) < 2 ? 0.0f : signed_speed);

    return state;
}

void uniformThroughflow(BoundarySide side, Phase phase, int width, int height)
{
    FluidGridGPU grid(width, height, 0.7);
    materials(grid, 300000.0);
    grid.setBoundaryFlows(through(grid, side, phase, 2.0));
    const auto initial = filled(grid, phase, side, 2.0f);
    grid.upload(initial);

    for (int i = 0; i < 8; ++i)
        require(grid.step(0.02, -1, 1e-4).accepted(), "Uniform viscous throughflow projection failed");
    const auto state = grid.download();
    double error = 0.0;

    for (size_t i = 0; i < state.u.size(); ++i)
        error = std::max(error, std::abs(double(state.u[i] - initial.u[i])));
    for (size_t i = 0; i < state.v.size(); ++i)
        error = std::max(error, std::abs(double(state.v[i] - initial.v[i])));

    require(error < 2e-4, "Viscosity or transport changed uniform driven flow");
    require(state.liquid_fraction == initial.liquid_fraction, "Uniform flow changed phase geometry");
    std::cout << "PASS viscous throughflow side=" << int(side) << " phase=" << int(phase) << " size=" << width << 'x'
              << height << " velocity error=" << error << '\n';
}

void displacement(BoundarySide side, Phase phase, bool relaxed)
{
    FluidGridGPU grid(16, 12, 0.7);
    materials(grid);
    grid.setPressureRecoveryTime(relaxed ? 0.1 : 0.0);
    grid.setBoundaryFlows(through(grid, side, phase, 2.0));
    grid.upload(filled(grid, phase == Phase::Liquid ? Phase::Gas : Phase::Liquid, side, 2.0f));
    const int cross_section = int(side) < 2 ? grid.height : grid.width;
    const int length = int(side) < 2 ? grid.width : grid.height;
    const double area = grid.cell_size * grid.cell_size;

    for (int step = 1; step <= 160; ++step)
    {
        require(grid.step(0.05, -1, 1e-4).accepted(), "Displacement projection failed");

        if (step % 10)
            continue;

        const auto state = grid.download();
        const auto& mass = phase == Phase::Liquid ? state.liquid_mass[4] : state.gas_mass[4];
        const double volume = std::accumulate(mass.begin(), mass.end(), 0.0) / (phase == Phase::Liquid ? 14.0 : 0.1);
        const double expected = std::min(double(length), step * 0.05 * 2.0 / grid.cell_size) * cross_section * area;

        // Before anything can reach the outlet, all injected volume must remain.
        // The existing wall-fragment fallback later diffuses the front near the
        // side walls, so a sharp plug is not an exact outlet-flux reference.
        if (step == 10)
            require(std::abs(volume - expected) < 2e-3, "Incoming material flux is wrong");
        if (step == 160)
            require(std::abs(volume - expected) < 0.02, "Throughflow did not replace the original phase");

        for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
        {
            const double liquid = state.liquid_mass[4][i] / (14.0 * area);
            const double gas = state.gas_mass[4][i] / (0.1 * area);

            require(std::abs(liquid + gas - 1.0) < 2e-4, "Boundary displacement lost occupied volume");
            require(std::abs(liquid - state.liquid_fraction[i]) < 2e-4, "Boundary fraction disagrees with material");
        }
    }

    std::cout << "PASS eight-second phase displacement side=" << int(side) << " phase=" << int(phase)
              << " relaxed=" << relaxed << '\n';
}

void configurationAndResume()
{
    FluidGridGPU grid(16, 12, 1.0);
    materials(grid, 1000.0);
    grid.upload(filled(grid, Phase::Gas, BoundarySide::Top, 0.0f));
    const BoundaryFlows flows{{{{BoundarySide::Top, 6, 4}, Phase::Liquid, 4, 2.0}},
        {{{BoundarySide::Top, 0, 6}, 2.0 / 3.0}, {{BoundarySide::Top, 10, 6}, 2.0 / 3.0}}};
    grid.setBoundaryFlows(flows);
    const auto initial = grid.download();

    require(std::accumulate(initial.liquid_mass[4].begin(), initial.liquid_mass[4].end(), 0.0) == 0.0,
        "Configuring an inlet replaced cell material");

    auto bad = flows;
    bad.outflows.clear();
    bool rejected = false;
    try
    {
        grid.setBoundaryFlows(bad);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(rejected && grid.getBoundaryFlows() == flows && grid.download().v == initial.v,
        "Unbalanced configuration was not rejected atomically");

    for (int pending = 1; pending >= 0; --pending)
    {
        FluidCheckpoint checkpoint;
        grid.saveCheckpoint(checkpoint);

        require(checkpoint.metadata.at("version") == 8, "Open boundary checkpoint version missing");

        FluidGridGPU restored(16, 12, 1.0);
        restored.loadCheckpoint(checkpoint);

        require(restored.getBoundaryFlows() == flows, "Boundary configuration did not restore");

        for (int i = 0; i < 4; ++i)
        {
            require(
                grid.step(0.02, -1, 1e-3).accepted() && restored.step(0.02, -1, 1e-3).accepted(), "Resume step failed");
        }

        const auto a = grid.download();
        const auto b = restored.download();

        require(a.u == b.u && a.v == b.v && a.liquid_mass == b.liquid_mass && a.gas_mass == b.gas_mass,
            "Boundary continuation differs after restore");
    }

    auto edit = grid.beginEdit();
    edit.setCell(7, 0, Phase::Liquid, 4);
    edit.commit(bl::DVec2{50, 50});

    require(grid.download().v[7] == 2.0f && grid.getBoundaryFlows() == flows, "Cell edit changed prescribed inlet");
    grid.setBoundaryFlows({});

    require(grid.step(0.02, -1, 1e-3).accepted(), "Closing boundary failed");

    const auto closed = grid.download();

    for (int x = 0; x < grid.width; ++x)
        require(closed.v[x] == 0.0, "Cleared opening remains open");
    FluidGrid cpu(16, 12, 1.0);
    rejected = false;
    try
    {
        cpu.setBoundaryFlows(flows);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(rejected, "Unsupported backend silently accepted inflow");
    std::cout << "PASS boundary configuration, edits, clearing and exact checkpoint continuation\n";
}

void mixedOutflow(BoundarySide side)
{
    FluidGridGPU grid(11, 9, 0.7);
    materials(grid);
    grid.setBoundaryFlows(through(grid, side, Phase::Liquid, 2.0));
    auto state = filled(grid, Phase::Liquid, side, 2.0f);

    for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
    {
        state.liquid_mass[0][i] = 0.3f * state.liquid_mass[4][i];
        state.liquid_mass[4][i] -= state.liquid_mass[0][i];
    }

    grid.upload(state);

    require(grid.project(0.02, -1, 1e-5).accepted(), "Mixed flow seed failed");

    std::array<double, 5> expected{};

    for (int k = 0; k < 5; ++k)
        expected[k] = std::accumulate(state.liquid_mass[k].begin(), state.liquid_mass[k].end(), 0.0);
    const int count = int(side) < 2 ? grid.height : grid.width;

    for (int step = 0; step < 200; ++step)
    {
        // Pure liquid has an exact swept-volume flux. Account for each outgoing
        // ingredient independently from the pre-step boundary cell contents.
        const double swept = 2.0 * 0.02 / grid.cell_size;
        for (int along = 0; along < count; ++along)
        {
            const int x = int(side) < 2 ? (side == BoundarySide::Left ? grid.width - 1 : 0) : along;
            const int y = int(side) >= 2 ? (side == BoundarySide::Top ? grid.height - 1 : 0) : along;
            for (int k = 0; k < 5; ++k)
                expected[k] -= swept * state.liquid_mass[k][y * grid.width + x];
        }

        expected[4] += count * 2.0 * 0.02 * grid.cell_size * 14.0;
        grid.advect(0.02);
        state = grid.download();

        for (int k = 0; k < 5; ++k)
        {
            require(std::abs(std::accumulate(state.liquid_mass[k].begin(), state.liquid_mass[k].end(), 0.0) -
                             expected[k]) < 0.002,
                "Outlet did not carry its actual local mixture conservatively");
        }
    }

    std::cout << "PASS four-second independently accounted mixed outlet side=" << int(side) << '\n';
}
}

int main()
{
    try
    {
        for (int side = 0; side < 4; ++side)
        {
            for (const auto phase : {Phase::Liquid, Phase::Gas})
            {
                uniformThroughflow(BoundarySide(side), phase, 17, 13);
                displacement(BoundarySide(side), phase, false);
            }
        }

        uniformThroughflow(BoundarySide::Top, Phase::Liquid, 1, 9);
        uniformThroughflow(BoundarySide::Left, Phase::Gas, 9, 1);
        displacement(BoundarySide::Top, Phase::Liquid, true);

        for (int side = 0; side < 4; ++side)
            mixedOutflow(BoundarySide(side));
        configurationAndResume();

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
