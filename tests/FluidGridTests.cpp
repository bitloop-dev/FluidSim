#include "FluidGridCPU.h"
#include "FluidTestSetup.h"
#include "FluidGridEditTests.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using fluid::FluidGridCPU;
using fluid::test::initializeTestMaterials;

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double liquidVolume(const FluidGridCPU& grid)
{
    return std::accumulate(grid.liquid_fraction.begin(), grid.liquid_fraction.end(), 0.0) * grid.cell_size * grid.cell_size;
}

void requireValidState(const FluidGridCPU& grid, double initial_volume)
{
    for (double value : grid.u)
        require(std::isfinite(value), "non-finite horizontal velocity");
    for (double value : grid.v)
        require(std::isfinite(value), "non-finite vertical velocity");
    for (double value : grid.liquid_fraction)
        require(std::isfinite(value) && value >= -1e-6 && value <= 1.0 + 1e-6, "material fraction left [0,1]");

    require(std::abs(liquidVolume(grid) - initial_volume) <= initial_volume * 1e-10, "liquid volume changed in a closed tank");
    for (int y = 0; y < grid.height; ++y)
        require(grid.u[grid.uIndex(0, y)] == 0.0 && grid.u[grid.uIndex(grid.width, y)] == 0.0, "flow crossed a side wall");
    for (int x = 0; x < grid.width; ++x)
        require(grid.v[grid.vIndex(x, 0)] == 0.0 && grid.v[grid.vIndex(x, grid.height)] == 0.0, "flow crossed a horizontal wall");
}

void samplesStaggeredVelocityInWorldSpace()
{
    FluidGridCPU grid(9, 7, 2.5);
    grid.origin = {-7.0, 11.0};
    auto horizontal = [](double x, double y) { return 0.3 * x + 0.7 * y - 2.0; };
    auto vertical = [](double x, double y) { return -0.4 * x + 0.2 * y + 1.0; };

    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x <= grid.width; ++x)
            grid.u[grid.uIndex(x, y)] = horizontal(grid.origin.x + x * grid.cell_size, grid.origin.y + (y + 0.5) * grid.cell_size);
    for (int y = 0; y <= grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
            grid.v[grid.vIndex(x, y)] = vertical(grid.origin.x + (x + 0.5) * grid.cell_size, grid.origin.y + y * grid.cell_size);

    // bilinear sampling reproduces an affine field exactly inside the sample domain
    for (bl::DVec2 pos : {bl::DVec2{-2.3, 16.7}, bl::DVec2{8.1, 24.2}})
    {
        const auto velocity = grid.sampleVelocity(pos);
        require(std::abs(velocity.x - horizontal(pos.x, pos.y)) < 1e-12, "incorrect horizontal staggered sample");
        require(std::abs(velocity.y - vertical(pos.x, pos.y)) < 1e-12, "incorrect vertical staggered sample");
    }
}

void stateTransferPreservesCPUBehaviour()
{
    FluidGridCPU direct(32, 24, 4), uploaded(32, 24, 4);
    for (int y = 0; y < 24; ++y)
    for (int x = 0; x < 32; ++x)
        direct.liquid_fraction[y * 32 + x] = std::clamp(y + 1.0 - (x + 12.0), 0.0, 1.0);
    const auto initial = initializeTestMaterials(direct);
    initializeTestMaterials(uploaded);
    uploaded.upload(initial);
    // Re-uploading the initial fields must reproduce the same CPU trajectory.
    for (int tick = 0; tick < 120; ++tick)
    {
        require(direct.step(1.0 / 120, 2000, 1e-8).converged, "direct CPU initialization failed");
        require(uploaded.step(1.0 / 120, 2000, 1e-8).converged, "uploaded CPU initialization failed");
        require(direct.u == uploaded.u && direct.v == uploaded.v && direct.liquid_fraction == uploaded.liquid_fraction && direct.pressure == uploaded.pressure,
                "state upload changed the CPU trajectory");
        require(direct.liquid_mass == uploaded.liquid_mass && direct.gas_mass == uploaded.gas_mass,
                "state upload changed component transport");
    }
    const auto state = uploaded.download();
    require(state.u == uploaded.u && state.v == uploaded.v && state.liquid_fraction == uploaded.liquid_fraction && state.pressure == uploaded.pressure,
            "CPU download omitted authoritative fields");
    require(state.liquid_mass == uploaded.liquid_mass && state.gas_mass == uploaded.gas_mass,
            "CPU download omitted component masses");
    uploaded.upload(initial);
    FluidGridCPU restarted(32, 24, 4);
    initializeTestMaterials(restarted);
    restarted.upload(initial);
    require(uploaded.step(1.0 / 120, 2000, 1e-8).converged && restarted.step(1.0 / 120, 2000, 1e-8).converged,
            "CPU reinitialization failed");
    require(uploaded.pressure == restarted.pressure && uploaded.u == restarted.u && uploaded.v == restarted.v,
            "CPU reinitialization retained stale pressure history");
    require(uploaded.liquid_mass == restarted.liquid_mass && uploaded.gas_mass == restarted.gas_mass,
            "CPU reinitialization retained stale component state");
}

void pressureBalancesGravity()
{
    FluidGridCPU grid(12, 10, 2.0);
    std::fill(grid.liquid_fraction.begin(), grid.liquid_fraction.end(), 1.0);
    initializeTestMaterials(grid);
    const double initial_volume = liquidVolume(grid);
    const auto result = grid.step(1.0 / 120.0, 2000, 1e-8);
    require(result.converged, "hydrostatic projection did not converge");

    for (double velocity : grid.v)
        require(std::abs(velocity) < 1e-7, "pressure did not cancel gravity in a filled tank");
    // hydrostatic pressure rises by density * gravity * depth
    const double pressure_rise = grid.pressure[(grid.height - 1) * grid.width] - grid.pressure[0];
    const double expected_rise = grid.liquid_mixture.components[0].density * grid.gravity_y * grid.cell_size * (grid.height - 1);
    require(std::abs(pressure_rise - expected_rise) < expected_rise * 1e-8, "incorrect hydrostatic pressure gradient");
    requireValidState(grid, initial_volume);
}

void explicitCacheInvalidationMatchesFreshState()
{
    for (const auto preconditioner : {FluidGridCPU::PressurePreconditioner::Multigrid, FluidGridCPU::PressurePreconditioner::Diagonal})
    {
        FluidGridCPU edited(33, 25, 2.0);
        edited.pressure_preconditioner = preconditioner;
        initializeTestMaterials(edited);
        edited.applyGravity(0.01);
        require(edited.project(0.01, 2000, 1e-8).converged, "cache fixture failed to project");

        // Change the density distribution after all pressure caches have been populated.
        for (int y = 0; y < edited.height; ++y)
        for (int x = 0; x < edited.width; ++x)
        {
            const int i = y * edited.width + x;
            const double f = edited.liquid_fraction[i] = std::clamp(y + 1.0 - (0.3 * x + 6.0), 0.0, 1.0);
            edited.liquid_mass[0][i] = 400.0 * f;
            edited.gas_mass[0][i] = 0.4 * (1.0 - f);
        }
        edited.invalidateCaches();
        FluidGridCPU fresh(edited.width, edited.height, edited.cell_size);
        initializeTestMaterials(fresh);
        fresh.pressure_preconditioner = preconditioner;
        fresh.upload(edited.download());
        edited.addVortex({30, 24}, 12, 0.1);
        fresh.addVortex({30, 24}, 12, 0.1);
        require(edited.project(0.01, 2000, 1e-8).converged && fresh.project(0.01, 2000, 1e-8).converged,
            "projection after direct mass edits failed");
        require(edited.density == fresh.density && edited.inv_density_u == fresh.inv_density_u && edited.inv_density_v == fresh.inv_density_v,
            "explicit invalidation retained stale density or pressure coefficients");
        for (size_t i = 0; i < edited.u.size(); ++i)
            require(std::abs(edited.u[i] - fresh.u[i]) < 2e-7, "edited horizontal pressure solution differs from fresh upload");
        for (size_t i = 0; i < edited.v.size(); ++i)
            require(std::abs(edited.v[i] - fresh.v[i]) < 2e-7, "edited vertical pressure solution differs from fresh upload");

        FluidGridCPU forced = edited;
        for (int tick = 0; tick < 20; ++tick)
        {
            forced.invalidateCaches();
            require(edited.step(0.004, 2000, 1e-8).converged && forced.step(0.004, 2000, 1e-8).converged,
                "cached/forced-refresh transport failed");
            require(edited.u == forced.u && edited.v == forced.v && edited.pressure == forced.pressure &&
                edited.liquid_fraction == forced.liquid_fraction && edited.liquid_mass == forced.liquid_mass && edited.gas_mass == forced.gas_mass,
                "automatic transport invalidation differs from forced cache refresh");
        }
    }
}

void transportConservesLiquidAndMovesItsCentroid()
{
    FluidGridCPU grid(64, 32, 1.0);
    std::fill(grid.u.begin(), grid.u.end(), 1.0);
    for (int y = 6; y < 18; ++y)
        for (int x = 8; x < 16; ++x)
            grid.liquid_fraction[y * grid.width + x] = 1.0;

    initializeTestMaterials(grid);
    // a long transport interval also exercises its internal CFL substeps
    grid.advect(8.0);
    double mass = 0.0, moment_x = 0.0, moment_y = 0.0;
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
        {
            const double fraction = grid.liquid_fraction[y * grid.width + x];
            // Corner intersections can leave double roundoff in otherwise empty/full cells.
            require(fraction >= -1e-14 && fraction <= 1.0 + 1e-14, "uniform transport violated fraction bounds");
            mass += fraction;
            moment_x += (x + 0.5) * fraction;
            moment_y += (y + 0.5) * fraction;
        }

    require(std::abs(mass - 96.0) < 1e-10, "uniform transport lost liquid");
    require(std::abs(moment_x / mass - 20.0) < 1e-10, "liquid did not travel velocity * time");
    require(std::abs(moment_y / mass - 12.0) < 1e-10, "horizontal transport moved liquid vertically");
}

void transportKeepsStraightInterfacesSharp()
{
    FluidGridCPU grid(40, 8, 1.0);
    std::fill(grid.u.begin(), grid.u.end(), 1.0);
    for (int y = 0; y < grid.height; ++y)
        for (int x = 8; x < 16; ++x)
            grid.liquid_fraction[y * grid.width + x] = 1.0;

    initializeTestMaterials(grid);
    // uniform motion by eight whole cells should preserve both straight interfaces
    grid.advect(8.0);
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
        {
            const double expected = x >= 16 && x < 24 ? 1.0 : 0.0;
            require(std::abs(grid.liquid_fraction[y * grid.width + x] - expected) < 1e-12,
                    "uniform transport blurred a straight material interface");
        }
}

void transportKeepsShallowInterfacesStraight()
{
    double largest_error = 0.0;
    for (bool vertical : {false, true})
    for (bool reverse_phase : {false, true})
    for (double slope : {0.0, -0.1, -0.05, 0.05, 0.1})
    for (double speed : {-1.0, 1.0})
    {
        FluidGridCPU grid(96, 96, 1.0);
        auto& flow = vertical ? grid.v : grid.u;
        std::fill(flow.begin(), flow.end(), speed);

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
            grid.liquid_fraction[y * grid.width + x] = vertical ? exactFraction(y, x) : exactFraction(x, y);
        initializeTestMaterials(grid);

        for (int step = 0; step < 20; ++step)
            grid.advect(0.25);

        // Inspect the interior, where imposed uniform translation is unaffected by the closed walls.
        for (int y = 24; y < 72; ++y)
        for (int x = 24; x < 72; ++x)
        {
            const double expected = vertical ? exactFraction(y - 5.0 * speed, x) : exactFraction(x - 5.0 * speed, y);
            largest_error = std::max(largest_error, std::abs(grid.liquid_fraction[y * grid.width + x] - expected));
        }
    }
    std::cout << "shallow-interface fraction error=" << largest_error << '\n';
    require(largest_error < 1e-6, "uniform transport corrugated a shallow straight interface");
}

void closedPoolRemainsValidOverRepeatedMotion()
{
    FluidGridCPU grid(32, 32, 4.0);
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
        {
            const double surface = grid.height * (0.5 + 0.04 * std::cos(3.14159265358979323846 * (x + 0.5) / grid.width));
            grid.liquid_fraction[y * grid.width + x] = std::clamp(y + 1.0 - surface, 0.0, 1.0);
        }

    const double initial_volume = liquidVolume(grid);
    initializeTestMaterials(grid, 1000.0, 1.0);
    // twenty simulated seconds cover several crossings and repeated wave reflections
    for (int tick = 0; tick < 2400; ++tick)
    {
        const auto result = grid.step(1.0 / 120.0, 2000, 1e-8);
        require(result.converged, "closed-pool projection did not converge");
        require(result.max_divergence <= 1e-8, "closed-pool divergence exceeded tolerance");
        requireValidState(grid, initial_volume);
    }
}

void transportPreservesUniformMotionAcrossDensityChanges()
{
    for (const bl::DVec2 velocity : {bl::DVec2{1.0, 0.75}, bl::DVec2{-1.0, 0.75},
                                    bl::DVec2{1.0, -0.75}, bl::DVec2{-1.0, -0.75}})
    {
        FluidGridCPU grid(48, 48, 1.0);
        std::fill(grid.u.begin(), grid.u.end(), velocity.x);
        std::fill(grid.v.begin(), grid.v.end(), velocity.y);

        for (int y = 18; y < 30; ++y)
            for (int x = 18; x < 30; ++x)
                grid.liquid_fraction[y * grid.width + x] =
                    std::clamp(6.0 - std::hypot(x - 23.5, y - 23.5), 0.0, 1.0);

        const double initial_volume = liquidVolume(grid);
        initializeTestMaterials(grid, 1000.0, 1.0);
        // diagonal transport exercises both momentum components in both passes and CFL substeps
        grid.advect(1.0);
        require(std::abs(liquidVolume(grid) - initial_volume) < 1e-10, "uniform transport changed liquid volume");

        // inspect the material and its surroundings, beyond the influence of the outer walls
        for (int y = 8; y < grid.height - 8; ++y)
            for (int x = 8; x < grid.width - 8; ++x)
            {
                require(std::abs(grid.u[grid.uIndex(x, y)] - velocity.x) < 1e-10,
                        "density changes altered uniform horizontal motion");
                require(std::abs(grid.v[grid.vIndex(x, y)] - velocity.y) < 1e-10,
                        "density changes altered uniform vertical motion");
            }
    }
}

void zeroTimeAndFailedProjectionAreHandled()
{
    FluidGridCPU grid(8, 8, 1.0);
    grid.gravity_y = 1.0;
    grid.liquid_fraction[3 * grid.width + 3] = 1.0;
    initializeTestMaterials(grid);

    const auto material = grid.liquid_fraction;
    const auto horizontal = grid.u;
    const auto vertical = grid.v;
    const auto liquid_mass = grid.liquid_mass;
    const auto gas_mass = grid.gas_mass;

    require(!grid.step(0.0).converged,
        "zero timestep unexpectedly advanced");

    require(grid.u == horizontal && grid.v == vertical &&
        grid.liquid_fraction == material && grid.liquid_mass == liquid_mass && grid.gas_mass == gas_mass,
        "zero timestep changed the state");

    // start with valid zero velocity; gravity creates the need for projection
    require(!grid.step(0.01, 0, 1e-8).converged,
        "zero-iteration projection unexpectedly converged");
}

void miscibleComponentsConserveMassAndVolume()
{
    FluidGridCPU grid(32, 32, 2.0);
    const auto water = grid.substances.add({"Water"});
    const auto red = grid.substances.add({"Red"});
    const auto heavy = grid.substances.add({"Heavy"});
    const auto unused = grid.substances.add({"Unused"});
    const auto air = grid.substances.add({"Air"});
    grid.liquid_mixture.components = {{water, 100.0}, {red, 100.0}, {heavy, 200.0}, {unused, 1000.0}};
    // The same substance can occupy either phase without sharing its mass field.
    grid.gas_mixture.components = {{air, 0.1}, {water, 0.2}};
    FluidGridCPU::State state;
    const size_t count = size_t(grid.width) * grid.height;
    const double cell_volume = grid.cell_size * grid.cell_size;
    state.liquid_fraction.resize(count);
    state.liquid_mass.assign(4, std::vector<double>(count));
    state.gas_mass.assign(2, std::vector<double>(count));
    for (int y = 0; y < grid.height; ++y)
    for (int x = 0; x < grid.width; ++x)
    {
        const int i = y * grid.width + x;
        const double surface = grid.height * (0.5 + 0.04 * std::cos(3.141592653589793 * (x + 0.5) / grid.width));
        const double fraction = state.liquid_fraction[i] = std::clamp(y + 1.0 - surface, 0.0, 1.0);
        const double heavy_share = x >= grid.width / 2 ? 0.25 : 0.0;
        const double water_share = (1.0 - heavy_share) * (y + 0.5) / grid.height;
        const double air_share = (x + 0.5) / grid.width;
        // Shares partition phase volume, even when ingredient densities differ.
        state.liquid_mass[0][i] = 100.0 * cell_volume * fraction * water_share;
        state.liquid_mass[1][i] = 100.0 * cell_volume * fraction * (1.0 - water_share - heavy_share);
        state.liquid_mass[2][i] = 200.0 * cell_volume * fraction * heavy_share;
        state.gas_mass[0][i] = 0.1 * cell_volume * (1.0 - fraction) * air_share;
        state.gas_mass[1][i] = 0.2 * cell_volume * (1.0 - fraction) * (1.0 - air_share);
    }
    grid.upload(state);
    const double initial_volume = liquidVolume(grid);
    double largest_mass_drift = 0.0, largest_volume_error = 0.0;
    auto checkMixture = [&](const auto& fields, const auto& initial, const fluid::FluidMixture& mixture, bool liquid)
    {
        for (size_t k = 0; k < fields.size(); ++k)
        {
            const double initial_mass = std::accumulate(initial[k].begin(), initial[k].end(), 0.0);
            const double mass = std::accumulate(fields[k].begin(), fields[k].end(), 0.0);
            const double drift = std::abs(mass - initial_mass) / std::max(1.0, initial_mass);
            largest_mass_drift = std::max(largest_mass_drift, drift);
            require(drift < 1e-8, "closed mixture lost component mass");
            for (double value : fields[k])
                require(std::isfinite(value) && value >= -1e-10 * mixture.components[k].density * cell_volume,
                        "component mass became negative or non-finite");
        }
        for (size_t i = 0; i < count; ++i)
        {
            double occupied_fraction = 0.0;
            for (size_t k = 0; k < fields.size(); ++k)
                occupied_fraction += fields[k][i] / (mixture.components[k].density * cell_volume);
            const double expected = liquid ? grid.liquid_fraction[i] : 1.0 - grid.liquid_fraction[i];
            largest_volume_error = std::max(largest_volume_error, std::abs(occupied_fraction - expected));
            require(std::abs(occupied_fraction - expected) < 1e-9, "component volumes disagree with phase geometry");
        }
    };
    // Twenty seconds cover repeated crossings of this 64-unit tank, including a longer 10-to-20s confirmation window.
    for (int tick = 0; tick < 2400; ++tick)
    {
        const auto result = grid.step(1.0 / 120.0, 2000, 1e-8);
        require(result.converged && result.max_divergence <= 1e-8, "multi-component pressure did not converge");
        requireValidState(grid, initial_volume);
        checkMixture(grid.liquid_mass, state.liquid_mass, grid.liquid_mixture, true);
        checkMixture(grid.gas_mass, state.gas_mass, grid.gas_mixture, false);
        for (double mass : grid.liquid_mass[3]) require(mass == 0.0, "an absent component appeared spontaneously");
    }
    std::cout << "component mass drift=" << largest_mass_drift << " phase-volume error=" << largest_volume_error << '\n';
}

void multigridPreservesPCGContract()
{
    for (const auto size : {bl::IVec2{1, 1}, {1, 129}, {131, 1}, {2, 67}, {13, 9}, {64, 64}, {65, 37}, {512, 512}})
    {
        const int w = size.x, h = size.y, count = w * h;
        fluid::PressureMultigrid multigrid(w, h);
        std::vector<double> horizontal((w + 1) * h), vertical(w * (h + 1));
        auto density = [](int x, int y) { return (x / 3 + y / 4) % 2 == 0 ? 0.1 : 100.0; };
        for (int y = 0; y < h; ++y)
            for (int x = 1; x < w; ++x)
                horizontal[y * (w + 1) + x] = 2.0 / (density(x - 1, y) + density(x, y));
        for (int y = 1; y < h; ++y)
            for (int x = 0; x < w; ++x)
                vertical[y * w + x] = 2.0 / (density(x, y - 1) + density(x, y));
        require(multigrid.rebuild(horizontal, vertical), "multigrid coarse factorization failed");
        std::vector<double> a(count), b(count), combined(count), ma, mb, mc, repeat;
        for (int i = 0; i < count; ++i)
        {
            a[i] = std::sin(i * 0.31) + std::cos(i * 0.073);
            b[i] = std::cos(i * 0.17) - std::sin(i * 0.019);
        }
        auto projectMean = [](std::vector<double>& values)
        {
            const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            for (double& value : values) value -= mean;
        };
        projectMean(a);
        projectMean(b);
        for (int i = 0; i < count; ++i) combined[i] = 1.3 * a[i] - 0.7 * b[i];
        multigrid.apply(a, ma, 0.0);
        multigrid.apply(b, mb, 0.0);
        multigrid.apply(combined, mc, 0.0);
        multigrid.apply(a, repeat, 0.0);
        require(ma == repeat, "multigrid retained a previous correction");
        projectMean(ma); projectMean(mb); projectMean(mc);
        double error = 0.0, scale = 0.0;
        for (int i = 0; i < count; ++i)
        {
            error = std::max(error, std::abs(mc[i] - (1.3 * ma[i] - 0.7 * mb[i])));
            scale = std::max(scale, std::abs(mc[i]));
        }
        require(error <= 1e-10 * std::max(1.0, scale), "multigrid is not a linear preconditioner");
        const double a_mb = std::inner_product(a.begin(), a.end(), mb.begin(), 0.0);
        const double b_ma = std::inner_product(b.begin(), b.end(), ma.begin(), 0.0);
        const double energy_a = std::inner_product(a.begin(), a.end(), ma.begin(), 0.0);
        const double energy_b = std::inner_product(b.begin(), b.end(), mb.begin(), 0.0);
        require(std::abs(a_mb - b_ma) <= 1e-10 * std::max(1.0, std::sqrt(energy_a * energy_b)), "multigrid is not symmetric");
        require(count == 1 ? energy_a == 0.0 : energy_a > 0.0 && energy_b > 0.0, "multigrid is not positive on mean-zero residuals");
        std::fill(a.begin(), a.end(), 7.0);
        multigrid.apply(a, ma, 7.0);
        require(std::all_of(ma.begin(), ma.end(), [](double value) { return value == 0.0; }), "multigrid did not remove the constant nullspace");
    }
}

void multigridMatchesDiagonalProjection()
{
    for (const auto size : {bl::IVec2{1, 129}, {131, 1}, {2, 67}, {65, 47}, {128, 128}})
    {
        FluidGridCPU multigrid(size.x, size.y, 4.0);
        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x)
                multigrid.liquid_fraction[y * size.x + x] = std::clamp(y + 1.0 - (0.35 * x + 0.2 * size.y), 0.0, 1.0);
        for (size_t i = 0; i < multigrid.u.size(); ++i) multigrid.u[i] = 3.0 * std::sin(i * 0.07);
        for (size_t i = 0; i < multigrid.v.size(); ++i) multigrid.v[i] = 4.0 * std::cos(i * 0.11);
        initializeTestMaterials(multigrid);
        FluidGridCPU diagonal = multigrid;
        diagonal.pressure_preconditioner = FluidGridCPU::PressurePreconditioner::Diagonal;
        const auto mg_result = multigrid.project(1.0 / 120.0, 4000, 1e-8);
        const auto diagonal_result = diagonal.project(1.0 / 120.0, 4000, 1e-8);
        require(mg_result.converged && diagonal_result.converged, "preconditioner comparison did not converge");
        for (size_t i = 0; i < multigrid.u.size(); ++i)
            require(std::abs(multigrid.u[i] - diagonal.u[i]) < 2e-6, "multigrid changed the horizontal pressure solution");
        for (size_t i = 0; i < multigrid.v.size(); ++i)
            require(std::abs(multigrid.v[i] - diagonal.v[i]) < 2e-6, "multigrid changed the vertical pressure solution");
    }
}

void largeMixedPressureConvergesWithinDefaultBudget()
{
    FluidGridCPU grid(512, 512, 4.0);
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
            grid.liquid_fraction[y * grid.width + x] = std::clamp(y + 1.0 - (0.35 * x + 0.2 * grid.height), 0.0, 1.0);
    for (size_t i = 0; i < grid.u.size(); ++i) grid.u[i] = 3.0 * std::sin(i * 0.07);
    for (size_t i = 0; i < grid.v.size(); ++i) grid.v[i] = 4.0 * std::cos(i * 0.11);
    initializeTestMaterials(grid);
    // Protect large-grid convergence without making the regression suite replay a whole wave.
    const auto result = grid.project(1.0 / 120.0);
    require(result.converged && result.max_divergence <= 1e-6, "512x512 mixed-density pressure exhausted the default iteration budget");
}

void parallelExecutionMatchesSerialOnOddGrid()
{
    FluidGridCPU serial(513, 257, 1.0);
    for (int y = 0; y < serial.height; ++y)
        for (int x = 0; x < serial.width; ++x)
            serial.liquid_fraction[y * serial.width + x] = std::clamp(y + 1.0 - (0.3 * x + 30.0), 0.0, 1.0);
    for (size_t i = 0; i < serial.u.size(); ++i) serial.u[i] = std::sin(i * 0.03);
    for (size_t i = 0; i < serial.v.size(); ++i) serial.v[i] = std::cos(i * 0.017);
    initializeTestMaterials(serial);
    FluidGridCPU parallel = serial;
    const int previous_threads = int(bl::Thread::threadCount());
    bl::Thread::setMaxThreads(1);
    const auto a = serial.project(0.004, 2000, 1e-8);
    bl::Thread::setMaxThreads(8);
    const auto b = parallel.project(0.004, 2000, 1e-8);
    bl::Thread::setMaxThreads(previous_threads);
    require(a.converged && b.converged, "serial/parallel odd-grid projection failed");
    for (size_t i = 0; i < serial.u.size(); ++i)
        require(std::abs(serial.u[i] - parallel.u[i]) < 2e-7, "parallel horizontal projection changed the solution");
    for (size_t i = 0; i < serial.v.size(); ++i)
        require(std::abs(serial.v[i] - parallel.v[i]) < 2e-7, "parallel vertical projection changed the solution");

    // Cross row-batch boundaries during both transport directions and interface reconstruction.
    for (int tick = 0; tick < 3; ++tick)
    {
        bl::Thread::setMaxThreads(1);
        const auto serial_step = serial.step(0.004, 2000, 1e-8);
        bl::Thread::setMaxThreads(8);
        const auto parallel_step = parallel.step(0.004, 2000, 1e-8);
        bl::Thread::setMaxThreads(previous_threads);
        require(serial_step.converged && parallel_step.converged, "serial/parallel transport failed");
        for (size_t i = 0; i < serial.liquid_fraction.size(); ++i)
        {
            require(std::abs(serial.liquid_fraction[i] - parallel.liquid_fraction[i]) < 2e-7,
                "parallel transport changed phase geometry");
            require(std::abs(serial.liquid_mass[0][i] - parallel.liquid_mass[0][i]) < 2e-5 &&
                    std::abs(serial.gas_mass[0][i] - parallel.gas_mass[0][i]) < 2e-8,
                "parallel transport changed component masses");
        }
        for (size_t i = 0; i < serial.u.size(); ++i)
            require(std::abs(serial.u[i] - parallel.u[i]) < 2e-7, "parallel transport changed horizontal momentum");
        for (size_t i = 0; i < serial.v.size(); ++i)
            require(std::abs(serial.v[i] - parallel.v[i]) < 2e-7, "parallel transport changed vertical momentum");
    }
}

}

int main()
{
    const struct { const char* name; void (*run)(); } tests[] = {
        {"staggered world-space sampling", samplesStaggeredVelocityInWorldSpace},
        {"state transfer and CPU initialization equivalence", stateTransferPreservesCPUBehaviour},
        {"sparse reusable material edit sessions", fluid::test::materialEditingIsSparseAndReusable<FluidGridCPU>},
        {"bulk material edit batches", fluid::test::bulkMaterialEditsPreserveEveryCell<FluidGridCPU>},
        {"repeated material edits and phase volumes", fluid::test::repeatedMaterialEditsPreservePhaseVolumes<FluidGridCPU>},
        {"hydrostatic equilibrium", pressureBalancesGravity},
        {"explicit and automatic cache invalidation", explicitCacheInvalidationMatchesFreshState},
        {"conservative material transport", transportConservesLiquidAndMovesItsCentroid},
        {"sharp straight-interface transport", transportKeepsStraightInterfacesSharp},
        {"shallow straight-interface transport", transportKeepsShallowInterfacesStraight},
        {"repeated closed-pool motion", closedPoolRemainsValidOverRepeatedMotion},
        {"uniform motion across density changes", transportPreservesUniformMotionAcrossDensityChanges},
        {"zero timestep and projection failure", zeroTimeAndFailedProjectionAreHandled},
        {"miscible component conservation and phase volumes", miscibleComponentsConserveMassAndVolume},
        {"multigrid linearity, symmetry and positivity", multigridPreservesPCGContract},
        {"multigrid and diagonal pressure agreement", multigridMatchesDiagonalProjection},
        {"large mixed-density pressure convergence", largeMixedPressureConvergesWithinDefaultBudget},
        {"parallel odd-grid pressure and transport agreement", parallelExecutionMatchesSerialOnOddGrid}
    };
    int failures = 0;
    for (const auto& test : tests)
    {
        try { test.run(); std::cout << "PASS " << test.name << '\n'; }
        catch (const std::exception& error) { ++failures; std::cerr << "FAIL " << test.name << ": " << error.what() << '\n'; }
    }
    return failures == 0 ? 0 : 1;
}
