#include "FluidGrid.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using fluid::FluidGrid;

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double waterVolume(const FluidGrid& grid)
{
    return std::accumulate(grid.water_fraction.begin(), grid.water_fraction.end(), 0.0) * grid.cell_size * grid.cell_size;
}

void requireValidState(const FluidGrid& grid, double initial_volume)
{
    for (double value : grid.u)
        require(std::isfinite(value), "non-finite horizontal velocity");
    for (double value : grid.v)
        require(std::isfinite(value), "non-finite vertical velocity");
    for (double value : grid.water_fraction)
        require(std::isfinite(value) && value >= -1e-6 && value <= 1.0 + 1e-6, "material fraction left [0,1]");

    require(std::abs(waterVolume(grid) - initial_volume) <= initial_volume * 1e-10, "water volume changed in a closed tank");
    for (int y = 0; y < grid.height; ++y)
        require(grid.u[grid.uIndex(0, y)] == 0.0 && grid.u[grid.uIndex(grid.width, y)] == 0.0, "flow crossed a side wall");
    for (int x = 0; x < grid.width; ++x)
        require(grid.v[grid.vIndex(x, 0)] == 0.0 && grid.v[grid.vIndex(x, grid.height)] == 0.0, "flow crossed a horizontal wall");
}

void samplesStaggeredVelocityInWorldSpace()
{
    FluidGrid grid(9, 7, 2.5);
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

void pressureBalancesGravity()
{
    FluidGrid grid(12, 10, 2.0);
    std::fill(grid.water_fraction.begin(), grid.water_fraction.end(), 1.0);
    const double initial_volume = waterVolume(grid);
    const auto result = grid.step(1.0 / 120.0, 2000, 1e-8);
    require(result.converged, "hydrostatic projection did not converge");

    for (double velocity : grid.v)
        require(std::abs(velocity) < 1e-7, "pressure did not cancel gravity in a filled tank");
    // hydrostatic pressure rises by density * gravity * depth
    const double pressure_rise = grid.pressure[(grid.height - 1) * grid.width] - grid.pressure[0];
    const double expected_rise = grid.water_density * grid.gravity_y * grid.cell_size * (grid.height - 1);
    require(std::abs(pressure_rise - expected_rise) < expected_rise * 1e-8, "incorrect hydrostatic pressure gradient");
    requireValidState(grid, initial_volume);
}

void transportConservesWaterAndMovesItsCentroid()
{
    FluidGrid grid(64, 32, 1.0);
    std::fill(grid.u.begin(), grid.u.end(), 1.0);
    for (int y = 6; y < 18; ++y)
        for (int x = 8; x < 16; ++x)
            grid.water_fraction[y * grid.width + x] = 1.0;

    // a long transport interval also exercises its internal CFL substeps
    grid.advect(8.0);
    double mass = 0.0, moment_x = 0.0, moment_y = 0.0;
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
        {
            const double fraction = grid.water_fraction[y * grid.width + x];
            require(fraction >= 0.0 && fraction <= 1.0, "uniform transport violated fraction bounds");
            mass += fraction;
            moment_x += (x + 0.5) * fraction;
            moment_y += (y + 0.5) * fraction;
        }

    require(std::abs(mass - 96.0) < 1e-10, "uniform transport lost water");
    require(std::abs(moment_x / mass - 20.0) < 1e-10, "water did not travel velocity * time");
    require(std::abs(moment_y / mass - 12.0) < 1e-10, "horizontal transport moved water vertically");
}

void transportKeepsStraightInterfacesSharp()
{
    FluidGrid grid(40, 8, 1.0);
    std::fill(grid.u.begin(), grid.u.end(), 1.0);
    for (int y = 0; y < grid.height; ++y)
        for (int x = 8; x < 16; ++x)
            grid.water_fraction[y * grid.width + x] = 1.0;

    // uniform motion by eight whole cells should preserve both straight interfaces
    grid.advect(8.0);
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
        {
            const double expected = x >= 16 && x < 24 ? 1.0 : 0.0;
            require(std::abs(grid.water_fraction[y * grid.width + x] - expected) < 1e-12,
                    "uniform transport blurred a straight material interface");
        }
}

void closedPoolRemainsValidOverRepeatedMotion()
{
    FluidGrid grid(32, 32, 4.0);
    grid.air_density = 1.0;
    grid.water_density = 1000.0;
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
        {
            const double surface = grid.height * (0.5 + 0.04 * std::cos(3.14159265358979323846 * (x + 0.5) / grid.width));
            grid.water_fraction[y * grid.width + x] = std::clamp(y + 1.0 - surface, 0.0, 1.0);
        }

    const double initial_volume = waterVolume(grid);
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
        FluidGrid grid(48, 48, 1.0);
        grid.air_density = 1.0;
        grid.water_density = 1000.0;
        std::fill(grid.u.begin(), grid.u.end(), velocity.x);
        std::fill(grid.v.begin(), grid.v.end(), velocity.y);

        for (int y = 18; y < 30; ++y)
            for (int x = 18; x < 30; ++x)
                grid.water_fraction[y * grid.width + x] =
                    std::clamp(6.0 - std::hypot(x - 23.5, y - 23.5), 0.0, 1.0);

        const double initial_volume = waterVolume(grid);
        // diagonal transport exercises both momentum components in both passes and CFL substeps
        grid.advect(1.0);
        require(std::abs(waterVolume(grid) - initial_volume) < 1e-10, "uniform transport changed water volume");

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
    FluidGrid grid(8, 8, 1.0);
    grid.gravity_y = 1.0;
    grid.water_fraction[3 * grid.width + 3] = 1.0;

    const auto material = grid.water_fraction;
    const auto horizontal = grid.u;
    const auto vertical = grid.v;

    require(!grid.step(0.0).converged,
        "zero timestep unexpectedly advanced");

    require(grid.u == horizontal && grid.v == vertical &&
        grid.water_fraction == material,
        "zero timestep changed the state");

    // start with valid zero velocity; gravity creates the need for projection
    require(!grid.step(0.01, 0, 1e-8).converged,
        "zero-iteration projection unexpectedly converged");
}

}

int main()
{
    const struct { const char* name; void (*run)(); } tests[] = {
        {"staggered world-space sampling", samplesStaggeredVelocityInWorldSpace},
        {"hydrostatic equilibrium", pressureBalancesGravity},
        {"conservative material transport", transportConservesWaterAndMovesItsCentroid},
        {"sharp straight-interface transport", transportKeepsStraightInterfacesSharp},
        {"repeated closed-pool motion", closedPoolRemainsValidOverRepeatedMotion},
        {"uniform motion across density changes", transportPreservesUniformMotionAcrossDensityChanges},
        {"zero timestep and projection failure", zeroTimeAndFailedProjectionAreHandled}
    };
    int failures = 0;
    for (const auto& test : tests)
    {
        try { test.run(); std::cout << "PASS " << test.name << '\n'; }
        catch (const std::exception& error) { ++failures; std::cerr << "FAIL " << test.name << ": " << error.what() << '\n'; }
    }
    return failures == 0 ? 0 : 1;
}
