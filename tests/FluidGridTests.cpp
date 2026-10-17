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

void stateTransferPreservesCPUBehaviour()
{
    FluidGrid direct(32, 24, 4), uploaded(32, 24, 4);
    FluidGrid::State initial;
    initial.water_fraction.resize(32 * 24);
    for (int y = 0; y < 24; ++y)
    for (int x = 0; x < 32; ++x)
        direct.water_fraction[y * 32 + x] = initial.water_fraction[y * 32 + x] = std::clamp(y + 1.0 - (x + 12.0), 0.0, 1.0);
    uploaded.upload(initial);
    // uploading the same initial fields must preserve the original CPU trajectory exactly
    for (int tick = 0; tick < 120; ++tick)
    {
        require(direct.step(1.0 / 120, 2000, 1e-8).converged, "direct CPU initialization failed");
        require(uploaded.step(1.0 / 120, 2000, 1e-8).converged, "uploaded CPU initialization failed");
        require(direct.u == uploaded.u && direct.v == uploaded.v && direct.water_fraction == uploaded.water_fraction && direct.pressure == uploaded.pressure,
                "state upload changed the CPU trajectory");
    }
    const auto state = uploaded.download();
    require(state.u == uploaded.u && state.v == uploaded.v && state.water_fraction == uploaded.water_fraction && state.pressure == uploaded.pressure,
            "CPU download omitted authoritative fields");
    uploaded.upload(initial);
    FluidGrid restarted(32, 24, 4);
    restarted.upload(initial);
    require(uploaded.step(1.0 / 120, 2000, 1e-8).converged && restarted.step(1.0 / 120, 2000, 1e-8).converged,
            "CPU reinitialization failed");
    require(uploaded.pressure == restarted.pressure && uploaded.u == restarted.u && uploaded.v == restarted.v,
            "CPU reinitialization retained stale pressure history");
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
        FluidGrid multigrid(size.x, size.y, 4.0);
        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x)
                multigrid.water_fraction[y * size.x + x] = std::clamp(y + 1.0 - (0.35 * x + 0.2 * size.y), 0.0, 1.0);
        for (size_t i = 0; i < multigrid.u.size(); ++i) multigrid.u[i] = 3.0 * std::sin(i * 0.07);
        for (size_t i = 0; i < multigrid.v.size(); ++i) multigrid.v[i] = 4.0 * std::cos(i * 0.11);
        FluidGrid diagonal = multigrid;
        diagonal.pressure_preconditioner = FluidGrid::PressurePreconditioner::Diagonal;
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
    FluidGrid grid(512, 512, 4.0);
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
            grid.water_fraction[y * grid.width + x] = std::clamp(y + 1.0 - (0.35 * x + 0.2 * grid.height), 0.0, 1.0);
    for (size_t i = 0; i < grid.u.size(); ++i) grid.u[i] = 3.0 * std::sin(i * 0.07);
    for (size_t i = 0; i < grid.v.size(); ++i) grid.v[i] = 4.0 * std::cos(i * 0.11);
    // Protect large-grid convergence without making the regression suite replay a whole wave.
    const auto result = grid.project(1.0 / 120.0);
    require(result.converged && result.max_divergence <= 1e-6, "512x512 mixed-density pressure exhausted the default iteration budget");
}

}

int main()
{
    const struct { const char* name; void (*run)(); } tests[] = {
        {"staggered world-space sampling", samplesStaggeredVelocityInWorldSpace},
        {"state transfer and CPU initialization equivalence", stateTransferPreservesCPUBehaviour},
        {"hydrostatic equilibrium", pressureBalancesGravity},
        {"conservative material transport", transportConservesWaterAndMovesItsCentroid},
        {"sharp straight-interface transport", transportKeepsStraightInterfacesSharp},
        {"repeated closed-pool motion", closedPoolRemainsValidOverRepeatedMotion},
        {"uniform motion across density changes", transportPreservesUniformMotionAcrossDensityChanges},
        {"zero timestep and projection failure", zeroTimeAndFailedProjectionAreHandled},
        {"multigrid linearity, symmetry and positivity", multigridPreservesPCGContract},
        {"multigrid and diagonal pressure agreement", multigridMatchesDiagonalProjection},
        {"large mixed-density pressure convergence", largeMixedPressureConvergesWithinDefaultBudget}
    };
    int failures = 0;
    for (const auto& test : tests)
    {
        try { test.run(); std::cout << "PASS " << test.name << '\n'; }
        catch (const std::exception& error) { ++failures; std::cerr << "FAIL " << test.name << ": " << error.what() << '\n'; }
    }
    return failures == 0 ? 0 : 1;
}
