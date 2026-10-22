#include "FluidAmbientMask.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

using fluid::FluidGrid;
using Mask = fluid::FluidAmbientMask;
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
FluidGrid gas(int w = 11, int h = 9, double spacing = 1) {
    FluidGrid grid(w, h, spacing);
    grid.elements.add({"Liquid"}); grid.elements.add({"Gas A"}); grid.elements.add({"Gas B"});
    grid.liquid_mixture.components = {{0, 300}};
    grid.gas_mixture.components = {{1, .1}, {2, .1}};
    FluidGrid::State state;
    state.liquid_mass.assign(1, std::vector<double>(w * h));
    state.gas_mass.assign(2, std::vector<double>(w * h));
    state.gas_mass[0].assign(w * h, .1 * spacing * spacing);
    grid.upload(state);
    return grid;
}
Mask::Settings settings() {
    Mask::Settings s;
    s.liquid_fractions = {0}; s.gas_fractions = {1, 0};
    s.halo_cells = 0; s.quiet_seconds = 0; s.reservoir_edges = Mask::Top;
    return s;
}
void liquid(FluidGrid& grid, int i) {
    grid.liquid_fraction[i] = 1;
    grid.liquid_mass[0][i] = 300 * grid.cell_size * grid.cell_size;
    grid.gas_mass[0][i] = grid.gas_mass[1][i] = 0;
}
void referenceAndComposition() {
    for (auto [w, h] : {std::pair{1, 1}, {1, 17}, {19, 1}, {33, 25}}) {
        auto grid = gas(w, h, 2);
        auto s = settings(); s.pressure = 37; s.pressure_gradient = {2, -3};
        grid.origin = {7, 11};
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
            grid.pressure[y * w + x] = 37 + 2 * (7 + (x + .5) * 2) - 3 * (11 + (y + .5) * 2);
        const auto before = grid.download();
        Mask mask; mask.observe(grid, 0, s);
        require(mask.statistics.candidates == w * h, "matching affine ambient background was rejected");
        grid.gravity_y = -1234;
        mask.observe(grid, .1, s);
        require(mask.statistics.candidates == w * h, "classifier inferred its background from gravity");
        require(grid.download().pressure == before.pressure && grid.download().gas_mass == before.gas_mass, "observer mutated grid");
    }
    auto grid = gas(); auto s = settings(); Mask mask;
    grid.gas_mass[0][35] = .098; grid.gas_mass[1][35] = .002;
    mask.observe(grid, 0, s);
    require(mask.cells[35] == Mask::Material, "equal-density ingredient mismatch was missed");
    s.gas_fractions = {.98, .02};
    for (size_t i = 0; i < grid.pressure.size(); ++i) { grid.gas_mass[0][i] = .098; grid.gas_mass[1][i] = .002; }
    mask.observe(grid, 0, s);
    require(mask.statistics.candidates == 99, "ambient compound was not supported");
    for (int i = 0; i < 99; ++i) liquid(grid, i);
    s.liquid_fractions = {1}; s.gas_fractions = {0, 0};
    mask.observe(grid, 0, s);
    require(mask.statistics.candidates == 99, "classifier hard-coded a gas-only background");
}
void cloudColoursShareAtmosphere() {
    auto grid = gas(); auto s = settings(); Mask mask;
    // Each cell has a different colour split, but the same total atmospheric volume and density.
    for (int i = 0; i < 99; ++i) {
        const double cloud = (i % 11) / 10.0;
        grid.gas_mass[0][i] = .1 * (1 - cloud);
        grid.gas_mass[1][i] = .1 * cloud;
    }
    s.gas_tracer_elements = {1, 2};
    const auto before = grid.download();
    mask.observe(grid, .1, s);
    require(mask.statistics.candidates == 99, "cloud colour alone excluded quiet atmosphere");
    require(grid.download().gas_mass == before.gas_mass, "mask erased cloud colours");

    grid.u[grid.uIndex(5, 4)] = 4;
    mask.observe(grid, .1, s);
    require(mask.cells[4 * 11 + 5] == Mask::Disturbed, "cloud grouping hid atmospheric motion");
    std::fill(grid.u.begin(), grid.u.end(), 0);
    grid.pressure[4 * 11 + 5] = 2;
    mask.observe(grid, .1, s);
    require(mask.cells[4 * 11 + 5] == Mask::Disturbed, "cloud grouping hid pressure disturbance");
    std::fill(grid.pressure.begin(), grid.pressure.end(), 0);

    grid.gas_mass[0][20] = grid.gas_mass[1][20] = .1;
    liquid(grid, 21);
    const auto other = grid.elements.add({"Unrelated gas"});
    grid.gas_mixture.components.push_back({other, .1});
    grid.gas_mass.emplace_back(99, 0);
    grid.gas_mass[0][22] = grid.gas_mass[1][22] = 0;
    grid.gas_mass[2][22] = .1;
    s.gas_fractions.push_back(0);
    mask.observe(grid, .1, s);
    require(mask.cells[20] == Mask::Material, "cloud grouping hid an invalid doubled gas volume");
    require(mask.cells[21] == Mask::Material, "cloud grouping accepted liquid");
    require(mask.cells[22] == Mask::Material, "cloud grouping accepted an unrelated gas");
}
void motionPressureAndHistory() {
    auto grid = gas(); auto s = settings(); Mask mask;
    const int i = 4 * 11 + 5;
    grid.u[grid.uIndex(5, 4)] = 3; grid.u[grid.uIndex(6, 4)] = -3;
    mask.observe(grid, 0, s);
    require(mask.cells[i] == Mask::Disturbed, "opposing velocities cancelled");
    std::fill(grid.u.begin(), grid.u.end(), 0);
    std::fill(grid.pressure.begin(), grid.pressure.end(), 2);
    mask.observe(grid, 0, s);
    require(mask.statistics.disturbed == 99, "unchanging pressure departure escaped detection");
    s.pressure = 2;
    mask.observe(grid, 0, s);
    require(mask.statistics.candidates == 99, "matching pressure gauges were not respected");
    s.acceleration_tolerance = .1f;
    grid.pressure[i] += .03;
    mask.observe(grid, 0, s);
    require(mask.cells[i] == Mask::Disturbed, "small but steep pressure disturbance was missed");
    std::fill(grid.pressure.begin(), grid.pressure.end(), 2);
    s.quiet_seconds = .2f;
    mask.observe(grid, .1, s);
    require(mask.statistics.waiting == 99, "quiet delay was skipped");
    mask.observe(grid, .11, s);
    require(mask.statistics.candidates == 99, "sustained quiet did not become eligible");
    std::fill(grid.u.begin(), grid.u.end(), 1.5);
    mask.observe(grid, .1, s);
    require(mask.statistics.candidates == 99, "hysteresis failed to retain candidates");
    std::fill(grid.u.begin(), grid.u.end(), 3);
    mask.observe(grid, .1, s);
    require(mask.statistics.candidates == 0, "disturbance did not wake candidates");
    std::fill(grid.u.begin(), grid.u.end(), 1.5);
    mask.observe(grid, 1, s);
    require(mask.statistics.candidates == 0, "re-entry ignored lower threshold");
    std::fill(grid.u.begin(), grid.u.end(), 0);
    mask.observe(grid, .21, s);
    require(mask.statistics.candidates == 99, "candidates never recovered after disturbance");
    mask.observe(grid, 0, s);
    require(mask.statistics.waiting == 99, "reset retained quiet histories");
}
void connectivityBufferAndOrientation() {
    auto grid = gas(13, 13); auto s = settings(); Mask mask;
    for (int y = 3; y <= 9; ++y) for (int x = 3; x <= 9; ++x)
        if (x == 3 || x == 9 || y == 3 || y == 9) liquid(grid, y * 13 + x);
    mask.observe(grid, 0, s);
    require(mask.cells[6 * 13 + 6] == Mask::Isolated && mask.cells[0] == Mask::Candidate,
        "connectivity merged a confined quiet patch with the reservoir");
    s.halo_cells = 1;
    mask.observe(grid, 0, s);
    require(mask.cells[2 * 13 + 5] == Mask::Buffer && mask.cells[3 * 13 + 5] == Mask::Material,
        "buffer did not surround incompatible material");
    liquid(grid, 10 * 13 + 1);
    grid.u[grid.uIndex(8, 1)] = 5;
    mask.observe(grid, 0, s);
    const auto original = mask.cells;
    auto transposed = gas(13, 13);
    for (int y = 0; y < 13; ++y) for (int x = 0; x < 13; ++x)
        if (grid.liquid_fraction[y * 13 + x] > 0) liquid(transposed, x * 13 + y);
    for (int y = 0; y < 13; ++y) for (int x = 0; x <= 13; ++x)
        transposed.v[transposed.vIndex(y, x)] = grid.u[grid.uIndex(x, y)];
    s.reservoir_edges = Mask::Left;
    mask.observe(transposed, 0, s);
    for (int y = 0; y < 13; ++y) for (int x = 0; x < 13; ++x)
        require(original[y * 13 + x] == mask.cells[x * 13 + y], "classification preferred a grid direction");
    s.reservoir_edges = 0;
    mask.observe(transposed, 0, s);
    require(mask.statistics.candidates == 0 && mask.statistics.isolated > 0, "classifier invented a reservoir connection");
}
void localHysteresisBeforeBuffer() {
    auto grid = gas(); auto s = settings(); Mask mask;
    s.halo_cells = 1; s.quiet_seconds = .2f; s.acceleration_tolerance = 1000;
    const int source = 4 * 11 + 5, neighbour = source + 1;
    mask.observe(grid, .21, s);
    grid.pressure[source] = 1.2;
    mask.observe(grid, .01, s);
    require(mask.cells[source] == Mask::Disturbed && mask.cells[neighbour] == Mask::Buffer,
        "local wake did not produce a buffer");

    // Both cells now have the same score, but only the source crossed the wake threshold.
    // Being buffered must not turn the neighbour into another disturbance source.
    grid.pressure[source] = grid.pressure[neighbour] = .75;
    for (double pressure : {.75, .99, .51, .5}) {
        grid.pressure[source] = pressure;
        mask.observe(grid, .01, s);
        require(mask.statistics.disturbed == 1 && mask.cells[source] == Mask::Disturbed,
            "local disturbance did not retain its hysteresis state");
        require(mask.statistics.buffer == 8, "buffer expanded due to a receiving cell's history");
        for (int y = 0; y < 9; ++y) for (int x = 0; x < 11; ++x) {
            const int i = y * 11 + x;
            const bool rim = std::abs(x - 5) <= 1 && std::abs(y - 4) <= 1 && i != source;
            require((mask.cells[i] == Mask::Buffer) == rim, "orange was not a fixed rim around the source");
        }
    }
    grid.pressure[source] = .49;
    mask.observe(grid, .01, s);
    require(mask.statistics.disturbed == 0 && mask.statistics.buffer == 0 && mask.statistics.waiting == 9,
        "cleared disturbance left a stale buffer or skipped quiet time");
    mask.observe(grid, .21, s);
    require(mask.statistics.candidates == 99, "quiet buffered cells failed to recover");
    grid.pressure[neighbour] = 1;
    mask.observe(grid, .01, s);
    require(mask.cells[neighbour] == Mask::Disturbed, "full tolerance did not wake a locally quiet cell");

    std::fill(grid.pressure.begin(), grid.pressure.end(), 0);
    grid.gas_mass[0][source] = .1 * (1 - .0004);
    grid.gas_mass[1][source] = .1 * .0004;
    mask.observe(grid, .21, s);
    require(mask.statistics.candidates == 99, "compatible composition created a buffer without a source");
    grid.pressure[source] = .75;
    mask.observe(grid, .01, s);
    require(mask.cells[source] == Mask::Candidate, "local history was lost after buffer recovery");
    mask.reset();
    mask.observe(grid, .01, s);
    require(mask.cells[source] == Mask::Disturbed, "explicit reset retained old local disturbance history");
}
void unchangedPhysics() {
    auto plain = gas(33, 25), observed = gas(33, 25); auto s = settings(); Mask mask;
    for (auto* grid : {&plain, &observed}) {
        grid->top_ambient = FluidGrid::AmbientBoundary{0, {1, 0}};
        for (int y = 15; y < 25; ++y) for (int x = 0; x < 33; ++x)
            if (y > 15 + 2 * std::sin(x * .3)) liquid(*grid, y * 33 + x);
        grid->upload(grid->download());
    }
    s.pressure_gradient.y = .1 * observed.gravity_y;
    for (int tick = 0; tick < 30; ++tick) {
        require(plain.step(.01, 2000, 1e-8).converged && observed.step(.01, 2000, 1e-8).converged, "observed pool failed pressure solve");
        mask.observe(observed, .01, s);
    }
    const auto a = plain.download(), b = observed.download();
    require(a.u == b.u && a.v == b.v && a.pressure == b.pressure && a.liquid_fraction == b.liquid_fraction &&
        a.liquid_mass == b.liquid_mass && a.gas_mass == b.gas_mass, "ambient mask altered physics");
}
}
int main() {
    bl::Thread::setMaxThreads(8);
    try {
        referenceAndComposition(); cloudColoursShareAtmosphere(); motionPressureAndHistory(); connectivityBufferAndOrientation();
        localHysteresisBeforeBuffer(); unchangedPhysics();
        std::cout << "PASS ambient composition, reference pressure, disturbances, history, connectivity, orientation and unchanged physics\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
