#include "FluidGrid.h"
#include "FluidTestSetup.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using fluid::FluidGrid;
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
double maximum(const std::vector<double>& field) {
    double value = 0; for (double x : field) value = std::max(value, std::abs(x)); return value;
}
double sum(const std::vector<double>& field) { return std::accumulate(field.begin(), field.end(), 0.0); }
double kineticEnergy(const FluidGrid& grid) {
    double energy = 0;
    auto rho = [&](int i) {
        double mass = 0;
        for (const auto& field : grid.liquid_mass) mass += field[i];
        for (const auto& field : grid.gas_mass) mass += field[i];
        return mass; // cell mass; face ownership below supplies the area weighting
    };
    for (int y = 0; y < grid.height; ++y)
        for (int x = 1; x < grid.width; ++x)
            energy += .25 * (rho(y * grid.width + x - 1) + rho(y * grid.width + x)) * std::pow(grid.u[grid.uIndex(x, y)], 2);
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x) {
            int i = y * grid.width + x;
            energy += .25 * (rho(i) + (y > 0 ? rho(i - grid.width) : 0.0)) * grid.v[i] * grid.v[i];
        }
    return energy;
}

void preconditionerContract() {
    const int w = 35, h = 27;
    fluid::PressureMultigrid mg(w, h);
    std::vector<double> u((w + 1) * h), v(w * (h + 1));
    for (int y = 0; y < h; ++y) for (int x = 1; x < w; ++x) u[y * (w + 1) + x] = y > h / 2 ? .003 : 10;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) v[y * w + x] = y == 0 ? 20 : y > h / 2 ? .003 : 10;
    require(mg.rebuild(u, v, true), "ambient multigrid factorisation failed");
    std::vector<double> a(w * h), b(w * h), ma, mb;
    for (int i = 0; i < w * h; ++i) { a[i] = std::sin(i * .13); b[i] = std::cos(i * .31); }
    mg.apply(a, ma, 0); mg.apply(b, mb, 0);
    const double amb = std::inner_product(a.begin(), a.end(), mb.begin(), 0.0);
    const double bma = std::inner_product(b.begin(), b.end(), ma.begin(), 0.0);
    require(std::abs(amb - bma) < 1e-10 * std::max({1.0, std::abs(amb), std::abs(bma)}), "ambient preconditioner is not symmetric");
    require(std::inner_product(a.begin(), a.end(), ma.begin(), 0.0) > 0, "ambient preconditioner is not positive");
}

void equilibrium() {
    for (double damping : {0.0, 0.2})
    for (auto size : {std::pair{1, 1}, {1, 17}, {33, 25}, {65, 33}}) {
        FluidGrid grid(size.first, size.second, 0.7);
        grid.gas_damping_rate = damping;
        grid.top_ambient = FluidGrid::AmbientBoundary{37, {1}};
        for (int y = 0; y < grid.height; ++y)
            for (int x = 0; x < grid.width; ++x)
                grid.liquid_fraction[y * grid.width + x] = y >= grid.height * 2 / 3 ? 1 : 0;
        fluid::test::initializeTestMaterials(grid, 300, 0.1);
        for (int i = 0; i < 20; ++i) {
            auto result = grid.step(0.01, 2000, 1e-10);
            require(result.converged, "ambient hydrostatic projection failed");
            require(std::max(maximum(grid.u), maximum(grid.v)) < 2e-8, "ambient boundary created equilibrium motion");
        }
        double face_pressure = 37;
        for (int y = 0; y < grid.height; ++y) {
            const double rho = y >= grid.height * 2 / 3 ? 300 : .1;
            const double drop = rho * grid.gravity_y * grid.cell_size;
            for (int x = 0; x < grid.width; ++x)
                require(std::abs(grid.pressure[y * grid.width + x] - face_pressure - .5 * drop) < 1e-6,
                    "gas damping changed the hydrostatic pressure background");
            face_pressure += drop;
        }
    }
}

void gasDamping() {
    FluidGrid grid(6, 5, 1);
    grid.top_ambient = FluidGrid::AmbientBoundary{0, {1}};
    for (int y = 0; y < grid.height; ++y) for (int x = 3; x < grid.width; ++x)
        grid.liquid_fraction[y * grid.width + x] = 1;
    fluid::test::initializeTestMaterials(grid, 300, .1);
    for (int y = 0; y < grid.height; ++y) for (int x = 1; x < grid.width; ++x) grid.u[grid.uIndex(x, y)] = 2;
    for (int y = 0; y < grid.height; ++y) for (int x = 0; x < grid.width; ++x) grid.v[grid.vIndex(x, y)] = -3;
    const auto before = grid.download();
    grid.dampGasVelocity(.5);
    require(grid.u == before.u && grid.v == before.v, "zero damping changed velocities");
    grid.gas_damping_rate = .2;
    grid.dampGasVelocity(0);
    require(grid.u == before.u && grid.v == before.v, "zero-time damping changed velocities");
    const double energy_before = kineticEnergy(grid), retention = std::exp(-.2 * .5);
    grid.dampGasVelocity(.5);
    for (int y = 0; y < grid.height; ++y) for (int x = 1; x < grid.width; ++x)
        require(std::abs(grid.u[grid.uIndex(x, y)] - 2 * (x < 3 ? retention : 1)) < 1e-14,
            "gas damping affected liquid/interface faces or missed gas");
    for (int y = 0; y < grid.height; ++y) for (int x = 0; x < grid.width; ++x)
        require(std::abs(grid.v[grid.vIndex(x, y)] + 3 * (x < 3 ? retention : 1)) < 1e-14,
            "gas damping treated vertical/open faces differently");
    require(kineticEnergy(grid) < energy_before, "gas drag did not dissipate energy");
    require(grid.liquid_mass == before.liquid_mass && grid.gas_mass == before.gas_mass && grid.liquid_fraction == before.liquid_fraction,
        "gas drag changed material");
    const auto once = grid.download();
    grid.upload(before);
    grid.dampGasVelocity(.25); grid.dampGasVelocity(.25);
    for (size_t i = 0; i < grid.u.size(); ++i) require(std::abs(grid.u[i] - once.u[i]) < 1e-14, "damping depends on timestep subdivision");
    for (size_t i = 0; i < grid.v.size(); ++i) require(std::abs(grid.v[i] - once.v[i]) < 1e-14, "damping depends on timestep subdivision");
    require(grid.project(.5, 2000, 1e-8).converged, "pressure failed after local gas damping");
}

void projectionAgreement() {
    FluidGrid mg(35, 27, 1), diagonal(35, 27, 1);
    for (auto* grid : {&mg, &diagonal}) {
        grid->top_ambient = FluidGrid::AmbientBoundary{0, {1}};
        for (int y = 0; y < grid->height; ++y)
            for (int x = 0; x < grid->width; ++x)
                grid->liquid_fraction[y * grid->width + x] = y > 15 + 3 * std::sin(x * .3) ? 1 : 0;
        fluid::test::initializeTestMaterials(*grid, 300, 0.1);
        for (size_t i = 0; i < grid->u.size(); ++i) grid->u[i] = std::sin(i * .13);
        for (size_t i = 0; i < grid->v.size(); ++i) grid->v[i] = std::cos(i * .17);
    }
    diagonal.pressure_preconditioner = FluidGrid::PressurePreconditioner::Diagonal;
    const double energy_before = kineticEnergy(mg);
    require(mg.project(.01, 4000, 1e-9).converged, "ambient multigrid failed");
    require(kineticEnergy(mg) <= energy_before * (1 + 1e-10), "ambient projection injected kinetic energy");
    require(diagonal.project(.01, 4000, 1e-9).converged, "ambient diagonal reference failed");
    for (size_t i = 0; i < mg.u.size(); ++i) require(std::abs(mg.u[i] - diagonal.u[i]) < 1e-7, "ambient u differs from reference");
    for (size_t i = 0; i < mg.v.size(); ++i) require(std::abs(mg.v[i] - diagonal.v[i]) < 1e-7, "ambient v differs from reference");
    require(maximum(std::vector<double>(mg.v.begin(), mg.v.begin() + mg.width)) > .01, "ambient face behaved like a wall");
}

void materialExchange() {
    FluidGrid grid(24, 18, 1);
    const auto a = grid.elements.add({"A"}), b = grid.elements.add({"B"});
    grid.gas_mixture.components = {{a, .1}, {b, .3}};
    grid.top_ambient = FluidGrid::AmbientBoundary{0, {.25, .75}};
    FluidGrid::State state;
    state.gas_mass.assign(2, std::vector<double>(grid.width * grid.height));
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x) {
            double f = .2 + .6 * x / (grid.width - 1.0);
            state.gas_mass[0][y * grid.width + x] = .1 * f;
            state.gas_mass[1][y * grid.width + x] = .3 * (1 - f);
        }
    grid.upload(state);
    auto psi = [&](int x, int y) {
        return x == 0 || x == grid.width ? 0.0 : 2 * std::sin(3.141592653589793 * x / grid.width) * std::pow(1.0 - double(y) / grid.height, 2);
    };
    for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x <= grid.width; ++x) grid.u[grid.uIndex(x, y)] = psi(x, y + 1) - psi(x, y);
    for (int y = 0; y <= grid.height; ++y)
        for (int x = 0; x < grid.width; ++x) grid.v[grid.vIndex(x, y)] = -(psi(x + 1, y) - psi(x, y));
    require(grid.maxDivergence() < 1e-14, "exchange fixture is not divergence free");
    const double before[] = {sum(grid.gas_mass[0]), sum(grid.gas_mass[1])};
    grid.advect(.2);
    for (int k = 0; k < 2; ++k)
        require(std::abs(sum(grid.gas_mass[k]) - before[k] - grid.boundary_gas_mass[k]) < 1e-12, "unaccounted ingredient exchange");
    require(std::abs(grid.boundary_gas_mass[0]) > 1e-5, "exchange test did not replace material");
    for (size_t i = 0; i < grid.liquid_fraction.size(); ++i) {
        require(std::abs(grid.gas_mass[0][i] / .1 + grid.gas_mass[1][i] / .3 - 1) < 1e-12, "ambient gas volume mismatch");
        require(grid.gas_mass[0][i] >= 0 && grid.gas_mass[1][i] >= 0, "negative ambient ingredient");
    }
}

}

int main() {
    bl::Thread::setMaxThreads(8);
    try {
        equilibrium(); std::cout << "PASS ambient equilibrium and thin grids\n";
        gasDamping(); std::cout << "PASS optional gas drag, phase selection and timestep independence\n";
        preconditionerContract(); std::cout << "PASS ambient multigrid symmetry and positivity\n";
        projectionAgreement(); std::cout << "PASS ambient pressure reference agreement\n";
        materialExchange(); std::cout << "PASS ambient ingredient exchange and volume\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
