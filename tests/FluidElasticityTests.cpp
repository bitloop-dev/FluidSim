#include "UniformCPU/FluidGridCPU.h"
#include "FluidGrid.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

using fluid::ElasticityCPU;
using fluid::FluidGridCPU;

namespace
{
void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

void configure(FluidGridCPU& grid, double modulus)
{
    fluid::SubstanceCatalogue substances;
    const auto solid = substances.add({"Elastic", {0.2f, 0.8f, 0.6f}});
    const auto gas = substances.add({"Gas", {0.0f, 0.0f, 0.0f}});
    grid.configureMaterials(std::move(substances), {{{solid, 10.0, 0.0, modulus}}}, {{{gas, 0.01}}});
    grid.gravity_y = 0.0;
}

FluidGridCPU::State block(int size, double dx, double stretch, bool history = true)
{
    FluidGridCPU::State state;
    const int count = size * size;
    state.liquid_fraction.resize(count);
    state.liquid_mass.assign(1, std::vector<double>(count));
    state.gas_mass.assign(1, std::vector<double>(count));
    if (history)
    {
        state.elastic_log_xx.assign(count, 2.0 * std::log(stretch));
        state.elastic_log_xy.assign(count, 0.0);
        state.elastic_log_yy.assign(count, -2.0 * std::log(stretch));
    }
    const double side = 0.3 * size;
    const double left = 0.5 * (size - side * stretch);
    const double top = 0.5 * (size - side / stretch);
    auto overlap = [](double x, double start, double length)
    {
        return std::max(0.0, std::min(x + 1.0, start + length) - std::max(x, start));
    };
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            const int i = y * size + x;
            const double f = overlap(x, left, side * stretch) * overlap(y, top, side / stretch);
            state.liquid_fraction[i] = f;
            state.liquid_mass[0][i] = 10.0 * f * dx * dx;
            state.gas_mass[0][i] = 0.01 * (1.0 - f) * dx * dx;
        }
    }
    return state;
}

void affineStress()
{
    const int size = 32;
    ElasticityCPU elastic(size, size, 0.5, 7.0);
    auto state = block(size, 0.5, 1.2);
    elastic.upload(state);
    elastic.rebuildStress(state.liquid_fraction);
    const double expected = 3.5 * (1.2 * 1.2 - 1.0 / (1.2 * 1.2));
    for (int i = 0; i < size * size; ++i)
    {
        require(
            std::abs(elastic.normal_stress[i] - state.liquid_fraction[i] * expected) < 1e-10, "affine stretch stress");
    }

    // A rotated stretch transforms dev(log B) through twice the physical angle.
    const double angle = 0.7;
    const double strain = 2.0 * std::log(1.2);
    std::fill(state.liquid_fraction.begin(), state.liquid_fraction.end(), 1.0);
    std::fill(state.elastic_log_xx.begin(), state.elastic_log_xx.end(), strain * std::cos(2.0 * angle));
    std::fill(state.elastic_log_xy.begin(), state.elastic_log_xy.end(), strain * std::sin(2.0 * angle));
    std::fill(state.elastic_log_yy.begin(), state.elastic_log_yy.end(), -strain * std::cos(2.0 * angle));
    elastic.upload(state);
    elastic.rebuildStress(state.liquid_fraction);
    for (double value : elastic.normal_stress)
        require(std::abs(value - expected * std::cos(2.0 * angle)) < 1e-12, "rotated stretch normal stress");
    require(std::abs(elastic.shear_stress[16 * (size + 1) + 16] - expected * std::sin(2.0 * angle)) < 1e-12,
        "rotated stretch shear stress");
}

void translatedHistory()
{
    FluidGridCPU grid(64, 64, 0.5);
    configure(grid, 7.0);
    auto state = block(64, 0.5, 1.2);
    for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
    {
        state.elastic_log_xx[i] += 0.07;
        state.elastic_log_xy[i] = 0.125;
        state.elastic_log_yy[i] += 0.07;
    }
    state.u.assign(65 * 64, 0.25);
    state.v.assign(64 * 65, -0.125);
    grid.upload(state);
    for (int step = 0; step < 60; ++step)
        grid.advect(0.1);
    state = grid.download();
    double error = 0.0;
    for (int i = 0; i < 64 * 64; ++i)
    {
        if (state.liquid_fraction[i] <= 0.0)
            continue;
        error = std::max({error, std::abs(state.elastic_log_xx[i] - 0.07 - 2.0 * std::log(1.2)),
            std::abs(state.elastic_log_xy[i] - 0.125), std::abs(state.elastic_log_yy[i] - 0.07 + 2.0 * std::log(1.2))});
    }
    std::cout << "translated strain error=" << error << '\n';
    require(error < 1e-10, "phase-flux translation changed uniform strain");
}

void internalForcesBalance()
{
    constexpr int size = 32;
    auto state = block(size, 1.0, 1.2);
    for (int i = 0; i < size * size; ++i)
        state.elastic_log_xy[i] = 0.15;
    ElasticityCPU elastic(size, size, 1.0, 7.0);
    elastic.upload(state);
    elastic.rebuildStress(state.liquid_fraction);
    std::vector<double> u((size + 1) * size, 0.0);
    std::vector<double> v(size * (size + 1), 0.0);
    std::vector<double> inverse_u(u.size(), 0.0);
    std::vector<double> inverse_v(v.size(), 0.0);
    auto density = [&](int i)
    {
        return state.liquid_mass[0][i] + state.gas_mass[0][i];
    };
    for (int y = 0; y < size; ++y)
    {
        for (int x = 1; x < size; ++x)
            inverse_u[y * (size + 1) + x] = 2.0 / (density(y * size + x) + density(y * size + x - 1));
    }
    for (int y = 1; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
            inverse_v[y * size + x] = 2.0 / (density(y * size + x) + density((y - 1) * size + x));
    }
    elastic.applyForces(0.01, u, v, inverse_u, inverse_v);
    double px = 0.0;
    double py = 0.0;
    double torque = 0.0;
    for (int y = 0; y < size; ++y)
    {
        for (int x = 1; x < size; ++x)
        {
            const int i = y * (size + 1) + x;
            const double momentum = u[i] / inverse_u[i];
            px += momentum;
            torque -= (y + 0.5) * momentum;
        }
    }
    for (int y = 1; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            const int i = y * size + x;
            const double momentum = v[i] / inverse_v[i];
            py += momentum;
            torque += (x + 0.5) * momentum;
        }
    }
    require(std::abs(px) < 1e-12 && std::abs(py) < 1e-12 && std::abs(torque) < 1e-11,
        "internal elastic stress introduced net force or torque");
}

void workCompatibleSource()
{
    constexpr int size = 12;
    constexpr double dx = 0.7;
    constexpr double modulus = 7.0;
    auto state = block(size, dx, 1.2);
    std::vector<double> u((size + 1) * size);
    std::vector<double> v(size * (size + 1));
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            const int i = y * size + x;
            state.liquid_fraction[i] = x > 1 && x < 10 && y > 1 && y < 10 ? 0.6 + 0.3 * std::sin(x + 2.0 * y) : 0.0;
            state.elastic_log_xx[i] = 0.04 + 0.2 * std::sin(x + y);
            state.elastic_log_xy[i] = 0.13 * std::cos(x - y);
            state.elastic_log_yy[i] = 0.04 - 0.2 * std::sin(x + y);
        }
        for (int x = 1; x < size; ++x)
            u[y * (size + 1) + x] = std::sin(0.3 * x + 0.7 * y);
    }
    for (int y = 1; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
            v[y * size + x] = std::cos(0.8 * x - 0.2 * y);
    }

    ElasticityCPU elastic(size, size, dx, modulus);
    elastic.upload(state);
    elastic.rebuildStress(state.liquid_fraction);
    std::vector<double> force_u(u.size());
    std::vector<double> force_v(v.size());
    elastic.applyForces(1.0, force_u, force_v, std::vector<double>(u.size(), 1.0), std::vector<double>(v.size(), 1.0));
    const double force_work = dx * dx *
                              (std::inner_product(u.begin(), u.end(), force_u.begin(), 0.0) +
                                  std::inner_product(v.begin(), v.end(), force_v.begin(), 0.0));
    auto strain_energy = [&]
    {
        double total = 0.0;
        for (size_t i = 0; i < elastic.history.size(); ++i)
        {
            const auto& strain = elastic.history[i];
            total += state.liquid_fraction[i] * dx * dx * modulus *
                     (std::exp(strain[2]) * std::cosh(std::hypot(strain[0], strain[1])) - 1.0);
        }
        return total;
    };
    constexpr double dt = 1e-6;
    elastic.deform(dt, state.liquid_fraction, u, v);
    const double forward = strain_energy();
    elastic.upload(state);
    elastic.deform(-dt, state.liquid_fraction, u, v);
    const double backward = strain_energy();
    const double strain_work = (forward - backward) / (2.0 * dt);
    const double error = std::abs(strain_work + force_work);
    std::cout << "elastic work-adjoint error=" << error << '\n';
    require(error < 1e-7 * std::max(1.0, std::abs(force_work)),
        "deformation source is not the adjoint of MAC elastic stress divergence");
}

void restAndReplay()
{
    FluidGridCPU grid(32, 32, 1.0);
    configure(grid, 100.0);
    grid.upload(block(32, 1.0, 1.0, false));
    for (int i = 0; i < 30; ++i)
        require(grid.step(0.01, 500, 1e-9).converged, "rest projection failed");
    for (double value : grid.u)
        require(std::abs(value) < 1e-10, "rest generated velocity");

    grid.upload(block(32, 1.0, 1.15));
    for (int i = 0; i < 20; ++i)
        require(grid.step(0.01, 500, 1e-8).converged, "initial stretch failed");
    FluidGridCPU restored(32, 32, 1.0);
    configure(restored, 100.0);
    restored.upload(grid.download());
    for (int i = 0; i < 20; ++i)
    {
        require(grid.step(0.01, 500, 1e-8).converged, "continued stretch failed");
        require(restored.step(0.01, 500, 1e-8).converged, "restored stretch failed");
    }
    const auto a = grid.download();
    const auto b = restored.download();
    double error = 0.0;
    auto compare = [&](const auto& left, const auto& right)
    {
        for (size_t i = 0; i < left.size(); ++i)
            error = std::max(error, std::abs(left[i] - right[i]));
    };
    compare(a.u, b.u);
    compare(a.v, b.v);
    compare(a.elastic_log_xx, b.elastic_log_xx);
    compare(a.elastic_log_xy, b.elastic_log_xy);
    compare(a.elastic_log_yy, b.elastic_log_yy);
    std::cout << "replay error=" << error << '\n';

    // The pressure predictor is a solver cache, absent from the portable state.
    require(error < 1e-7, "upload/download changed the elastic trajectory beyond pressure tolerance");
}

void rotationAndGasSlip()
{
    constexpr int size = 32;
    constexpr double omega = 0.3;
    ElasticityCPU elastic(size, size, 1.0, 7.0);
    auto state = block(size, 1.0, 1.0);
    state.liquid_fraction.assign(size * size, 1.0);
    state.elastic_log_xx.assign(size * size, 0.37);
    state.elastic_log_xy.assign(size * size, 0.1);
    state.elastic_log_yy.assign(size * size, -0.23);
    std::vector<double> u((size + 1) * size);
    std::vector<double> v(size * (size + 1));
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x <= size; ++x)
            u[y * (size + 1) + x] = -omega * (y + 0.5);
    }
    for (int y = 0; y <= size; ++y)
    {
        for (int x = 0; x < size; ++x)
            v[y * size + x] = omega * (x + 0.5);
    }
    elastic.upload(state);

    // A full rotation tests accumulation as well as every intermediate orientation.
    const double duration = 2.0 * std::acos(-1.0) / omega;
    const double dt = duration / 1000;
    for (int step = 1; step <= 1000; ++step)
    {
        elastic.deform(dt, state.liquid_fraction, u, v);
        const double angle = 2.0 * omega * step * dt;
        const double expected_d = 0.3 * std::cos(angle) - 0.1 * std::sin(angle);
        const double expected_s = 0.3 * std::sin(angle) + 0.1 * std::cos(angle);
        for (const auto& strain : elastic.history)
        {
            require(std::abs(strain[0] - expected_d) < 1e-11 && std::abs(strain[1] - expected_s) < 1e-11,
                "material strain did not rotate objectively");
            require(std::abs(strain[2] - 0.07) < 1e-15, "rotation changed existing volume memory");
        }
    }

    // Fast tangential gas cannot deform or rotate a prestretched free-slip slab.
    for (int y = 0; y < size; ++y)
    {
        const double fraction = y < 14 ? 1.0 : (y == 14 ? 0.5 : 0.0);
        for (int x = 0; x < size; ++x)
            state.liquid_fraction[y * size + x] = fraction;
        for (int x = 0; x <= size; ++x)
            u[y * (size + 1) + x] = y > 14 ? 500.0 : 0.0;
    }
    std::fill(v.begin(), v.end(), 0.0);
    elastic.upload(state);
    const auto initial = elastic.history;
    for (int step = 0; step < 100; ++step)
        elastic.deform(0.01, state.liquid_fraction, u, v);
    require(elastic.history == initial, "gas slip contaminated elastic history");
    std::cout << "full rotation and prestretched gas slip passed\n";
}

void affineExtensionPreservesVolume()
{
    constexpr int size = 32;
    ElasticityCPU elastic(size, size, 1.0, 7.0);
    auto state = block(size, 1.0, 1.0, false);
    state.liquid_fraction.assign(size * size, 1.0);
    elastic.upload(state);
    std::vector<double> u((size + 1) * size);
    std::vector<double> v(size * (size + 1));
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x <= size; ++x)
            u[y * (size + 1) + x] = x - size * 0.5;
    }
    for (int y = 0; y <= size; ++y)
    {
        for (int x = 0; x < size; ++x)
            v[y * size + x] = -(y - size * 0.5);
    }
    for (int step = 0; step < 100; ++step)
        elastic.deform(0.01, state.liquid_fraction, u, v);
    for (const auto& strain : elastic.history)
    {
        require(std::abs(strain[0] - 2.0) < 1e-12 && std::abs(strain[1]) < 1e-12,
            "affine extension strain differs from exact exponential");
        require(strain[2] == 0.0, "affine extension changed volume memory");
    }
}

double shearWave(int size)
{
    constexpr double amplitude = 0.001;
    FluidGridCPU grid(size, size, 1.0);
    configure(grid, 100.0);
    auto state = block(size, 1.0, 1.0, false);
    std::fill(state.liquid_fraction.begin(), state.liquid_fraction.end(), 1.0);
    std::fill(state.liquid_mass[0].begin(), state.liquid_mass[0].end(), 10.0);
    std::fill(state.gas_mass[0].begin(), state.gas_mass[0].end(), 0.0);
    state.u.resize((size + 1) * size);
    state.v.resize(size * (size + 1));
    const double k = std::acos(-1.0) / size;
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x <= size; ++x)
            state.u[y * (size + 1) + x] = amplitude * std::sin(k * x) * std::cos(k * (y + 0.5));
    }
    for (int y = 0; y <= size; ++y)
    {
        for (int x = 0; x < size; ++x)
            state.v[y * size + x] = -amplitude * std::cos(k * (x + 0.5)) * std::sin(k * y);
    }
    grid.upload(state);
    const double frequency = std::sqrt(20.0) * k;
    const double period = 2.0 * std::acos(-1.0) / frequency;
    const int steps = 1800;
    double max_error = 0.0;
    const double norm = std::inner_product(state.u.begin(), state.u.end(), state.u.begin(), 0.0) +
                        std::inner_product(state.v.begin(), state.v.end(), state.v.begin(), 0.0);
    for (int step = 1; step <= steps; ++step)
    {
        require(grid.step(period / steps, 500, 1e-10).converged, "shear-wave pressure failed");
        const double projection = (std::inner_product(state.u.begin(), state.u.end(), grid.u.begin(), 0.0) +
                                      std::inner_product(state.v.begin(), state.v.end(), grid.v.begin(), 0.0)) /
                                  norm;
        max_error = std::max(max_error, std::abs(projection - std::cos(frequency * period * step / steps)));
    }
    std::cout << "shear-wave size=" << size << " normalized velocity error=" << max_error << '\n';
    require(max_error < 0.035, "elastic wave frequency/amplitude differs from analytic small-strain solution");
    return max_error;
}

void stiffnessChanges()
{
    FluidGridCPU grid(24, 24, 1.0);
    FluidGridCPU reference(24, 24, 1.0);
    configure(grid, 100.0);
    grid.upload(block(24, 1.0, 1.05));
    require(grid.step(.002, 1000, 1e-8).converged, "CPU stiffness warmup failed");

    for (double coefficient : {1.0, 1000000.0, 50000.0})
    {
        const auto state = grid.download();
        grid.setElasticStiffness(coefficient);
        require(grid.download().elastic_log_xx == state.elastic_log_xx, "CPU stiffness edit discarded deformation");
        configure(reference, coefficient);
        reference.upload(state);
        require(grid.step(.002, 1000, 1e-8).converged && reference.step(.002, 1000, 1e-8).converged,
            "CPU stiffness continuation failed");
        double error = 0;
        double scale = 1;
        for (const auto& pair : {std::pair{&grid.u, &reference.u}, std::pair{&grid.v, &reference.v}})
        {
            for (size_t i = 0; i < pair.first->size(); ++i)
            {
                error = std::max(error, std::abs((*pair.first)[i] - (*pair.second)[i]));
                scale = std::max(scale, std::abs((*pair.second)[i]));
            }
        }
        std::cout << "CPU stiffness=" << coefficient << " relative error=" << error / scale << '\n';
        require(error < 1e-8 * scale, "Live CPU stiffness differs from a freshly configured material");
    }
}

void capabilitiesAndDisabledState()
{
    FluidGridCPU first(24, 24, 1.0);
    FluidGridCPU second(24, 24, 1.0);
    configure(first, 0.0);
    configure(second, 100.0);
    second.upload(block(24, 1.0, 1.15));
    configure(second, 0.0);
    const auto state = block(24, 1.0, 1.0, false);
    first.upload(state);
    second.upload(state);
    first.gravity_y = second.gravity_y = 1.0;
    for (int step = 0; step < 15; ++step)
    {
        require(first.step(0.01, 500, 1e-9).converged, "ordinary fluid step failed");
        require(second.step(0.01, 500, 1e-9).converged, "zero-modulus fluid step failed");
    }
    require(first.u == second.u && first.v == second.v && first.liquid_fraction == second.liquid_fraction,
        "disabled elasticity changed the fluid path");
    require(second.download().elastic_log_xx.empty(), "disabled elasticity retained history");

    fluid::FluidGrid unsupported(24, 24, 1.0);
    configure(first, 10.0);
    unsupported.configureMaterials(first.substances, first.liquid_mixture, first.gas_mixture);
    bool rejected = false;
    try
    {
        unsupported.setAdaptiveMode(fluid::AdaptiveMode::MAX_2);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected && unsupported.getDevice() == fluid::FluidDevice::CPU, "unsupported backend discarded elasticity");
}

void invalidStateAndStiffSubsteps()
{
    FluidGridCPU grid(256, 256, 1.0);
    configure(grid, 100.0);
    auto state = block(256, 1.0, 1.0);
    state.elastic_log_xy[128 * 256 + 128] = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try
    {
        grid.upload(state);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected, "nonfinite imported strain was silently accepted");

    // Finite history can still overflow stress; join workers before reporting this numerical failure.
    state = block(256, 1.0, 1.0);
    state.elastic_log_xx[128 * 256 + 128] = 1500.0;
    grid.upload(state);
    rejected = false;
    try
    {
        grid.step(0.01, 500, 1e-8);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    require(rejected, "overflowing elastic stress was silently accepted");
    grid.upload(block(256, 1.0, 1.0));
    require(grid.step(0.01, 500, 1e-8).converged, "restart after parallel elastic failure did not recover");

    FluidGridCPU stiff(24, 24, 1.0);
    configure(stiff, 100000.0);
    stiff.upload(block(24, 1.0, 1.01));
    const auto result = stiff.step(0.01, 1000, 1e-8);
    require(result.converged && result.physical_steps > 1, "stiff elasticity skipped complete wave-limited substeps");
}

double aspect(const FluidGridCPU& grid)
{
    double xx = 0.0;
    double yy = 0.0;
    for (int y = 0; y < grid.height; ++y)
    {
        for (int x = 0; x < grid.width; ++x)
        {
            const double f = grid.liquid_fraction[y * grid.width + x];
            xx += f * (x + 0.5 - grid.width * 0.5) * (x + 0.5 - grid.width * 0.5);
            yy += f * (y + 0.5 - grid.height * 0.5) * (y + 0.5 - grid.height * 0.5);
        }
    }
    return std::sqrt(xx / yy);
}

struct Health
{
    double energy = 0.0;
    double distortion_energy = 0.0;
    double maximum_volume_error = 0.0;
    double rms_volume_error = 0.0;
    double speed = 0.0;
};

Health inspect(const FluidGridCPU& grid)
{
    auto state = grid.download();
    Health result;
    const double area = grid.cell_size * grid.cell_size;
    const double modulus = grid.liquid_mixture.components.front().shear_modulus;
    double volume = 0.0;
    for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
    {
        const double fraction = state.liquid_fraction[i];
        if (fraction <= 0.0)
            continue;

        const double difference = 0.5 * (state.elastic_log_xx[i] - state.elastic_log_yy[i]);
        const double shear = state.elastic_log_xy[i];
        const double mean = 0.5 * (state.elastic_log_xx[i] + state.elastic_log_yy[i]);
        require(std::isfinite(difference) && std::isfinite(shear) && std::isfinite(mean), "nonfinite elastic strain");
        const double radius = std::hypot(difference, shear);
        const double trace = std::exp(mean + radius) + std::exp(mean - radius);
        const double volume_ratio = std::exp(mean);
        require(std::isfinite(trace) && volume_ratio > 0.0, "invalid elastic tensor");
        result.energy += fraction * area * 0.5 * modulus * (trace - 2.0);
        result.distortion_energy += fraction * area * modulus * (std::cosh(radius) - 1.0);
        result.maximum_volume_error = std::max(result.maximum_volume_error, std::abs(volume_ratio - 1.0));
        result.rms_volume_error += fraction * (volume_ratio - 1.0) * (volume_ratio - 1.0);
        volume += fraction;
    }
    result.rms_volume_error = std::sqrt(result.rms_volume_error / volume);
    auto density = [&](int i)
    {
        return (grid.liquid_mass[0][i] + grid.gas_mass[0][i]) / area;
    };
    for (int y = 0; y < grid.height; ++y)
    {
        for (int x = 1; x < grid.width; ++x)
        {
            const int i = y * grid.width + x;
            const double velocity = grid.u[y * (grid.width + 1) + x];
            require(std::isfinite(velocity), "nonfinite elastic velocity");
            result.speed = std::max(result.speed, std::abs(velocity));
            result.energy += 0.25 * area * (density(i) + density(i - 1)) * velocity * velocity;
            result.distortion_energy += 0.25 * area * (density(i) + density(i - 1)) * velocity * velocity;
        }
    }
    for (int y = 1; y < grid.height; ++y)
    {
        for (int x = 0; x < grid.width; ++x)
        {
            const int i = y * grid.width + x;
            const double velocity = grid.v[i];
            require(std::isfinite(velocity), "nonfinite elastic velocity");
            result.speed = std::max(result.speed, std::abs(velocity));
            result.energy += 0.25 * area * (density(i) + density(i - grid.width)) * velocity * velocity;
            result.distortion_energy += 0.25 * area * (density(i) + density(i - grid.width)) * velocity * velocity;
        }
    }
    return result;
}

void oscillation(int size, double duration, double dt)
{
    FluidGridCPU grid(size, size, 48.0 / size);
    configure(grid, 100.0);
    grid.upload(block(size, grid.cell_size, 1.15));
    const double initial_mass = std::accumulate(grid.liquid_mass[0].begin(), grid.liquid_mass[0].end(), 0.0);
    double minimum = aspect(grid);
    double maximum = minimum;
    double max_volume_error = 0.0;
    int substeps = 0;
    const double initial_energy = inspect(grid).energy;
    double max_energy = initial_energy;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < int(std::ceil(duration / dt)); ++step)
    {
        const auto result = [&]
        {
            try
            {
                return grid.step(dt, 1000, 1e-9);
            }
            catch (...)
            {
                std::cerr << "failed at time=" << (step + 1) * dt << '\n';
                inspect(grid);
                throw;
            }
        }();
        require(result.converged, "oscillation pressure failed");
        substeps += result.physical_steps;
        for (size_t i = 0; i < grid.liquid_fraction.size(); ++i)
        {
            const double f = grid.liquid_fraction[i];
            require(std::isfinite(f) && f >= -1e-8 && f <= 1.0 + 1e-8, "oscillation lost bounded fraction");
            const double volume = grid.liquid_mass[0][i] / (10.0 * grid.cell_size * grid.cell_size);
            max_volume_error = std::max(max_volume_error, std::abs(volume - f));
        }
        const double current = aspect(grid);
        minimum = std::min(minimum, current);
        maximum = std::max(maximum, current);
        const auto health = inspect(grid);
        max_energy = std::max(max_energy, health.energy);
        require(
            health.distortion_energy < 1.2 * initial_energy, "elastic distortion energy grew without external work");
        if (duration <= 6.0)
        {
            // Accuracy budget for the first recovery cycle; longer CLI runs report cumulative drift separately.
            require(health.rms_volume_error < 0.025 && health.maximum_volume_error < 0.15,
                "first-cycle volume memory exceeded the experimental budget");
            require(health.distortion_energy > 0.6 * initial_energy, "first-cycle numerical damping erased elasticity");
        }
        if (step % std::max(1, int(1.0 / dt)) == 0)
        {
            std::cout << "t=" << (step + 1) * dt << " aspect=" << current << " substeps=" << substeps
                      << " energy=" << health.energy / initial_energy
                      << " distortion=" << health.distortion_energy / initial_energy
                      << " Jmax=" << health.maximum_volume_error << " Jrms=" << health.rms_volume_error
                      << " speed=" << health.speed << std::endl;
            require(health.energy < 1.2 * initial_energy, "elastic energy grew without external work");
        }
    }
    const double mass = std::accumulate(grid.liquid_mass[0].begin(), grid.liquid_mass[0].end(), 0.0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "size=" << size << " simulated=" << duration << " wall=" << seconds << " substeps=" << substeps
              << " aspect_range=" << minimum << ',' << maximum << " mass_drift=" << mass / initial_mass - 1.0
              << " volume_error=" << max_volume_error << '\n';
    require(std::abs(mass / initial_mass - 1.0) < 1e-9 && max_volume_error < 1e-6, "oscillation mass/geometry drift");
    if (duration >= 5.0)
        require(minimum < 1.0 && maximum < 1.4, "block did not recover through its relaxed shape");
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc > 1)
            oscillation(std::stoi(argv[1]), argc > 2 ? std::stod(argv[2]) : 10.0, argc > 3 ? std::stod(argv[3]) : 0.01);
        else
        {
            affineStress();
            internalForcesBalance();
            workCompatibleSource();
            translatedHistory();
            rotationAndGasSlip();
            affineExtensionPreservesVolume();
            const double coarse_wave = shearWave(20);
            const double fine_wave = shearWave(40);
            require(fine_wave < 0.6 * coarse_wave, "elastic wave did not converge with grid refinement");
            capabilitiesAndDisabledState();
            stiffnessChanges();
            invalidStateAndStiffSubsteps();
            restAndReplay();
            oscillation(32, 6.0, 0.01);
        }
        std::cout << "PASS elasticity\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL elasticity: " << error.what() << '\n';
        return 1;
    }
}
