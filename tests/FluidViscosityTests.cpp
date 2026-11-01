#include "UniformWebGPU/ViscositySolverGPU.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace fluid::webgpu;

namespace
{
constexpr double pi = 3.14159265358979323846;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct Case
{
    int w;
    int h;
    double dx = 1;
    std::vector<float> material;
    std::vector<float> u;
    std::vector<float> v;

    Case(int w, int h, double rho, double mu) : w(w), h(h), material(2 * w * h), u((w + 1) * h), v(w * (h + 1))
    {
        for (int i = 0; i < w * h; ++i)
        {
            material[2 * i] = float(rho);
            material[2 * i + 1] = float(mu);
        }
    }

    int ui(int x, int y) const { return y * (w + 1) + x; }

    int vi(int x, int y) const { return y * w + x; }

    double rho(int x, int y) const { return material[2 * (y * w + x)]; }

    double mu(int x, int y) const { return material[2 * (y * w + x) + 1]; }

    double um(int x, int y) const { return 0.5 * (rho(x - 1, y) + rho(x, y)) * dx * dx; }

    double vm(int x, int y) const { return 0.5 * (rho(x, y - 1) + rho(x, y)) * dx * dx; }

    double energy() const
    {
        double sum = 0;
        for (int y = 0; y < h; ++y)
        {
            for (int x = 1; x < w; ++x)
                sum += um(x, y) * u[ui(x, y)] * u[ui(x, y)];
        }
        for (int y = 1; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
                sum += vm(x, y) * v[vi(x, y)] * v[vi(x, y)];
        }

        return 0.5 * sum;
    }

    void walls() const
    {
        for (int y = 0; y < h; ++y)
            require(u[ui(0, y)] == 0 && u[ui(w, y)] == 0, "Viscosity opened an X wall");
        for (int x = 0; x < w; ++x)
            require(v[vi(x, 0)] == 0 && v[vi(x, h)] == 0, "Viscosity opened a Y wall");
        for (const auto* field : {&u, &v})
        {
            for (auto value : *field)
                require(std::isfinite(value), "Nonfinite viscous velocity");
        }
    }
};

// Assemble forces by differentiating independent cell/vertex strain energies.
// This oracle does not use the GPU's gather stencil or preconditioner.
double residual(const Case& old, const Case& next, double dt)
{
    std::vector<double> ku(next.u.size()), kv(next.v.size());
    for (int y = 0; y < next.h; ++y)
    {
        for (int x = 0; x < next.w; ++x)
        {
            const double tx = 2 * next.mu(x, y) * (next.u[next.ui(x + 1, y)] - next.u[next.ui(x, y)]);
            const double ty = 2 * next.mu(x, y) * (next.v[next.vi(x, y + 1)] - next.v[next.vi(x, y)]);
            ku[next.ui(x, y)] -= tx;
            ku[next.ui(x + 1, y)] += tx;
            kv[next.vi(x, y)] -= ty;
            kv[next.vi(x, y + 1)] += ty;
        }
    }
    for (int y = 1; y < next.h; ++y)
    {
        for (int x = 1; x < next.w; ++x)
        {
            const double mu = 0.25 * (next.mu(x - 1, y - 1) + next.mu(x, y - 1) + next.mu(x - 1, y) + next.mu(x, y));
            const double t = mu * (next.u[next.ui(x, y)] - next.u[next.ui(x, y - 1)] + next.v[next.vi(x, y)] -
                                      next.v[next.vi(x - 1, y)]);
            ku[next.ui(x, y)] += t;
            ku[next.ui(x, y - 1)] -= t;
            kv[next.vi(x, y)] += t;
            kv[next.vi(x - 1, y)] -= t;
        }
    }

    double error = 0;

    for (int y = 0; y < next.h; ++y)
    {
        for (int x = 1; x < next.w; ++x)
        {
            const int i = next.ui(x, y);
            const double m = next.um(x, y);
            const double r = m * (next.u[i] - old.u[i]) + dt * ku[i];

            error += r * r / m;
        }
    }
    for (int y = 1; y < next.h; ++y)
    {
        for (int x = 0; x < next.w; ++x)
        {
            const int i = next.vi(x, y);
            const double m = next.vm(x, y);
            const double r = m * (next.v[i] - old.v[i]) + dt * kv[i];

            error += r * r / m;
        }
    }

    return std::sqrt(error / std::max(2 * old.energy(), 1e-30));
}

struct Fixture
{
    Compute& compute;
    Field u;
    Field v;
    Field material;
    ViscositySolverGPU solver;

    Fixture(Compute& compute, const Case& value)
        : compute(compute),
          u(compute.field(value.w + 1, value.h)),
          v(compute.field(value.w, value.h + 1)),
          material(compute.field(value.w, value.h, 2)),
          solver(compute, value.w, value.h)
    {
        upload(value);
    }

    void upload(const Case& value)
    {
        compute.write(u, value.u);
        compute.write(v, value.v);
        compute.write(material, value.material);
    }

    int step(Case& value, double dt, double tolerance = 1e-4)
    {
        const int count = solver.solve(u, v, material, dt, value.dx, tolerance);
        value.u = compute.read(u);
        value.v = compute.read(v);
        value.walls();

        return count;
    }
};

void decay(Compute& compute, int w, int h, double mu, double dt, bool transverse)
{
    Case value(w, h, 2.0, mu);
    value.dx = 0.7;
    const double kx = pi / w;
    const double ky = h > 1 ? pi / h : 0.0;
    const double sx = 2 * std::sin(0.5 * kx);
    const double sy = 2 * std::sin(0.5 * ky);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 1; x < w; ++x)
            value.u[value.ui(x, y)] = float(std::sin(kx * x) * (transverse ? std::cos(ky * (y + 0.5)) : 1.0));
    }

    if (transverse)
    {
        for (int y = 1; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
                value.v[value.vi(x, y)] = float(-sx / sy * std::cos(kx * (x + 0.5)) * std::sin(ky * y));
        }
    }

    const auto old = value;
    Fixture fixture(compute, value);
    const int count = fixture.step(value, dt);
    const double factor =
        1.0 / (1.0 + dt * mu * (transverse ? sx * sx + sy * sy : 2 * sx * sx) / (2 * value.dx * value.dx));
    double error = 0;

    for (size_t i = 0; i < value.u.size(); ++i)
        error = std::max(error, std::abs(value.u[i] - factor * old.u[i]));
    for (size_t i = 0; i < value.v.size(); ++i)
        error = std::max(error, std::abs(value.v[i] - factor * old.v[i]));
    const double r = residual(old, value, dt);
    std::cout << "decay " << w << 'x' << h << " mu=" << mu << " steps=" << count << " error=" << error
              << " residual=" << r << '\n';

    require(error < 3e-4, "Incorrect analytic backward-Euler viscous decay");
    require(r < 1.5e-4, "Viscosity true residual exceeds its requested accuracy");
    require(value.energy() <= old.energy() * (1 + 1e-6), "Viscosity increased kinetic energy");
}

void zeroAndRotation(Compute& compute)
{
    Case value(25, 19, 2, 0);
    for (int y = 0; y < value.h; ++y)
    {
        for (int x = 1; x < value.w; ++x)
            value.u[value.ui(x, y)] = float(0.3 * (y - 8));
    }
    for (int y = 1; y < value.h; ++y)
    {
        for (int x = 0; x < value.w; ++x)
            value.v[value.vi(x, y)] = float(-0.3 * (x - 10));
    }

    const auto old = value;
    Fixture fixture(compute, value);

    require(fixture.step(value, 0.01) == 0 && value.u == old.u && value.v == old.v, "Zero viscosity changed velocity");

    for (int y = 4; y < value.h - 4; ++y)
    {
        for (int x = 4; x < value.w - 4; ++x)
            value.material[2 * (y * value.w + x) + 1] = 500;
    }

    fixture.upload(value);
    fixture.step(value, 0.01);
    double error = 0;

    for (size_t i = 0; i < value.u.size(); ++i)
        error = std::max(error, std::abs(double(value.u[i] - old.u[i])));
    for (size_t i = 0; i < value.v.size(); ++i)
        error = std::max(error, std::abs(double(value.v[i] - old.v[i])));

    require(error < 3e-5, "Symmetric stress damped rigid rotation inside the viscous region");
    std::fill(value.u.begin(), value.u.end(), 0.0f);
    std::fill(value.v.begin(), value.v.end(), 0.0f);
    fixture.upload(value);

    require(fixture.step(value, 0.01) == 0 && value.energy() == 0.0, "Viscosity introduced motion at rest");
    std::cout << "PASS zero viscosity, zero flow and rigid rotation\n";
}

void variable(Compute& compute, int w, int h)
{
    Case value(w, h, 1, 0);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const bool dense = x < w / 2;
            value.material[2 * (y * w + x)] = dense ? 10.0f : 0.1f;
            value.material[2 * (y * w + x) + 1] = dense ? 10000.0f : 0.0f;
        }
    }
    for (int y = 0; y < h; ++y)
    {
        for (int x = 1; x < w; ++x)
            value.u[value.ui(x, y)] = float(std::sin(0.7 * x + 0.3 * y));
    }
    for (int y = 1; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
            value.v[value.vi(x, y)] = float(std::cos(0.2 * x + 0.8 * y));
    }

    Fixture fixture(compute, value);
    int maximum = 0;

    for (int i = 0; i < 40; ++i)
    {
        const auto old = value;
        maximum = std::max(maximum, fixture.step(value, 0.02));

        require(value.energy() <= old.energy() * (1 + 2e-6), "Variable viscosity increased kinetic energy");
        require(residual(old, value, 0.02) < 1.8e-4, "Variable-viscosity CPU residual failed");
    }

    std::cout << "PASS variable " << w << 'x' << h << " 40 steps; max iterations=" << maximum << '\n';
}

void iterationLimit(Compute& compute)
{
    Case value(33, 25, 1, 10000);
    for (int y = 0; y < value.h; ++y)
    {
        for (int x = 1; x < value.w; ++x)
            value.u[value.ui(x, y)] = float(std::sin(0.7 * x + 0.3 * y));
    }

    Fixture fixture(compute, value);

    require(fixture.step(value, 0.0) == 0, "Zero timestep performed viscous work");

    bool failed = false;
    try
    {
        fixture.solver.solve(fixture.u, fixture.v, fixture.material, 0.02, 1.0, 1e-4, 1);
    }
    catch (const std::runtime_error&)
    {
        failed = true;
    }

    require(failed, "An unconverged viscosity solve silently accepted its iteration cap");
    require(compute.read(fixture.u) == value.u && compute.read(fixture.v) == value.v,
        "Failed viscosity solve partially committed face velocities");

    const Case old = value;
    const int iterations = fixture.step(value, 0.02);

    require(iterations > 1 && residual(old, value, 0.02) < 1.8e-4,
        "Default viscosity solve did not continue to the original accuracy");
    fixture.upload(old);
    auto invalid = old.u;
    invalid[value.ui(5, 5)] = std::numeric_limits<float>::quiet_NaN();
    compute.write(fixture.u, invalid);
    failed = false;
    try
    {
        fixture.solver.solve(fixture.u, fixture.v, fixture.material, 0.02, 1.0);
    }
    catch (const std::runtime_error&)
    {
        failed = true;
    }

    require(failed && compute.read(fixture.v) == old.v, "Nonfinite input was accepted or partially committed");
    std::cout << "PASS zero timestep, explicit caller budget, unbounded default accuracy and nonfinite failure\n";
}

void drivenBoundary(Compute& compute)
{
    Case value(33, 25, 14.0, 300000.0);
    for (int y = 0; y < value.h; ++y)
    {
        for (int x = 1; x < value.w; ++x)
            value.u[value.ui(x, y)] = float(0.5 * std::sin(0.2 * x + 0.3 * y));
    }
    for (int y = 0; y <= value.h; ++y)
    {
        for (int x = 0; x < value.w; ++x)
            value.v[value.vi(x, y)] = float(20.0 + 2.0 * std::sin(pi * y / value.h) * std::cos(0.3 * x));
    }

    Fixture fixture(compute, value);

    for (const double dt : {0.001, 0.01, 0.02})
    {
        const auto old = value;
        const int iterations =
            fixture.solver.solve(fixture.u, fixture.v, fixture.material, dt, value.dx, 1e-4, 1024, true);
        value.u = compute.read(fixture.u);
        value.v = compute.read(fixture.v);

        for (int x = 0; x < value.w; ++x)
        {
            require(value.v[value.vi(x, 0)] == old.v[old.vi(x, 0)] &&
                        value.v[value.vi(x, value.h)] == old.v[old.vi(x, value.h)],
                "Viscosity changed a prescribed normal velocity");
        }

        const double error = residual(old, value, dt);

        require(error < 1.5e-4, "Driven viscosity failed independent strain-energy residual");
        std::cout << "PASS driven viscosity dt=" << dt << " iterations=" << iterations
                  << " independent residual=" << error << '\n';
    }
}

void benchmark(Compute& compute)
{
    using Clock = std::chrono::steady_clock;
    Case value(512, 420, 10, 0);
    for (int y = 0; y < value.h; ++y)
    {
        for (int x = 1; x < value.w; ++x)
            value.u[value.ui(x, y)] = float(100 * std::sin(0.07 * x + 0.03 * y));
    }
    for (int y = 1; y < value.h; ++y)
    {
        for (int x = 0; x < value.w; ++x)
            value.v[value.vi(x, y)] = float(100 * std::cos(0.02 * x + 0.08 * y));
    }

    Fixture fixture(compute, value);

    for (double mu : {100.0, 1000.0, 10000.0})
    {
        for (int y = 0; y < value.h; ++y)
        {
            for (int x = 0; x < value.w; ++x)
            {
                const bool liquid = y > value.h * 7 / 10;
                value.material[2 * (y * value.w + x)] = liquid ? 10.0f : 0.1f;
                value.material[2 * (y * value.w + x) + 1] = liquid ? float(mu) : 0.0f;
            }
        }

        double total = 0;
        int iterations = 0;

        for (int repeat = 0; repeat < 13; ++repeat)
        {
            fixture.upload(value);
            const auto start = Clock::now();
            const int count = fixture.solver.solve(fixture.u, fixture.v, fixture.material, 0.01, 1.0);
            compute.read(fixture.u, 0, sizeof(float));
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            if (repeat > 0)
            {
                total += ms;
                iterations += count;
            }
        }

        std::cout << "benchmark 512x420 mu=" << mu << " mean_ms=" << total / 12
                  << " mean_iterations=" << iterations / 12.0
                  << " relative_residual=" << fixture.solver.lastRelativeResidual() << '\n';
    }
}
}

int main(int argc, char** argv)
{
    try
    {
        std::cout << std::unitbuf;
        Compute compute;
        if (argc == 2 && std::string_view(argv[1]) == "--benchmark")
        {
            benchmark(compute);
            return 0;
        }

        zeroAndRotation(compute);
        decay(compute, 24, 16, 100, 0.02, false);
        decay(compute, 31, 23, 1000, 0.02, true);
        decay(compute, 65, 49, 100000, 0.1, true);
        decay(compute, 47, 1, 1000, 0.02, false);
        decay(compute, 2, 33, 1000, 0.02, false);
        variable(compute, 33, 25);
        variable(compute, 64, 48);
        iterationLimit(compute);
        drivenBoundary(compute);
        std::cout << "PASS implicit viscosity\n";

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
