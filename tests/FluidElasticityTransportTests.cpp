#include "FluidFieldInternal.h"
#include "UniformWebGPU/ElasticityGPU.h"
#include "UniformWebGPU/FluidGridGPU.h"
#include "UniformCPU/ElasticityCPU.h"
#include <cmath>
#include <iostream>
#include <numeric>

using namespace fluid;
using namespace fluid::webgpu;

namespace
{
void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

FluidGridGPU::State block(int n, double stretch = 1.15)
{
    FluidGridGPU::State s;
    s.liquid_fraction.resize(n * n);
    s.liquid_mass.assign(1, std::vector<float>(n * n));
    s.gas_mass.assign(1, std::vector<float>(n * n));
    s.elastic_log_xx.assign(n * n, float(2 * std::log(stretch)));
    s.elastic_log_xy.assign(n * n, 0);
    s.elastic_log_yy.assign(n * n, float(-2 * std::log(stretch)));
    auto overlap = [n](int x, double length)
    {
        return std::max(0., std::min(x + 1., n * .5 + length * .5) - std::max(double(x), n * .5 - length * .5));
    };
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            const int i = y * n + x;
            const float f = float(overlap(x, .3 * n * stretch) * overlap(y, .3 * n / stretch));
            s.liquid_fraction[i] = f;
            s.liquid_mass[0][i] = 10 * f;
            s.gas_mass[0][i] = .01f * (1 - f);
        }
    }
    return s;
}

void configure(FluidGridGPU& grid, const FluidWebGPUContext& context)
{
    grid.setWebGPUContext(context);
    const auto solid = grid.substances.add({"Elastic", {.2f, .8f, .6f}});
    const auto gas = grid.substances.add({"Gas", {0, 0, 0}});
    grid.liquid_mixture = {{{solid, 10, 0, 1000}}};
    grid.gas_mixture = {{{gas, .01}}};
    grid.gravity_y = 0;
}

void strainKinematics(Compute& compute)
{
    constexpr int n = 32;
    auto f = compute.field(n, n);
    auto u = compute.field(n + 1, n);
    auto v = compute.field(n, n + 1);
    ElasticityGPU elastic(compute, n, n, 1, 1000);
    auto state = block(n);
    state.liquid_fraction.assign(n * n, 1);
    state.elastic_log_xx.assign(n * n, .37f);
    state.elastic_log_xy.assign(n * n, .1f);
    state.elastic_log_yy.assign(n * n, -.23f);
    compute.write(f, state.liquid_fraction);
    std::vector<float> horizontal((n + 1) * n);
    std::vector<float> vertical(n * (n + 1));
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x <= n; ++x)
            horizontal[y * (n + 1) + x] = float(-.3 * (y + .5));
    }
    for (int y = 0; y <= n; ++y)
    {
        for (int x = 0; x < n; ++x)
            vertical[y * n + x] = float(.3 * (x + .5));
    }
    compute.write(u, horizontal);
    compute.write(v, vertical);
    elastic.upload(state);
    const double duration = 2 * std::acos(-1.0) / .3;
    for (int i = 0; i < 1000; ++i)
        elastic.deform(duration / 1000, f, u, v);
    const auto values = compute.read(elastic.history);
    const double expected_d = .3;
    const double expected_s = .1;
    double error = 0;
    for (int i = 0; i < n * n; ++i)
    {
        error = std::max({error, std::abs(values[3 * i] - expected_d), std::abs(values[3 * i + 1] - expected_s)});
        require(std::abs(values[3 * i + 2] - .07) < 1e-7, "rotation changed existing volume memory");
    }
    require(error < 5e-5, "material log strain did not rotate objectively");

    // Fast tangential gas must not deform or rotate a prestretched free-slip slab.
    for (int y = 0; y < n; ++y)
    {
        const float fraction = y < 14 ? 1.0f : (y == 14 ? .5f : 0.0f);
        for (int x = 0; x < n; ++x)
            state.liquid_fraction[y * n + x] = fraction;
        for (int x = 0; x <= n; ++x)
            horizontal[y * (n + 1) + x] = y > 14 ? 500.0f : 0.0f;
    }
    std::fill(vertical.begin(), vertical.end(), 0.0f);
    compute.write(f, state.liquid_fraction);
    compute.write(u, horizontal);
    compute.write(v, vertical);
    elastic.upload(state);
    const auto initial = compute.read(elastic.history);
    for (int i = 0; i < 100; ++i)
        elastic.deform(.01, f, u, v);
    require(compute.read(elastic.history) == initial, "gas slip contaminated elastic history");
    std::cout << "rotation error=" << error << "; prestretched gas slip exact\n";
}

void translation(Compute& compute)
{
    constexpr int n = 32;
    auto f = compute.field(n, n);
    auto u = compute.field(n + 1, n);
    auto v = compute.field(n, n + 1);
    auto tx = compute.field(n + 1, n, 3);
    auto ty = compute.field(n, n + 1, 3);
    ElasticityGPU elastic(compute, n, n, 1, 1000);
    auto state = block(n);
    auto fractions = [](double shift)
    {
        std::vector<float> result(n * n);
        for (int y = 0; y < n; ++y)
        {
            for (int x = 0; x < n; ++x)
            {
                result[y * n + x] =
                    float(std::max(0., std::min(x + 1., 20.6 + shift) - std::max(double(x), 4.2 + shift)));
            }
        }
        return result;
    };
    state.liquid_fraction = fractions(0);
    elastic.upload(state);
    compute.write(u, std::vector<float>((n + 1) * n, .5f));
    compute.write(v, std::vector<float>(n * (n + 1), 0));
    compute.write(ty, std::vector<float>(3 * n * (n + 1), 0));
    double shift = 0;
    for (int step = 0; step < 50; ++step)
    {
        compute.write(f, state.liquid_fraction);
        std::vector<float> flux(3 * (n + 1) * n);
        for (int y = 0; y < n; ++y)
        {
            for (int x = 1; x < n; ++x)
            {
                flux[3 * (y * (n + 1) + x)] =
                    float(std::max(0., std::min(double(x), 20.6 + shift) - std::max(x - .1, 4.2 + shift)));
            }
        }
        compute.write(tx, flux);
        elastic.advect(.2, true, u, f, tx);
        for (int y = 0; y < n; ++y)
        {
            for (int x = 0; x < n; ++x)
                state.liquid_fraction[y * n + x] += flux[3 * (y * (n + 1) + x)] - flux[3 * (y * (n + 1) + x + 1)];
        }
        compute.write(f, state.liquid_fraction);
        elastic.advect(.2, false, v, f, ty);
        shift += .1;
    }
    elastic.download(state);
    double error = 0;
    for (int i = 0; i < n * n; ++i)
    {
        if (state.liquid_fraction[i] > 1e-4)
            error = std::max(error, std::abs(state.elastic_log_xx[i] - 2 * std::log(1.15)));
    }
    require(error < 2e-5, "material flux transport changed uniform strain at a cut interface");
    std::cout << "cut-interface translation error=" << error << '\n';
}

void discreteCoupling(Compute& compute)
{
    // Odd rectangular dimensions exercise the last reduction group and both MAC face layouts.
    constexpr int width = 37;
    constexpr int height = 19;
    constexpr double dx = .7;
    constexpr double modulus = 7;
    FluidGridState<float> state;
    const int cells = width * height;
    state.liquid_fraction.resize(cells);
    state.elastic_log_xx.resize(cells);
    state.elastic_log_xy.resize(cells);
    state.elastic_log_yy.resize(cells);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int i = y * width + x;
            state.liquid_fraction[i] =
                x > 1 && x < width - 2 && y > 1 && y < height - 2 ? float(.6 + .3 * std::sin(x + 2.0 * y)) : 0;
            state.elastic_log_xx[i] = float(.04 + .2 * std::sin(x + y));
            state.elastic_log_xy[i] = float(.13 * std::cos(x - y));
            state.elastic_log_yy[i] = float(.04 - .2 * std::sin(x + y));
        }
    }
    auto fraction = compute.field(width, height);
    auto density = compute.field(width, height, 2);
    auto u = compute.field(width + 1, height);
    auto v = compute.field(width, height + 1);
    auto next_u = compute.field(width + 1, height);
    auto next_v = compute.field(width, height + 1);
    compute.write(fraction, state.liquid_fraction);
    std::vector<float> rho(2 * cells, 0);
    for (int i = 0; i < cells; ++i)
        rho[2 * i] = 1;
    compute.write(density, rho);
    std::vector<float> horizontal((width + 1) * height, 0);
    std::vector<float> vertical(width * (height + 1), 0);
    compute.write(u, horizontal);
    compute.write(v, vertical);
    ElasticityGPU gpu(compute, width, height, float(dx), float(modulus));
    ElasticityCPU cpu(width, height, float(dx), modulus);
    gpu.upload(state);
    cpu.upload(convertFluidState<double>(state));
    gpu.rebuildStress(fraction);
    cpu.rebuildStress(std::vector<double>(state.liquid_fraction.begin(), state.liquid_fraction.end()));
    gpu.applyForces(1, u, v, next_u, next_v, density);
    const auto force_u = compute.read(u);
    const auto force_v = compute.read(v);
    std::vector<double> cpu_u(horizontal.size(), 0);
    std::vector<double> cpu_v(vertical.size(), 0);
    std::vector<double> inverse_u(cpu_u.size(), 1);
    std::vector<double> inverse_v(cpu_v.size(), 1);
    cpu.applyForces(1, cpu_u, cpu_v, inverse_u, inverse_v);
    double error = 0;
    for (size_t i = 0; i < force_u.size(); ++i)
        error = std::max(error, std::abs(force_u[i] - cpu_u[i]));
    for (size_t i = 0; i < force_v.size(); ++i)
        error = std::max(error, std::abs(force_v[i] - cpu_v[i]));
    require(error < 1e-5, "rectangular GPU stress forces differ from CPU");
    const double limit = cpu.stepLimit(inverse_u, inverse_v);
    require(std::abs(gpu.stepLimit(density) / limit - 1) < 1e-6, "GPU elastic wave reduction differs from CPU");

    for (int y = 0; y < height; ++y)
    {
        for (int x = 1; x < width; ++x)
            horizontal[y * (width + 1) + x] = float(std::sin(.3 * x + .7 * y));
    }
    for (int y = 1; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
            vertical[y * width + x] = float(std::cos(.8 * x - .2 * y));
    }
    compute.write(u, horizontal);
    compute.write(v, vertical);
    const double force_work = dx * dx *
                              (std::inner_product(horizontal.begin(), horizontal.end(), force_u.begin(), 0.0) +
                                  std::inner_product(vertical.begin(), vertical.end(), force_v.begin(), 0.0));
    auto energy = [&]
    {
        const auto values = compute.read(gpu.history);
        double total = 0;
        for (int i = 0; i < cells; ++i)
        {
            total +=
                state.liquid_fraction[i] * dx * dx * modulus *
                (std::exp(values[3 * i + 2]) * std::cosh(std::hypot(double(values[3 * i]), double(values[3 * i + 1]))) -
                    1);
        }
        return total;
    };
    constexpr double dt = .002;
    gpu.deform(dt, fraction, u, v);
    const double forward = energy();
    gpu.upload(state);
    gpu.deform(-dt, fraction, u, v);
    const double backward = energy();
    const double work_error = std::abs((forward - backward) / (2 * dt) + force_work);
    require(work_error < .002 * std::max(1.0, std::abs(force_work)), "GPU strain source violates stress-work balance");
    std::cout << "rectangular force error=" << error << " work-adjoint error=" << work_error << '\n';
}

void smallStrainStress(Compute& compute)
{
    constexpr int n = 8;
    auto state = block(n, 1);
    state.liquid_fraction.assign(n * n, 1);
    state.elastic_log_xx.assign(n * n, 1e-8f);
    state.elastic_log_xy.assign(n * n, 2e-8f);
    state.elastic_log_yy.assign(n * n, -1e-8f);
    auto fraction = compute.field(n, n);
    compute.write(fraction, state.liquid_fraction);
    ElasticityGPU elastic(compute, n, n, 1, 7);
    elastic.upload(state);
    elastic.rebuildStress(fraction);
    const auto normal = compute.read(elastic.normal);
    const auto shear = compute.read(elastic.shear);
    require(std::abs(normal[0] / 7e-8 - 1) < 1e-6 && std::abs(shear[3 * (4 * (n + 1) + 4)] / 1.4e-7 - 1) < 1e-6,
        "GPU stress lost small represented elastic strains");
}
}

int main()
{
    try
    {
        FluidWebGPUContext context;
        auto& compute = fluid::detail::FluidFieldAccess::commands(context);
        strainKinematics(compute);
        translation(compute);
        discreteCoupling(compute);
        smallStrainStress(compute);
        std::cout << "PASS elastic material transport\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
