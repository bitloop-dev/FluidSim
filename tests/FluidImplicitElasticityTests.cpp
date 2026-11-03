#include "UniformWebGPU/ImplicitElasticityGPU.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <chrono>

using namespace fluid::webgpu;

namespace fluid::webgpu
{
struct ImplicitElasticityGPUTestAccess
{
    static void denseReference(ImplicitElasticityGPU& solver)
    {
        auto& compute = solver.compute;
        const int width = solver.parameters.grid[0];
        const int height = solver.parameters.grid[1];
        const auto computed = compute.read(solver.solution);
        std::vector<int> indices;
        for (int y = 0; y <= height; ++y)
        {
            for (int x = 0; x <= width; ++x)
            {
                const int i = 3 * (y * (width + 1) + x);
                if (x > 0 && x < width && y < height)
                    indices.push_back(i);
                if (y > 0 && y < height && x < width)
                    indices.push_back(i + 1);
                // Pin one pressure unknown to remove only the constant gauge.
                if (x < width && y < height && (x != 0 || y != 0))
                    indices.push_back(i + 2);
            }
        }
        const int n = int(indices.size());
        std::vector<double> matrix(size_t(n) * n);
        std::vector<float> basis(computed.size());
        for (int col = 0; col < n; ++col)
        {
            basis[indices[col]] = 1;
            compute.write(solver.w1, basis);
            solver.apply(solver.w1, solver.product);
            const auto product = compute.read(solver.product);
            for (int row = 0; row < n; ++row)
                matrix[row * n + col] = product[indices[row]];
            basis[indices[col]] = 0;
        }
        compute.write(solver.solution, basis);
        auto u = compute.field(width + 1, height);
        auto v = compute.field(width, height + 1);
        solver.trueResidual(solver.r2, u, v);
        const auto rhs = compute.read(solver.r2);
        std::vector<double> reference(n);
        for (int row = 0; row < n; ++row)
            reference[row] = rhs[indices[row]];

        // Independent double-precision elimination of the sampled frozen operator.
        for (int col = 0; col < n; ++col)
        {
            int pivot = col;
            for (int row = col + 1; row < n; ++row)
            {
                if (std::abs(matrix[row * n + col]) > std::abs(matrix[pivot * n + col]))
                    pivot = row;
            }
            if (std::abs(matrix[pivot * n + col]) < 1e-12)
                throw std::runtime_error("dense coupled reference is singular");
            for (int j = col; j < n; ++j)
                std::swap(matrix[col * n + j], matrix[pivot * n + j]);
            std::swap(reference[col], reference[pivot]);
            for (int row = col + 1; row < n; ++row)
            {
                const double factor = matrix[row * n + col] / matrix[col * n + col];
                for (int j = col + 1; j < n; ++j)
                    matrix[row * n + j] -= factor * matrix[col * n + j];
                reference[row] -= factor * reference[col];
            }
        }
        for (int row = n - 1; row >= 0; --row)
        {
            for (int col = row + 1; col < n; ++col)
                reference[row] -= matrix[row * n + col] * reference[col];
            reference[row] /= matrix[row * n + row];
        }
        double maximum = 0;
        double scale = 0;
        for (int row = 0; row < n; ++row)
        {
            if (indices[row] % 3 == 2)
                continue;
            maximum = std::max(maximum, std::abs(reference[row] - computed[indices[row]]));
            scale = std::max(scale, std::abs(reference[row]));
        }
        std::cout << "dense reference velocity error=" << maximum << " scale=" << scale << '\n';
        if (maximum > 2e-4 * std::max(1.0, scale))
            throw std::runtime_error("coupled solve differs from double-precision dense reference");
    }

    static void kinematics(Compute& compute, double strain_rate, double spin)
    {
        constexpr int width = 12;
        constexpr int height = 10;
        constexpr float dt = .2f;
        auto fraction = compute.field(width, height);
        auto u = compute.field(width + 1, height);
        auto v = compute.field(width, height + 1);
        auto pressure = compute.field(width, height);
        compute.write(fraction, std::vector<float>(width * height, 1));
        fluid::FluidGridState<float> state;
        state.liquid_fraction.assign(width * height, 1);
        state.elastic_log_xx.assign(width * height, .37f);
        state.elastic_log_xy.assign(width * height, .1f);
        state.elastic_log_yy.assign(width * height, -.23f);
        ElasticityGPU elastic(compute, width, height, 1, 1000);
        elastic.upload(state);
        ImplicitElasticityGPU solver(compute, width, height, 1, 1000);
        auto& params = solver.parameters;
        params.value[1] = dt;
        std::vector<float> horizontal((width + 1) * height);
        std::vector<float> vertical(width * (height + 1));
        std::vector<float> velocity((width + 1) * (height + 1) * 3);
        for (int y = 0; y <= height; ++y)
        {
            for (int x = 0; x <= width; ++x)
            {
                const float ux = float(strain_rate * x - spin * (y + .5));
                const float vy = float(spin * (x + .5) - strain_rate * y);
                if (y < height)
                    horizontal[y * (width + 1) + x] = ux;
                if (x < width)
                    vertical[y * width + x] = vy;
                velocity[3 * (y * (width + 1) + x)] = ux;
                velocity[3 * (y * (width + 1) + x) + 1] = vy;
            }
        }
        compute.write(u, horizontal);
        compute.write(v, vertical);
        compute.write(solver.solution, velocity);
        compute.run(solver.corner, {&solver.corners}, {{"fraction", &fraction}}, params);
        compute.run(solver.prepare, {&solver.material},
            {{"history", &elastic.history}, {"fraction", &fraction}, {"corners", &solver.corners}, {"u", &u},
                {"v", &v}},
            params);
        compute.run(solver.strain, {&solver.stress},
            {{"source", &solver.solution}, {"material", &solver.material}, {"corners", &solver.corners}}, params);
        compute.run(solver.commit, {&u, &v, &solver.next_history, &pressure},
            {{"solution", &solver.solution}, {"material", &solver.material}, {"stress", &solver.stress},
                {"history", &elastic.history}},
            params, {width + 1, height + 1, 1});
        const auto history = compute.read(solver.next_history);
        const double radius = std::hypot(.3, .1);
        const double angle = 2 * spin * dt;
        const double old_scale = std::exp(.07) * std::sinh(radius) / radius;
        const double x = old_scale * (.3 * std::cos(angle) - .1 * std::sin(angle)) +
                         2 * dt * std::exp(.07) * std::cosh(radius) * strain_rate;
        const double y = old_scale * (.3 * std::sin(angle) + .1 * std::cos(angle));
        double maximum = 0;
        for (int row = 2; row < height - 2; ++row)
        {
            for (int col = 2; col < width - 2; ++col)
            {
                const auto i = 3 * (row * width + col);
                const double r = std::hypot(double(history[i]), double(history[i + 1]));
                const double scale = std::exp(double(history[i + 2])) * std::sinh(r) / r;
                maximum = std::max({maximum, std::abs(history[i] * scale - x), std::abs(history[i + 1] * scale - y),
                    std::abs(history[i + 2] - .07)});
            }
        }
        if (maximum > 2e-6)
            throw std::runtime_error("implicit affine/spin/volume constitutive update is inconsistent");
        std::cout << "kinematics strain=" << strain_rate << " spin=" << spin << " error=" << maximum << '\n';
    }

    static void symmetry(ImplicitElasticityGPU& solver)
    {
        auto& compute = solver.compute;
        const int width = solver.parameters.grid[0];
        const int height = solver.parameters.grid[1];
        std::vector<float> a(size_t(width + 1) * (height + 1) * 3);
        std::vector<float> b(a.size());
        for (int y = 0; y <= height; ++y)
        {
            for (int x = 0; x <= width; ++x)
            {
                const auto i = (y * (width + 1) + x) * 3;
                for (int lane = 0; lane < 3; ++lane)
                {
                    const bool valid = lane == 0   ? x > 0 && x < width && y < height
                                       : lane == 1 ? y > 0 && y < height && x < width
                                                   : x < width && y < height;
                    if (valid)
                    {
                        a[i + lane] = float(std::sin((i + lane) * 1.37));
                        b[i + lane] = float(std::cos((i + lane) * .71));
                    }
                }
            }
        }
        compute.write(solver.w1, a);
        compute.write(solver.w2, b);
        solver.apply(solver.w1, solver.product);
        const auto aa = compute.read(solver.product);
        solver.apply(solver.w2, solver.product);
        const auto ab = compute.read(solver.product);
        const double left = std::inner_product(a.begin(), a.end(), ab.begin(), 0.0);
        const double right = std::inner_product(b.begin(), b.end(), aa.begin(), 0.0);
        double scale = 0;
        for (size_t i = 0; i < a.size(); ++i)
            scale += std::abs(double(a[i]) * ab[i]) + std::abs(double(b[i]) * aa[i]);
        std::cout << "operator symmetry=" << std::abs(left - right) / scale << '\n';
        if (std::abs(left - right) > 2e-7 * scale)
            throw std::runtime_error("implicit saddle operator is not symmetric");

        compute.write(solver.r2, a);
        solver.preconditionResidual();
        const auto pa = compute.read(solver.z);
        compute.write(solver.r2, b);
        solver.preconditionResidual();
        const auto pb = compute.read(solver.z);
        const double pre_left = std::inner_product(a.begin(), a.end(), pb.begin(), 0.0);
        const double pre_right = std::inner_product(b.begin(), b.end(), pa.begin(), 0.0);
        const double energy = std::inner_product(a.begin(), a.end(), pa.begin(), 0.0);
        double pre_scale = 0;
        for (size_t i = 0; i < a.size(); ++i)
            pre_scale += std::abs(double(a[i]) * pb[i]) + std::abs(double(b[i]) * pa[i]);
        std::cout << "preconditioner symmetry=" << std::abs(pre_left - pre_right) / pre_scale << " energy=" << energy
                  << '\n';
        if (std::abs(pre_left - pre_right) > 2e-6 * pre_scale || !(energy > 0))
            throw std::runtime_error("implicit preconditioner lost symmetry or positivity");

        // Compare the fused tile/halo calculation with three independently staged sweeps of the actual operator.
        const auto diagonal = compute.read(solver.diagonal);
        std::vector<float> reference(a.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (i % 3 != 2)
                reference[i] = .5f * a[i] / diagonal[(i / 3) * 4 + i % 3];
        }
        for (int sweep = 0; sweep < 2; ++sweep)
        {
            compute.write(solver.w1, reference);
            solver.apply(solver.w1, solver.product);
            const auto applied = compute.read(solver.product);
            for (size_t i = 0; i < a.size(); ++i)
            {
                if (i % 3 != 2)
                    reference[i] += .5f * (a[i] - applied[i]) / diagonal[(i / 3) * 4 + i % 3];
            }
        }
        double error_squared = 0;
        double reference_squared = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (i % 3 != 2)
            {
                error_squared += std::pow(double(pa[i]) - reference[i], 2);
                reference_squared += double(reference[i]) * reference[i];
            }
        }
        std::cout << "preconditioner polynomial error=" << std::sqrt(error_squared / reference_squared) << '\n';
        if (error_squared > 4e-12 * reference_squared)
            throw std::runtime_error("fused preconditioner differs from staged velocity operator");

        std::vector<float> sum(a.size());
        for (size_t i = 0; i < sum.size(); ++i)
            sum[i] = a[i] + b[i];
        compute.write(solver.r2, sum);
        solver.preconditionResidual();
        const auto combined = compute.read(solver.z);
        error_squared = 0;
        reference_squared = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            const double expected = double(pa[i]) + pb[i];
            error_squared += std::pow(double(combined[i]) - expected, 2);
            reference_squared += expected * expected;
        }
        if (error_squared > 16e-12 * reference_squared)
            throw std::runtime_error("implicit preconditioner is not linear");
        compute.write(solver.r2, a);
        solver.preconditionResidual();
        if (compute.read(solver.z) != pa)
            throw std::runtime_error("implicit preconditioner depends on previous workspace contents");

        std::vector<float> constant(a.size());
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
                constant[3 * (y * (width + 1) + x) + 2] = 7;
        }
        compute.write(solver.r2, constant);
        solver.preconditionResidual();
        for (float value : compute.read(solver.z))
        {
            if (std::abs(value) > 1e-6)
                throw std::runtime_error("implicit preconditioner retained constant pressure gauge");
        }
    }

    static void tangentContrast(ImplicitElasticityGPU& solver, const Field& density)
    {
        auto& compute = solver.compute;
        auto material = compute.read(solver.material);
        const int width = solver.parameters.grid[0];
        const int height = solver.parameters.grid[1];
        material[4 * ((height / 2) * width + width / 2) + 2] *= 10000;
        compute.write(solver.material, material);
        compute.run(solver.build_diagonal, {&solver.diagonal},
            {{"material", &solver.material}, {"corners", &solver.corners}, {"density", &density}}, solver.parameters);
        compute.run(solver.weights, {&solver.coefficients}, {{"diagonal", &solver.diagonal}}, solver.parameters);
        solver.multigrid.rebuild(solver.coefficients);
        symmetry(solver);
    }
};
}

namespace
{
void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

double energy(Compute& compute, ElasticityGPU& elastic, const Field& u, const Field& v, double modulus)
{
    double result = 0;
    const auto history = compute.read(elastic.history);
    for (size_t i = 0; i < history.size(); i += 3)
    {
        result += modulus * std::exp(double(history[i + 2])) *
                  (std::cosh(std::hypot(double(history[i]), double(history[i + 1]))) - 1);
    }
    for (const auto* field : {&u, &v})
    {
        const auto values = compute.read(*field);
        for (float value : values)
            result += 0.5 * 10 * value * value;
    }
    return result;
}

double wave(Compute& compute, double dt)
{
    constexpr int width = 24;
    constexpr int height = 20;
    constexpr double modulus = 50000;
    const double pi = std::acos(-1.0);
    Field fraction = compute.field(width, height);
    Field density = compute.field(width, height);
    Field u = compute.field(width + 1, height);
    Field v = compute.field(width, height + 1);
    Field pressure = compute.field(width, height);
    compute.write(fraction, std::vector<float>(width * height, 1));
    compute.write(density, std::vector<float>(width * height, 10));
    std::vector<float> initial_u((width + 1) * height);
    std::vector<float> initial_v(width * (height + 1));
    const double kx = pi / width;
    const double ky = pi / height;
    const double sx = std::sin(kx / 2);
    const double sy = std::sin(ky / 2);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x <= width; ++x)
            initial_u[y * (width + 1) + x] = float(0.01 * sy * std::sin(kx * x) * std::cos(ky * (y + 0.5)));
    }
    for (int y = 0; y <= height; ++y)
    {
        for (int x = 0; x < width; ++x)
            initial_v[y * width + x] = float(-0.01 * sx * std::cos(kx * (x + 0.5)) * std::sin(ky * y));
    }
    compute.write(u, initial_u);
    compute.write(v, initial_v);
    ElasticityGPU elastic(compute, width, height, 1, modulus);
    fluid::FluidGridState<float> state;
    state.liquid_fraction.assign(width * height, 1);
    elastic.upload(state);
    ImplicitElasticityGPU solver(compute, width, height, 1, modulus);
    double previous = energy(compute, elastic, u, v, modulus);
    const double initial = previous;
    int iterations = 0;
    const int steps = dt <= .01 ? int(std::round(.2 / dt)) : 20;
    for (int step = 0; step < steps; ++step)
    {
        const auto result = solver.solve(dt, 1e-7, 1000, elastic, fraction, density, u, v, pressure);
        require(result.converged || result.precision_limited, "implicit wave solve failed");
        const double current = energy(compute, elastic, u, v, modulus);
        require(current <= previous * 1.0002 + 1e-10, "implicit wave added energy");
        previous = current;
        iterations += result.iterations;
    }
    std::cout << "wave dt=" << dt << " energy_ratio=" << previous / initial << " iterations=" << iterations << '\n';
    if (dt <= .01)
    {
        const double omega2 = modulus / 10 * (kx * kx + ky * ky);
        const double expected = std::pow(1 + omega2 * dt * dt, -steps);
        require(std::abs(previous / initial - expected) < .01, "wave damping differs from backward Euler");
        const double amplitude = std::inner_product(initial_u.begin(), initial_u.end(), compute.read(u).begin(), 0.0) /
                                 std::inner_product(initial_u.begin(), initial_u.end(), initial_u.begin(), 0.0);
        const double expected_amplitude = std::sqrt(expected) * std::cos(steps * std::atan(std::sqrt(omega2) * dt));
        require(std::abs(amplitude - expected_amplitude) < .01, "wave phase differs from backward Euler");
        return std::abs(amplitude - std::cos(steps * dt * std::sqrt(omega2)));
    }
    return 0;
}

void block(Compute& compute, int width, int height, bool filled = false)
{
    constexpr double modulus = 20000;
    Field fraction = compute.field(width, height);
    Field density = compute.field(width, height);
    Field u = compute.field(width + 1, height);
    Field v = compute.field(width, height + 1);
    Field pressure = compute.field(width, height);
    fluid::FluidGridState<float> state;
    state.liquid_fraction.resize(width * height);
    state.elastic_log_xx.assign(width * height, float(2 * std::log(1.15)));
    state.elastic_log_xy.assign(width * height, 0);
    state.elastic_log_yy.assign(width * height, float(-2 * std::log(1.15)));
    std::vector<float> rho(width * height);
    const double side = .3 * std::min(width, height);
    auto overlap = [](int x, double side, double center)
    {
        return std::max(0.0, std::min(x + 1.0, center + side / 2) - std::max(double(x), center - side / 2));
    };
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const auto i = y * width + x;
            const float f =
                filled ? 1 : float(overlap(x, side * 1.15, width * .5) * overlap(y, side / 1.15, height * .5));
            state.liquid_fraction[i] = f;
            rho[i] = .01f + 9.99f * f;
        }
    }
    compute.write(fraction, state.liquid_fraction);
    compute.write(density, rho);
    ElasticityGPU elastic(compute, width, height, 1, modulus);
    elastic.upload(state);
    ImplicitElasticityGPU solver(compute, width, height, 1, modulus);
    const auto start = std::chrono::steady_clock::now();
    const auto result = solver.solve(.04 / 3, 1e-5, 1000, elastic, fraction, density, u, v, pressure);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "block " << width << 'x' << height << " it=" << result.iterations
              << " residual=" << result.relative_residual << " div=" << result.divergence << " ms=" << ms << '\n';
    require(result.converged || result.precision_limited, "implicit block solve failed");
    if (width == 9)
        ImplicitElasticityGPUTestAccess::denseReference(solver);
    ImplicitElasticityGPUTestAccess::symmetry(solver);
    if (filled)
        ImplicitElasticityGPUTestAccess::tangentContrast(solver, density);
}
}

int main()
{
    try
    {
        Compute compute;
        ImplicitElasticityGPUTestAccess::kinematics(compute, 0, .3);
        ImplicitElasticityGPUTestAccess::kinematics(compute, .2, 0);
        const double coarse = wave(compute, 0.01);
        const double fine = wave(compute, 0.005);
        require(fine < .65 * coarse, "implicit wave did not improve with timestep refinement");
        wave(compute, 0.1);
        block(compute, 9, 7);
        block(compute, 3, 17, true);
        block(compute, 17, 11, true);
        block(compute, 63, 51);
        block(compute, 512, 400);
        std::cout << "PASS implicit elasticity\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL implicit elasticity: " << error.what() << '\n';
        return 1;
    }
}
