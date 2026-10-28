#include "Adaptive/AmrPressure2D.h"
#include "AdaptiveWebGPU/AmrPressureGPU.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
using CPU = fluid::AmrPressure2D;
using GPU = fluid::AmrPressureGPU;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

std::vector<CPU::Level> layout(int n, int kind)
{
    if (kind == 0) return {{n, n, 1., {{0, 0, n, n}}}};
    if (kind == 7) return {{n, n - 2, 1., {{0, 0, n, n - 2}}}};
    if (kind == 8) return {{1, n, 1., {{0, 0, 1, n}}}};
    std::vector<CPU::Level> levels{{n / 2, n / 2, 2., {{0, 0, n / 2, n / 2}}}};
    std::vector<CPU::Box> fine;
    if (kind == 1) fine = {{n / 4, n / 4, n / 2, n / 2}};
    if (kind == 2) fine = {{0, 0, n / 2, n / 2}, {n / 2, 0, n / 4, n / 4}};
    if (kind == 3)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                if ((x + y) % 2 == 0) fine.push_back({x * n / 4, y * n / 4, n / 4, n / 4});
    levels.push_back({n, n, 1., fine});
    if (kind == 4 || kind == 5)
    {
        levels[1].boxes = {{n / 4, n / 4, n / 2, n / 2}};
        levels.push_back({2 * n, 2 * n, .5, {{3 * n / 4, 3 * n / 4, n / 2, n / 2}}});
        if (kind == 5)
            for (int l = 3; l < 6; ++l)
            {
                int width = levels.back().width * 2, origin = (width - n / 2) / 2;
                levels.push_back({width, width, levels.back().cell_size / 2., {{origin, origin, n / 2, n / 2}}});
            }
    }
    if (kind == 6)
        for (int y = 0; y < n; y += 4)
            for (int x = 0; x < n; x += 4)
                if ((7 * (x / 4) + 11 * (y / 4)) % 5 < 2) levels[1].boxes.push_back({x, y, 4, 4});
    return levels;
}

void upload(fluid::webgpu::Compute& compute, GPU::Field& gpu, const CPU::Field& cpu)
{
    for (int l = 0; l < int(cpu.size()); ++l)
    {
        const int plane = gpu[l].width * gpu[l].height;
        std::vector<float> values(gpu[l].channels == 2 ? 2 * plane : cpu[l].size());
        for (int i = 0; i < int(cpu[l].size()); ++i)
        {
            values[i] = float(cpu[l][i]);
            if (gpu[l].channels == 2) values[plane + i] = float(cpu[l][i] - values[i]);
        }
        compute.write(gpu[l], values);
    }
}
CPU::Field download(fluid::webgpu::Compute& compute, const GPU& solver, const GPU::Field& field)
{
    CPU::Field result;
    for (int l = 0; l < int(field.size()); ++l)
    {
        const int size = solver.topology().grids[l].size, plane = field[l].width * field[l].height;
        auto values = compute.read(field[l], 0, (field[l].channels == 2 ? 2 * plane : size) * sizeof(float));
        result.emplace_back(values.begin(), values.begin() + size);
        if (field[l].channels == 2)
            for (int i = 0; i < size; ++i) result.back()[i] += values[plane + i];
    }
    return result;
}
double difference(const fluid::AmrPressureTopology2D& topology, const CPU::Field& a, const CPU::Field& b, bool relative = false)
{
    double error = 0., scale = 0.;
    for (int l = 0; l < int(a.size()); ++l)
        for (const auto& row : topology.grids[l].rows)
            for (int i = row.begin; i < row.end; ++i)
            {
                require(std::isfinite(a[l][i]), "Nonfinite GPU pressure field");
                error = std::max(error, std::abs(a[l][i] - b[l][i])); scale = std::max(scale, std::abs(b[l][i]));
            }
    return relative ? error / std::max(1., scale) : error;
}

void reductionCases(fluid::webgpu::Compute& compute)
{
    for (auto [n, kind] : {std::pair{15, 7}, std::pair{32, 5}, std::pair{520, 0}, std::pair{1031, 0}})
    {
        GPU gpu(compute, layout(n, kind)); auto input = gpu.makeField(); CPU::Field samples;
        double squared = 0.;
        for (const auto& grid : gpu.topology().grids)
        {
            samples.emplace_back(grid.size, std::numeric_limits<double>::quiet_NaN());
            for (const auto& row : grid.rows)
                for (int i = row.begin; i < row.end; ++i)
                    if (grid.active[i]) {
                        double value = i % 5 - 2; samples.back()[i] = value; squared += value * value;
                    }
        }
        // Poisoned ghosts/covered cells must not enter the active-cell norm.
        upload(compute, input, samples);
        const double expected = std::sqrt(squared), actual = gpu.norm(input);
        require(std::abs(actual - expected) < 2.e-6 * expected, "GPU reduction lost tail cells or included inactive cells");
        std::cout << "reduction,n=" << n << ",layout=" << kind << ",norm=" << actual << ",expected=" << expected << std::endl;
    }
    GPU gpu(compute, layout(15, 0)); CPU::Field rhs, beta;
    for (const auto& grid : gpu.topology().grids) { rhs.emplace_back(grid.size); beta.emplace_back(grid.size, 1.); }
    rhs[0][gpu.topology().grids[0].rows[0].begin] = std::numeric_limits<double>::quiet_NaN();
    auto forcing = gpu.makeField(); GPU::Faces coefficients{gpu.makeField(), gpu.makeField()};
    upload(compute, forcing, rhs);
    for (int d = 0; d < 2; ++d) upload(compute, coefficients[d], beta);
    gpu.setCoefficients(coefficients);
    for (bool multigrid : {false, true}) {
        auto pressure = gpu.makeField(true);
        const auto result = multigrid ? gpu.solveMultigrid(pressure, forcing, 1.e-3, 8) : gpu.solve(pressure, forcing, 1.e-3, 8);
        require(!result.converged && !std::isfinite(result.residual), "GPU reduction concealed a nonfinite active value");
    }
    std::cout << "reduction,nonfinite_forcing_reported" << std::endl;
}

void coarseFallbackCase(fluid::webgpu::Compute& compute)
{
    const auto levels = layout(49, 7); CPU cpu(levels); GPU gpu(compute, levels);
    auto rhs = cpu.makeField(), expected = cpu.makeField(); CPU::Faces beta{cpu.makeField(), cpu.makeField()};
    for (const auto& row : gpu.topology().grids[0].rows)
        for (int i = row.begin; i < row.end; ++i) rhs[0][i] = float(std::sin(.1 * i));
    for (auto& axis : beta) for (auto& level : axis) std::fill(level.begin(), level.end(), 1.);
    auto forcing = gpu.makeField(), output = gpu.makeField(); GPU::Faces coefficients{gpu.makeField(), gpu.makeField()};
    upload(compute, forcing, rhs);
    for (int d = 0; d < 2; ++d) upload(compute, coefficients[d], beta[d]);
    cpu.setCoefficients(beta); gpu.setCoefficients(coefficients);
    // This odd single-level grid exceeds the shared tail's capacity.
    cpu.precondition(expected, rhs, false); gpu.precondition(output, forcing, false);
    const double error = difference(gpu.topology(), download(compute, gpu, output), expected, true);
    require(error < 2.e-4, "Generic GPU coarse-cycle fallback differs from CPU");
    std::cout << "coarse_fallback,odd49x47,relative=" << error << std::endl;
}

void breakdownCase(fluid::webgpu::Compute& compute)
{
    auto levels = layout(8, 0); CPU cpu(levels); GPU gpu(compute, levels);
    auto initial = cpu.makeField(), rhs = cpu.makeField(); CPU::Faces beta{cpu.makeField(), cpu.makeField()};
    for (auto& level : initial) std::fill(level.begin(), level.end(), 3.);
    for (auto& level : rhs) std::fill(level.begin(), level.end(), 1.);
    for (auto& axis : beta) for (auto& level : axis) std::fill(level.begin(), level.end(), 1.);
    auto pressure = gpu.makeField(true), forcing = gpu.makeField(); GPU::Faces coefficients{gpu.makeField(), gpu.makeField()};
    upload(compute, pressure, initial); upload(compute, forcing, rhs);
    for (int d = 0; d < 2; ++d) upload(compute, coefficients[d], beta[d]);
    gpu.setCoefficients(coefficients); cpu.setCoefficients(beta);
    auto expected = initial; auto reference = cpu.solve(expected, rhs, 1.e-3, 30);
    auto status = gpu.solve(pressure, forcing, 1.e-3, 30);
    require(!reference.converged && !status.converged && status.iterations == reference.iterations, "Incompatible forcing did not report Krylov breakdown");
    require(difference(gpu.topology(), download(compute, gpu, pressure), initial) == 0., "Krylov breakdown committed pressure");
    std::cout << "incompatible_rhs,breakdown_reported,pressure_unchanged,iterations=" << status.iterations << std::endl;
}

void waveCase(fluid::webgpu::Compute& compute, bool mixed = false)
{
    constexpr int n = 512;
    std::vector<CPU::Level> levels;
    if (mixed) levels.push_back({n / 2, n / 2, 2., {}});
    levels.push_back({n, n, 1., {}});
    for (int l = 0; l < int(levels.size()); ++l)
        for (int y = mixed && l == 1 ? n / 2 : 0; y < levels[l].height; y += 64)
            for (int x = 0; x < levels[l].width; x += 64) levels[l].boxes.push_back({x, y, 64, 64});
    CPU cpu(levels); GPU gpu(compute, levels);
    auto rhs = cpu.makeField(); CPU::Faces beta{cpu.makeField(), cpu.makeField()};
    auto density = [](int x, int y)
    {
        x = std::clamp(x, 0, n - 1); y = std::clamp(y, 0, n - 1);
        const double surface = .7 * n + (double(x) / n) * .5 * n;
        const double alpha = std::clamp(y + 1. - surface, 0., 1.);
        return float(100. * alpha) + float(.1 * (1. - alpha));
    };
    for (int l = 0; l < int(levels.size()); ++l)
    {
        const int scale = int(levels[l].cell_size);
        auto averageDensity = [&](int x, int y)
        {
            float sum = 0.;
            for (int j = 0; j < scale; ++j) for (int i = 0; i < scale; ++i) sum += density(scale * x + i, scale * y + j);
            return sum / float(scale * scale);
        };
        for (const auto& p : cpu.patches(l))
        {
            for (int y = p.box.y; y < p.box.y + p.box.height; ++y)
                for (int x = p.box.x; x < p.box.x + p.box.width; ++x)
                    rhs[l][p.index(x, y)] = float((y == 0 ? -1.6 : y == levels[l].height - 1 ? 1.6 : 0.) / scale);
            for (int d = 0; d < 2; ++d)
                for (int y = p.box.y; y < p.box.y + p.box.height + (d == 1); ++y)
                    for (int x = p.box.x; x < p.box.x + p.box.width + (d == 0); ++x)
                        beta[d][l][p.index(x, y)] = 2.f / (averageDensity(x, y) + averageDensity(x - (d == 0), y - (d == 1)));
        }
    }
    auto pressure = gpu.makeField(true), forcing = gpu.makeField();
    GPU::Faces coefficients{gpu.makeField(), gpu.makeField()}, flux{gpu.makeField(), gpu.makeField()};
    upload(compute, forcing, rhs);
    for (int d = 0; d < 2; ++d) upload(compute, coefficients[d], beta[d]);
    gpu.setCoefficients(coefficients); cpu.setCoefficients(beta);
    auto result = gpu.solve(pressure, forcing, 1.e-3, 2000);
    auto values = download(compute, gpu, pressure); gpu.fluxes(flux, pressure);
    auto fx = download(compute, gpu, flux[0]), fy = download(compute, gpu, flux[1]);
    double divergence = 0., minimum = 0., maximum = 0.;
    for (int l = 0; l < int(levels.size()); ++l)
        for (const auto& row : gpu.topology().grids[l].rows)
            for (int i = row.begin; i < row.end; ++i)
                if (gpu.topology().grids[l].active[i])
                {
                    const int stride = cpu.patches(l)[row.patch].stride;
                    const double div = (fx[l][i + 1] - fx[l][i] + fy[l][i + stride] - fy[l][i]) / levels[l].cell_size;
                    divergence = std::max(divergence, std::abs(div - rhs[l][i]));
                    minimum = std::min(minimum, values[l][i]); maximum = std::max(maximum, values[l][i]);
                }
    std::cout << "wave512,mixed=" << mixed << ",iterations=" << result.iterations << ",residual=" << result.residual << ",converged=" << result.converged
              << ",divergence=" << divergence << ",pressure_min=" << minimum << ",pressure_max=" << maximum << std::endl;
    auto expected = cpu.makeField(); auto reference = cpu.solve(expected, rhs, 1.e-3, 2000);
    auto rounded = expected; for (auto& values : rounded) for (auto& value : values) value = float(value);
    auto error = cpu.makeField(); cpu.apply(error, rounded);
    double square = 0., roundedMax = 0.;
    for (int l = 0; l < int(levels.size()); ++l)
        for (const auto& row : gpu.topology().grids[l].rows)
            for (int i = row.begin; i < row.end; ++i)
                if (gpu.topology().grids[l].active[i])
                {
                    const double r = error[l][i] - rhs[l][i]; square += r * r; roundedMax = std::max(roundedMax, std::abs(r));
                }
    std::cout << "wave512,mixed=" << mixed << ",cpu_iterations=" << reference.iterations << ",cpu_residual=" << reference.residual
              << ",rounded_cpu_l2=" << std::sqrt(square) << ",rounded_cpu_max=" << roundedMax << std::endl;
    require(result.converged, "512-square wave pressure failed its requested tolerance");
    require(divergence <= 1.e-3, "512-square wave returned flux failed actual divergence target");
    const auto warm = gpu.solve(pressure, forcing, 1.e-3, 2000);
    require(warm.converged && warm.iterations == 0, "Compensated wave warm start lost its low plane");
    auto limited = gpu.makeField(true); const auto capped = gpu.solve(limited, forcing, 1.e-3, 6);
    require(!capped.converged && capped.iterations == 6, "Residual corrections exceeded the total Krylov iteration cap");
    std::cout << "wave512,warm_iterations=0,total_iteration_cap=6,enforced" << std::endl;
}

void runCase(fluid::webgpu::Compute& compute, int n, int kind, double contrast, bool solveCase, bool duplicateFaces = false, double tolerance = 1.e-3)
{
    const auto levels = layout(n, kind); CPU cpu(levels); GPU gpu(compute, levels);
    auto phi = cpu.makeField(), result = cpu.makeField(); CPU::Faces beta{cpu.makeField(), cpu.makeField()};
    for (int l = 0; l < int(levels.size()); ++l)
    {
        std::fill(phi[l].begin(), phi[l].end(), 12345.);
        const double dx = levels[l].cell_size;
        for (int pindex = 0; pindex < int(cpu.patches(l).size()); ++pindex)
        {
            const auto& p = cpu.patches(l)[pindex];
            for (int y = p.box.y; y < p.box.y + p.box.height; ++y)
                for (int x = p.box.x; x < p.box.x + p.box.width; ++x)
                    phi[l][p.index(x, y)] = float(std::sin(.11 * (x + .5) * dx) * std::cos(.07 * (y + .5) * dx) + .01 * std::sin(1.7 * x + 2.3 * y));
            auto density = [&](double x, double y)
            {
                bool dense = kind == 6 ? std::sin(1.7 * x + 2.3 * y) > 0. : y + .3 * x < .6 * n;
                return (dense ? contrast : 1.) * (1. + .1 * std::cos(.03 * x));
            };
            for (int d = 0; d < 2; ++d)
                for (int y = p.box.y; y < p.box.y + p.box.height + (d == 1); ++y)
                    for (int x = p.box.x; x < p.box.x + p.box.width + (d == 0); ++x)
                    {
                        double px = (x + .5) * dx, py = (y + .5) * dx;
                        double value = 2. / (density(px, py) + density(px - (d == 0) * dx, py - (d == 1) * dx));
                        if (duplicateFaces) value *= 1. + .003 * pindex;
                        beta[d][l][p.index(x, y)] = float(value);
                    }
        }
    }
    auto input = gpu.makeField(), output = gpu.makeField(); GPU::Faces coefficients{gpu.makeField(), gpu.makeField()};
    upload(compute, input, phi);
    for (int d = 0; d < 2; ++d) upload(compute, coefficients[d], beta[d]);
    cpu.setCoefficients(beta); gpu.setCoefficients(coefficients); cpu.apply(result, phi); gpu.apply(output, input);
    auto actual = download(compute, gpu, output);
    double operatorError = difference(gpu.topology(), actual, result, true);
    std::cout << "operator,n=" << n << ",layout=" << kind << ",contrast=" << contrast << ",duplicates=" << duplicateFaces << ",relative=" << operatorError << std::endl;
    require(operatorError < 3.e-6, "GPU composite pressure operator differs from CPU");
    auto rhs = result;
    for (auto& level : rhs) for (auto& value : level) value = float(value);
    auto forcing = gpu.makeField(); upload(compute, forcing, rhs);
    for (bool bottom : {false, true})
    {
        cpu.precondition(result, rhs, bottom); gpu.precondition(output, forcing, bottom); actual = download(compute, gpu, output);
        double cycleError = difference(gpu.topology(), actual, result, true);
        std::cout << "cycle,bottom=" << bottom << ",relative=" << cycleError << std::endl;
        require(cycleError < 2.e-4, "GPU multigrid preconditioner differs from CPU");
    }
    CPU::Faces cpuFlux{cpu.makeField(), cpu.makeField()}; GPU::Faces gpuFlux{gpu.makeField(), gpu.makeField()};
    cpu.fluxes(cpuFlux, phi); gpu.fluxes(gpuFlux, input);
    double fluxError = 0.;
    for (int d = 0; d < 2; ++d)
    {
        auto values = download(compute, gpu, gpuFlux[d]);
        for (int l = 0; l < int(levels.size()); ++l)
            for (const auto& p : cpu.patches(l))
                for (int y = p.box.y; y < p.box.y + p.box.height + (d == 1); ++y)
                    for (int x = p.box.x; x < p.box.x + p.box.width + (d == 0); ++x)
                        fluxError = std::max(fluxError, std::abs(values[l][p.index(x, y)] - cpuFlux[d][l][p.index(x, y)]));
    }
    std::cout << "flux,max_error=" << fluxError << std::endl; require(fluxError < 2.e-5, "GPU composite pressure flux differs from CPU");

    auto shifted = phi;
    for (auto& level : shifted) for (auto& value : level) value += 1048576.;
    auto compensated = gpu.makeField(true); upload(compute, compensated, shifted);
    cpu.apply(result, shifted); gpu.apply(output, compensated);
    const double gaugeError = difference(gpu.topology(), download(compute, gpu, output), result, true);
    std::cout << "compensated_gauge,relative=" << gaugeError << std::endl;
    require(gaugeError < 3.e-6,
            "Compensated pressure lost a small gradient across a large constant gauge");

    auto constant = cpu.makeField();
    for (auto& level : constant) std::fill(level.begin(), level.end(), 1.);
    upload(compute, input, constant); gpu.apply(output, input);
    require(gpu.norm(output) < 1.e-6, "GPU pressure constant nullspace failed");
    if (!solveCase) return;
    for (bool multigrid : {false, true})
    {
        auto cpuSolved = cpu.makeField();
        const auto reference = multigrid ? cpu.solveMultigrid(cpuSolved, rhs, tolerance, 200) : cpu.solve(cpuSolved, rhs, tolerance, 200);
        auto solved = gpu.makeField(true);
        auto start = std::chrono::steady_clock::now();
        auto status = multigrid ? gpu.solveMultigrid(solved, forcing, tolerance, 200) : gpu.solve(solved, forcing, tolerance, 200);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "solve,mlmg=" << multigrid << ",iterations=" << status.iterations << ",residual=" << status.residual
                  << ",cpu_iterations=" << reference.iterations << ",cpu_residual=" << reference.residual << ",ms=" << ms << std::endl;
        if (!reference.converged)
        {
            require(!status.converged && status.iterations == reference.iterations, "GPU solver hid CPU nonconvergence");
            std::cout << "nonconvergence_matches_cpu" << std::endl;
            continue;
        }
        require(status.converged, "GPU pressure solve did not meet its requested tolerance");
        if (!multigrid && n == 128 && kind == 6) require(status.iterations > 30, "Restart fixture did not exercise a GMRES restart");
        gpu.fluxes(gpuFlux, solved);
        auto fx = download(compute, gpu, gpuFlux[0]), fy = download(compute, gpu, gpuFlux[1]);
        double divergence = 0.;
        for (int l = 0; l < int(levels.size()); ++l)
            for (const auto& row : gpu.topology().grids[l].rows)
                for (int i = row.begin; i < row.end; ++i)
                    if (gpu.topology().grids[l].active[i])
                    {
                        int stride = cpu.patches(l)[row.patch].stride;
                        double div = (fx[l][i + 1] - fx[l][i] + fy[l][i + stride] - fy[l][i]) / levels[l].cell_size;
                        divergence = std::max(divergence, std::abs(div - rhs[l][i]));
                    }
        std::cout << "returned_flux,max_divergence=" << divergence << std::endl;
        require(divergence <= tolerance, "Returned GPU flux fails actual composite divergence target");
        auto warm = multigrid ? gpu.solveMultigrid(solved, forcing, tolerance, 200) : gpu.solve(solved, forcing, tolerance, 200);
        require(warm.converged && warm.iterations == 0, "GPU warm solve did not retain converged pressure");
        auto zero = gpu.makeField(true), zeroRhs = gpu.makeField();
        auto zeroStatus = multigrid ? gpu.solveMultigrid(zero, zeroRhs, tolerance, 200) : gpu.solve(zero, zeroRhs, tolerance, 200);
        require(zeroStatus.converged && zeroStatus.iterations == 0, "GPU zero solve iterated");
        auto limited = gpu.makeField(true);
        auto cap = multigrid ? gpu.solveMultigrid(limited, forcing, 1.e-10, 1) : gpu.solve(limited, forcing, 1.e-10, 1);
        require(!cap.converged && cap.iterations == 1, "GPU solver did not enforce iteration cap");
        auto noIterations = gpu.makeField(true);
        auto cappedZero = multigrid ? gpu.solveMultigrid(noIterations, forcing, 1.e-10, 0) : gpu.solve(noIterations, forcing, 1.e-10, 0);
        require(!cappedZero.converged && cappedZero.iterations == 0, "GPU zero iteration cap changed pressure");
        if (!multigrid && n == 16)
        {
            auto tooTight = gpu.makeField();
            auto floor = gpu.solve(tooTight, forcing, 1.e-12, 120);
            std::cout << "precision_floor,requested=1e-12,residual=" << floor.residual << ",iterations=" << floor.iterations << std::endl;
            require(!floor.converged && floor.residual > 1.e-12, "GPU pressure hid its float precision floor");
        }
    }
    // A changed coefficient field must invalidate every AMR and base-MG diagonal.
    for (int d = 0; d < 2; ++d)
    {
        for (auto& level : beta[d]) for (auto& value : level) value = float(value * 1.25);
        upload(compute, coefficients[d], beta[d]);
    }
    cpu.setCoefficients(beta); gpu.setCoefficients(coefficients); upload(compute, input, phi);
    cpu.apply(result, phi); gpu.apply(output, input);
    require(difference(gpu.topology(), download(compute, gpu, output), result, true) < 3.e-6, "GPU coefficient refresh left stale operator state");
    cpu.precondition(result, rhs, false); gpu.precondition(output, forcing, false);
    require(difference(gpu.topology(), download(compute, gpu, output), result, true) < 2.e-4, "GPU coefficient refresh left stale coarse-cycle state");
    auto constantRhs = cpu.makeField();
    for (auto& level : constantRhs) std::fill(level.begin(), level.end(), 3.);
    upload(compute, forcing, constantRhs); gpu.precondition(output, forcing, false);
    const auto constantResult = download(compute, gpu, output);
    for (int l = 0; l < int(constantResult.size()); ++l)
        for (const auto& row : gpu.topology().grids[l].rows)
            for (int i = row.begin; i < row.end; ++i)
                require(std::abs(constantResult[l][i]) < 1.e-7, "GPU coarse cycle lost the constant nullspace");
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::cout << std::setprecision(10);
        fluid::webgpu::Compute compute(argc < 3 || std::string(argv[2]) != "portable");
        const bool subgroups = compute.gpu.device().HasFeature(wgpu::FeatureName::Subgroups);
        std::cout << "subgroup_reductions=" << subgroups << '\n';
        if (argc > 2 && std::string(argv[2]) == "subgroups") require(subgroups, "Requested subgroup test path is unavailable");
        breakdownCase(compute);
        const std::string mode = argc > 1 ? argv[1] : "all";
        if (mode == "reductions") reductionCases(compute);
        else if (mode == "wave") { waveCase(compute); waveCase(compute, true); }
        else if (mode == "smoke") runCase(compute, 16, 1, 1000., true);
        else if (mode == "restart") runCase(compute, 128, 6, 1000., true, false, 1.e-5);
        else
        {
            reductionCases(compute);
            runCase(compute, 16, 1, 1000., true);
            for (double contrast : {1., 1000.})
                for (int kind = 0; kind < 7; ++kind) runCase(compute, 32, kind, contrast, true);
            runCase(compute, 15, 7, 1., true);
            runCase(compute, 13, 8, 1., true);
            coarseFallbackCase(compute);
            runCase(compute, 32, 2, 1000., false, true);
            runCase(compute, 128, 6, 1000., true, false, 1.e-5);
            waveCase(compute); waveCase(compute, true);
        }
        std::cout << "AMR WebGPU pressure tests passed" << std::endl; return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
}
