#include "ViscositySolverGPU.h"
#include "ViscosityKernels.h"
#include <cmath>
#include <bitloop/core/debug.h>

namespace fluid::webgpu
{

struct ViscositySolverGPU::Impl : ViscosityKernels
{
    struct Level
    {
        int width;
        int height;
        Field material;
        Field coefficients;
        Field rhs;
        Field x;
        Field next;
        Field residual;

        Level(Compute& compute, int w, int h, bool fine)
            : width(w),
              height(h),
              material(fine ? Field{} : compute.field(w, h, 2)),
              coefficients(compute.field(w + 1, h + 1, 4)),
              rhs(fine ? Field{} : compute.field(w + 1, h + 1, 2)),
              x(compute.field(w + 1, h + 1, 2)),
              next(compute.field(w + 1, h + 1, 2)),
              residual(compute.field(w + 1, h + 1, 2))
        {
        }
    };

    // Persistent solver workspaces; no per-iteration allocations.
    Compute& compute;
    int width;
    int height;
    std::vector<Level> levels;
    Field solution;
    Field next_solution;
    Field rhs;
    Field residual;
    Field next_residual;
    Field direction;
    Field next_direction;
    Field product;
    Field truth;
    Field partials;
    Field stats;
    Field state;
    Field next_state;
    bool prescribed_boundary_velocity = false;
    float dt = 0;
    float dx = 1;
    double relative_residual = 0;
    size_t next_work_warning = 100;

    static int groups(int count) { return std::min((count + 511) / 512, 1024); }

    Impl(Compute& compute, int w, int h)
        : compute(compute),
          width(w),
          height(h),
          solution(compute.field(w + 1, h + 1, 2)),
          next_solution(compute.field(w + 1, h + 1, 2)),
          rhs(compute.field(w + 1, h + 1, 2)),
          residual(compute.field(w + 1, h + 1, 2)),
          next_residual(compute.field(w + 1, h + 1, 2)),
          direction(compute.field(w + 1, h + 1, 2)),
          next_direction(compute.field(w + 1, h + 1, 2)),
          product(compute.field(w + 1, h + 1, 2)),
          truth(compute.field(w + 1, h + 1, 2)),
          partials(compute.field(groups((w + 1) * (h + 1)), 1, 4)),
          stats(compute.field(1, 1, 4)),
          state(compute.field(2, 1, 4)),
          next_state(compute.field(2, 1, 4))
    {
        for (;;)
        {
            levels.emplace_back(compute, w, h, levels.empty());

            if (w * h <= 64)
                break;

            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }

        build(compute);
    }

    void run(Kernel& kernel, int level, std::initializer_list<const Field*> outputs,
        std::initializer_list<Compute::Input> inputs, int mode = 0)
    {
        Parameters p;
        p.grid = {levels[level].width, levels[level].height, mode, 0};
        p.value = {std::ldexp(dx, level), dt, 0, 0};
        p.level[3] = prescribed_boundary_velocity ? 1.0f : 0.0f;

        compute.run(kernel, outputs, inputs, p);
    }

    void reduce(const Field& a, const Field& b, int mode)
    {
        const int count = a.width * a.height;
        const int group_count = groups(count);

        Parameters p;
        p.grid = {count, group_count, mode, 0};

        compute.run(first_reduce, {&partials},
            {{"source", &a}, {"other", &b}, {"coeff", &levels[0].coefficients}, {"state", &state}}, p,
            {unsigned(group_count * lanes), 1, 1});

        p.grid[0] = group_count;

        compute.run(finish_reduce, {&stats}, {{"source", &partials}}, p, {lanes, 1, 1});
    }

    void scalars(int mode)
    {
        Parameters p;
        p.grid[2] = mode;

        compute.run(scalar, {&next_state}, {{"state", &state}, {"stats", &stats}}, p, {1, 1, 1});
        state.swap(next_state);
    }

    void smoothLevel(int level, const Field& b, bool zero)
    {
        auto& l = levels[level];
        run(smooth, level, {&l.next}, {{"source", &l.x}, {"rhs", &b}, {"coeff", &l.coefficients}, {"state", &state}},
            zero ? 1 : 0);
        l.x.swap(l.next);
    }

    // Symmetric pre/post smoothing preserves the CG preconditioner.
    void cycle(int level, const Field& b)
    {
        auto& l = levels[level];
        if (level + 1 == int(levels.size()))
        {
            // A fixed symmetric polynomial is an SPD coarse approximate inverse.
            Parameters p;
            p.grid = {l.width, l.height, 0, 0};
            p.value = {std::ldexp(dx, level), dt, 0, 0};

            compute.run(coarse_smooth, {&l.x}, {{"rhs", &b}, {"weights", &l.coefficients}}, p, {256, 1, 1});

            return;
        }

        smoothLevel(level, b, true);
        smoothLevel(level, b, false);
        run(residual_pass, level, {&l.residual}, {{"source", &l.x}, {"rhs", &b}, {"coeff", &l.coefficients}});
        auto& coarse = levels[level + 1];
        run(restrict_pass, level + 1, {&coarse.rhs}, {{"source", &l.residual}, {"state", &state}});
        cycle(level + 1, coarse.rhs);
        run(prolong, level, {&l.next}, {{"source", &l.x}, {"coarse", &coarse.x}, {"state", &state}});
        l.x.swap(l.next);
        smoothLevel(level, b, false);
        smoothLevel(level, b, false);
    }

    // Rebuild coefficients, iterate, and accept only the recomputed residual.
    int solve(Field& u, Field& v, const Field& material, double timestep, double spacing, double tolerance, int maximum,
        bool prescribed)
    {
        prescribed_boundary_velocity = prescribed;

        if (timestep <= 0.0)
        {
            relative_residual = 0.0;
            return 0;
        }

        dt = float(timestep);
        dx = float(spacing);
        const Field* properties = &material;

        for (int level = 0; level < int(levels.size()); ++level)
        {
            auto& l = levels[level];
            if (level > 0)
            {
                run(coarsen, level, {&l.material}, {{"fine", properties}});
                properties = &l.material;
            }

            run(coefficients, level, {&l.coefficients}, {{"material", properties}});
        }

        run(pack, 0, {&solution, &rhs}, {{"u", &u}, {"v", &v}, {"coeff", &levels[0].coefficients}});

        auto trueResidual = [&](Field& output)
        {
            if (prescribed_boundary_velocity)
            {
                run(driven_residual, 0, {&output},
                    {{"source", &solution}, {"u", &u}, {"v", &v}, {"coeff", &levels[0].coefficients}});
            }
            else
            {
                run(residual_pass, 0, {&output},
                    {{"source", &solution}, {"rhs", &rhs}, {"coeff", &levels[0].coefficients}});
            }
        };

        trueResidual(residual);
        reduce(residual, solution, 0);
        const auto initial = compute.read(stats);
        const double target = tolerance * tolerance * initial[1] + 1e-12 * initial[2];
        if (initial[3] != 0.0)
            throw std::runtime_error("Viscosity received nonfinite velocity/material state");

        relative_residual = std::sqrt(initial[0] / std::max(double(initial[1]), 1e-30));

        if (initial[0] <= target)
            return 0;

        Parameters p;
        p.value[2] = float(tolerance);

        compute.run(initialize, {&state}, {{"stats", &stats}}, p, {1, 1, 1});
        int iterations = 0;

        while (maximum < 0 || iterations < maximum)
        {
            const int batch = maximum < 0 ? 4 : std::min(4, maximum - iterations);
            for (int i = 0; i < batch; ++i)
            {
                cycle(0, residual);
                reduce(residual, levels[0].x, 1);
                scalars(1);
                run(direction_pass, 0, {&next_direction},
                    {{"correction", &levels[0].x}, {"previous", &direction}, {"state", &state}});
                direction.swap(next_direction);
                run(apply, 0, {&product},
                    {{"source", &direction}, {"coeff", &levels[0].coefficients}, {"state", &state}});
                reduce(direction, product, 2);
                scalars(2);
                run(update, 0, {&next_solution, &next_residual},
                    {{"solution", &solution}, {"residual", &residual}, {"direction", &direction}, {"product", &product},
                        {"state", &state}});
                solution.swap(next_solution);
                residual.swap(next_residual);
            }

            iterations += batch;
            trueResidual(truth);
            reduce(truth, truth, 3);
            const auto check = compute.read(stats);
            relative_residual = std::sqrt(check[0] / std::max(double(initial[1]), 1e-30));

            if (check[3] != 0.0)
                throw std::runtime_error("Implicit viscosity solver breakdown");

            if (check[0] <= check[1])
            {
                run(unpack, 0, {&u}, {{"source", &solution}}, 0);
                run(unpack, 0, {&v}, {{"source", &solution}}, 1);
                return iterations;
            }

            if (maximum < 0 && size_t(iterations) >= next_work_warning)
            {
                blPrint() << "Warning: viscosity solve continuing after " << iterations
                          << " iterations; relative residual " << relative_residual;
                next_work_warning *= 2;
            }

            // Float recurrence can appear converged before the recomputed residual.
            if (check[2] != 0.0)
            {
                run(refresh, 0, {&residual, &next_state}, {{"truth", &truth}, {"state", &state}, {"stats", &stats}});
                state.swap(next_state);
            }
        }

        throw std::runtime_error(
            "Implicit viscosity did not converge; relative residual " + std::to_string(relative_residual));
    }
};

ViscositySolverGPU::ViscositySolverGPU(Compute& compute, int width, int height)
    : impl(std::make_unique<Impl>(compute, width, height))
{
}

ViscositySolverGPU::~ViscositySolverGPU() = default;

int ViscositySolverGPU::solve(Field& u, Field& v, const Field& material, double dt, double cell_size, double tolerance,
    int maximum, bool prescribed_boundary_velocity)
{
    return impl->solve(u, v, material, dt, cell_size, tolerance, maximum, prescribed_boundary_velocity);
}

double ViscositySolverGPU::lastRelativeResidual() const
{
    return impl->relative_residual;
}

}
