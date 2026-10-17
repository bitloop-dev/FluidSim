#include "PressureMultigrid.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace fluid
{

namespace
{
template<typename Operation>
void forRows(int width, int height, Operation&& operation)
{
    // Keep coarse grids serial; dispatch workers only when each pass has enough work.
    const int workers = std::min(width * height >= 1048576 ? 8 : 4, int(bl::Thread::threadCount()));
    if (width * height >= 65536 && workers > 1)
        bl::Thread::forEachBatch(height, std::forward<Operation>(operation), workers);
    else
        operation(0, height);
}
}

PressureMultigrid::Level::Level(int w, int h, int spacing, int domain_width, int domain_height) :
    width(w), height(h), stride(w + 2), center_x(w), center_y(h)
{
    const size_t count = size_t(stride) * (height + 2);
    for (auto* values : {&right, &down, &inverse_diagonal, &rhs, &solution, &residual})
        values->resize(count);

    // A shorter final cell keeps odd grid sizes inside the original domain.
    for (int x = 0; x < width; ++x)
        center_x[x] = 0.5 * (x * spacing + std::min((x + 1) * spacing, domain_width));
    for (int y = 0; y < height; ++y)
        center_y[y] = 0.5 * (y * spacing + std::min((y + 1) * spacing, domain_height));
}

PressureMultigrid::PressureMultigrid(int width, int height)
{
    const int domain_width = width, domain_height = height;
    int spacing = 1;
    for (;;)
    {
        levels.emplace_back(width, height, spacing, domain_width, domain_height);
        if (width * height <= 64)
            break;
        width = (width + 1) / 2;
        height = (height + 1) / 2;
        spacing *= 2;
    }
    for (size_t i = 0; i + 1 < levels.size(); ++i)
        buildTransfers(levels[i], levels[i + 1]);
    const int unknowns = levels.back().width * levels.back().height - 1;
    coarse_factor.resize(size_t(unknowns) * unknowns);
    coarse_rhs.resize(unknowns);
}

void PressureMultigrid::buildTransfers(Level& fine, const Level& coarse)
{
    auto buildAxis = [](const std::vector<double>& fine_centers, const std::vector<double>& coarse_centers,
                        std::vector<Interpolation>& prolong, std::vector<Restriction>& restrict)
    {
        prolong.resize(fine_centers.size());
        restrict.resize(coarse_centers.size());
        for (int f = 0; f < int(fine_centers.size()); ++f)
        {
            const int parent = f / 2;
            const int first = std::max(0, parent - int(fine_centers[f] < coarse_centers[parent]));
            const int second = std::min(first + 1, int(coarse_centers.size()) - 1);
            const double weight = first == second ? 0.0 :
                std::clamp((fine_centers[f] - coarse_centers[first]) / (coarse_centers[second] - coarse_centers[first]), 0.0, 1.0);
            prolong[f] = {first, second, weight};

            // Restriction is exactly the transpose of prolongation, as PCG requires.
            auto append = [&](int c, double value)
            {
                if (value == 0.0) return;
                auto& stencil = restrict[c];
                stencil.indices[stencil.count] = f;
                stencil.weights[stencil.count++] = value;
            };
            append(first, 1.0 - weight);
            append(second, weight);
        }
    };
    buildAxis(fine.center_x, coarse.center_x, fine.prolong_x, fine.restrict_x);
    buildAxis(fine.center_y, coarse.center_y, fine.prolong_y, fine.restrict_y);
}

void PressureMultigrid::Level::rebuildDiagonal()
{
    for (int y = 0; y < height; ++y)
        for (int x = 0, i = (y + 1) * stride + 1; x < width; ++x, ++i)
        {
            const double diagonal = right[i - 1] + right[i] + down[i - stride] + down[i];
            inverse_diagonal[i] = diagonal > 0.0 ? 1.0 / diagonal : 0.0;
        }
}

bool PressureMultigrid::rebuild(const std::vector<double>& inv_density_u, const std::vector<double>& inv_density_v)
{
    auto& fine = levels.front();
    for (int y = 0; y < fine.height; ++y)
        for (int x = 0, i = (y + 1) * fine.stride + 1; x < fine.width; ++x, ++i)
        {
            fine.right[i] = inv_density_u[y * (fine.width + 1) + x + 1];
            fine.down[i] = inv_density_v[(y + 1) * fine.width + x];
        }
    fine.rebuildDiagonal();
    for (size_t i = 0; i + 1 < levels.size(); ++i)
        coarsen(levels[i], levels[i + 1]);
    return factorCoarsest();
}

void PressureMultigrid::coarsen(const Level& fine, Level& coarse)
{
    // Add parallel paths, but combine conductances in series across density jumps.
    for (int y = 0; y < coarse.height; ++y)
        for (int x = 0; x + 1 < coarse.width; ++x)
        {
            const int edge = 2 * x + 2;
            const double left = (fine.center_x[edge - 1] - coarse.center_x[x]) /
                                (fine.center_x[edge - 1] - fine.center_x[edge - 2]);
            const double right = edge + 1 < fine.width ?
                (coarse.center_x[x + 1] - fine.center_x[edge]) / (fine.center_x[edge + 1] - fine.center_x[edge]) : 0.0;
            double conductance = 0.0;
            for (int fy = 2 * y; fy < std::min(2 * y + 2, fine.height); ++fy)
            {
                const int i = (fy + 1) * fine.stride + edge;
                double resistance = left / fine.right[i - 1] + 1.0 / fine.right[i];
                if (right != 0.0) resistance += right / fine.right[i + 1];
                conductance += 1.0 / resistance;
            }
            coarse.right[(y + 1) * coarse.stride + x + 1] = conductance;
        }
    for (int y = 0; y + 1 < coarse.height; ++y)
        for (int x = 0; x < coarse.width; ++x)
        {
            const int edge = 2 * y + 2;
            const double up = (fine.center_y[edge - 1] - coarse.center_y[y]) /
                              (fine.center_y[edge - 1] - fine.center_y[edge - 2]);
            const double down = edge + 1 < fine.height ?
                (coarse.center_y[y + 1] - fine.center_y[edge]) / (fine.center_y[edge + 1] - fine.center_y[edge]) : 0.0;
            double conductance = 0.0;
            for (int fx = 2 * x; fx < std::min(2 * x + 2, fine.width); ++fx)
            {
                const int i = edge * fine.stride + fx + 1;
                double resistance = up / fine.down[i - fine.stride] + 1.0 / fine.down[i];
                if (down != 0.0) resistance += down / fine.down[i + fine.stride];
                conductance += 1.0 / resistance;
            }
            coarse.down[(y + 1) * coarse.stride + x + 1] = conductance;
        }
    coarse.rebuildDiagonal();
}

void PressureMultigrid::smooth(Level& level, bool reverse, bool from_zero)
{
    const int stride = level.stride;
    // Reverse the colour order after coarse correction to make the V-cycle symmetric.
    for (int pass = 0; pass < 2; ++pass)
    {
        const int color = reverse ? 1 - pass : pass;
        forRows(level.width, level.height, [&](int first_row, int last_row)
        {
            for (int y = first_row; y < last_row; ++y)
                for (int x = (color ^ (y & 1)), i = (y + 1) * stride + x + 1; x < level.width; x += 2, i += 2)
                {
                    double value = level.rhs[i];
                    if (!(from_zero && pass == 0))
                        value += (level.right[i - 1] * level.solution[i - 1] + level.right[i] * level.solution[i + 1]) +
                                 (level.down[i - stride] * level.solution[i - stride] + level.down[i] * level.solution[i + stride]);
                    level.solution[i] = value * level.inverse_diagonal[i];
                }
        });
    }
}

void PressureMultigrid::restrictResidual(Level& fine, Level& coarse)
{
    const int stride = fine.stride;
    forRows(fine.width, fine.height, [&](int first_row, int last_row)
    {
        for (int y = first_row; y < last_row; ++y)
            for (int x = 0, i = (y + 1) * stride + 1; x < fine.width; ++x, ++i)
            {
                const double center = fine.solution[i];
                const double product =
                    (fine.right[i - 1] * (center - fine.solution[i - 1]) + fine.right[i] * (center - fine.solution[i + 1])) +
                    (fine.down[i - stride] * (center - fine.solution[i - stride]) + fine.down[i] * (center - fine.solution[i + stride]));
                fine.residual[i] = fine.rhs[i] - product;
            }
    });

    forRows(coarse.width, coarse.height, [&](int first_row, int last_row)
    {
        for (int y = first_row; y < last_row; ++y)
            for (int x = 0; x < coarse.width; ++x)
            {
                const auto& rx = fine.restrict_x[x];
                const auto& ry = fine.restrict_y[y];
                double value = 0.0;
                for (int iy = 0; iy < ry.count; ++iy)
                {
                    const double* row = fine.residual.data() + (ry.indices[iy] + 1) * stride + 1;
                    double horizontal = 0.0;
                    for (int ix = 0; ix < rx.count; ++ix)
                        horizontal += rx.weights[ix] * row[rx.indices[ix]];
                    value += ry.weights[iy] * horizontal;
                }
                coarse.rhs[(y + 1) * coarse.stride + x + 1] = value;
            }
    });
}

void PressureMultigrid::prolongCorrection(const Level& coarse, Level& fine)
{
    forRows(fine.width, fine.height, [&](int first_row, int last_row)
    {
        for (int y = first_row; y < last_row; ++y)
        {
            const auto& py = fine.prolong_y[y];
            const double* top = coarse.solution.data() + (py.first + 1) * coarse.stride + 1;
            const double* bottom = coarse.solution.data() + (py.second + 1) * coarse.stride + 1;
            for (int x = 0, i = (y + 1) * fine.stride + 1; x < fine.width; ++x, ++i)
            {
                const auto& px = fine.prolong_x[x];
                const double a = (1.0 - px.second_weight) * top[px.first] + px.second_weight * top[px.second];
                const double b = (1.0 - px.second_weight) * bottom[px.first] + px.second_weight * bottom[px.second];
                fine.solution[i] += (1.0 - py.second_weight) * a + py.second_weight * b;
            }
        }
    });
}

bool PressureMultigrid::factorCoarsest()
{
    const auto& level = levels.back();
    const int n = int(coarse_rhs.size());
    std::fill(coarse_factor.begin(), coarse_factor.end(), 0.0);
    // Fix one pressure to zero; this removes the closed tank's constant nullspace.
    for (int cell = 1; cell <= n; ++cell)
    {
        const int x = cell % level.width, y = cell / level.width;
        const int i = (y + 1) * level.stride + x + 1, row = cell - 1;
        coarse_factor[row * n + row] = level.right[i - 1] + level.right[i] + level.down[i - level.stride] + level.down[i];
        if (x > 0 && cell - 1 > 0) coarse_factor[row * n + row - 1] = -level.right[i - 1];
        if (y > 0 && cell - level.width > 0) coarse_factor[row * n + row - level.width] = -level.down[i - level.stride];
    }
    // Cholesky is rebuilt once per projection, then reused by every V-cycle.
    for (int row = 0; row < n; ++row)
        for (int col = 0; col <= row; ++col)
        {
            double value = coarse_factor[row * n + col];
            for (int k = 0; k < col; ++k)
                value -= coarse_factor[row * n + k] * coarse_factor[col * n + k];
            if (row == col)
            {
                // A failed pivot is a real numerical breakdown, not a converged solve.
                if (value <= 0.0 || !std::isfinite(value)) return false;
                coarse_factor[row * n + col] = std::sqrt(value);
            }
            else coarse_factor[row * n + col] = value / coarse_factor[col * n + col];
        }
    return true;
}

void PressureMultigrid::solveCoarsest()
{
    auto& level = levels.back();
    const int n = int(coarse_rhs.size());
    double mean = 0.0;
    for (int y = 0; y < level.height; ++y)
        for (int x = 0; x < level.width; ++x)
            mean += level.rhs[(y + 1) * level.stride + x + 1];
    mean /= n + 1;
    for (int row = 0; row < n; ++row)
    {
        const int cell = row + 1;
        double value = level.rhs[(cell / level.width + 1) * level.stride + cell % level.width + 1] - mean;
        for (int col = 0; col < row; ++col) value -= coarse_factor[row * n + col] * coarse_rhs[col];
        coarse_rhs[row] = value / coarse_factor[row * n + row];
    }
    for (int row = n - 1; row >= 0; --row)
    {
        double value = coarse_rhs[row];
        for (int col = row + 1; col < n; ++col) value -= coarse_factor[col * n + row] * coarse_rhs[col];
        coarse_rhs[row] = value / coarse_factor[row * n + row];
    }
    // Choose a mean-zero gauge before interpolating this solution to finer grids.
    mean = std::accumulate(coarse_rhs.begin(), coarse_rhs.end(), 0.0) / (n + 1);
    level.solution[level.stride + 1] = -mean;
    for (int cell = 1; cell <= n; ++cell)
        level.solution[(cell / level.width + 1) * level.stride + cell % level.width + 1] = coarse_rhs[cell - 1] - mean;
}

void PressureMultigrid::cycle(int index)
{
    if (index + 1 == int(levels.size()))
    {
        solveCoarsest();
        return;
    }
    auto& fine = levels[index];
    auto& coarse = levels[index + 1];
    constexpr int smoothing_sweeps = 4;
    smooth(fine, false, true);
    for (int sweep = 1; sweep < smoothing_sweeps; ++sweep)
        smooth(fine, false);
    restrictResidual(fine, coarse);
    cycle(index + 1);
    prolongCorrection(coarse, fine);
    for (int sweep = 0; sweep < smoothing_sweeps; ++sweep)
        smooth(fine, true);
}

void PressureMultigrid::apply(const std::vector<double>& residual, std::vector<double>& correction, double residual_mean)
{
    auto& fine = levels.front();
    for (int y = 0; y < fine.height; ++y)
        for (int x = 0; x < fine.width; ++x)
            fine.rhs[(y + 1) * fine.stride + x + 1] = residual[y * fine.width + x] - residual_mean;
    cycle(0);
    correction.resize(residual.size());
    for (int y = 0; y < fine.height; ++y)
        std::copy_n(fine.solution.data() + (y + 1) * fine.stride + 1, fine.width, correction.data() + y * fine.width);
}

}
