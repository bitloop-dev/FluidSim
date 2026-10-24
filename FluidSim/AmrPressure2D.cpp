// Derived from AMReX (c) 2024 The Regents of the University of California.
// Redistribution terms and the pinned source map: reference/pressure2d/.
#include "AmrPressure2D.h"
#include <algorithm>
#include <cmath>

namespace fluid
{
namespace
{
std::array<double, 3> interpolationWeights(double position, const std::array<double, 3>& points, int count)
{
    std::array<double, 3> weights{};
    for (int i = 0; i < count; ++i)
    {
        weights[i] = 1.0;
        for (int j = 0; j < count; ++j)
            if (i != j) weights[i] *= (position - points[j]) / (points[i] - points[j]);
    }
    return weights;
}
} // namespace

AmrPressure2D::Grid AmrPressure2D::makeGrid(const Level& level)
{
    Grid grid;
    grid.width = level.width;
    grid.height = level.height;
    grid.cell_size = level.cell_size;
    for (const auto& box : level.boxes)
    {
        Patch patch{box, grid.size, box.width + 2};
        int p = int(grid.patches.size());
        grid.patches.push_back(patch);
        for (int y = box.y; y < box.y + box.height; ++y)
            grid.rows.push_back({p, y, patch.index(box.x, y), patch.index(box.x + box.width, y)});
        grid.size += (box.width + 2) * (box.height + 2);
        grid.cells += box.width * box.height;
    }
    grid.active.resize(grid.size, 1);
    grid.delta.resize(grid.size);
    grid.diagonal.resize(grid.size);
    for (auto& beta : grid.beta)
        beta.resize(grid.size);
    return grid;
}

int AmrPressure2D::findCell(const Grid& grid, int x, int y)
{
    for (const auto& p : grid.patches)
        if (x >= p.box.x && x < p.box.x + p.box.width && y >= p.box.y && y < p.box.y + p.box.height)
            return p.index(x, y);
    return -1;
}

void AmrPressure2D::buildBoundary(Grid& grid, const Grid* coarse)
{
    for (const auto& p : grid.patches)
        for (int axis = 0; axis < 2; ++axis)
            for (int side = 0; side < 2; ++side)
            {
                int length = axis == 0 ? p.box.width : p.box.height;
                int start = axis == 0 ? p.box.y : p.box.x;
                int end = start + (axis == 0 ? p.box.height : p.box.width);
                int normal = (axis == 0 ? p.box.x : p.box.y) + (side ? length : -1);
                int step = side ? -1 : 1;
                int dx = axis == 0 ? step : 0, dy = axis == 1 ? step : 0;
                for (int t = start; t < end; ++t)
                {
                    int x = axis == 0 ? normal : t, y = axis == 1 ? normal : t;
                    Ghost g{p.index(x, y),
                            p.index(x + dx, y + dy),
                            p.index(x + 2 * dx, y + 2 * dy),
                            findCell(grid, x, y),
                            0.,
                            0.,
                            0.};
                    if (g.source < 0)
                    {
                        bool wall = x < 0 || y < 0 || x >= grid.width || y >= grid.height;
                        if (wall)
                            g.first_weight = 1.0;
                        else
                        {
                            // AMReX's order-three normal Dirichlet stencil at the coarse cell centre.
                            auto weights = interpolationWeights(-.5, {-1., .5, 1.5}, std::min(length + 1, 3));
                            g.boundary_weight = weights[0];
                            g.first_weight = weights[1];
                            g.second_weight = weights[2];
                            int cx = x / 2, cy = y / 2;
                            std::array<double, 3> points{0., 0., 0.};
                            g.coarse[0] = findCell(*coarse, cx, cy);
                            g.count = 1;
                            auto available = [&](int offset)
                            {
                                int fx = x + (axis == 1 ? offset : 0), fy = y + (axis == 0 ? offset : 0);
                                return fx >= 0 && fy >= 0 && fx < grid.width && fy < grid.height &&
                                       findCell(grid, fx, fy) < 0;
                            };
                            auto add = [&](int offset)
                            {
                                int index =
                                    findCell(*coarse, cx + (axis == 1 ? offset : 0), cy + (axis == 0 ? offset : 0));
                                if (index >= 0)
                                {
                                    points[g.count] = double(offset);
                                    g.coarse[g.count++] = index;
                                }
                            };
                            bool low = available(-2), high = available(2);
                            if (low)
                                add(-1);
                            else if (available(4))
                                add(2);
                            if (high)
                                add(1);
                            else if (available(-4))
                                add(-2);
                            if (!low && !high) g.count = 1;
                            g.weights = interpolationWeights(-.5 + (.5 + t % 2) / 2., points, g.count);
                            for (const auto& cp : coarse->patches)
                                if (cx >= cp.box.x && cx < cp.box.x + cp.box.width && cy >= cp.box.y &&
                                    cy < cp.box.y + cp.box.height)
                                {
                                    int c = cp.index(cx, cy), cs = axis == 0 ? 1 : cp.stride;
                                    int fs = axis == 0 ? 1 : p.stride;
                                    grid.joins.push_back({c, step * cs, step > 0 ? c + cs : c, g.first, step * fs,
                                                          step > 0 ? g.first : g.target, axis, step});
                                }
                        }
                    }
                    grid.ghosts.push_back(g);
                }
            }
}

void AmrPressure2D::buildTransfer(Grid& fine, const Grid& coarse)
{
    for (const auto& p : fine.patches)
    {
        for (int y = p.box.y; y < p.box.y + p.box.height; y += 2)
            for (int x = p.box.x; x < p.box.x + p.box.width; x += 2)
                fine.averages.push_back({findCell(coarse, x / 2, y / 2),
                                         {p.index(x, y), p.index(x + 1, y), p.index(x, y + 1), p.index(x + 1, y + 1)}});
        for (int axis = 0; axis < 2; ++axis)
            for (int y = p.box.y; y < p.box.y + p.box.height + (axis == 1); y += 2)
                for (int x = p.box.x; x < p.box.x + p.box.width + (axis == 0); x += 2)
                {
                    int cx = x / 2, cy = y / 2;
                    for (const auto& cp : coarse.patches)
                        if (cx >= cp.box.x && cx < cp.box.x + cp.box.width + (axis == 0) && cy >= cp.box.y &&
                            cy < cp.box.y + cp.box.height + (axis == 1))
                            fine.face_averages[axis].push_back(
                                {cp.index(cx, cy), {p.index(x, y), p.index(x + (axis == 1), y + (axis == 0))}});
                }
        for (int y = p.box.y; y < p.box.y + p.box.height; ++y)
            for (int x = p.box.x; x < p.box.x + p.box.width; ++x)
            {
                int cx = x / 2, cy = y / 2, ox = 2 * (x % 2) - 1, oy = 2 * (y % 2) - 1;
                fine.interpolation.push_back({p.index(x, y),
                                              {findCell(coarse, cx, cy), findCell(coarse, cx + ox, cy),
                                               findCell(coarse, cx, cy + oy), findCell(coarse, cx + ox, cy + oy)}});
            }
    }
}

AmrPressure2D::AmrPressure2D(const std::vector<Level>& levels)
{
    for (const auto& level : levels)
        grids.push_back(makeGrid(level));
    for (int l = 0; l < int(grids.size()); ++l)
    {
        buildBoundary(grids[l], l ? &grids[l - 1] : nullptr);
        if (l)
        {
            buildTransfer(grids[l], grids[l - 1]);
            for (const auto& a : grids[l].averages)
                grids[l - 1].active[a.coarse] = 0;
        }
    }
    Grid base = grids[0];
    while (base.width % 2 == 0 && base.height % 2 == 0 && base.width >= 4 && base.height >= 4)
    {
        Level level{base.width / 2, base.height / 2, base.cell_size * 2., {{0, 0, base.width / 2, base.height / 2}}};
        auto coarse = makeGrid(level);
        buildBoundary(coarse, nullptr);
        base.averages.clear();
        base.interpolation.clear();
        base.face_averages = {};
        buildTransfer(base, coarse);
        multigrid.push_back(std::move(base));
        base = std::move(coarse);
    }
    multigrid.push_back(std::move(base));
    residual = makeField();
    correction = makeField();
    held = makeField();
    solution = makeField();
    rhs_work = makeField();
    scratch = makeField();
    for (const auto& grid : multigrid)
    {
        mg_rhs.emplace_back(grid.size);
        mg_correction.emplace_back(grid.size);
        mg_residual.emplace_back(grid.size);
    }
}

AmrPressure2D::Field AmrPressure2D::makeField() const
{
    Field field;
    for (const auto& grid : grids)
        field.emplace_back(grid.size);
    return field;
}

void AmrPressure2D::fillBoundary(const Grid& grid, std::vector<double>& field, const std::vector<double>* coarse)
{
    for (const auto& g : grid.ghosts)
    {
        if (g.source >= 0)
            field[g.target] = field[g.source];
        else
        {
            double value = g.first_weight * field[g.first];
            if (g.second_weight != 0.) value += g.second_weight * field[g.second];
            if (coarse)
            {
                double boundary = 0.;
                for (int i = 0; i < g.count; ++i)
                    boundary += g.weights[i] * (*coarse)[g.coarse[i]];
                value += g.boundary_weight * boundary;
            }
            field[g.target] = value;
        }
    }
}

void AmrPressure2D::updateDiagonal(Grid& grid)
{
    std::fill(grid.delta.begin(), grid.delta.end(), 0.);
    double inverse_square = 1. / (grid.cell_size * grid.cell_size);
    for (const auto& row : grid.rows)
        for (int i = row.begin; i < row.end; ++i)
        {
            int s = grid.patches[row.patch].stride;
            grid.diagonal[i] =
                inverse_square * (grid.beta[0][i] + grid.beta[0][i + 1] + grid.beta[1][i] + grid.beta[1][i + s]);
        }
    for (const auto& g : grid.ghosts)
    {
        if (g.source >= 0) continue;
        int stride = g.first - g.target;
        int axis = std::abs(stride) == 1 ? 0 : 1;
        int face = stride > 0 ? g.first : g.target;
        grid.delta[g.first] += inverse_square * grid.beta[axis][face] * g.first_weight;
    }
}

void AmrPressure2D::setCoefficients(const Faces& coefficients)
{
    for (int l = 0; l < int(grids.size()); ++l)
        for (int d = 0; d < 2; ++d)
            grids[l].beta[d] = coefficients[d][l];
    for (int l = int(grids.size()) - 1; l > 0; --l)
        for (int d = 0; d < 2; ++d)
            for (const auto& a : grids[l].face_averages[d])
                grids[l - 1].beta[d][a.coarse] = .5 * (grids[l].beta[d][a.fine[0]] + grids[l].beta[d][a.fine[1]]);
    for (auto& grid : grids)
        updateDiagonal(grid);
    multigrid[0].beta = grids[0].beta;
    for (int m = 0; m < int(multigrid.size()); ++m)
    {
        if (m)
            for (int d = 0; d < 2; ++d)
                for (const auto& a : multigrid[m - 1].face_averages[d])
                    multigrid[m].beta[d][a.coarse] =
                        .5 * (multigrid[m - 1].beta[d][a.fine[0]] + multigrid[m - 1].beta[d][a.fine[1]]);
        updateDiagonal(multigrid[m]);
    }
}

void AmrPressure2D::applyGrid(const Grid& grid, std::vector<double>& output, std::vector<double>& pressure,
                              const std::vector<double>* coarse)
{
    fillBoundary(grid, pressure, coarse);
    double factor = 1. / (grid.cell_size * grid.cell_size);
    const auto& bx = grid.beta[0];
    const auto& by = grid.beta[1];
#pragma omp parallel for schedule(static) if (grid.cells >= 65536)
    for (int r = 0; r < int(grid.rows.size()); ++r)
    {
        const auto& row = grid.rows[r];
        int s = grid.patches[row.patch].stride;
        for (int i = row.begin; i < row.end; ++i)
            output[i] =
                -factor * (bx[i + 1] * (pressure[i + 1] - pressure[i]) - bx[i] * (pressure[i] - pressure[i - 1])) -
                factor * (by[i + s] * (pressure[i + s] - pressure[i]) - by[i] * (pressure[i] - pressure[i - s]));
    }
}

void AmrPressure2D::smooth(const Grid& grid, std::vector<double>& pressure, const std::vector<double>& rhs,
                           int iterations)
{
    double factor = 1. / (grid.cell_size * grid.cell_size);
    const auto& bx = grid.beta[0];
    const auto& by = grid.beta[1];
    for (int iteration = 0; iteration < iterations; ++iteration)
        for (int color = 0; color < 2; ++color)
        {
            fillBoundary(grid, pressure, nullptr);
#pragma omp parallel for schedule(static) if (grid.cells >= 65536)
            for (int r = 0; r < int(grid.rows.size()); ++r)
            {
                const auto& row = grid.rows[r];
                const auto& patch = grid.patches[row.patch];
                int s = patch.stride;
                int first = row.begin + ((patch.box.x + row.y + color) & 1);
                for (int i = first; i < row.end; i += 2)
                {
                    double neighbors = factor * (bx[i] * pressure[i - 1] + bx[i + 1] * pressure[i + 1]) +
                                       factor * (by[i] * pressure[i - s] + by[i + s] * pressure[i + s]);
                    pressure[i] =
                        (rhs[i] + neighbors - pressure[i] * grid.delta[i]) / (grid.diagonal[i] - grid.delta[i]);
                }
            }
        }
}

void AmrPressure2D::restrictCells(const Grid& fine, std::vector<double>& coarse, const std::vector<double>& values)
{
    for (const auto& a : fine.averages)
        coarse[a.coarse] = .25 * (values[a.fine[0]] + values[a.fine[1]] + values[a.fine[2]] + values[a.fine[3]]);
}

void AmrPressure2D::restrictField(Field& field) const
{
    for (int l = int(grids.size()) - 1; l > 0; --l)
        restrictCells(grids[l], field[l - 1], field[l]);
}

void AmrPressure2D::fluxes(Faces& output, Field& pressure)
{
    for (int l = 0; l < int(grids.size()); ++l)
    {
        const auto& grid = grids[l];
        auto& q = pressure[l];
        fillBoundary(grid, q, l ? &pressure[l - 1] : nullptr);
        double factor = 1. / grid.cell_size;
        for (const auto& p : grid.patches)
            for (int d = 0; d < 2; ++d)
            {
                int stride = d ? p.stride : 1;
#pragma omp parallel for schedule(static) if (p.box.width * p.box.height >= 65536)
                for (int y = p.box.y; y < p.box.y + p.box.height + (d == 1); ++y)
                    for (int x = p.box.x; x < p.box.x + p.box.width + (d == 0); ++x)
                    {
                        int i = p.index(x, y);
                        output[d][l][i] = -factor * grid.beta[d][i] * (q[i] - q[i - stride]);
                    }
            }
    }
    for (int l = int(grids.size()) - 1; l > 0; --l)
        for (int d = 0; d < 2; ++d)
            for (const auto& a : grids[l].face_averages[d])
                output[d][l - 1][a.coarse] = .5 * (output[d][l][a.fine[0]] + output[d][l][a.fine[1]]);
}

void AmrPressure2D::apply(Field& output, Field& pressure)
{
    for (int l = 0; l < int(grids.size()); ++l)
        applyGrid(grids[l], output[l], pressure[l], l ? &pressure[l - 1] : nullptr);
    for (int l = int(grids.size()) - 2; l >= 0; --l)
        reflux(l, output, pressure, 1.);
    restrictField(output);
}

void AmrPressure2D::reflux(int coarse_level, Field& output, Field& pressure, double sign)
{
    int l = coarse_level + 1;
    const auto& fine = grids[l];
    const auto& coarse = grids[coarse_level];
    fillBoundary(coarse, pressure[coarse_level], coarse_level ? &pressure[coarse_level - 1] : nullptr);
    fillBoundary(fine, pressure[l], &pressure[coarse_level]);
    const auto& c = pressure[coarse_level];
    const auto& f = pressure[l];
    // Reflux only the coarse/fine perimeter; interior pressure needs one stencil pass.
    for (const auto& j : fine.joins)
    {
        double fc = -j.sign * fine.beta[j.axis][j.fine_face] * (f[j.fine] - f[j.fine - j.fine_step]) / fine.cell_size;
        double cc = -j.sign * coarse.beta[j.axis][j.coarse_face] * (c[j.coarse + j.coarse_step] - c[j.coarse]) /
                    coarse.cell_size;
        output[coarse_level][j.coarse] += sign * .5 * j.sign * (fc - cc) / coarse.cell_size;
    }
}

double AmrPressure2D::dot(const Field& first, const Field& second) const
{
    double sum = 0.;
    for (int l = 0; l < int(grids.size()); ++l)
    {
        const auto& grid = grids[l];
        double level_sum = 0.;
#pragma omp parallel for reduction(+ : level_sum) schedule(static) if (grid.cells >= 65536)
        for (int r = 0; r < int(grid.rows.size()); ++r)
            for (int i = grid.rows[r].begin; i < grid.rows[r].end; ++i)
                if (grid.active[i]) level_sum += first[l][i] * second[l][i];
        sum += level_sum;
    }
    return sum;
}

double AmrPressure2D::norm(const Field& field) const
{
    return std::sqrt(dot(field, field));
}

void AmrPressure2D::increment(Field& destination, const Field& source, double factor) const
{
    for (int l = 0; l < int(grids.size()); ++l)
    {
        const auto& grid = grids[l];
#pragma omp parallel for schedule(static) if (grid.cells >= 65536)
        for (int r = 0; r < int(grid.rows.size()); ++r)
            for (int i = grid.rows[r].begin; i < grid.rows[r].end; ++i)
                destination[l][i] += factor * source[l][i];
    }
}

void AmrPressure2D::scale(Field& field, double factor) const
{
    for (int l = 0; l < int(grids.size()); ++l)
    {
        const auto& grid = grids[l];
#pragma omp parallel for schedule(static) if (grid.cells >= 65536)
        for (int r = 0; r < int(grid.rows.size()); ++r)
            for (int i = grid.rows[r].begin; i < grid.rows[r].end; ++i)
                field[l][i] *= factor;
    }
}

void AmrPressure2D::subtractMean(const Grid& grid, std::vector<double>& field)
{
    double sum = 0.;
    for (const auto& row : grid.rows)
        for (int i = row.begin; i < row.end; ++i)
            sum += field[i];
    double mean = sum / grid.cells;
    for (const auto& row : grid.rows)
        for (int i = row.begin; i < row.end; ++i)
            field[i] -= mean;
}

void AmrPressure2D::residualLevel(int level, Field& output, Field& pressure, const Field& rhs)
{
    const auto& grid = grids[level];
    applyGrid(grid, output[level], pressure[level], level ? &pressure[level - 1] : nullptr);
    for (const auto& row : grid.rows)
        for (int i = row.begin; i < row.end; ++i)
            output[level][i] = rhs[level][i] - output[level][i];
}

void AmrPressure2D::cycleBase(bool bottomKrylov)
{
    int last = int(multigrid.size()) - 1;
    mg_rhs[0] = residual[0];
    for (int m = 0; m < last; ++m)
    {
        const auto& grid = multigrid[m];
        std::fill(mg_correction[m].begin(), mg_correction[m].end(), 0.);
        smooth(grid, mg_correction[m], mg_rhs[m], 2);
        applyGrid(grid, mg_residual[m], mg_correction[m]);
        for (const auto& row : grid.rows)
            for (int i = row.begin; i < row.end; ++i)
                mg_residual[m][i] = mg_rhs[m][i] - mg_residual[m][i];
        restrictCells(grid, mg_rhs[m + 1], mg_residual[m]);
    }
    // GMRESMLMG selects the eight-sweep bottom smoother in the upstream path.
    std::fill(mg_correction[last].begin(), mg_correction[last].end(), 0.);
    if (!bottomKrylov || !solveBottom(multigrid[last], mg_correction[last], mg_rhs[last]))
        smooth(multigrid[last], mg_correction[last], mg_rhs[last], 8);
    for (int m = last - 1; m >= 0; --m)
    {
        const auto& grid = multigrid[m];
        for (const auto& a : grid.averages)
            for (int i : a.fine)
                mg_correction[m][i] += mg_correction[m + 1][a.coarse];
        smooth(grid, mg_correction[m], mg_rhs[m], 2);
    }
    correction[0] = mg_correction[0];
}

void AmrPressure2D::precondition(Field& output, const Field& rhs, bool bottomKrylov)
{
    rhs_work = rhs;
    restrictField(rhs_work);
    double sum = 0.;
    for (const auto& row : grids[0].rows)
        for (int i = row.begin; i < row.end; ++i)
            sum += rhs_work[0][i];
    double mean = sum / grids[0].cells;
    for (int l = 0; l < int(grids.size()); ++l)
    {
        std::fill(solution[l].begin(), solution[l].end(), 0.);
        for (const auto& row : grids[l].rows)
            for (int i = row.begin; i < row.end; ++i)
                rhs_work[l][i] -= mean;
    }
    residual = rhs_work;
    int last = int(grids.size()) - 1;
    for (int l = last; l > 0; --l)
    {
        const auto& grid = grids[l];
        std::fill(correction[l].begin(), correction[l].end(), 0.);
        smooth(grid, correction[l], residual[l], 2);
        for (const auto& row : grid.rows)
            for (int i = row.begin; i < row.end; ++i)
                solution[l][i] += correction[l][i];
        residualLevel(l - 1, residual, solution, rhs_work);
        applyGrid(grid, scratch[l], correction[l]);
        for (const auto& row : grid.rows)
            for (int i = row.begin; i < row.end; ++i)
                residual[l][i] -= scratch[l][i];
        reflux(l - 1, residual, solution, -1.);
        restrictCells(grid, residual[l - 1], residual[l]);
        if (l != last) held[l] = correction[l];
    }
    subtractMean(grids[0], residual[0]);
    cycleBase(bottomKrylov);
    for (const auto& row : grids[0].rows)
        for (int i = row.begin; i < row.end; ++i)
            solution[0][i] += correction[0][i];
    for (int l = 1; l <= last; ++l)
    {
        const auto& grid = grids[l];
        const auto& coarse = correction[l - 1];
        // AMR correction uses bilinear assignment; the base V-cycle uses constant addition.
        for (const auto& a : grid.interpolation)
        {
            constexpr double weights[]{.5625, .1875, .1875, .0625};
            double value = 0.;
            for (int k = 0; k < 4; ++k)
                if (a.coarse[k] >= 0) value += weights[k] * coarse[a.coarse[k]];
            correction[l][a.fine] = value;
            solution[l][a.fine] += value;
            if (l != last) held[l][a.fine] += value;
        }
        applyGrid(grid, scratch[l], correction[l], &correction[l - 1]);
        for (const auto& row : grid.rows)
            for (int i = row.begin; i < row.end; ++i)
                residual[l][i] -= scratch[l][i];
        std::fill(correction[l].begin(), correction[l].end(), 0.);
        smooth(grid, correction[l], residual[l], 2);
        for (const auto& row : grid.rows)
            for (int i = row.begin; i < row.end; ++i)
            {
                solution[l][i] += correction[l][i];
                if (l != last) correction[l][i] += held[l][i];
            }
    }
    restrictField(solution);
    output = solution;
}

bool AmrPressure2D::solveBottom(const Grid& grid, std::vector<double>& pressure, const std::vector<double>& rhs)
{
    // MLCGSolver::solve_bicgstab: diagonal normalization and a 1e-4 relative bottom target.
    auto r=rhs;
    subtractMean(grid,r);
    auto each=[&](auto&& f){for(const auto& row:grid.rows)for(int i=row.begin;i<row.end;++i)f(i);};
    auto maximum=[&](const auto& a){double m=0.;each([&](int i){m=std::isfinite(a[i])?std::max(m,std::abs(a[i])):std::numeric_limits<double>::infinity();});return m;};
    auto product=[&](const auto& a,const auto& b){double s=0.;each([&](int i){s+=a[i]*b[i];});return s;};
    auto normalize=[&](auto& a){each([&](int i){a[i]/=grid.diagonal[i];});};
    normalize(r);
    const auto shadow=r;
    std::vector<double> p(grid.size),v(grid.size),t(grid.size);
    const double initial=maximum(r),target=1.e-4*initial;
    if(initial==0.)return true;
    double previous=0.,alpha=0.,omega=0.,magnitude=initial;
    bool breakdown=false;
    for(int iteration=0;iteration<200;++iteration) {
        const double rho=product(shadow,r);
        if(rho==0.){breakdown=true;break;}
        if(iteration==0)p=r;
        else {const double beta=(rho/previous)*(alpha/omega);each([&](int i){p[i]=r[i]+beta*(p[i]-omega*v[i]);});}
        applyGrid(grid,v,p);normalize(v);
        const double denominator=product(shadow,v);
        if(denominator==0.){breakdown=true;break;}
        alpha=rho/denominator;
        each([&](int i){pressure[i]+=alpha*p[i];r[i]-=alpha*v[i];});
        magnitude=maximum(r);
        if(magnitude<target)return true;
        applyGrid(grid,t,r);normalize(t);
        const double tt=product(t,t);
        if(tt==0.){breakdown=true;break;}
        omega=product(t,r)/tt;
        each([&](int i){pressure[i]+=omega*r[i];r[i]-=omega*t[i];});
        magnitude=maximum(r);
        if(magnitude<target)return true;
        if(omega==0. || !std::isfinite(magnitude)){breakdown=true;break;}
        previous=rho;
    }
    // Upstream retains useful nonconverged progress, otherwise restarts with smoothing.
    if(breakdown || !(magnitude<initial))std::fill(pressure.begin(),pressure.end(),0.);
    return false;
}

AmrPressure2D::Result AmrPressure2D::solveMultigrid(Field& pressure,const Field& rhs,double tolerance,int max_iterations)
{
    Field right=rhs,error=makeField(),update=makeField();
    restrictField(right);
    double sum=0.;for(const auto& row:grids[0].rows)for(int i=row.begin;i<row.end;++i)sum+=right[0][i];
    const double mean=sum/grids[0].cells;
    for(int l=0;l<int(grids.size());++l)for(const auto& row:grids[l].rows)for(int i=row.begin;i<row.end;++i)right[l][i]-=mean;
    double magnitude=0.;int iterations=0;
    for(;;) {
        apply(error,pressure);scale(error,-1.);increment(error,right,1.);
        magnitude=0.;
        for(int l=0;l<int(grids.size());++l)for(const auto& row:grids[l].rows)for(int i=row.begin;i<row.end;++i)
            if(grids[l].active[i])magnitude=std::isfinite(error[l][i])?std::max(magnitude,std::abs(error[l][i])):std::numeric_limits<double>::infinity();
        if(!std::isfinite(magnitude) || magnitude<=tolerance || iterations==max_iterations)break;
        precondition(update,error,true);increment(pressure,update,1.);++iterations;
    }
    return {iterations,magnitude,std::isfinite(magnitude)&&magnitude<=tolerance};
}

AmrPressure2D::Result AmrPressure2D::solve(Field& pressure, const Field& rhs, double tolerance, int max_iterations)
{
    // AMReX GMRES uses right preconditioning and two classical Gram-Schmidt passes.
    constexpr int restart = 30;
    std::vector<Field> basis;
    basis.reserve(restart + 1);
    basis.push_back(makeField());
    Field error = makeField(), update = makeField(), work = makeField(), combination = makeField();
    apply(error, pressure);
    increment(error, rhs, -1.);
    int iterations = 0;
    double magnitude = norm(error);
    while (magnitude >= tolerance && iterations < max_iterations)
    {
        apply(basis[0], update);
        scale(basis[0], -1.);
        increment(basis[0], error, 1.);
        magnitude = norm(basis[0]);
        if (magnitude < tolerance) break;
        if (!std::isfinite(magnitude)) return {iterations, magnitude, false};
        scale(basis[0], 1. / magnitude);
        std::array<std::array<double, restart>, restart + 1> h{};
        std::array<double, restart + 1> g{}, cosines{}, sines{};
        g[0] = magnitude;
        int count = 0;
        for (; count < restart && iterations < max_iterations && magnitude >= tolerance;)
        {
            int k = count;
            if (int(basis.size()) <= k + 1) basis.push_back(makeField());
            precondition(work, basis[k]);
            apply(basis[k + 1], work);
            for (int pass = 0; pass < 2; ++pass)
            {
                std::array<double, restart> products{};
                for (int j = 0; j <= k; ++j)
                    products[j] = dot(basis[k + 1], basis[j]);
                for (int j = 0; j <= k; ++j)
                {
                    increment(basis[k + 1], basis[j], -products[j]);
                    h[j][k] += products[j];
                }
            }
            double length = norm(basis[k + 1]);
            bool happy = length < 1.e-99;
            if (!happy) scale(basis[k + 1], 1. / length);
            h[k + 1][k] = length;
            for (int j = 0; j < k; ++j)
            {
                double old = h[j][k];
                h[j][k] = cosines[j] * old + sines[j] * h[j + 1][k];
                h[j + 1][k] = cosines[j] * h[j + 1][k] - sines[j] * old;
            }
            if (!happy)
            {
                double rotation = std::sqrt(h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]);
                cosines[k] = h[k][k] / rotation;
                sines[k] = h[k + 1][k] / rotation;
                g[k + 1] = -sines[k] * g[k];
                g[k] *= cosines[k];
                h[k][k] = cosines[k] * h[k][k] + sines[k] * h[k + 1][k];
                magnitude = std::abs(g[k + 1]);
            }
            else
                magnitude = 0.;
            ++count;
            ++iterations;
            if (!std::isfinite(magnitude)) return {iterations, magnitude, false};
            if (happy) break;
        }
        for (int k = count - 1; k >= 0; --k)
        {
            double value = g[k];
            for (int j = k + 1; j < count; ++j)
                value -= h[k][j] * g[j];
            // A singular Krylov basis can break down before reaching the requested tolerance.
            if (h[k][k] == 0.) return {iterations, magnitude, false};
            g[k] = value / h[k][k];
        }
        for (auto& level : combination)
            std::fill(level.begin(), level.end(), 0.);
        for (int k = 0; k < count; ++k)
            increment(combination, basis[k], g[k]);
        precondition(work, combination);
        increment(update, work, 1.);
    }
    increment(pressure, update, -1.);
    apply(work, pressure);
    increment(work, rhs, -1.);
    magnitude = norm(work);
    return {iterations, magnitude, std::isfinite(magnitude) && magnitude <= tolerance};
}

} // namespace fluid
