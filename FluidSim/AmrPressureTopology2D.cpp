// Derived from AMReX (c) 2024 The Regents of the University of California.
// Redistribution terms and the pinned source map: reference/pressure2d/.
#include "AmrPressureTopology2D.h"
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

AmrPressureTopology2D::Grid AmrPressureTopology2D::makeGrid(const Level& level)
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
    return grid;
}

int AmrPressureTopology2D::findCell(const Grid& grid, int x, int y)
{
    for (const auto& p : grid.patches)
        if (x >= p.box.x && x < p.box.x + p.box.width && y >= p.box.y && y < p.box.y + p.box.height)
            return p.index(x, y);
    return -1;
}

void AmrPressureTopology2D::buildBoundary(Grid& grid, const Grid* coarse)
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

void AmrPressureTopology2D::buildTransfer(Grid& fine, const Grid& coarse)
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

AmrPressureTopology2D::AmrPressureTopology2D(const std::vector<Level>& levels)
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
}

} // namespace fluid
