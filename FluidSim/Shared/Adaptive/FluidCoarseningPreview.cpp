#include "Shared/Adaptive/FluidCoarseningPreview.h"
#include "Shared/Adaptive/FluidCellView.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#include "AmrHierarchy2D.h"
#include <bit>

namespace fluid
{
namespace
{
using Cell = FluidCellView::Cell;

struct Assessment
{
    FluidCoarseningPreview::Label label;
    bool quiet = false;
};

void flatMaterialError(const FluidCellView& view, int x, int y, FluidCoarseningPreview::Error& error)
{
    // Uniform liquid previews retain their flat-mean representation contract.
    // Native gas uses the bounded plane below.
    const int offsets[5][2] = {{0, 0}, {-2, 0}, {2, 0}, {0, -2}, {0, 2}};
    std::array<Cell, 4> groups[5];
    for (int g = 0; g < 5; ++g)
    {
        const int gx = x + offsets[g][0];
        const int gy = y + offsets[g][1];
        if (gx < 0 || gy < 0 || gx + 1 >= view.width || gy + 1 >= view.height)
            continue;

        for (int j = 0; j < 2; ++j)
        {
            for (int i = 0; i < 2; ++i)
                groups[g][2 * j + i] = view.at(gx + i, gy + j);
        }
    }

    for (int k = -1; k < view.components(); ++k)
    {
        auto value = [&](const Cell& c)
        {
            return k < 0 ? c.density() : c.fraction(k);
        };

        double mean[5]{};

        for (int g = 0; g < 5; ++g)
        {
            if (groups[g][0])
            {
                for (const auto& c : groups[g])
                    mean[g] += 0.25 * value(c);
            }
        }

        double detail = 0.0;
        double slope[2]{};

        for (const auto& c : groups[0])
            detail = std::max(detail, std::abs(value(c) - mean[0]));
        for (int side = 0; side < 4; ++side)
        {
            if (groups[side + 1][0])
            {
                slope[side / 2] = std::max(slope[side / 2], std::abs(mean[side + 1] - mean[0]));

                // Averages alone hide opposing fine samples on an adjoining face.
                constexpr int face[4][2] = {{1, 3}, {0, 2}, {2, 3}, {0, 1}};
                const auto& neighbour = groups[side + 1];
                detail =
                    std::max(detail, 0.5 * std::abs(value(neighbour[face[side][0]]) - value(neighbour[face[side][1]])));
            }
        }

        // Resolved neighbour slopes predict variation at child centres. Apply
        // this before merging too, so a stationary step cannot merge/refine forever.
        detail = std::max(detail, 0.25 * (slope[0] + slope[1]));
        if (k < 0)
            error.density = detail / mean[0];
        else
            error.ingredient = std::max(error.ingredient, detail);
    }
}

void nativeRepresentationError(const FluidCellView& view, int x, int y, int scale, FluidCoarseningPreview::Error& error,
    FluidCellView::MaterialPlane& plane)
{
    view.reconstruct(plane, x, y, scale, true);
    const int stride = view.minimumScale(x, y, scale);
    constexpr int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    for (int k = -1; k < view.components(); ++k)
    {
        double mean = 0.0;
        LinearSlope slope;
        for (int n = 0; n < view.components(); ++n)
        {
            if (k < 0 || k == n)
            {
                const double divisor = k < 0 ? 1.0 : view.mixture->components[n].density;
                mean += plane.mean(n) / divisor;
                slope.x += plane.slopes[n].x / divisor;
                slope.y += plane.slopes[n].y / divisor;
            }
        }

        auto value = [&](const Cell& c)
        {
            return k < 0 ? c.density() : c.fraction(k);
        };

        auto residual = [&](const Cell& c)
        {
            return value(c) - mean - (c.cx() / view.cellSize - x - 0.5 * scale) * slope.x / scale -
                   (c.cy() / view.cellSize - y - 0.5 * scale) * slope.y / scale;
        };

        double detail = 0.0;
        if (stride < scale)
        {
            for (int dy = 0; dy < scale; dy += stride)
            {
                for (int dx = 0; dx < scale; dx += stride)
                    detail = std::max(detail, std::abs(residual(view.at(x + dx, y + dy))));
            }
        }

        for (int side = 0; side < 4; ++side)
        {
            const int gx = x + scale * offsets[side][0];
            const int gy = y + scale * offsets[side][1];
            if (gx < 0 || gy < 0 || gx + scale > view.width || gy + scale > view.height)
                continue;

            double neighbour = 0.0;

            for (int n = 0; n < view.components(); ++n)
            {
                if (k < 0 || k == n)
                    neighbour += plane.mean(n, side + 1) / (k < 0 ? 1.0 : view.mixture->components[n].density);
            }

            detail = std::max(
                detail, 0.25 * std::abs(neighbour - mean - offsets[side][0] * slope.x - offsets[side][1] * slope.y));

            // Preserve all resolved subface variation, including opposing samples
            // whose mean and endpoints alone would hide fine detail.
            double low = 0.0;
            double high = 0.0;

            for (int j = 0; j < scale; ++j)
            {
                const auto c = view.at(gx + (side == 0      ? scale - 1
                                                : side == 1 ? 0
                                                            : j),
                    gy + (side == 2      ? scale - 1
                             : side == 3 ? 0
                                         : j));
                const double r = residual(c);
                if (j == 0)
                    low = high = r;
                else
                {
                    low = std::min(low, r);
                    high = std::max(high, r);
                }
            }

            detail = std::max(detail, 0.5 * (high - low));
        }

        if (k < 0)
            error.density = detail / mean;
        else
            error.ingredient = std::max(error.ingredient, detail);
    }

    auto velocity = [&](int d, double dx, double dy)
    {
        double rho = 0.0;
        for (int k = 0; k < view.components(); ++k)
            rho += plane.value(k, dx, dy);

        return plane.value(view.components() + d, dx, dy) / rho;
    };

    for (int d = 0; d < 2; ++d)
    {
        double rho = 0.0;
        LinearSlope densitySlope;
        for (int k = 0; k < view.components(); ++k)
        {
            rho += plane.mean(k);
            densitySlope.x += plane.slopes[k].x;
            densitySlope.y += plane.slopes[k].y;
        }

        const double mean = plane.mean(view.components() + d) / rho;
        const LinearSlope slope{(plane.slopes[view.components() + d].x - mean * densitySlope.x) / rho,
            (plane.slopes[view.components() + d].y - mean * densitySlope.y) / rho};

        // Neighbour centres lie outside the bounded density plane. Extend the
        // velocity slope there, rather than divide by extrapolated density.
        auto extended = [&](double dx, double dy)
        {
            return mean + dx * slope.x + dy * slope.y;
        };

        auto residual = [&](const Cell& c)
        {
            return c.velocity(d) -
                   extended((c.cx() / view.cellSize - x) / scale - 0.5, (c.cy() / view.cellSize - y) / scale - 0.5);
        };

        double detail = 0.0;

        for (int dy = 0; dy < scale; dy += stride)
        {
            for (int dx = 0; dx < scale; dx += stride)
            {
                const auto c = view.at(x + dx, y + dy);
                detail =
                    std::max(detail, std::abs(c.velocity(d) - velocity(d, (c.cx() / view.cellSize - x) / scale - 0.5,
                                                                  (c.cy() / view.cellSize - y) / scale - 0.5)));

                // MAC faces are independent of cell momentum. Check them too:
                // cancelling face fluctuations must not disappear in a good cell fit.
                for (int side = 0; side < 2; ++side)
                {
                    const double px = c.cx() / view.cellSize + (d == 0 ? (side - 0.5) * c.patch->scale : 0.0);
                    const double py = c.cy() / view.cellSize + (d == 1 ? (side - 0.5) * c.patch->scale : 0.0);
                    detail = std::max(
                        detail, std::abs(c.face(d, side) - velocity(d, (px - x) / scale - 0.5, (py - y) / scale - 0.5)));
                }
            }
        }

        for (int side = 0; side < 4; ++side)
        {
            const int gx = x + scale * offsets[side][0];
            const int gy = y + scale * offsets[side][1];
            if (gx < 0 || gy < 0 || gx + scale > view.width || gy + scale > view.height)
                continue;

            double neighbourDensity = 0.0;
            for (int k = 0; k < view.components(); ++k)
                neighbourDensity += plane.mean(k, side + 1);

            const double neighbour = plane.mean(view.components() + d, side + 1) / neighbourDensity;
            detail = std::max(detail, 0.25 * std::abs(neighbour - extended(offsets[side][0], offsets[side][1])));

            double low = 0.0;
            double high = 0.0;

            for (int j = 0; j < scale; ++j)
            {
                const auto c = view.at(gx + (side == 0      ? scale - 1
                                                : side == 1 ? 0
                                                            : j),
                    gy + (side == 2      ? scale - 1
                             : side == 3 ? 0
                                         : j));
                const double r = residual(c);
                if (j == 0)
                    low = high = r;
                else
                {
                    low = std::min(low, r);
                    high = std::max(high, r);
                }

                if (d == side / 2)
                {
                    const double px = c.cx() / view.cellSize + (d == 0 ? (side == 0 ? 0.5 : -0.5) * c.patch->scale : 0.0);
                    const double py = c.cy() / view.cellSize + (d == 1 ? (side == 2 ? 0.5 : -0.5) * c.patch->scale : 0.0);
                    const double face =
                        c.face(d, side % 2 == 0 ? 1 : 0) - velocity(d, (px - x) / scale - 0.5, (py - y) / scale - 0.5);

                    low = std::min(low, face);
                    high = std::max(high, face);
                }
            }

            detail = std::max(detail, 0.5 * (high - low));
        }

        error.velocity = std::max(error.velocity, detail);
    }
}

Assessment boundaryRefinement(const FluidCellView& grid, int fx, int fy, int scale,
    const FluidCoarseningPreview::Tolerances& tolerance, double quiet_ratio, FluidCellView::MaterialPlane& plane,
    FluidCoarseningPreview::Trace::Cell* observed)
{
    using P = FluidCoarseningPreview;
    const auto c = grid.at(fx, fy);
    if (c.liquid())
        return {P::Refine};

    std::array<Cell, 8> neighbours;

    // Average the two fine neighbours on a coarse face, retaining each sample
    // too: otherwise opposing fine variations could disappear at the boundary.
    for (int j = 0; j < 2; ++j)
    {
        neighbours[j] = grid.at(fx - 1, fy + j * (scale / 2));
        neighbours[2 + j] = grid.at(fx + scale, fy + j * (scale / 2));
        neighbours[4 + j] = grid.at(fx + j * (scale / 2), fy - 1);
        neighbours[6 + j] = grid.at(fx + j * (scale / 2), fy + scale);
    }

    for (const auto& n : neighbours)
    {
        if (bool(n) && n.liquid())
            return {P::Refine};
    }
    for (int dy : {-1, scale})
    {
        for (int dx : {-1, scale})
        {
            if (const auto n = grid.at(fx + dx, fy + dy); bool(n) && n.liquid())
                return {P::Refine};
        }
    }

    P::Error e;
    nativeRepresentationError(grid, fx, fy, scale, e, plane);

    double gradient[4]{};
    bool available[4]{};

    for (int side = 0; side < 4; ++side)
    {
        const auto& a = neighbours[2 * side];
        const auto& z = neighbours[2 * side + 1];
        if (!bool(a) || !bool(z))
            continue;

        const double distance = side < 2 ? 0.5 * (a.cx() + z.cx()) - c.cx() : 0.5 * (a.cy() + z.cy()) - c.cy();
        gradient[side] = (0.5 * (a.pressure() + z.pressure()) - c.pressure()) / distance;
        available[side] = true;
    }

    double slope[2]{};

    for (int axis = 0; axis < 2; ++axis)
    {
        const int a = 2 * axis;
        const int z = a + 1;
        const int count = int(available[a]) + int(available[z]);
        if (count)
            slope[axis] = (gradient[a] + gradient[z]) / count;
        if (count == 2)
            e.pressure_gradient = std::max(e.pressure_gradient, 0.5 * std::abs(gradient[a] - gradient[z]));
    }

    // Check each fine neighbour against the local plane, preserving affine
    // pressure across a 2:1 join while detecting cancelling sub-face patterns.
    for (const auto& n : neighbours)
    {
        if (bool(n))
        {
            const double dx = n.cx() - c.cx();
            const double dy = n.cy() - c.cy();
            const double residual = n.pressure() - c.pressure() - slope[0] * dx - slope[1] * dy;

            e.pressure_gradient = std::max(e.pressure_gradient, std::abs(residual) / (std::abs(dx) + std::abs(dy)));
        }
    }

    const double score = e.score(tolerance);
    if (observed)
    {
        observed->error = e;
        observed->evaluated = true;
    }

    if (score > 1)
        return {P::Refine};

    return {P::Coarse, score <= quiet_ratio};
}

FluidCoarseningPreview::Error fineError(
    const FluidCellView& view, int x, int y, int scale, FluidCellView::MaterialPlane& plane)
{
    FluidCoarseningPreview::Error e;
    double rho = 0.0;
    double mean[2]{};

    double pressureHalf[2][2]{};

    for (int dy = 0; dy < scale; ++dy)
    {
        for (int dx = 0; dx < scale; ++dx)
        {
            const auto c = view.at(x + dx, y + dy);
            rho += c.density();
            for (int d = 0; d < 2; ++d)
                mean[d] += c.momentum(d);
            pressureHalf[0][dx >= scale / 2] += c.pressure();
            pressureHalf[1][dy >= scale / 2] += c.pressure();
        }
    }
    for (auto& v : mean)
        v /= rho;

    if (view.isUniform())
    {
        for (int dy = 0; dy < scale; ++dy)
        {
            for (int dx = 0; dx < scale; ++dx)
            {
                const auto c = view.at(x + dx, y + dy);
                for (int d = 0; d < 2; ++d)
                {
                    e.velocity = std::max(e.velocity, std::abs(c.velocity(d) - mean[d]));
                    for (int side = 0; side < 2; ++side)
                        e.velocity = std::max(e.velocity, std::abs(c.face(d, side) - mean[d]));
                }
            }
        }
    }

    if (view.isUniform())
        flatMaterialError(view, x, y, e);
    else
        nativeRepresentationError(view, x, y, scale, e, plane);

    double gradient[2]{};
    for (int d = 0; d < 2; ++d)
        gradient[d] = 4.0 * (pressureHalf[d][1] - pressureHalf[d][0]) / (scale * scale * scale * view.cellSize);

    for (int d = 0; d < 2; ++d)
    {
        for (int along = -1; along < scale; ++along)
        {
            for (int across = 0; across < scale; ++across)
            {
                const auto a = view.at(x + (d ? across : along), y + (d ? along : across));
                const auto b = view.at(x + (d ? across : along) + !d, y + (d ? along : across) + d);
                if (!a || !b)
                    continue;

                const double distance = d ? b.cy() - a.cy() : b.cx() - a.cx();
                if (distance == 0.0)
                    continue; // Both sample positions may belong to one actual coarse cell.

                const double tangent = d ? b.cx() - a.cx() : b.cy() - a.cy();
                e.pressure_gradient = std::max(e.pressure_gradient,
                    std::abs((b.pressure() - a.pressure() - gradient[1 - d] * tangent) / distance - gradient[d]));
            }
        }
    }

    return e;
}
}

double FluidCoarseningPreview::Error::score(const Tolerances& t) const
{
    // Positive tolerances are supplied by the scene controls.
    return std::max({velocity / t.velocity, pressure_gradient / t.pressure_gradient, density / t.density,
        ingredient / t.ingredient});
}

FluidCoarseningPreview::Error FluidCoarseningPreview::estimate(const FluidCellView& grid, int x, int y, int scale)
{
    FluidCellView::MaterialPlane plane;
    return fineError(grid, x, y, scale, plane);
}

void FluidCoarseningPreview::reset()
{
    assessments.clear();
    observed_scales.clear();
    statistics = {};
    counts.fill(0);
    labels.clear();
}

AmrLayout FluidCoarseningPreview::proposeLayout(int finestWidth, int finestHeight, Trace* trace) const
{
    if (trace)
        trace->fallback_scale = 0;

    auto mark = [&](std::vector<std::uint8_t>& tags, int tile, int x0, int y0, int x1, int y1)
    {
        const int nx = (finestWidth + tile - 1) / tile;
        for (int y = std::max(0, y0) / tile; y <= std::min(finestHeight - 1, y1) / tile; ++y)
        {
            for (int x = std::max(0, x0) / tile; x <= std::min(finestWidth - 1, x1) / tile; ++x)
                tags[y * nx + x] = 1;
        }
    };

    auto requests = [&](const AssessmentLevel& level, Trace::Level* observed, int tile)
    {
        const int nx = (finestWidth + tile - 1) / tile;
        const int ny = (finestHeight + tile - 1) / tile;
        if (observed)
        {
            observed->tile = tile;
            observed->tiles_x = nx;
            observed->tiles_y = ny;
        }

        std::vector<std::uint8_t> tags(size_t(nx) * ny, 0);

        for (int y = 0; y < level.height; ++y)
        {
            for (int x = 0; x < level.width; ++x)
            {
                const int i = y * level.width + x;
                const auto label = level.labels[i];
                const bool needed =
                    level.history[i].coarse ? label == Refine || label == Approaching : label != Candidate;
                if (observed)
                    observed->cells[i].requested = needed;

                if (!needed)
                    continue;

                const auto raw = level.history[i].raw;
                const int buffer =
                    (raw == Detail || raw == LiquidBuffer || raw == Refine || raw == Approaching) ? level.scale : 0;

                // Quiet dwell retains its own cells without seeding fresh padding.
                mark(tags, tile, x * level.scale - buffer, y * level.scale - buffer, (x + 1) * level.scale - 1 + buffer,
                    (y + 1) * level.scale - 1 + buffer);
            }
        }

        if (observed)
            observed->padded = observed->nested = tags;

        return tags;
    };

    std::vector<Requests> masks;

    for (size_t lev = 0; lev < assessments.size(); ++lev)
    {
        const auto& level = assessments[lev];
        masks.push_back(
            {level.scale, requests(level, trace ? &trace->levels[lev] : nullptr, std::max(16, level.scale))});
    }

    return clusterRequests(finestWidth, finestHeight, masks, trace);
}

void FluidCoarseningPreview::update(const FluidCellView& grid, const Tolerances& tolerance, const Stability& stability,
    double time, double lookahead, int maxCellScale, Trace* trace)
{
    if (grid.isUniform())
        maxCellScale = 2;

    const int levelCount = std::countr_zero(static_cast<unsigned>(maxCellScale));
    if (trace)
    {
        *trace = {};
        trace->levels.resize(levelCount);
    }

    // Controls and observation identity determine which history can be reused.
    const bool fresh = assessments.size() != levelCount || observed_grid != grid.identity ||
                       width != (grid.width + 1) / 2 || height != (grid.height + 1) / 2 ||
                       observed_cell_size != grid.cellSize || observed_tolerances != tolerance ||
                       observed_stability != stability || observed_lookahead != lookahead || time < last_time;

    width = (grid.width + 1) / 2;
    height = (grid.height + 1) / 2;
    if (fresh)
        assessments.assign(levelCount, {});

    observed_grid = grid.identity;
    observed_cell_size = grid.cellSize;
    observed_tolerances = tolerance;
    observed_stability = stability;
    observed_lookahead = lookahead;
    last_time = time;

    // Do not pack a group's footprint into a fixed-width integer: larger
    // groups would silently lose changes in their earlier samples.
    std::vector<std::uint8_t> scales(size_t(width) * height);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
            scales[y * width + x] = std::uint8_t(grid.at(2 * x, 2 * y).patch->scale);
    }

    // Assess each candidate resolution against the same committed state.
    for (int lev = 0; lev < levelCount; ++lev)
    {
        auto& level = assessments[lev];
        const int scale = level.scale = 2 << lev;
        level.width = (grid.width + scale - 1) / scale;
        level.height = (grid.height + scale - 1) / scale;
        const int count = level.width * level.height;
        auto* observed = trace ? &trace->levels[lev] : nullptr;
        if (observed)
        {
            observed->scale = scale;
            observed->width = level.width;
            observed->height = level.height;
            observed->cells.resize(count);
        }

        level.labels.assign(count, Unassessed);
        if (fresh)
            level.history.assign(count, {});

        bl::Thread::forEachBatch(count, [&](int first, int last)
        {
            FluidCellView::MaterialPlane plane;
            for (int index = first; index < last; ++index)
            {
                const int fx = scale * (index % level.width);
                const int fy = scale * (index / level.width);
                auto& h = level.history[index];
                if (!fresh)
                {
                    for (int dy = 0; dy < scale && fy + dy < grid.height; dy += 2)
                    {
                        for (int dx = 0; dx < scale && fx + dx < grid.width; dx += 2)
                        {
                            const int i = ((fy + dy) / 2) * width + (fx + dx) / 2;
                            if (scales[i] != observed_scales[i])
                                h = {};
                        }
                    }
                }

                const int actual = grid.at(fx, fy).patch->scale;
                auto* sample = observed ? &observed->cells[index] : nullptr;
                if (sample)
                    sample->observed_scale = actual;
                h.quiet = false;
                h.coarse = actual >= scale;

                if (actual > scale)
                {
                    level.labels[index] = Coarse;
                    continue;
                }

                if (h.coarse)
                {
                    const auto a =
                        boundaryRefinement(grid, fx, fy, scale, tolerance, stability.quiet_ratio, plane, sample);
                    level.labels[index] = a.label;
                    h.quiet = a.quiet;
                }
                else if (fx + scale <= grid.width && fy + scale <= grid.height)
                {
                    bool liquid = false;
                    bool complete = true;
                    for (int dy = -1; dy <= scale; ++dy)
                    {
                        for (int dx = -1; dx <= scale; ++dx)
                        {
                            const auto c = grid.at(fx + dx, fy + dy);
                            if (!c)
                                complete &= !grid.isUniform();
                            else
                                liquid |= c.liquid();
                        }
                    }

                    if (liquid)
                        level.labels[index] = LiquidBuffer;
                    else if (complete)
                    {
                        const auto error = fineError(grid, fx, fy, scale, plane);
                        const double score = error.score(tolerance);
                        if (sample)
                        {
                            sample->error = error;
                            sample->evaluated = true;
                        }

                        level.labels[index] = score <= 1 ? Candidate : Detail;
                        h.quiet = score <= stability.quiet_ratio;
                    }
                }
            }
        }, count >= 8192 ? 8 : 1);

        if (observed)
        {
            for (int i = 0; i < count; ++i)
                observed->cells[i].source = level.labels[i];
        }

        if (lookahead > 0.0)
            markApproaching(level, grid, tolerance, lookahead);
        stabilise(level, stability, time, fresh);

        if (observed)
        {
            for (int i = 0; i < count; ++i)
            {
                auto& sample = observed->cells[i];
                const auto& h = level.history[i];
                sample.raw = h.raw;
                sample.decision = h.decision;
                sample.quiet = h.quiet;
                sample.quiet_since = h.quiet_since;
            }
        }
    }

    observed_scales = std::move(scales);

    // Project decisions only for display; physics and history remain per level.
    labels = assessments.front().labels;
    counts.fill(0);
    statistics = {};

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto& label = labels[y * width + x];
            for (int lev = 1; lev < levelCount && (label == Coarse || label == Candidate); ++lev)
                label = assessments[lev].labels[(y >> lev) * assessments[lev].width + (x >> lev)];
        }
    }

    for (auto label : labels)
        ++counts[label];

    for (const auto& level : assessments)
    {
        statistics.raw_changes += level.statistics.raw_changes;
        statistics.decision_changes += level.statistics.decision_changes;
        statistics.waiting += level.statistics.waiting;
        statistics.held += level.statistics.held;
    }

    ++revision;
}

void FluidCoarseningPreview::stabilise(AssessmentLevel& level, const Stability& stability, double time, bool fresh)
{
    level.statistics = {};

    auto refine = [](std::uint8_t label)
    {
        return label == Refine || label == Approaching;
    };

    auto action = [&](std::uint8_t label)
    {
        return label == Candidate ? -1 : refine(label) ? 1 : 0;
    };

    for (size_t i = 0; i < level.labels.size(); ++i)
    {
        auto& h = level.history[i];
        const auto raw = level.labels[i];
        if (!h.quiet || refine(raw))
            h.quiet_since = -1;
        else if (h.quiet_since < 0)
            h.quiet_since = time;

        const bool settled = h.quiet_since >= 0 && time >= h.quiet_since + stability.quiet_seconds;
        auto decision = raw;
        if (raw == Candidate && h.decision != Candidate && !settled)
        {
            decision = Detail;
            ++level.statistics.waiting;
        }
        else if ((raw == Coarse || raw == Unassessed) && refine(h.decision) && !settled)
        {
            decision = h.decision;
            ++level.statistics.held;
        }

        if (!fresh)
        {
            level.statistics.raw_changes += action(raw) != action(h.raw);
            level.statistics.decision_changes += action(decision) != action(h.decision);
        }

        h.raw = raw;
        h.decision = decision;
        level.labels[i] = decision;
    }
}

void FluidCoarseningPreview::markApproaching(
    AssessmentLevel& level, const FluidCellView& grid, const Tolerances& tolerance, double lookahead)
{
    FluidCellView::MaterialPlane plane;
    const int width = level.width;
    const int height = level.height;
    const int scale = level.scale;
    const int stride = width + 1;
    auto& swept = level.swept;

    swept.assign(size_t(stride) * (height + 1), 0);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int fx = scale * x;
            const int fy = scale * y;
            const int index = y * width + x;
            const auto label = level.labels[index];
            bool source = label == Detail || label == LiquidBuffer || label == Refine;
            if (!source && label == Unassessed && !level.history[index].coarse && fx + scale <= grid.width &&
                fy + scale <= grid.height)
            {
                source = fineError(grid, fx, fy, scale, plane).score(tolerance) > 1;
            }

            if (!source)
                continue;

            double low[2]{};
            double high[2]{};

            for (int dy = 0; dy < scale; ++dy)
            {
                for (int dx = 0; dx < scale; ++dx)
                {
                    const auto c = grid.at(fx + dx, fy + dy);
                    if (!c)
                        continue;

                    for (int d = 0; d < 2; ++d)
                    {
                        for (int side = 0; side < 2; ++side)
                        {
                            const double v = c.face(d, side);
                            const double accelerated = v + grid.acceleration[d] * lookahead;
                            low[d] = std::min({low[d], v, accelerated});
                            high[d] = std::max({high[d], v, accelerated});
                        }
                    }
                }
            }

            const double travel = lookahead / grid.cellSize;
            const int x0 = int(std::floor(std::clamp((fx + low[0] * travel) / scale, 0.0, double(width))));
            const int x1 = int(std::ceil(std::clamp((fx + scale + high[0] * travel) / scale, 0.0, double(width))));
            const int y0 = int(std::floor(std::clamp((fy + low[1] * travel) / scale, 0.0, double(height))));
            const int y1 = int(std::ceil(std::clamp((fy + scale + high[1] * travel) / scale, 0.0, double(height))));

            ++swept[y0 * stride + x0];
            --swept[y0 * stride + x1];
            --swept[y1 * stride + x0];
            ++swept[y1 * stride + x1];
        }
    }

    for (int y = 0; y < height; ++y)
    {
        int row = 0;
        for (int x = 0; x < width; ++x)
        {
            row += swept[y * stride + x];
            const int coverage = row + (y ? swept[(y - 1) * stride + x] : 0);
            swept[y * stride + x] = coverage;
            const int index = y * width + x;
            if (coverage && level.history[index].coarse && level.labels[index] != Refine)
                level.labels[index] = Approaching;
        }
    }
}
}
