#include "FluidGrid.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace fluid
{

using bl::DVec2;
namespace Thread = bl::Thread;

namespace
{
int batchWorkers(int count)
{
    if (count < 65536) return 1;
    return std::min(8, int(Thread::threadCount()));
}

template<typename Operation>
void forBatches(int count, Operation&& operation, int work_per_item = 1)
{
    const int workers = batchWorkers(count * work_per_item);
    if (workers > 1)
        Thread::forEachBatch(count, std::forward<Operation>(operation), workers);
    else
        operation(0, count);
}

template<typename Operation, typename Merge>
auto reduceBatches(int count, Operation&& operation, Merge&& merge, int work_per_item = 1)
{
    const int workers = batchWorkers(count * work_per_item);
    if (workers == 1) return operation(0, count);
    const auto partials = Thread::forEachBatch(count, std::forward<Operation>(operation), workers);
    auto result = partials.front();
    // Merge in batch order rather than worker completion order.
    for (size_t i = 1; i < partials.size(); ++i) result = merge(result, partials[i]);
    return result;
}

double sumValues(const std::vector<double>& values)
{
    return reduceBatches(int(values.size()), [&](int first, int last) {
        return std::reduce(values.begin() + first, values.begin() + last, 0.0);
    }, std::plus<double>{});
}

double sampleScalar(
    const std::vector<double>& data,
    int width,
    int height,
    double x,
    double y)
{
    // blend the four surrounding values in the supplied grid at fractional array indices
    x = std::clamp(x, 0.0, double(width - 1));
    y = std::clamp(y, 0.0, double(height - 1));

    int x0 = int(x);
    int y0 = int(y);
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);

    double tx = x - x0;
    double ty = y - y0;

    auto at = [&](int x, int y) {
        return data[y * width + x];
    };

    double a = std::lerp(at(x0, y0), at(x1, y0), tx);
    double b = std::lerp(at(x0, y1), at(x1, y1), tx);

    return std::lerp(a, b, ty);
}
}

double liquidArea(
    DVec2 normal,
    double offset,
    DVec2 rect_min,
    DVec2 rect_max)
{
    const double width = rect_max.x - rect_min.x;
    const double height = rect_max.y - rect_min.y;
    const double area = width * height;

    // measure the line from the rectangle's furthest liquid-side corner
    const double min_projection =
        normal.x * (normal.x >= 0.0 ? rect_min.x : rect_max.x) +
        normal.y * (normal.y >= 0.0 ? rect_min.y : rect_max.y);

    const double cut = offset - min_projection;

    // measure how far each rectangle edge extends along the normal
    const double span_x = std::abs(normal.x) * width;
    const double span_y = std::abs(normal.y) * height;
    const double small_span = std::min(span_x, span_y);
    const double large_span = std::max(span_x, span_y);

    if (cut <= 0.0)
        return 0.0;

    if (cut >= small_span + large_span)
        return area;

    // an axis-aligned interface cuts out a rectangular strip
    if (small_span == 0.0)
        return area * cut / large_span;

    // the liquid occupies a triangle near the liquid-side corner
    if (cut < small_span)
        return area * cut * cut / (2.0 * small_span * large_span);

    // after reaching one edge, the liquid region becomes a trapezoid
    if (cut < large_span)
        return area * (cut - 0.5 * small_span) / large_span;

    // near full coverage, subtract the remaining dry triangle
    const double dry_cut = small_span + large_span - cut;
    return area * (1.0 -
        dry_cut * dry_cut / (2.0 * small_span * large_span));
}

double interfaceOffset(DVec2 normal, double fraction)
{
    // find the smallest projection inside the local unit square
    const double min_projection =
        std::min(normal.x, 0.0) +
        std::min(normal.y, 0.0);

    const double span_x = std::abs(normal.x);
    const double span_y = std::abs(normal.y);
    const double small_span = std::min(span_x, span_y);
    const double large_span = std::max(span_x, span_y);

    // liquid fraction where the liquid triangle first becomes a trapezoid
    const double triangle_fraction =
        0.5 * small_span / large_span;

    double cut;

    if (fraction < triangle_fraction)
    {
        // reverse the liquid-triangle area formula
        cut = std::sqrt(
            2.0 * small_span * large_span * fraction);
    }
    else if (fraction <= 1.0 - triangle_fraction)
    {
        // reverse the trapezoid area formula
        cut = large_span * fraction + 0.5 * small_span;
    }
    else
    {
        // position the line using the remaining dry triangle
        cut = small_span + large_span - std::sqrt(
            2.0 * small_span * large_span * (1.0 - fraction));
    }

    return min_projection + cut;
}

FluidGrid::FluidGrid(int w, int h, double cell_size) :
    width(w),
    height(h),
    cell_size(cell_size),
    u((w + 1)* h, 0.0),
    v(w* (h + 1), 0.0),
    next_u((w + 1)* h, 0.0),
    next_v(w* (h + 1), 0.0),
    liquid_fraction(w* h, 0.0),
    next_liquid_fraction(w* h, 0.0),
    density(w* h, 0.0),
    inv_density_u((w + 1)* h, 0.0),
    inv_density_v(w* (h + 1), 0.0),
    inv_pressure_diagonal(w* h, 0.0),
    pressure_multigrid(w, h),
    interface_normals(w* h),
    interface_offsets(w* h, 0.0), 
    split_weights(w* h, 0),
    mass_correction_weights(w* h, 0.0),
    liquid_transfer_u((w + 1)* h, 0.0),
    liquid_transfer_v(w* (h + 1), 0.0),
    mass_transfer_u((w + 1)* h, 0.0),
    mass_transfer_v(w* (h + 1), 0.0),
    momentum_u((w + 1)* h, 0.0),
    momentum_v(w* (h + 1), 0.0),
    component_transfer(std::max(u.size(), v.size()), 0.0)
{
}

void FluidGrid::upload(const State& state)
{
    const size_t cell_count = size_t(width) * height;

    auto assign = [](std::vector<double>& field, const std::vector<double>& values, size_t count)
    {
        if (values.empty()) { field.assign(count, 0.0); return; }
        // imported arrays must match the fixed grid to keep later stencil accesses in bounds
        if (values.size() != count)
            throw std::invalid_argument("FluidGrid upload dimensions do not match the grid");
        field = values;
    };

    assign(liquid_fraction, state.liquid_fraction, size_t(width) * height);
    assign(u, state.u, size_t(width + 1) * height);
    assign(v, state.v, size_t(width) * (height + 1));
    assign(pressure, state.pressure, size_t(width) * height);

    auto assign_components = [&](auto& fields, const auto& values, size_t component_count)
    {
        if (values.empty())
        {
            fields.assign(component_count, std::vector<Scalar>(cell_count, 0.0));
            return;
        }

        // Imported component arrays must match the configured mixture.
        if (values.size() != component_count)
            throw std::invalid_argument("FluidGrid upload component count does not match the mixture");

        fields.resize(component_count);

        for (size_t k = 0; k < component_count; ++k)
            assign(fields[k], values[k], cell_count);
    };

    assign_components(liquid_mass, state.liquid_mass, liquid_mixture.components.size());
    assign_components(gas_mass, state.gas_mass, gas_mixture.components.size());

    liquid_correction_weights.assign(
        liquid_mass.size(), std::vector<Scalar>(cell_count, 0.0));

    gas_correction_weights.assign(
        gas_mass.size(), std::vector<Scalar>(cell_count, 0.0));

    previous_pressure = pressure;
    invalidateCaches();
}

void FluidGrid::commitEdits(detail::CellEditBatch& batch)
{
    batch.prepare(width, height, cell_size * cell_size, liquid_mixture, gas_mixture);
    if (liquid_mass.size() != liquid_mixture.components.size() || gas_mass.size() != gas_mixture.components.size())
        throw std::logic_error("Upload the configured fluid mixtures before committing edits");

    forBatches(int(batch.cells.size()), [&](int first, int last)
    {
        for (int n = first; n < last; ++n)
        {
            const auto& cell = batch.cells[n];
            const size_t i = size_t(cell.y) * width + cell.x;
            const bool liquid = cell.component < liquid_mass.size();
            liquid_fraction[i] = liquid ? 1.0 : 0.0;
            for (auto& mass : liquid_mass) mass[i] = 0.0;
            for (auto& mass : gas_mass) mass[i] = 0.0;
            if (liquid) liquid_mass[cell.component][i] = cell.mass;
            else gas_mass[cell.component - liquid_mass.size()][i] = cell.mass;
        }
    }, int(liquid_mass.size() + gas_mass.size()));
    invalidateCaches();
}

void FluidGrid::invalidateCaches()
{
    density_valid = false;
    pressure_coefficients_valid = false;
    multigrid_valid = false;
}

FluidGrid::State FluidGrid::download() const
{
    return {
        liquid_fraction, u, v, pressure,
        liquid_mass, gas_mass
    };
}

FluidGrid::ProjectionResult FluidGrid::step(
    double dt,
    int max_iterations,
    double divergence_tolerance)
{
    if (dt <= 0.0)
        return {};

    // carry material and momentum using the previous projected velocity
    advect(dt);

    applyGravity(dt);

    // correct the transported velocity after applying forces
    return project(dt, max_iterations, divergence_tolerance);
}

void FluidGrid::advect(double dt)
{
    rebuildMomentum();

    const double max_outflow = reduceBatches(height, [&](int y0, int y1)
    {
        double maximum = 0.0;
        for (int y = y0; y < y1; ++y)
        {
            const double* horizontal = u.data() + y * (width + 1);
            const double* vertical = v.data() + y * width;
            for (int x = 0; x < width; ++x)
            {
                const double outflow = std::max(horizontal[x + 1], 0.0) + std::max(-horizontal[x], 0.0) +
                    std::max(vertical[x + width], 0.0) + std::max(-vertical[x], 0.0);
                maximum = std::max(maximum, outflow);
            }
        }
        return maximum;
    }, [](double a, double b) { return std::max(a, b); }, width);

    // Limit transport so no cell exports more than half its volume per substep.
    const double max_step =
        max_outflow > 0.0 ? 0.5 * cell_size / max_outflow : dt;

    double remaining = dt;

    while (remaining > 0.0)
    {
        const double step = std::min(remaining, max_step);
        const double scale = step / cell_size;

        // keep the same correction weights throughout both directional passes
        freezeComponentCorrections();

        // reconstruct the current interface before horizontal transport
        rebuildInterfaces();
        rebuildTransfersX(step);
        advectComponents(step, true);
        advectMomentumX(step, u);

        forBatches(height, [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            for (int x = 0, i = y * width; x < width; ++x, ++i)
            {
                const int left_face = uIndex(x, y);

                // signed liquid leaving through the right face minus the left face
                const double net_liquid_transfer =
                    liquid_transfer_u[left_face + 1] -
                    liquid_transfer_u[left_face];

                // total fluid leaving minus entering during this directional pass
                const double net_fluid_transfer = scale * (
                    u[uIndex(x + 1, y)] -
                    u[uIndex(x, y)]);

                // account for the temporary expansion or compression of an X-only pass
                next_liquid_fraction[i] =
                    liquid_fraction[i] - net_liquid_transfer +
                    split_weights[i] * net_fluid_transfer;
            }
        }, width);

        liquid_fraction.swap(next_liquid_fraction);
        recoverVelocity();

        // horizontal transport changed the fractions, so reconstruct before vertical transport
        rebuildInterfaces();
        rebuildTransfersY(step);
        advectComponents(step, false);
        advectMomentumY(step, v);

        forBatches(height, [&](int y0, int y1)
        {
            for (int y = y0; y < y1; ++y)
            for (int x = 0, i = y * width; x < width; ++x, ++i)
            {
                // signed liquid leaving through the bottom face minus the top face
                const double net_liquid_transfer =
                    liquid_transfer_v[i + width] -
                    liquid_transfer_v[i];

                const double net_fluid_transfer = scale * (
                    v[i + width] -
                    v[i]);

                // reuse the weights chosen before either pass changed the fractions
                next_liquid_fraction[i] =
                    liquid_fraction[i] - net_liquid_transfer +
                    split_weights[i] * net_fluid_transfer;
            }
        }, width);

        liquid_fraction.swap(next_liquid_fraction);
        recoverVelocity();

        remaining -= step;
    }

    u.swap(next_u);
    v.swap(next_v);
}

DVec2 FluidGrid::sampleVelocity(DVec2 pos) const
{
    double x = (pos.x - origin.x) / cell_size;
    double y = (pos.y - origin.y) / cell_size;

    return {
        sampleScalar(u, width + 1, height, x, y - 0.5),
        sampleScalar(v, width, height + 1, x - 0.5, y)
    };
}

DVec2 FluidGrid::interfaceNormal(int x, int y) const
{
    // use the nearest boundary cell when a neighbour lies outside the grid
    const int left_x = std::max(x - 1, 0), right_x = std::min(x + 1, width - 1);
    const double* above = liquid_fraction.data() + std::max(y - 1, 0) * width;
    const double* row = liquid_fraction.data() + y * width;
    const double* below = liquid_fraction.data() + std::min(y + 1, height - 1) * width;
    const double top_left = above[left_x], top = above[x], top_right = above[right_x];
    const double left = row[left_x], right = row[right_x];
    const double bottom_left = below[left_x], bottom = below[x], bottom_right = below[right_x];

    // Column/row totals measure surface height rather than one cell's abrupt fraction jump.
    const double left_sum = top_left + left + bottom_left;
    const double right_sum = top_right + right + bottom_right;
    const double top_sum = top_left + top + top_right;
    const double bottom_sum = bottom_left + bottom + bottom_right;
    DVec2 centered{0.5 * (left_sum - right_sum), 0.5 * (top_sum - bottom_sum)};

    // Youngs' weighted gradient supplies the alternative estimate near corners and thin features.
    const DVec2 youngs{
        left_sum + left - right_sum - right,
        top_sum + top - bottom_sum - bottom
    };

    // A symmetric neighbourhood (such as an isolated cell) has no unique interface direction.
    if (centered.x == 0.0 && centered.y == 0.0)
        return youngs.x == 0.0 && youngs.y == 0.0 ? DVec2{1.0, 0.0} : youngs;

    // MYC chooses the dominant height direction, then the steeper of its two slope estimates.
    if (std::abs(centered.x) <= std::abs(centered.y))
    {
        centered.y = std::copysign(1.0, centered.y);
        if (std::abs(youngs.x) > std::abs(centered.x * youngs.y))
            return youngs;
    }
    else
    {
        centered.x = std::copysign(1.0, centered.x);
        if (std::abs(youngs.y) > std::abs(centered.y * youngs.x))
            return youngs;
    }
    return centered;
}

void FluidGrid::rebuildInterfaces()
{
    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, i = y * width; x < width; ++x, ++i)
        {
            const double fraction = liquid_fraction[i];
            if (fraction <= 0.0 || fraction >= 1.0) continue;
            const DVec2 normal = interfaceNormal(x, y);
            interface_normals[i] = normal;
            interface_offsets[i] = interfaceOffset(normal, fraction);
        }
    }, width);
}

double FluidGrid::componentMassTransfer(
    double volume_transfer,
    double donor_fraction,
    double donor_mass)
{
    // A cell containing none of this mixture cannot supply its ingredients.
    if (donor_fraction <= 0.0)
        return 0.0;

    // Transfer the same proportion of each ingredient as of its mixture.
    return donor_mass * (volume_transfer / donor_fraction);
}

double FluidGrid::momentumTransfer(
    double mass_transfer,
    double velocity_before,
    double velocity_after)
{
    // the transferred mass carries the donor's velocity
    const double donor_velocity =
        mass_transfer >= 0.0 ? velocity_before : velocity_after;

    return mass_transfer * donor_velocity;
}

void FluidGrid::advectMomentumX(
    double step,
    const std::vector<double>& flow_u)
{
    const double scale = step / cell_size;

    auto massCorrection = [&](int x, int y)
    {
        const int cell = y * width + x;
        const int left_face = uIndex(x, y);

        // use the density counterpart of the frozen material correction
        const double weight = mass_correction_weights[cell];

        return weight * scale * (
            flow_u[left_face + 1] - flow_u[left_face]);
    };

    // carry horizontal momentum horizontally
    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * (width + 1); x < width + 1; ++x, ++face)
        {
            if (x == 0 || x == width)
            {
                momentum_u[face] = 0.0;
                continue;
            }

            // average transfers onto the boundaries between u samples
            const double left_mass = 0.5 * (
                mass_transfer_u[face - 1] +
                mass_transfer_u[face]);

            const double right_mass = 0.5 * (
                mass_transfer_u[face] +
                mass_transfer_u[face + 1]);

            const double left_momentum = momentumTransfer(
                left_mass, next_u[face - 1], next_u[face]);

            const double right_momentum = momentumTransfer(
                right_mass, next_u[face], next_u[face + 1]);

            // this momentum region spans half of each neighbouring cell
            const double correction = 0.5 * (
                massCorrection(x - 1, y) +
                massCorrection(x, y) );

            momentum_u[face] +=
                left_momentum - right_momentum +
                u[face] * correction;
        }
    }, width + 1);

    // carry vertical momentum horizontally
    forBatches(height + 1, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * width; x < width; ++x, ++face)
        {
            if (face < width || face >= width * height)
            {
                momentum_v[face] = 0.0;
                continue;
            }

            const int above_left = uIndex(x, y - 1);
            const int below_left = uIndex(x, y);

            // this region spans the cells above and below the v sample
            const double left_mass = 0.5 * (
                mass_transfer_u[above_left] +
                mass_transfer_u[below_left]);

            const double right_mass = 0.5 * (
                mass_transfer_u[above_left + 1] +
                mass_transfer_u[below_left + 1]);

            // outer walls have no transfer and no sample beyond them
            const double left_momentum = x > 0
                ? momentumTransfer(left_mass, next_v[face - 1], next_v[face])
                : 0.0;

            const double right_momentum = x + 1 < width
                ? momentumTransfer(right_mass, next_v[face], next_v[face + 1])
                : 0.0;

            const double correction = 0.5 * (
                massCorrection(x, y - 1) +
                massCorrection(x, y));

            momentum_v[face] +=
                left_momentum - right_momentum +
                v[face] * correction;
        }
    }, width);
}

void FluidGrid::advectMomentumY(
    double step,
    const std::vector<double>& flow_v)
{
    const double scale = step / cell_size;
    const int u_stride = width + 1;

    auto massCorrection = [&](int x, int y)
    {
        const int cell = y * width + x;
        const int top_face = vIndex(x, y);

        // reuse the correction weights chosen before either pass
        const double weight = mass_correction_weights[cell];

        return weight * scale * (
            flow_v[top_face + width] - flow_v[top_face]);
    };

    // carry horizontal momentum vertically
    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * (width + 1); x < width + 1; ++x, ++face)
        {
            if (x == 0 || x == width)
            {
                momentum_u[face] = 0.0;
                continue;
            }

            const int left_top = vIndex(x - 1, y);
            const int right_top = vIndex(x, y);

            // this region spans the cells left and right of the u sample
            const double top_mass = 0.5 * (
                mass_transfer_v[left_top] +
                mass_transfer_v[right_top]);

            const double bottom_mass = 0.5 * (
                mass_transfer_v[left_top + width] +
                mass_transfer_v[right_top + width]);

            // use the velocities recovered after the X pass
            const double top_momentum = y > 0
                ? momentumTransfer(
                top_mass, next_u[face - u_stride], next_u[face])
                : 0.0;

            const double bottom_momentum = y + 1 < height
                ? momentumTransfer(
                bottom_mass, next_u[face], next_u[face + u_stride])
                : 0.0;

            const double correction = 0.5 * (
                massCorrection(x - 1, y) +
                massCorrection(x, y));

            momentum_u[face] +=
                top_momentum - bottom_momentum +
                u[face] * correction;
        }
    }, width + 1);

    // carry vertical momentum vertically
    forBatches(height + 1, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * width; x < width; ++x, ++face)
        {
            if (face < width || face >= width * height)
            {
                momentum_v[face] = 0.0;
                continue;
            }

            // average transfers onto the boundaries between v samples
            const double top_mass = 0.5 * (
                mass_transfer_v[face - width] +
                mass_transfer_v[face]);

            const double bottom_mass = 0.5 * (
                mass_transfer_v[face] +
                mass_transfer_v[face + width]);

            const double top_momentum = momentumTransfer(
                top_mass, next_v[face - width], next_v[face]);

            const double bottom_momentum = momentumTransfer(
                bottom_mass, next_v[face], next_v[face + width]);

            const double correction = 0.5 * (
                massCorrection(x, y - 1) +
                massCorrection(x, y));

            momentum_v[face] +=
                top_momentum - bottom_momentum +
                v[face] * correction;
        }
    }, width);
}

void FluidGrid::recoverVelocity()
{
    rebuildDensity();

    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * (width + 1); x < width + 1; ++x, ++face)
        {
            if (x == 0 || x == width)
            {
                next_u[face] = 0.0;
                continue;
            }

            const int cell = y * width + x;
            const double face_density =
                0.5 * (density[cell - 1] + density[cell]);

            // recover velocity using the mass now occupying this region
            next_u[face] = momentum_u[face] / face_density;
        }
    }, width + 1);

    forBatches(height + 1, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * width; x < width; ++x, ++face)
        {
            if (face < width || face >= width * height)
            {
                next_v[face] = 0.0;
                continue;
            }

            const double face_density =
                0.5 * (density[face - width] + density[face]);

            next_v[face] = momentum_v[face] / face_density;
        }
    }, width);
}

double FluidGrid::liquidTransferX(int x, int y, double step) const
{
    if (x == 0 || x == width)
        return 0.0;

    const double speed = u[uIndex(x, y)];
    if (speed == 0.0)
        return 0.0;

    // positive speed draws from the left cell; negative speed draws from the right
    const int donor = y * width + (speed > 0.0 ? x - 1 : x);
    const double fraction = liquid_fraction[donor];

    if (fraction <= 0.0)
        return 0.0;

    // convert distance travelled into a fraction of one cell's width
    const double volume_transfer = speed * (step / cell_size);
    const double strip_width = std::abs(volume_transfer);

    // a full-liquid cell supplies liquid across the entire outgoing strip
    if (fraction >= 1.0)
        return volume_transfer;

    // select the strip beside the donor cell's outgoing face
    const DVec2 rect_min{ (speed > 0.0 ? 1.0 - strip_width : 0.0), 0.0 };
    const DVec2 rect_max{ (speed > 0.0 ? 1.0 : strip_width),       1.0 };

    const double area = liquidArea(
        interface_normals[donor],
        interface_offsets[donor],
        rect_min,
        rect_max);

    // Bound roundoff in tiny slivers so neither phase exports more than the donor contains.
    const double bounded_area = std::clamp(area,
        std::max(0.0, strip_width - (1.0 - fraction)),
        std::min(strip_width, fraction));

    // positive transfer moves liquid right; negative transfer moves it left
    return speed > 0.0 ? bounded_area : -bounded_area;
}

double FluidGrid::liquidTransferY(int x, int y, double step) const
{
    if (y == 0 || y == height)
        return 0.0;

    const double speed = v[vIndex(x, y)];
    if (speed == 0.0)
        return 0.0;

    // positive speed draws from above; negative speed draws from below
    const int donor = (speed > 0.0 ? y - 1 : y) * width + x;
    const double fraction = liquid_fraction[donor];

    if (fraction <= 0.0)
        return 0.0;

    // convert distance travelled into a fraction of one cell's height
    const double volume_transfer = speed * (step / cell_size);
    const double strip_height = std::abs(volume_transfer);

    // a full-liquid cell supplies liquid across the entire outgoing strip
    if (fraction >= 1.0)
        return volume_transfer;

    // select the bottom strip for downward motion or the top strip for upward motion
    const DVec2 rect_min{
        0.0,
        speed > 0.0 ? 1.0 - strip_height : 0.0
    };
    const DVec2 rect_max{
        1.0,
        speed > 0.0 ? 1.0 : strip_height
    };

    const double area = liquidArea(
        interface_normals[donor],
        interface_offsets[donor],
        rect_min,
        rect_max);

    // Bound roundoff in tiny slivers so neither phase exports more than the donor contains.
    const double bounded_area = std::clamp(area,
        std::max(0.0, strip_height - (1.0 - fraction)),
        std::min(strip_height, fraction));

    // positive transfer moves liquid down; negative transfer moves it up
    return speed > 0.0 ? bounded_area : -bounded_area;
}

void FluidGrid::applyGravity(double dt)
{
    // Gravity accelerates both materials equally; skip the two outer wall rows.
    forBatches(width * (height - 1), [&](int first, int last) {
        for (int i = first + width; i < last + width; ++i) v[i] += gravity_y * dt;
    });
}

void FluidGrid::addVortex(DVec2 center, double radius, double strength)
{
    auto vortexVelocity = [&](DVec2 pos) -> DVec2 {
        double dx = pos.x - center.x;
        double dy = pos.y - center.y;

        double r2 = dx * dx + dy * dy;
        double falloff = std::exp(-r2 / (2.0 * radius * radius));

        return {
            -dy * strength * falloff,
             dx * strength * falloff
        };
    };

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x <= width; ++x)
        {
            DVec2 pos{
                origin.x + x * cell_size,
                origin.y + (y + 0.5) * cell_size
            };

            u[uIndex(x, y)] += vortexVelocity(pos).x;
        }
    }

    for (int y = 0; y <= height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            DVec2 pos{
                origin.x + (x + 0.5) * cell_size,
                origin.y + y * cell_size
            };

            v[vIndex(x, y)] += vortexVelocity(pos).y;
        }
    }
}

FluidGrid::ProjectionResult FluidGrid::project(double dt, int max_iterations, double divergence_tolerance)
{
    if (dt <= 0.0 ||
        width <= 0 || height <= 0 ||
        max_iterations < 0)
    {
        return {};
    }

    const double pressure_scale = cell_size * cell_size / dt;

    const size_t count = size_t(width) * height;
    pressure.resize(count);
    previous_pressure.resize(count);
    pressure_rhs.resize(count);
    pressure_residual.resize(count);
    pressure_correction.resize(count);
    pressure_direction.resize(count);
    pressure_product.resize(count);

    rebuildPressureCoefficients();

    // prevent flow through the outer boundaries
    for (int y = 0; y < height; ++y) {
        u[uIndex(0, y)] = 0.0;
        u[uIndex(width, y)] = 0.0;
    }

    for (int x = 0; x < width; ++x) {
        v[vIndex(x, 0)] = 0.0;
        v[vIndex(x, height)] = 0.0;
    }

    // Form the compatible RHS and accumulate its mean in the same traversal.
    const double rhs_sum = reduceBatches(height, [&](int y0, int y1)
    {
        double sum = 0.0;
        for (int y = y0; y < y1; ++y)
        {
            const double* horizontal = u.data() + y * (width + 1);
            const double* vertical = v.data() + y * width;
            double* rhs = pressure_rhs.data() + y * width;
            for (int x = 0; x < width; ++x)
            {
                const double du = horizontal[x + 1] - horizontal[x];
                const double dv = vertical[x + width] - vertical[x];
                rhs[x] = -(du + dv) / cell_size * pressure_scale;
                sum += rhs[x];
            }
        }
        return sum;
    }, std::plus<double>{}, width);
    const double rhs_mean = rhs_sum / double(count);
    forBatches(int(count), [&](int first, int last) {
        for (int i = first; i < last; ++i) pressure_rhs[i] -= rhs_mean;
    });

    ProjectionResult result;
    result.iterations = solvePressure(max_iterations, divergence_tolerance * pressure_scale);
    if (result.iterations < 0) return {};

    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        {
            const double* row = pressure.data() + y * width;
            double* horizontal = u.data() + y * (width + 1);
            const double* horizontal_weight = inv_density_u.data() + y * (width + 1);
            for (int x = 1; x < width; ++x)
                horizontal[x] -= ((row[x] - row[x - 1]) / cell_size) * dt * horizontal_weight[x];
            if (y == 0) continue;
            double* vertical = v.data() + y * width;
            const double* vertical_weight = inv_density_v.data() + y * width;
            for (int x = 0; x < width; ++x)
                vertical[x] -= ((row[x] - row[x - width]) / cell_size) * dt * vertical_weight[x];
        }
    }, width);

    result.max_divergence = maxDivergence();
    result.converged = result.max_divergence <= divergence_tolerance;
    return result;
}

double FluidGrid::maxDivergence() const
{
    return reduceBatches(height, [&](int y0, int y1)
    {
        double maximum = 0.0;
        for (int y = y0; y < y1; ++y)
        {
            const double* horizontal = u.data() + y * (width + 1);
            const double* vertical = v.data() + y * width;
            for (int x = 0; x < width; ++x)
            {
                const double divergence = ((horizontal[x + 1] - horizontal[x]) + (vertical[x + width] - vertical[x])) / cell_size;
                if (!std::isfinite(divergence)) return std::numeric_limits<double>::infinity();
                maximum = std::max(maximum, std::abs(divergence));
            }
        }
        return maximum;
    }, [](double a, double b) { return std::max(a, b); }, width);
}

void FluidGrid::rebuildPressureCoefficients()
{
    if (pressure_coefficients_valid) return;
    rebuildDensity();

    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        {
            const double* row = density.data() + y * width;
            double* horizontal = inv_density_u.data() + y * (width + 1);
            double* vertical = inv_density_v.data() + y * width;
            // Zero only the walls; every interior weight is overwritten below.
            horizontal[0] = horizontal[width] = 0.0;
            for (int x = 1; x < width; ++x)
                horizontal[x] = 2.0 / (row[x - 1] + row[x]);
            if (y == 0) std::fill_n(vertical, width, 0.0);
            else for (int x = 0; x < width; ++x)
                vertical[x] = 2.0 / (row[x - width] + row[x]);
        }
    }, width);
    std::fill_n(inv_density_v.data() + height * width, width, 0.0);

    // The diagonal reads neighbouring rows, after all face weights have been written.
    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        {
            const double* horizontal = inv_density_u.data() + y * (width + 1);
            const double* vertical = inv_density_v.data() + y * width;
            double* diagonal = inv_pressure_diagonal.data() + y * width;
            for (int x = 0; x < width; ++x)
            {
                const double sum = horizontal[x] + horizontal[x + 1] + vertical[x] + vertical[x + width];
                diagonal[x] = sum > 0.0 ? 1.0 / sum : 0.0;
            }
        }
    }, width);
    pressure_coefficients_valid = true;
}

void FluidGrid::rebuildTransfersX(double step)
{
    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * (width + 1); x < width + 1; ++x, ++face)
        {
            liquid_transfer_u[face] = liquidTransferX(x, y, step);
        }
    }, width + 1);
}

void FluidGrid::rebuildTransfersY(double step)
{
    forBatches(height + 1, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * width; x < width; ++x, ++face)
        {
            liquid_transfer_v[face] = liquidTransferY(x, y, step);
        }
    }, width);
}

void FluidGrid::rebuildComponentTransfersX(
    double step, const std::vector<Scalar>& mass, bool is_liquid)
{
    const double scale = step / cell_size;
    const double inverse_cell_volume =
        1.0 / (cell_size * cell_size);

    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * (width + 1); x < width + 1; ++x, ++face)
        {
            // Closed walls cannot transfer any ingredient.
            if (x == 0 || x == width)
            {
                component_transfer[face] = 0.0;
                continue;
            }

            // Positive flow takes material from the left; negative from the right.
            const int donor =
                y * width + (u[face] > 0.0 ? x - 1 : x);

            const double total = u[face] * scale;
            const double liquid = liquid_transfer_u[face];
            const double fraction = liquid_fraction[donor];

            // Gas uses the volume left over after accounting for liquid.
            component_transfer[face] = componentMassTransfer(
                is_liquid ? liquid : total - liquid,
                is_liquid ? fraction : 1.0 - fraction,
                mass[donor]);

            // Momentum uses transferred mass divided by cell volume.
            mass_transfer_u[face] +=
                component_transfer[face] * inverse_cell_volume;
        }
    }, width + 1);
}

void FluidGrid::rebuildComponentTransfersY(
    double step, const std::vector<Scalar>& mass, bool is_liquid)
{
    const double scale = step / cell_size;
    const double inverse_cell_volume =
        1.0 / (cell_size * cell_size);

    forBatches(height + 1, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * width; x < width; ++x, ++face)
        {
            // Closed walls cannot transfer any ingredient.
            if (y == 0 || y == height)
            {
                component_transfer[face] = 0.0;
                continue;
            }

            // A v-face shares its index with the cell below it.
            const int donor =
                v[face] > 0.0 ? face - width : face;

            const double total = v[face] * scale;
            const double liquid = liquid_transfer_v[face];
            const double fraction = liquid_fraction[donor];

            // Gas uses the volume left over after accounting for liquid.
            component_transfer[face] = componentMassTransfer(
                is_liquid ? liquid : total - liquid,
                is_liquid ? fraction : 1.0 - fraction,
                mass[donor]);

            // Momentum uses transferred mass divided by cell volume.
            mass_transfer_v[face] +=
                component_transfer[face] * inverse_cell_volume;
        }
    }, width);
}

void FluidGrid::freezeComponentCorrections()
{
    const double inverse_cell_volume = 1.0 / (cell_size * cell_size);
    forBatches(width * height, [&](int i0, int i1)
    {
        // Small cell blocks keep ingredient traversals contiguous without dispatching each ingredient.
        constexpr int block_size = 128;
        double inverse_fractions[block_size];
        for (int first = i0; first < i1; first += block_size)
        {
            const int last = std::min(first + block_size, i1);
            for (int i = first; i < last; ++i)
            {
                const double fraction = liquid_fraction[i];
                const bool selected = fraction > 0.5;
                split_weights[i] = selected;
                // The selected phase fills at least half the cell.
                inverse_fractions[i - first] = 1.0 / (selected ? fraction : 1.0 - fraction);
                mass_correction_weights[i] = 0.0;
            }
            auto freeze = [&]<bool IsLiquid>(const auto& fields, auto& corrections)
            {
                for (size_t k = 0; k < fields.size(); ++k)
                {
                    const double* mass = fields[k].data();
                    double* weights = corrections[k].data();
                    for (int i = first; i < last; ++i)
                    {
                        const bool selected = bool(split_weights[i]) == IsLiquid;
                        const double weight = selected ? mass[i] * inverse_fractions[i - first] : 0.0;
                        weights[i] = weight;
                        mass_correction_weights[i] += weight * inverse_cell_volume;
                    }
                }
            };
            freeze.operator()<true>(liquid_mass, liquid_correction_weights);
            freeze.operator()<false>(gas_mass, gas_correction_weights);
        }
    });
}

void FluidGrid::advectComponents(double step, bool horizontal)
{
    const double scale = step / cell_size;
    const auto& flow = horizontal ? u : v;
    const int face_offset = horizontal ? 1 : width;

    auto& total_mass_transfer =
        horizontal ? mass_transfer_u : mass_transfer_v;

    std::fill(
        total_mass_transfer.begin(),
        total_mass_transfer.end(),
        0.0);

    auto advect_mixture = [&](auto& fields, const auto& corrections,
        bool is_liquid)
    {
        for (size_t k = 0; k < fields.size(); ++k)
        {
            auto& mass = fields[k];
            const auto& weights = corrections[k];

            // Finish all neighbour reads before changing any cell masses.
            if (horizontal)
                rebuildComponentTransfersX(step, mass, is_liquid);
            else
                rebuildComponentTransfersY(step, mass, is_liquid);

            forBatches(height, [&](int y0, int y1)
            {
                for (int y = y0; y < y1; ++y)
                for (int x = 0, i = y * width; x < width; ++x, ++i)
                {
                    // Before/after means left/right for X, or top/bottom for Y.
                    const int before = horizontal
                        ? uIndex(x, y)
                        : i;

                    const int after = before + face_offset;

                    const double net_fluid_transfer =
                        scale * (flow[after] - flow[before]);

                    // Reuse the coefficient frozen before either direction.
                    mass[i] +=
                        component_transfer[before] -
                        component_transfer[after] +
                        weights[i] * net_fluid_transfer;
                }
            }, width);
        }
    };

    advect_mixture(liquid_mass, liquid_correction_weights, true);
    advect_mixture(gas_mass, gas_correction_weights, false);
    invalidateCaches();
}

void FluidGrid::rebuildDensity()
{
    if (density_valid) return;
    const double inverse_cell_volume =
        1.0 / (cell_size * cell_size);

    forBatches(int(density.size()), [&](int i0, int i1)
    {
        for (int i = i0; i < i1; ++i)
        {
            double total_mass = 0.0;

            for (const auto& mass : liquid_mass)
                total_mass += mass[i];

            for (const auto& mass : gas_mass)
                total_mass += mass[i];

            // Cell density includes every ingredient in both mixtures.
            density[i] = total_mass * inverse_cell_volume;
        }
    });
    density_valid = true;
}

void FluidGrid::rebuildMomentum()
{
    rebuildDensity();

    forBatches(height, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * (width + 1); x < width + 1; ++x, ++face)
        {
            // the left and right walls have zero normal velocity
            if (x == 0 || x == width)
            {
                momentum_u[face] = 0.0;
                next_u[face] = 0.0;
                continue;
            }

            const int cell = y * width + x;

            // associate this velocity with half of each neighbouring cell
            const double face_density =
                0.5 * (density[cell - 1] + density[cell]);

            momentum_u[face] = face_density * u[face];
            next_u[face] = u[face];
        }
    }, width + 1);

    forBatches(height + 1, [&](int y0, int y1)
    {
        for (int y = y0; y < y1; ++y)
        for (int x = 0, face = y * width; x < width; ++x, ++face)
        {
            // the first and last face rows are the top and bottom walls
            if (face < width || face >= width * height)
            {
                momentum_v[face] = 0.0;
                next_v[face] = 0.0;
                continue;
            }

            // a v-face index also identifies the cell immediately below it
            const double face_density =
                0.5 * (density[face - width] + density[face]);

            momentum_v[face] = face_density * v[face];
            next_v[face] = v[face];
        }
    }, width);
}

void FluidGrid::removeMean(std::vector<double>& values)
{
    const double mean = sumValues(values) / double(values.size());
    forBatches(int(values.size()), [&](int first, int last) {
        for (int i = first; i < last; ++i) values[i] -= mean;
    });
}

double FluidGrid::maxAbs(const std::vector<double>& values)
{
    return reduceBatches(int(values.size()), [&](int first, int last) {
        return std::transform_reduce(values.begin() + first, values.begin() + last, 0.0,
            [](double a, double b) { return std::max(a, b); },
            [](double value) { return std::isfinite(value) ? std::abs(value) : std::numeric_limits<double>::infinity(); });
    }, [](double a, double b) { return std::max(a, b); });
}

double FluidGrid::applyPressureOperator(
    const std::vector<double>& input,
    std::vector<double>& output) const
{
    // General stencil: used only for boundaries and very thin grids.
    auto applyCell = [&](int x, int y)
    {
        const int i = y * width + x;
        const double center = input[i];

        double value = 0.0;
        if (x > 0)          value += inv_density_u[uIndex(x, y)]     * (center - input[i - 1]);
        if (x + 1 < width)  value += inv_density_u[uIndex(x + 1, y)] * (center - input[i + 1]);
        if (y > 0)          value += inv_density_v[i]                * (center - input[i - width]);
        if (y + 1 < height) value += inv_density_v[i + width]        * (center - input[i + width]);

        output[i] = value;
        return center * value;
    };

    // These grids have no interior.
    if (width < 3 || height < 3)
    {
        double dot = 0.0;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                dot += applyCell(x, y);
        return dot;
    }

    const int stride = width;
    auto applyRows = [&](int first_row, int last_row)
    {
        double dots[4] = {};
        for (int y = first_row + 1; y < last_row + 1; ++y) {
            const double* row = input.data() + y * stride;
            const double* above = row - stride;
            const double* below = row + stride;
            const double* horizontal = inv_density_u.data() + y * (stride + 1);
            const double* up = inv_density_v.data() + y * stride;
            const double* down = up + stride;
            double* result = output.data() + y * stride;
            auto apply = [&](int x) {
                const double center = row[x];
                result[x] =
                    (horizontal[x] * (center - row[x - 1]) + horizontal[x + 1] * (center - row[x + 1])) +
                    (up[x] * (center - above[x]) + down[x] * (center - below[x]));
                return center * result[x];
            };
            int x = 1;
            for (; x + 3 < stride - 1; x += 4)
            {
                dots[0] += apply(x); dots[1] += apply(x + 1);
                dots[2] += apply(x + 2); dots[3] += apply(x + 3);
            }
            for (; x < stride - 1; ++x) dots[0] += apply(x);
        }
        return (dots[0] + dots[1]) + (dots[2] + dots[3]);
    };
    double dot = reduceBatches(height - 2, applyRows, std::plus<double>{}, width);

    // Top and bottom, including corners.
    for (int x = 0; x < width; ++x)
    {
        dot += applyCell(x, 0);
        dot += applyCell(x, height - 1);
    }

    // Left and right, excluding the already processed corners.
    for (int y = 1; y < height - 1; ++y)
    {
        dot += applyCell(0, y);
        dot += applyCell(width - 1, y);
    }
    return dot;
}

FluidGrid::ResidualStats FluidGrid::updatePressureAndResidual(double alpha)
{
    auto stats = reduceBatches(int(pressure.size()), [&](int first, int last)
    {
        double sums[4] = {}, maxima[4] = {};
        int i = first;
        for (; i + 3 < last; i += 4)
            for (int lane = 0; lane < 4; ++lane)
            {
                pressure[i + lane] += alpha * pressure_direction[i + lane];
                const double value = pressure_residual[i + lane] -= alpha * pressure_product[i + lane];
                sums[lane] += value;
                maxima[lane] = std::max(maxima[lane], std::abs(value));
            }
        ResidualStats part{(sums[0] + sums[1]) + (sums[2] + sums[3]),
            std::max(std::max(maxima[0], maxima[1]), std::max(maxima[2], maxima[3]))};
        for (; i < last; ++i)
        {
            pressure[i] += alpha * pressure_direction[i];
            const double value = pressure_residual[i] -= alpha * pressure_product[i];
            part.sum += value;
            part.maximum = std::max(part.maximum, std::abs(value));
        }
        return part;
    }, [](ResidualStats a, ResidualStats b) { return ResidualStats{a.sum + b.sum, std::max(a.maximum, b.maximum)}; });
    if (!std::isfinite(stats.sum)) stats.maximum = std::numeric_limits<double>::infinity();
    return stats;
}

FluidGrid::ResidualStats FluidGrid::rebuildPressureResidual()
{
    applyPressureOperator(pressure, pressure_product);
    auto stats = reduceBatches(int(pressure.size()), [&](int first, int last)
    {
        double sums[4] = {}, maxima[4] = {};
        int i = first;
        for (; i + 3 < last; i += 4)
            for (int lane = 0; lane < 4; ++lane)
            {
                const double value = pressure_residual[i + lane] = pressure_rhs[i + lane] - pressure_product[i + lane];
                sums[lane] += value;
                maxima[lane] = std::max(maxima[lane], std::abs(value));
            }
        ResidualStats part{(sums[0] + sums[1]) + (sums[2] + sums[3]),
            std::max(std::max(maxima[0], maxima[1]), std::max(maxima[2], maxima[3]))};
        for (; i < last; ++i)
        {
            const double value = pressure_residual[i] = pressure_rhs[i] - pressure_product[i];
            part.sum += value;
            part.maximum = std::max(part.maximum, std::abs(value));
        }
        return part;
    }, [](ResidualStats a, ResidualStats b) { return ResidualStats{a.sum + b.sum, std::max(a.maximum, b.maximum)}; });
    if (!std::isfinite(stats.sum)) stats.maximum = std::numeric_limits<double>::infinity();
    return stats;
}

double FluidGrid::applyPreconditioner(const std::vector<double>& residual, std::vector<double>& correction, double residual_mean)
{
    // Project before and after preconditioning to preserve symmetry on mean-zero pressures.
    if (pressure_preconditioner == PressurePreconditioner::Multigrid)
        pressure_multigrid.apply(residual, correction, residual_mean);
    else
        forBatches(int(residual.size()), [&](int first, int last) {
            for (int i = first; i < last; ++i)
                correction[i] = (residual[i] - residual_mean) * inv_pressure_diagonal[i];
        });
    const double mean = sumValues(correction) / double(correction.size());

    // Accumulate residual dot correction while removing the correction's mean.
    return reduceBatches(int(residual.size()), [&](int first, int last)
    {
        double dots[4] = {};
        int i = first;
        for (; i + 3 < last; i += 4)
            for (int lane = 0; lane < 4; ++lane)
            {
                correction[i + lane] -= mean;
                dots[lane] += residual[i + lane] * correction[i + lane];
            }
        double dot = (dots[0] + dots[1]) + (dots[2] + dots[3]);
        for (; i < last; ++i)
        {
            correction[i] -= mean;
            dot += residual[i] * correction[i];
        }
        return dot;
    }, std::plus<double>{});
}

int FluidGrid::solvePressure(int max_iterations, double residual_tolerance)
{
    // A divergence-free field needs no correction, regardless of the previous pressure.
    if (maxAbs(pressure_rhs) <= residual_tolerance) {
        forBatches(int(pressure.size()), [&](int first, int last) {
            std::fill(pressure.begin() + first, pressure.begin() + last, 0.0);
            std::fill(previous_pressure.begin() + first, previous_pressure.begin() + last, 0.0);
        });
        return 0;
    }

    // Predict pressure from the last two solves, then rebuild its residual for the current system.
    forBatches(int(pressure.size()), [&](int first, int last) {
        for (int i = first; i < last; ++i) {
            const double latest = pressure[i];
            pressure[i] = 2.0 * latest - previous_pressure[i];
            previous_pressure[i] = latest;
        }
    });
    removeMean(pressure);
    const auto initial = rebuildPressureResidual();
    if (initial.maximum <= residual_tolerance || max_iterations == 0)
        return 0;

    // Only build the preconditioner if prediction leaves an error that needs solving.
    // Density changes invalidate it; it stays fixed throughout each PCG solve.
    if (pressure_preconditioner == PressurePreconditioner::Multigrid && !multigrid_valid)
    {
        multigrid_valid = pressure_multigrid.rebuild(inv_density_u, inv_density_v);
        if (!multigrid_valid) return -1;
    }

    double residual_mean = initial.sum / double(pressure_residual.size());
    double residual_dot_correction = applyPreconditioner(pressure_residual, pressure_correction, residual_mean);
    forBatches(int(pressure.size()), [&](int first, int last) {
        std::copy(pressure_correction.begin() + first, pressure_correction.begin() + last, pressure_direction.begin() + first);
    });
    int iterations = 0;

    while (iterations < max_iterations && residual_dot_correction > 0.0 && std::isfinite(residual_dot_correction))
    {
        const double denominator = applyPressureOperator(pressure_direction, pressure_product);
        if (denominator <= 0.0 || !std::isfinite(denominator))
            break;

        const double alpha = residual_dot_correction / denominator;
        if (!std::isfinite(alpha))
            break;

        const auto stats = updatePressureAndResidual(alpha);
        residual_mean = stats.sum / double(pressure_residual.size());

        ++iterations;

        bool restart = false;
        if (stats.maximum <= residual_tolerance)
        {
            // verify the true residual before accepting convergence; restart if recurrence drifted
            const auto actual = rebuildPressureResidual();
            if (actual.maximum <= residual_tolerance)
                break;

            residual_mean = actual.sum / double(pressure_residual.size());
            restart = true;
        }

        const double next_dot = applyPreconditioner(pressure_residual, pressure_correction, residual_mean);
        const double beta = restart ? 0.0 : next_dot / residual_dot_correction;

        if (!std::isfinite(beta))
            break;

        forBatches(int(pressure.size()), [&](int first, int last) {
            for (int i = first; i < last; ++i)
                pressure_direction[i] = pressure_correction[i] + beta * pressure_direction[i];
        });

        residual_dot_correction = next_dot;
    }

    removeMean(pressure);
    return iterations;
}

}
