#include "FluidGrid.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace fluid
{

using bl::DVec2;
namespace Thread = bl::Thread;

namespace
{
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

double waterArea(
    DVec2 normal,
    double offset,
    DVec2 rect_min,
    DVec2 rect_max)
{
    const double width = rect_max.x - rect_min.x;
    const double height = rect_max.y - rect_min.y;
    const double area = width * height;

    // measure the line from the rectangle's furthest water-side corner
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

    // the water occupies a triangle near the water-side corner
    if (cut < small_span)
        return area * cut * cut / (2.0 * small_span * large_span);

    // after reaching one edge, the water region becomes a trapezoid
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

    // water fraction where the water triangle first becomes a trapezoid
    const double triangle_fraction =
        0.5 * small_span / large_span;

    double cut;

    if (fraction < triangle_fraction)
    {
        // reverse the water-triangle area formula
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
    water_fraction(w* h, 0.0),
    next_water_fraction(w* h, 0.0),
    density(w* h, 0.0),
    inv_density_u((w + 1)* h, 0.0),
    inv_density_v(w* (h + 1), 0.0),
    inv_pressure_diagonal(w* h, 0.0),
    pressure_multigrid(w, h),
    interface_normals(w* h),
    interface_offsets(w* h, 0.0), 
    split_weights(w* h, 0),
    water_transfer_u((w + 1)* h, 0.0),
    water_transfer_v(w* (h + 1), 0.0),
    mass_transfer_u((w + 1)* h, 0.0),
    mass_transfer_v(w* (h + 1), 0.0),
    momentum_u((w + 1)* h, 0.0),
    momentum_v(w* (h + 1), 0.0)
{
}

void FluidGrid::upload(const State& state)
{
    auto assign = [](std::vector<double>& field, const std::vector<double>& values, size_t count)
    {
        if (values.empty()) { field.assign(count, 0.0); return; }
        // imported arrays must match the fixed grid to keep later stencil accesses in bounds
        if (values.size() != count)
            throw std::invalid_argument("FluidGrid upload dimensions do not match the grid");
        field = values;
    };
    assign(water_fraction, state.water_fraction, size_t(width) * height);
    assign(u, state.u, size_t(width + 1) * height);
    assign(v, state.v, size_t(width) * (height + 1));
    assign(pressure, state.pressure, size_t(width) * height);
    previous_pressure = pressure;
}

FluidGrid::State FluidGrid::download() const
{
    return { water_fraction, u, v, pressure };
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
    recoverVelocity();

    double max_outflow = 0.0;

    for (int y = 0; y < height; ++y) 
    {
        for (int x = 0; x < width; ++x)
        {
            const double outflow =
                std::max( u[uIndex(x + 1, y)], 0.0) +
                std::max(-u[uIndex(x, y)],     0.0) +
                std::max( v[vIndex(x, y + 1)], 0.0) +
                std::max(-v[vIndex(x, y)],     0.0);

            max_outflow = std::max(max_outflow, outflow);
        }
    }

    // Limit transport so no cell exports more than half its volume per substep.
    const double max_step =
        max_outflow > 0.0 ? 0.5 * cell_size / max_outflow : dt;

    double remaining = dt;
    int count = width * height;

    while (remaining > 0.0)
    {
        const double step = std::min(remaining, max_step);
        const double scale = step / cell_size;

        // keep the same correction weights throughout both directional passes
        for (int i = 0; i < count; ++i)
            split_weights[i] = water_fraction[i] > 0.5 ? 1 : 0;

        // reconstruct the current interface before horizontal transport
        rebuildInterfaces();
        rebuildTransfersX(step);
        advectMomentumX(step, u);

        Thread::forEachBatch(next_water_fraction, [&](int i0, int i1)
        {
            for (int i = i0; i < i1; ++i)
            {
                const int x = i % width;
                const int y = i / width;

                const int left_face = uIndex(x, y);

                // signed water leaving through the right face minus the left face
                const double net_water_transfer =
                    water_transfer_u[left_face + 1] -
                    water_transfer_u[left_face];

                // total fluid leaving minus entering during this directional pass
                const double net_fluid_transfer = scale * (
                    u[uIndex(x + 1, y)] -
                    u[uIndex(x, y)]);

                // account for the temporary expansion or compression of an X-only pass
                next_water_fraction[i] =
                    water_fraction[i] - net_water_transfer +
                    split_weights[i] * net_fluid_transfer;
            }
        });

        water_fraction.swap(next_water_fraction);
        recoverVelocity();

        // horizontal transport changed the fractions, so reconstruct before vertical transport
        rebuildInterfaces();
        rebuildTransfersY(step);
        advectMomentumY(step, v);

        Thread::forEachBatch(next_water_fraction, [&](int i0, int i1)
        {
            for (int i = i0; i < i1; ++i)
            {
                // signed water leaving through the bottom face minus the top face
                const double net_water_transfer =
                    water_transfer_v[i + width] -
                    water_transfer_v[i];

                const double net_fluid_transfer = scale * (
                    v[i + width] -
                    v[i]);

                // reuse the weights chosen before either pass changed the fractions
                next_water_fraction[i] =
                    water_fraction[i] - net_water_transfer +
                    split_weights[i] * net_fluid_transfer;
            }
        });

        water_fraction.swap(next_water_fraction);
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
    auto fraction = [&](int sx, int sy)
    {
        sx = std::clamp(sx, 0, width - 1);
        sy = std::clamp(sy, 0, height - 1);

        return water_fraction[sy * width + sx];
    };

    // point towards decreasing water fraction: from water towards air
    DVec2 normal{
        fraction(x - 1, y) - fraction(x + 1, y),
        fraction(x, y - 1) - fraction(x, y + 1)
    };

    // equal neighbouring fractions provide no direction, so choose a vertical interface
    if (normal.x == 0.0 && normal.y == 0.0)
        return { 1.0, 0.0 };

    return normal;
}

void FluidGrid::rebuildInterfaces()
{
    for (int i = 0; i < width * height; ++i)
    {
        const double fraction = water_fraction[i];

        // entirely air or entirely water needs no internal dividing line
        if (fraction <= 0.0 || fraction >= 1.0)
            continue;

        const int x = i % width;
        const int y = i / width;

        // neighbouring fractions estimate which way the interface faces
        const DVec2 normal = interfaceNormal(x, y);

        // this cell's fraction determines where the interface sits
        interface_normals[i] = normal;
        interface_offsets[i] = interfaceOffset(normal, fraction);
    }
}

double FluidGrid::massTransfer(double fluid_transfer, double water_transfer) const
{
    const double air_transfer = fluid_transfer - water_transfer;

    // combine the masses carried by the water and air portions
    return 
        water_density * water_transfer +
        air_density * air_transfer;
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
        const double weight =
            split_weights[cell] ? water_density : air_density;

        return weight * scale * (
            flow_u[left_face + 1] - flow_u[left_face]);
    };

    // carry horizontal momentum horizontally
    Thread::forEachBatch(momentum_u, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
        {
            const int x = face % (width + 1);

            if (x == 0 || x == width)
            {
                momentum_u[face] = 0.0;
                continue;
            }

            const int y = face / (width + 1);

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
    });

    // carry vertical momentum horizontally
    Thread::forEachBatch(momentum_v, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
        {
            if (face < width || face >= width * height)
            {
                momentum_v[face] = 0.0;
                continue;
            }

            const int x = face % width;
            const int y = face / width;
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
    });
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
        const double weight =
            split_weights[cell] ? water_density : air_density;

        return weight * scale * (
            flow_v[top_face + width] - flow_v[top_face]);
    };

    // carry horizontal momentum vertically
    Thread::forEachBatch(momentum_u, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
        {
            const int x = face % u_stride;

            if (x == 0 || x == width)
            {
                momentum_u[face] = 0.0;
                continue;
            }

            const int y = face / u_stride;
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
    });

    // carry vertical momentum vertically
    Thread::forEachBatch(momentum_v, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
        {
            if (face < width || face >= width * height)
            {
                momentum_v[face] = 0.0;
                continue;
            }

            const int x = face % width;
            const int y = face / width;

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
    });
}

void FluidGrid::recoverVelocity()
{
    rebuildDensity();

    Thread::forEachBatch(next_u, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
        {
            const int x = face % (width + 1);

            if (x == 0 || x == width)
            {
                next_u[face] = 0.0;
                continue;
            }

            const int y = face / (width + 1);
            const int cell = y * width + x;
            const double face_density =
                0.5 * (density[cell - 1] + density[cell]);

            // recover velocity using the mass now occupying this region
            next_u[face] = momentum_u[face] / face_density;
        }
    });

    Thread::forEachBatch(next_v, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
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
    });
}

double FluidGrid::waterTransferX(int x, int y, double step) const
{
    if (x == 0 || x == width)
        return 0.0;

    const double speed = u[uIndex(x, y)];
    if (speed == 0.0)
        return 0.0;

    // positive speed draws from the left cell; negative speed draws from the right
    const int donor = y * width + (speed > 0.0 ? x - 1 : x);
    const double fraction = water_fraction[donor];

    if (fraction <= 0.0)
        return 0.0;

    // convert distance travelled into a fraction of one cell's width
    const double strip_width = std::abs(speed) * step / cell_size;

    // a full-water cell supplies water across the entire outgoing strip
    if (fraction >= 1.0)
        return speed * step / cell_size;

    // select the strip beside the donor cell's outgoing face
    const DVec2 rect_min{ (speed > 0.0 ? 1.0 - strip_width : 0.0), 0.0 };
    const DVec2 rect_max{ (speed > 0.0 ? 1.0 : strip_width),       1.0 };

    const double area = waterArea(
        interface_normals[donor],
        interface_offsets[donor],
        rect_min,
        rect_max);

    // positive transfer moves water right; negative transfer moves it left
    return speed > 0.0 ? area : -area;
}

double FluidGrid::waterTransferY(int x, int y, double step) const
{
    if (y == 0 || y == height)
        return 0.0;

    const double speed = v[vIndex(x, y)];
    if (speed == 0.0)
        return 0.0;

    // positive speed draws from above; negative speed draws from below
    const int donor = (speed > 0.0 ? y - 1 : y) * width + x;
    const double fraction = water_fraction[donor];

    if (fraction <= 0.0)
        return 0.0;

    // convert distance travelled into a fraction of one cell's height
    const double strip_height = std::abs(speed) * step / cell_size;

    // a full-water cell supplies water across the entire outgoing strip
    if (fraction >= 1.0)
        return speed * step / cell_size;

    // select the bottom strip for downward motion or the top strip for upward motion
    const DVec2 rect_min{
        0.0,
        speed > 0.0 ? 1.0 - strip_height : 0.0
    };
    const DVec2 rect_max{
        1.0,
        speed > 0.0 ? 1.0 : strip_height
    };

    const double area = waterArea(
        interface_normals[donor],
        interface_offsets[donor],
        rect_min,
        rect_max);

    // positive transfer moves water down; negative transfer moves it up
    return speed > 0.0 ? area : -area;
}

void FluidGrid::applyGravity(double dt)
{
    // Gravity accelerates both materials equally; skip the two outer wall rows.
    for (int i = width; i < width * height; ++i)
        v[i] += gravity_y * dt;
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

    auto index = [&](int x, int y) {
        return y * width + x;
    };

    const size_t count = size_t(width) * height;
    pressure.resize(count);
    previous_pressure.resize(count);
    pressure_rhs.resize(count);
    pressure_residual.resize(count);
    pressure_correction.resize(count);
    pressure_direction.resize(count);
    pressure_product.resize(count);

    rebuildPressureCoefficients();

    // The hierarchy follows the current density field, but stays fixed during this PCG solve.
    if (pressure_preconditioner == PressurePreconditioner::Multigrid && !pressure_multigrid.rebuild(inv_density_u, inv_density_v))
        return {};

    // prevent flow through the outer boundaries
    for (int y = 0; y < height; ++y) {
        u[uIndex(0, y)] = 0.0;
        u[uIndex(width, y)] = 0.0;
    }

    for (int x = 0; x < width; ++x) {
        v[vIndex(x, 0)] = 0.0;
        v[vIndex(x, height)] = 0.0;
    }

    // solve the positive pressure operator against negative, scaled divergence
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = index(x, y);

            double du = u[uIndex(x + 1, y)] - u[uIndex(x, y)];
            double dv = v[vIndex(x, y + 1)] - v[vIndex(x, y)];

            pressure_rhs[i] = -(du + dv) / cell_size * pressure_scale;
        }
    }

    // closed walls require a compatible right-hand side; pressure is defined up to a constant
    removeMean(pressure_rhs);
    ProjectionResult result;
    result.iterations = solvePressure(max_iterations, divergence_tolerance * pressure_scale);

    // correct horizontal velocities
    for (int y = 0; y < height; ++y) {
        for (int x = 1; x < width; ++x) {
            // if the cell on the right has higher pressure than the cell on the left,
            // increases velocity across that boundary towards the left
            double gradient = (
                pressure[index(x, y)] -
                pressure[index(x - 1, y)]
            ) / cell_size;

            const int face = uIndex(x, y);
            u[face] -= gradient * dt * inv_density_u[face];
        }
    }

    // correct vertical velocities
    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // if the cell on the bottom has higher pressure than the cell on the top,
            // increases velocity across that boundary towards the top
            double gradient = (
                pressure[index(x, y)] -
                pressure[index(x, y - 1)]
                ) / cell_size;

            const int face = vIndex(x, y);
            v[face] -= gradient * dt * inv_density_v[face];
        }
    }

    result.max_divergence = maxDivergence();
    result.converged = result.max_divergence <= divergence_tolerance;
    return result;
}

double FluidGrid::maxDivergence() const
{
    double result = 0.0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double du = u[uIndex(x + 1, y)] - u[uIndex(x, y)];
            double dv = v[vIndex(x, y + 1)] - v[vIndex(x, y)];

            double divergence = (du + dv) / cell_size;

            if (!std::isfinite(divergence))
                return std::numeric_limits<double>::infinity();

            result = std::max(result, std::abs(divergence));
        }
    }

    return result;
}

void FluidGrid::rebuildPressureCoefficients()
{
    // Convert the material fraction into a cell-averaged density.
    for (size_t i = 0; i < water_fraction.size(); ++i)
    {
        const double fraction = std::clamp(water_fraction[i], 0.0, 1.0);
        density[i] =
            fraction * water_density +
            (1.0 - fraction) * air_density;
    }

    // Zero wall weights exclude pressure corrections through the tank boundary.
    std::fill(inv_density_u.begin(), inv_density_u.end(), 0.0);
    std::fill(inv_density_v.begin(), inv_density_v.end(), 0.0);

    // Each shared face uses the reciprocal of the neighboring cells' average density.
    for (int y = 0; y < height; ++y)
    {
        for (int x = 1; x < width; ++x)
        {
            const int i = y * width + x;
            inv_density_u[uIndex(x, y)] = 2.0 / (density[i - 1] + density[i]);
        }
    }
    for (int y = 1; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int i = y * width + x;
            inv_density_v[vIndex(x, y)] = 2.0 / (density[i - width] + density[i]);
        }
    }

    // The matrix diagonal is the sum of the four shared face weights.
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const double diagonal =
                inv_density_u[uIndex(x, y)] +
                inv_density_u[uIndex(x + 1, y)] +
                inv_density_v[vIndex(x, y)] +
                inv_density_v[vIndex(x, y + 1)];
            inv_pressure_diagonal[y * width + x] = diagonal > 0.0 ? 1.0 / diagonal : 0.0;
        }
    }
}

void FluidGrid::rebuildTransfersX(double step)
{
    const double scale = step / cell_size;

    Thread::forEachBatch(water_transfer_u, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
        {
            const int x = face % (width + 1);
            const int y = face / (width + 1);

            // closed walls transfer neither water nor air
            if (x == 0 || x == width)
            {
                water_transfer_u[face] = 0.0;
                mass_transfer_u[face] = 0.0;
                continue;
            }

            // geometry determines the water portion of the transferred fluid
            const double water = waterTransferX(x, y, step);
            const double fluid = u[face] * scale;

            water_transfer_u[face] = water;
            mass_transfer_u[face] = massTransfer(fluid, water);
        }
    });
}

void FluidGrid::rebuildTransfersY(double step)
{
    const double scale = step / cell_size;

    Thread::forEachBatch(water_transfer_v, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
        {
            const int x = face % width;
            const int y = face / width;

            // the top and bottom walls transfer neither water nor air
            if (y == 0 || y == height)
            {
                water_transfer_v[face] = 0.0;
                mass_transfer_v[face] = 0.0;
                continue;
            }

            // geometry determines the water portion of the transferred fluid
            const double water = waterTransferY(x, y, step);
            const double fluid = v[face] * scale;

            water_transfer_v[face] = water;
            mass_transfer_v[face] = massTransfer(fluid, water);
        }
    });
}

void FluidGrid::rebuildDensity()
{
    for (size_t i = 0; i < water_fraction.size(); ++i)
    {
        const double fraction = std::clamp(water_fraction[i], 0.0, 1.0);

        // combine the densities according to the cell's material proportions
        density[i] =
            fraction * water_density +
            (1.0 - fraction) * air_density;
    }
}

void FluidGrid::rebuildMomentum()
{
    rebuildDensity();

    Thread::forEachBatch(momentum_u, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
        {
            const int x = face % (width + 1);

            // the left and right walls have zero normal velocity
            if (x == 0 || x == width)
            {
                momentum_u[face] = 0.0;
                continue;
            }

            const int y = face / (width + 1);
            const int cell = y * width + x;

            // associate this velocity with half of each neighbouring cell
            const double face_density =
                0.5 * (density[cell - 1] + density[cell]);

            momentum_u[face] = face_density * u[face];
        }
    });

    Thread::forEachBatch(momentum_v, [&](int i0, int i1)
    {
        for (int face = i0; face < i1; ++face)
        {
            // the first and last face rows are the top and bottom walls
            if (face < width || face >= width * height)
            {
                momentum_v[face] = 0.0;
                continue;
            }

            // a v-face index also identifies the cell immediately below it
            const double face_density =
                0.5 * (density[face - width] + density[face]);

            momentum_v[face] = face_density * v[face];
        }
    });
}

void FluidGrid::removeMean(std::vector<double>& values)
{
    const double mean = std::reduce(values.begin(), values.end(), 0.0) / double(values.size());
    for (double& value : values)
        value -= mean;
}

double FluidGrid::maxAbs(const std::vector<double>& values)
{
    // Preserve non-finite failure detection while reducing absolute residuals.
    return std::transform_reduce(values.begin(), values.end(), 0.0,
        [](double a, double b) { return std::max(a, b); },
        [](double value) { return std::isfinite(value) ? std::abs(value) : std::numeric_limits<double>::infinity(); });
}

void FluidGrid::applyPressureOperator(
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
    };

    // These grids have no interior.
    if (width < 3 || height < 3)
    {
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                applyCell(x, y);

        return;
    }

    const int stride = width;
    auto applyRows = [&](int first_row, int last_row)
    {
        for (int y = first_row + 1; y < last_row + 1; ++y) {
            const double* row = input.data() + y * stride;
            const double* above = row - stride;
            const double* below = row + stride;
            const double* horizontal = inv_density_u.data() + y * (stride + 1);
            const double* up = inv_density_v.data() + y * stride;
            const double* down = up + stride;
            double* result = output.data() + y * stride;
            for (int x = 1; x < stride - 1; ++x) {
                const double center = row[x];
                result[x] =
                    (horizontal[x] * (center - row[x - 1]) + horizontal[x + 1] * (center - row[x + 1])) +
                    (up[x] * (center - above[x]) + down[x] * (center - below[x]));
            }
        }
    };
    // Large stencils amortize worker dispatch; small grids stay on the calling thread.
    const int workers = std::min(width * height >= 1048576 ? 8 : 4, int(Thread::threadCount()));
    if (width * height >= 262144 && workers > 1)
        Thread::forEachBatch(height - 2, applyRows, workers);
    else
        applyRows(0, height - 2);

    // Top and bottom, including corners.
    for (int x = 0; x < width; ++x)
    {
        applyCell(x, 0);
        applyCell(x, height - 1);
    }

    // Left and right, excluding the already processed corners.
    for (int y = 1; y < height - 1; ++y)
    {
        applyCell(0, y);
        applyCell(width - 1, y);
    }
}

FluidGrid::ResidualStats FluidGrid::updatePressureAndResidual(double alpha)
{
    double* pressure_values = pressure.data();
    double* residual_values = pressure_residual.data();
    const double* direction = pressure_direction.data();
    const double* product = pressure_product.data();
    const size_t count = pressure.size();
    double sums[4] = {}, maxima[4] = {};

    // Measure the new residual while updating it, with four independent accumulators.
    auto update = [&](size_t i) {
        pressure_values[i] += alpha * direction[i];
        residual_values[i] -= alpha * product[i];
        return residual_values[i];
    };
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        for (size_t lane = 0; lane < 4; ++lane) {
            const double value = update(i + lane);
            sums[lane] += value;
            maxima[lane] = std::max(maxima[lane], std::abs(value));
        }
    }
    ResidualStats stats{(sums[0] + sums[1]) + (sums[2] + sums[3]),
        std::max(std::max(maxima[0], maxima[1]), std::max(maxima[2], maxima[3]))};
    for (; i < count; ++i) {
        const double value = update(i);
        stats.sum += value;
        stats.maximum = std::max(stats.maximum, std::abs(value));
    }
    // Any non-finite residual also makes its sum non-finite.
    if (!std::isfinite(stats.sum)) stats.maximum = std::numeric_limits<double>::infinity();
    return stats;
}

double FluidGrid::applyPreconditioner(const std::vector<double>& residual, std::vector<double>& correction, double residual_mean)
{
    // Project before and after preconditioning to preserve symmetry on mean-zero pressures.
    if (pressure_preconditioner == PressurePreconditioner::Multigrid)
        pressure_multigrid.apply(residual, correction, residual_mean);
    else
        for (size_t i = 0; i < residual.size(); ++i)
            correction[i] = (residual[i] - residual_mean) * inv_pressure_diagonal[i];
    const double mean = std::reduce(correction.begin(), correction.end(), 0.0) / double(correction.size());

    // Accumulate residual dot correction while removing the correction's mean.
    double dots[4] = {};
    size_t i = 0;
    for (; i + 3 < residual.size(); i += 4) {
        for (size_t lane = 0; lane < 4; ++lane) {
            correction[i + lane] -= mean;
            dots[lane] += residual[i + lane] * correction[i + lane];
        }
    }
    double dot = (dots[0] + dots[1]) + (dots[2] + dots[3]);
    for (; i < residual.size(); ++i) {
        correction[i] -= mean;
        dot += residual[i] * correction[i];
    }
    return dot;
}

int FluidGrid::solvePressure(int max_iterations, double residual_tolerance)
{
    // A divergence-free field needs no correction, regardless of the previous pressure.
    if (maxAbs(pressure_rhs) <= residual_tolerance) {
        std::fill(pressure.begin(), pressure.end(), 0.0);
        std::fill(previous_pressure.begin(), previous_pressure.end(), 0.0);
        return 0;
    }

    // Predict pressure from the last two solves, then rebuild its residual for the current system.
    for (size_t i = 0; i < pressure.size(); ++i) {
        const double latest = pressure[i];
        pressure[i] = 2.0 * latest - previous_pressure[i];
        previous_pressure[i] = latest;
    }
    removeMean(pressure);
    applyPressureOperator(pressure, pressure_product);
    for (size_t i = 0; i < pressure.size(); ++i)
        pressure_residual[i] = pressure_rhs[i] - pressure_product[i];

    if (maxAbs(pressure_residual) <= residual_tolerance || max_iterations == 0)
        return 0;

    double residual_mean = std::reduce(pressure_residual.begin(), pressure_residual.end(), 0.0) / double(pressure_residual.size());
    double residual_dot_correction = applyPreconditioner(pressure_residual, pressure_correction, residual_mean);
    pressure_direction = pressure_correction;
    int iterations = 0;

    while (iterations < max_iterations && residual_dot_correction > 0.0 && std::isfinite(residual_dot_correction))
    {
        applyPressureOperator(pressure_direction, pressure_product);

        const double denominator = std::transform_reduce(pressure_direction.begin(), pressure_direction.end(), pressure_product.begin(), 0.0);
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
            applyPressureOperator(pressure, pressure_product);

            for (size_t i = 0; i < pressure.size(); ++i)
                pressure_residual[i] = pressure_rhs[i] - pressure_product[i];
            if (maxAbs(pressure_residual) <= residual_tolerance)
                break;

            residual_mean = std::reduce(pressure_residual.begin(), pressure_residual.end(), 0.0) / double(pressure_residual.size());
            restart = true;
        }

        const double next_dot = applyPreconditioner(pressure_residual, pressure_correction, residual_mean);
        const double beta = restart ? 0.0 : next_dot / residual_dot_correction;

        if (!std::isfinite(beta))
            break;

        for (size_t i = 0; i < pressure.size(); ++i)
            pressure_direction[i] = pressure_correction[i] + beta * pressure_direction[i];

        residual_dot_correction = next_dot;
    }

    removeMean(pressure);
    return iterations;
}

}
