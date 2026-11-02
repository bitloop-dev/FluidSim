#include "ElasticityCPU.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fluid
{
namespace
{
template <class Operation> void rows(int width, int height, Operation&& operation)
{
    if (width * height < 65536)
        operation(0, height);
    else
    {
        bl::Thread::forEachBatch(
            height, std::forward<Operation>(operation), std::min(8, int(bl::Thread::threadCount())));
    }
}
}

ElasticityCPU::ElasticityCPU(int width, int height, double cell_size, double shear_modulus)
    : width(width),
      height(height),
      cell_size(cell_size),
      shear_modulus(shear_modulus),
      history(width * height),
      normal_stress(width * height),
      shear_stress((width + 1) * (height + 1)),
      next_history(width * height),
      source_history(width * height),
      incoming_history(width * height),
      original_share(width * height),
      cell_shear(width * height),
      cell_stiffness(width * height),
      corner_stiffness((width + 1) * (height + 1))
{
}

void ElasticityCPU::upload(const FluidGridState<double>& state)
{
    const size_t count = history.size();
    const bool logs = !state.elastic_log_xx.empty() || !state.elastic_log_xy.empty() || !state.elastic_log_yy.empty();
    if (logs && (state.elastic_log_xx.size() != count || state.elastic_log_xy.size() != count ||
                    state.elastic_log_yy.size() != count))
    {
        throw std::invalid_argument("Elastic strain dimensions do not match the grid");
    }
    if (!state.liquid_fraction.empty() && state.liquid_fraction.size() != count)
        throw std::invalid_argument("Elastic fraction dimensions do not match the grid");

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int i = y * width + x;
            history[i] = {};
            if (state.liquid_fraction.empty() || state.liquid_fraction[i] <= 0.0)
                continue;

            double difference = 0.0;
            double shear = 0.0;
            double mean = 0.0;
            if (logs)
            {
                difference = 0.5 * (state.elastic_log_xx[i] - state.elastic_log_yy[i]);
                shear = state.elastic_log_xy[i];
                mean = 0.5 * (state.elastic_log_xx[i] + state.elastic_log_yy[i]);
            }

            // Imported history is external state; invalid values cannot be transported meaningfully.
            if (!std::isfinite(difference) || !std::isfinite(shear) || !std::isfinite(mean))
                throw std::invalid_argument("Elastic history must be finite");

            history[i] = {difference, shear, mean};
        }
    }
}

void ElasticityCPU::download(FluidGridState<double>& state) const
{
    state.elastic_log_xx.resize(history.size());
    state.elastic_log_xy.resize(history.size());
    state.elastic_log_yy.resize(history.size());
    for (size_t i = 0; i < history.size(); ++i)
    {
        const auto& strain = history[i];
        const bool occupied = state.liquid_fraction[i] > 0.0;
        state.elastic_log_xx[i] = occupied ? strain[2] + strain[0] : 0.0;
        state.elastic_log_xy[i] = occupied ? strain[1] : 0.0;
        state.elastic_log_yy[i] = occupied ? strain[2] - strain[0] : 0.0;
    }
}

double ElasticityCPU::coverage(int x, int y, std::span<const double> fraction) const
{
    if (x == 0 || y == 0 || x == width || y == height)
        return 0.0;

    const int i = y * width + x;
    const double a = fraction[i];
    const double b = fraction[i - 1];
    const double c = fraction[i - width];
    const double d = fraction[i - width - 1];
    const double minimum = std::min({a, b, c, d});
    if (minimum <= 0.0)
        return 0.0;

    return 4.0 * minimum / (minimum / a + minimum / b + minimum / c + minimum / d);
}

void ElasticityCPU::deform(
    double dt, std::span<const double> fraction, std::span<const double> u, std::span<const double> v)
{
    rows(width, height, [&](int first, int last)
    {
        for (int y = first; y < last; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int i = y * width + x;
                const double f = fraction[i];
                auto& strain = history[i];
                if (f <= 0.0)
                {
                    strain = {};
                    continue;
                }

                const int face = y * (width + 1) + x;
                const double a = 0.5 * (u[face + 1] - u[face] - v[i + width] + v[i]) / cell_size;
                double shear_rate = 0.0;
                double spin = 0.0;
                double weight = 0.0;
                for (int dy = 0; dy < 2; ++dy)
                {
                    for (int dx = 0; dx < 2; ++dx)
                    {
                        const int cx = x + dx;
                        const int cy = y + dy;
                        const double h = coverage(cx, cy, fraction);
                        if (h == 0.0)
                            continue;

                        const int horizontal = cy * (width + 1) + cx;
                        const int vertical = cy * width + cx;
                        const double uy = (u[horizontal] - u[horizontal - width - 1]) / cell_size;
                        const double vx = (v[vertical] - v[vertical - 1]) / cell_size;

                        // Adjoint of harmonic corner stress scatter; spin contributes no elastic work.
                        shear_rate += h * (uy + vx) / (8.0 * f);
                        spin += h * (uy - vx);
                        weight += h;
                    }
                }
                if (weight > 0.0)
                    spin /= 2.0 * weight;

                const double b = shear_rate + spin;
                const double c = shear_rate - spin;
                if (a == 0.0 && b == 0.0 && c == 0.0)
                    continue;

                // Exact exponential of the trace-free 2x2 velocity gradient.
                const double discriminant = a * a + b * c;
                const double z = std::abs(dt) * std::sqrt(std::abs(discriminant));
                double diagonal = 1.0;
                double scale = dt;
                if (z > 1e-4)
                {
                    if (discriminant > 0.0)
                    {
                        diagonal = std::cosh(z);
                        scale *= std::sinh(z) / z;
                    }
                    else
                    {
                        diagonal = std::cos(z);
                        scale *= std::sin(z) / z;
                    }
                }
                else
                {
                    const double q = discriminant * dt * dt;
                    diagonal += 0.5 * q;
                    scale *= 1.0 + q / 6.0;
                }

                // Scale out the large eigenvalue: small material remnants can have large finite stretches.
                const double radius = std::hypot(strain[0], strain[1]);
                const double relative_eigenvalue = std::exp(-2.0 * radius);
                const double mean = 0.5 * (1.0 + relative_eigenvalue);
                const double deviation_scale = radius > 0.0 ? -0.5 * std::expm1(-2.0 * radius) / radius : 1.0;
                const double difference = deviation_scale * strain[0];
                const double shear = deviation_scale * strain[1];
                const double square = scale * scale;
                const double cross = diagonal * scale;
                const double alpha = diagonal * diagonal;

                // Expand the deviator directly to retain small strains without diagonal cancellation.
                const double updated_difference = mean * (2.0 * cross * a + 0.5 * square * (b * b - c * c)) +
                                                  difference * (alpha + square * (a * a - 0.5 * (b * b + c * c))) +
                                                  shear * (cross * (b - c) + square * a * (b + c));
                const double updated_shear = mean * (cross * (b + c) + square * a * (c - b)) +
                                             difference * (cross * (c - b) + square * a * (b + c)) +
                                             shear * (alpha + square * (b * c - a * a));
                const double updated_mean = mean * (alpha + square * (a * a + 0.5 * (b * b + c * c))) +
                                            difference * (2.0 * cross * a + 0.5 * square * (c * c - b * b)) +
                                            shear * (cross * (b + c) + square * a * (b - c));
                const double magnitude = std::hypot(updated_difference, updated_shear);
                double extent = 0.0;
                if (magnitude > 0.0)
                {
                    extent = radius < 1.0 ? std::asinh(std::exp(radius) * magnitude)
                                          : radius + std::log(updated_mean + magnitude);
                }
                const double updated_scale = magnitude > 0.0 ? extent / magnitude : 0.0;

                // Trace-free L preserves the existing logarithmic volume memory, strain[2].
                strain[0] = updated_scale * updated_difference;
                strain[1] = updated_scale * updated_shear;
            }
        }
    });
}

void ElasticityCPU::advect(double dt, bool horizontal, std::span<const double> velocity,
    std::span<const double> fraction, std::span<const double> transfers, std::span<const unsigned char> split_weights)
{
    rows(width, height, [&](int first, int last)
    {
        for (int y = first; y < last; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int i = y * width + x;
                const int face = horizontal ? y * (width + 1) + x : i;
                const int stride = horizontal ? 1 : width;
                const double left = transfers[face];
                const double right = transfers[face + stride];
                const double in_left = std::max(left, 0.0);
                const double in_right = std::max(-right, 0.0);
                const double outflow = std::max(-left, 0.0) + std::max(right, 0.0);
                const double entering = in_left + in_right;
                const double dilation = (velocity[face + stride] - velocity[face]) * dt / cell_size;
                auto inflow = [&](int component, double fallback)
                {
                    if (in_left == 0.0 && in_right == 0.0)
                        return fallback;
                    if (in_left == 0.0)
                        return history[i + stride][component];

                    const double value = history[i - stride][component];
                    if (in_right == 0.0)
                        return value;

                    return value + (in_right / entering) * (history[i + stride][component] - value);
                };

                if (horizontal)
                {
                    // Keep original and entering material separate until the paired WY sources cancel.
                    const double original = std::max(0.0, fraction[i] - outflow + split_weights[i] * dilation);
                    const double share = original + entering > 0.0 ? original / (original + entering) : 1.0;
                    source_history[i] = history[i];
                    original_share[i] = share;
                    for (int component = 0; component < 3; ++component)
                    {
                        const double source = history[i][component];
                        const double carried = inflow(component, source);
                        incoming_history[i][component] = carried;
                        next_history[i][component] = source + (1.0 - share) * (carried - source);
                    }
                }
                else
                {
                    const double remaining = std::max(0.0, fraction[i] - outflow);

                    // The half-cell outflow bound makes both pools positive; max removes exhaustion roundoff.
                    const double original = std::max(0.0, remaining * original_share[i] + split_weights[i] * dilation);
                    const double previous_inflow = remaining * (1.0 - original_share[i]);
                    const double retained = original + previous_inflow;
                    for (int component = 0; component < 3; ++component)
                    {
                        double value = history[i][component];
                        const double source = source_history[i][component];
                        if (retained > 0.0)
                            value = source + (previous_inflow / retained) * (incoming_history[i][component] - source);
                        if (entering > 0.0)
                        {
                            const double carried = inflow(component, value);
                            value = retained == 0.0 ? carried
                                                    : value + (entering / (retained + entering)) * (carried - value);
                        }
                        next_history[i][component] = value;
                    }
                }
            }
        }
    });
    history.swap(next_history);
}

void ElasticityCPU::rebuildStress(std::span<const double> fraction)
{
    std::atomic<bool> invalid = false;
    rows(width, height, [&](int first, int last)
    {
        for (int i = first * width; i < last * width; ++i)
        {
            normal_stress[i] = 0.0;
            cell_shear[i] = 0.0;
            cell_stiffness[i] = 0.0;
            if (fraction[i] <= 0.0)
                continue;

            const auto& strain = history[i];
            const double radius = std::hypot(strain[0], strain[1]);
            const double positive = 0.5 * std::exp(std::log(fraction[i]) + strain[2] + radius);
            const double negative = 0.5 * std::exp(std::log(fraction[i]) + strain[2] - radius);
            const double deviator = radius > 0.0 ? positive * -std::expm1(-2.0 * radius) / radius : 0.0;
            normal_stress[i] = shear_modulus * deviator * strain[0];
            cell_shear[i] = shear_modulus * deviator * strain[1];
            cell_stiffness[i] = 2.0 * shear_modulus * (positive + negative);
            if (!std::isfinite(cell_stiffness[i]) || !std::isfinite(normal_stress[i]) || !std::isfinite(cell_shear[i]))
                invalid.store(true, std::memory_order_relaxed);
        }
    });

    // Throw after all workers join so no worker outlives the captured callback.
    if (invalid.load(std::memory_order_relaxed))
        throw std::runtime_error("Elastic strain produced non-finite stress or stability state");

    rows(width + 1, height + 1, [&](int first, int last)
    {
        for (int y = first; y < last; ++y)
        {
            for (int x = 0; x <= width; ++x)
            {
                const int corner = y * (width + 1) + x;
                shear_stress[corner] = 0.0;
                corner_stiffness[corner] = 0.0;
                const double h = coverage(x, y, fraction);
                if (h == 0.0)
                    continue;

                // Harmonic coverage supplies zero shear traction through inviscid gas.
                for (int dy = 0; dy < 2; ++dy)
                {
                    for (int dx = 0; dx < 2; ++dx)
                    {
                        const int i = (y - dy) * width + x - dx;
                        const double weight = 0.25 * h / fraction[i];
                        shear_stress[corner] += weight * cell_shear[i];
                        corner_stiffness[corner] += weight * cell_stiffness[i];
                    }
                }
            }
        }
    });
}

double ElasticityCPU::stepLimit(
    std::span<const double> inverse_density_u, std::span<const double> inverse_density_v) const
{
    double maximum = 0.0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 1; x < width; ++x)
        {
            const int face = y * (width + 1) + x;
            const int cell = y * width + x;
            const double stiffness = cell_stiffness[cell - 1] + cell_stiffness[cell] + corner_stiffness[face] +
                                     corner_stiffness[face + width + 1];
            maximum = std::max(maximum, stiffness * inverse_density_u[face]);
        }
    }
    for (int y = 1; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int face = y * width + x;
            const int corner = y * (width + 1) + x;
            const double stiffness = cell_stiffness[face - width] + cell_stiffness[face] + corner_stiffness[corner] +
                                     corner_stiffness[corner + 1];
            maximum = std::max(maximum, stiffness * inverse_density_v[face]);
        }
    }

    // Conservative safety factor for the coupled, deformation-first update, including light interface faces.
    return maximum > 0.0 ? 0.4 * cell_size / std::sqrt(maximum) : std::numeric_limits<double>::infinity();
}

void ElasticityCPU::applyForces(double dt, std::span<double> u, std::span<double> v,
    std::span<const double> inverse_density_u, std::span<const double> inverse_density_v) const
{
    const double scale = dt / cell_size;
    rows(width, height, [&](int first, int last)
    {
        for (int y = first; y < last; ++y)
        {
            for (int x = 1; x < width; ++x)
            {
                const int face = y * (width + 1) + x;
                const int cell = y * width + x;
                u[face] += scale * inverse_density_u[face] *
                           (normal_stress[cell] - normal_stress[cell - 1] + shear_stress[face + width + 1] -
                               shear_stress[face]);
            }
            if (y == 0)
                continue;

            for (int x = 0; x < width; ++x)
            {
                const int face = y * width + x;
                const int corner = y * (width + 1) + x;
                v[face] += scale * inverse_density_v[face] *
                           (normal_stress[face - width] - normal_stress[face] + shear_stress[corner + 1] -
                               shear_stress[corner]);
            }
        }
    });
}
}
