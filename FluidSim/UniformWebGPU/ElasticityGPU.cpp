#include "ElasticityGPU.h"
#include <array>
#include <cmath>
#include <limits>

namespace fluid::webgpu
{
ElasticityGPU::ElasticityGPU(Compute& compute, int width, int height, float cell_size, float shear_modulus)
    : history(compute.field(width, height, 3, "material log strain")),
      normal(compute.field(width, height, 3, "elastic normal stress, stiffness and failure")),
      shear(compute.field(width + 1, height + 1, 3, "elastic corner stress, stiffness and failure")),
      compute(compute),
      next_history(compute.field(width, height, 3)),
      correction(compute.field(width, height, 4)),
      incoming(compute.field(width, height, 4)),
      partials(compute.field(std::min((width * height + 511) / 512, 1024), 1, 4)),
      statistics(compute.field(1, 1, 4))
{
    parameters.grid = {width, height, 0, partials.width};
    parameters.value = {cell_size, 0, 0, shear_modulus};

    const std::string tensor_helpers = R"wgsl(
        fn coverage(p: vec2i) -> f32
        {
            if (any(p <= vec2i(0)) || any(p >= gridSize()))
            {
                return 0.0;
            }
            let f = vec4f(fraction(p).x, fraction(p - vec2i(1, 0)).x,
                fraction(p - vec2i(0, 1)).x, fraction(p - 1).x);
            let minimum = min(min(f.x, f.y), min(f.z, f.w));
            if (minimum <= 0.0)
            {
                return 0.0;
            }
            return 4.0 * minimum / dot(vec4f(minimum) / f, vec4f(1));
        }

        // Weighted deviator (xx, xy) and trace of B, without subtracting its large diagonal means.
        fn tensor(p: vec2i) -> vec3f
        {
            let f = fraction(p).x;
            if (f <= 0.0)
            {
                return vec3f(0);
            }
            let s = history(p).xyz;
            let radius = length(s.xy);
            let positive = 0.5 * exp(log(f) + s.z + radius);
            let negative = 0.5 * exp(log(f) + s.z - radius);
            var scale = 0.0;
            if (radius > 0.0)
            {
                scale = (positive - negative) / radius;
            }
            if (radius < 1e-3)
            {
                scale = positive * (2.0 - 2.0 * radius + (4.0 / 3.0) * radius * radius);
            }
            return vec3f(scale * s.xy, 2.0 * (positive + negative));
        }
    )wgsl";

    compute.build(deformation, "elastic material strain update", 1, {"history", "fraction", "u", "v"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let f = fraction(p).x;
                if (f <= 0.0)
                {
                    write0(p, vec4f(0));
                    return;
                }
                let s = history(p).xyz;
                let a = 0.5 * (u(p + vec2i(1, 0)).x - u(p).x -
                    v(p + vec2i(0, 1)).x + v(p).x) / cellSize();
                var shear_rate = 0.0;
                var spin = 0.0;
                var weight = 0.0;
                for (var y = 0; y < 2; y++)
                {
                    for (var x = 0; x < 2; x++)
                    {
                        let corner = p + vec2i(x, y);
                        let h = coverage(corner);
                        if (h == 0.0)
                        {
                            continue;
                        }
                        let uy = (u(corner).x - u(corner - vec2i(0, 1)).x) / cellSize();
                        let vx = (v(corner).x - v(corner - vec2i(1, 0)).x) / cellSize();
                        shear_rate += h * (uy + vx) / (8.0 * f);
                        spin += h * (uy - vx);
                        weight += h;
                    }
                }
                if (weight > 0.0)
                {
                    spin /= 2.0 * weight;
                }
                let b = shear_rate + spin;
                let c = shear_rate - spin;
                if (a == 0.0 && b == 0.0 && c == 0.0)
                {
                    write0(p, history(p));
                    return;
                }
                let discriminant = a * a + b * c;
                let z = timeStep() * sqrt(abs(discriminant));
                var diagonal = 1.0;
                var scale = timeStep();
                if (z > 1e-4)
                {
                    if (discriminant > 0.0)
                    {
                        diagonal = cosh(z);
                        scale *= sinh(z) / z;
                    }
                    else
                    {
                        diagonal = cos(z);
                        scale *= sin(z) / z;
                    }
                }
                else
                {
                    let q = discriminant * timeStep() * timeStep();
                    diagonal += 0.5 * q;
                    scale *= 1.0 + q / 6.0;
                }

                // Scale out the large eigenvalue before congruence. Tiny material remnants
                // can have large finite stretches even when their weighted stress is modest.
                let radius = length(s.xy);
                let small = exp(-2.0 * radius);
                var direction = vec2f(0);
                if (radius > 0.0)
                {
                    direction = s.xy / radius;
                }
                let mean = 0.5 * (1.0 + small);
                var deviation = 0.5 * (1.0 - small) * direction;
                if (radius < 1e-3)
                {
                    deviation = (1.0 - radius + (2.0 / 3.0) * radius * radius) * s.xy;
                }
                let square = scale * scale;
                let cross = diagonal * scale;
                let alpha = diagonal * diagonal;

                // Expand the deviator directly; subtracting nearly equal diagonals loses small strains.
                let deviator = vec2f(
                    mean * (2.0 * cross * a + 0.5 * square * (b * b - c * c)) +
                        deviation.x * (alpha + square * (a * a - 0.5 * (b * b + c * c))) +
                        deviation.y * (cross * (b - c) + square * a * (b + c)),
                    mean * (cross * (b + c) + square * a * (c - b)) +
                        deviation.x * (cross * (c - b) + square * a * (b + c)) +
                        deviation.y * (alpha + square * (b * c - a * a)));
                let updated_mean = mean * (alpha + square * (a * a + 0.5 * (b * b + c * c))) +
                    deviation.x * (2.0 * cross * a + 0.5 * square * (c * c - b * b)) +
                    deviation.y * (cross * (b + c) + square * a * (b - c));
                let magnitude = length(deviator);
                var updated = vec2f(0);
                if (magnitude > 0.0)
                {
                    var extent = radius + log(updated_mean + magnitude);
                    if (radius < 1.0)
                    {
                        let unscaled = exp(radius) * magnitude;
                        if (unscaled < 1e-3)
                        {
                            extent = unscaled * (1.0 - unscaled * unscaled / 6.0);
                        }
                        else
                        {
                            extent = log(unscaled + sqrt(1.0 + unscaled * unscaled));
                        }
                    }
                    updated = (extent / magnitude) * deviator;
                }

                // Trace-free velocity gradients preserve the existing logarithmic volume memory.
                write0(p, vec4f(updated, s.z, 0));
            }
        )wgsl",
        tensor_helpers);

    const std::string transport_helpers = R"wgsl(
        fn inflow(p: vec2i, axis: vec2i, left: f32, right: f32, fallback: vec3f) -> vec3f
        {
            if (left == 0.0 && right == 0.0)
            {
                return fallback;
            }
            if (left == 0.0)
            {
                return history(p + axis).xyz;
            }
            let a = history(p - axis).xyz;
            if (right == 0.0)
            {
                return a;
            }
            return a + (right / (left + right)) * (history(p + axis).xyz - a);
        }
    )wgsl";

    compute.build(transport_x, "elastic material history X", 3, {"history", "fraction", "velocity", "transfers", "weights"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let axis = vec2i(1, 0);
                let f = fraction(p).x;
                let source = history(p).xyz;
                let left = transfers(p)[u32(params.level.x)];
                let right = transfers(p + axis)[u32(params.level.x)];
                let in_left = max(left, 0.0);
                let in_right = max(-right, 0.0);
                let outflow = max(-left, 0.0) + max(right, 0.0);
                var c = select(0.0, 1.0, f > 0.5);
                if (params.level.x != 0.0)
                {
                    c = weights(p).z;
                }
                let dilation = (velocity(p + axis).x - velocity(p).x) * timeStep() / cellSize();

                // Separate original and incoming material until the paired WY sources cancel.
                let original = max(0.0, f - outflow + c * dilation);
                let entering = in_left + in_right;
                var share = 1.0;
                if (original + entering > 0.0)
                {
                    share = original / (original + entering);
                }
                let carried = inflow(p, axis, in_left, in_right, source);
                write0(p, vec4f(source + (1.0 - share) * (carried - source), 0));
                write1(p, vec4f(source, c));
                write2(p, vec4f(carried, share));
            }
        )wgsl",
        transport_helpers);
    compute.build(transport_y, "elastic material history Y", 1,
        {"history", "fraction", "velocity", "transfers", "correction", "incoming"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let axis = vec2i(0, 1);
                let left = transfers(p)[u32(params.level.x)];
                let right = transfers(p + axis)[u32(params.level.x)];
                let in_left = max(left, 0.0);
                let in_right = max(-right, 0.0);
                let outflow = max(-left, 0.0) + max(right, 0.0);
                let remaining = max(0.0, fraction(p).x - outflow);
                let source = correction(p);
                let earlier = incoming(p);
                let dilation = (velocity(p + axis).x - velocity(p).x) * timeStep() / cellSize();

                // Under the winning-phase outflow bound these are nonnegative material volumes.
                // Roundoff at exhausted donors must not produce unbounded intensive strain.
                let original = max(0.0, remaining * earlier.w + source.w * dilation);
                let previous_inflow = remaining * (1.0 - earlier.w);
                let retained = original + previous_inflow;
                var value = history(p).xyz;
                if (retained > 0.0)
                {
                    value = source.xyz + (previous_inflow / retained) * (earlier.xyz - source.xyz);
                }
                let entering = in_left + in_right;
                if (entering > 0.0)
                {
                    let carried = inflow(p, axis, in_left, in_right, value);
                    if (retained == 0.0)
                    {
                        value = carried;
                    }
                    else
                    {
                        value += (entering / (retained + entering)) * (carried - value);
                    }
                }
                write0(p, vec4f(value, 0));
            }
        )wgsl",
        transport_helpers);

    compute.build(stress, "elastic material stress", 2, {"history", "fraction"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                if (all(p < gridSize()))
                {
                    let t = tensor(p);
                    let stiffness = params.value.w * t.z;
                    let failure = select(0.0, 1.0, !finite(stiffness) || !finite(t.x) || !finite(t.y));
                    write0(p, vec4f(params.value.w * t.x, stiffness, failure, 0));
                }
                var value = vec4f(0);
                let h = coverage(p);
                if (h > 0.0)
                {
                    var t = vec3f(0);
                    for (var y = 0; y < 2; y++)
                    {
                        for (var x = 0; x < 2; x++)
                        {
                            let sample = p - vec2i(x, y);
                            t += (0.25 * h / fraction(sample).x) * tensor(sample);
                        }
                    }
                    let stiffness = params.value.w * t.z;
                    value = vec4f(params.value.w * t.y, stiffness,
                        select(0.0, 1.0, !finite(stiffness) || !finite(t.y)), 0);
                }
                write1(p, value);
            }
        )wgsl",
        tensor_helpers);

    const std::string_view density_helpers = R"wgsl(
        fn rho(p: vec2i) -> f32
        {
            let phases = density(p).xy;
            return phases.x + phases.y;
        }
    )wgsl";

    compute.build(forces, "elastic MAC forces", 2, {"normal", "shear", "density", "u", "v"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let scale = timeStep() / cellSize();
                if (p.y < gridSize().y)
                {
                    var value = 0.0;
                    if (p.x > 0 && p.x < gridSize().x)
                    {
                        let inverse_density = 2.0 / (rho(p) + rho(p - vec2i(1, 0)));
                        let traction = normal(p).x - normal(p - vec2i(1, 0)).x + shear(p + vec2i(0, 1)).x - shear(p).x;
                        value = u(p).x + scale * inverse_density * traction;
                    }
                    write0(p, vec4f(value));
                }
                if (p.x < gridSize().x)
                {
                    var value = 0.0;
                    if (p.y > 0 && p.y < gridSize().y)
                    {
                        let inverse_density = 2.0 / (rho(p) + rho(p - vec2i(0, 1)));
                        let traction = normal(p - vec2i(0, 1)).x - normal(p).x + shear(p + vec2i(1, 0)).x - shear(p).x;
                        value = v(p).x + scale * inverse_density * traction;
                    }
                    write1(p, vec4f(value));
                }
            }
        )wgsl",
        density_helpers);

    const std::string reduction_helpers = R"wgsl(
        var<workgroup> partial: array<vec4f, 128>;

        fn groupMaximum(lane: u32, value: vec4f) -> vec4f
        {
            partial[lane] = value;
            workgroupBarrier();
            for (var stride = 64u; stride > 0u; stride /= 2u)
            {
                if (lane < stride)
                {
                    partial[lane] = max(partial[lane], partial[lane + stride]);
                }
                workgroupBarrier();
            }
            return partial[0];
        }
    )wgsl";

    compute.build(reduce, "elastic stability partials", 1, {"normal", "shear", "density"},
        R"wgsl(
            fn execute()
            {
                let lane = u32(cell().x) % 128u;
                let group = u32(cell().x) / 128u;
                let count = u32(gridSize().x * gridSize().y);
                var total = vec4f(0);
                for (var base = group * 512u; base < count; base += u32(params.grid.w) * 512u)
                {
                    for (var offset = 0u; offset < 512u; offset += 128u)
                    {
                        let i = base + offset + lane;
                        if (i >= count)
                        {
                            continue;
                        }
                        let p = vec2i(i32(i % u32(gridSize().x)), i32(i / u32(gridSize().x)));
                        var rate = 0.0;
                        if (p.x > 0)
                        {
                            rate = (normal(p).y + normal(p - vec2i(1, 0)).y + shear(p).y + shear(p + vec2i(0, 1)).y) *
                                2.0 / (rho(p) + rho(p - vec2i(1, 0)));
                        }
                        if (p.y > 0)
                        {
                            let vertical = (normal(p).y + normal(p - vec2i(0, 1)).y + shear(p).y + shear(p + vec2i(1, 0)).y) *
                                2.0 / (rho(p) + rho(p - vec2i(0, 1)));
                            rate = max(rate, vertical);
                        }
                        var failure = max(normal(p).z, shear(p).z);
                        if (!finite(rate))
                        {
                            failure = 1.0;
                        }
                        total = max(total, vec4f(rate, 0, failure, 0));
                    }
                }
                let result = groupMaximum(lane, total);
                if (lane == 0u)
                {
                    write0(vec2i(i32(group), 0), result);
                }
            }
        )wgsl",
        std::string(density_helpers) + reduction_helpers, true, {128, 1});

    compute.build(finish, "elastic stability finish", 1, {"source"},
        R"wgsl(
            fn execute()
            {
                let lane = u32(cell().x);
                var total = vec4f(0);
                for (var i = lane; i < u32(params.grid.w); i += 128u)
                {
                    total = max(total, source(vec2i(i32(i), 0)));
                }
                let result = groupMaximum(lane, total);
                if (lane == 0u)
                {
                    write0(vec2i(0), result);
                }
            }
        )wgsl",
        reduction_helpers, true, {128, 1});
}

void ElasticityGPU::upload(const FluidGridState<float>& state)
{
    const int width = history.width;
    const int height = history.height;
    const size_t count = size_t(width) * height;
    const bool logs = !state.elastic_log_xx.empty() || !state.elastic_log_xy.empty() || !state.elastic_log_yy.empty();
    if (logs && (state.elastic_log_xx.size() != count || state.elastic_log_xy.size() != count ||
                    state.elastic_log_yy.size() != count))
    {
        throw std::invalid_argument("Elastic strain dimensions do not match the grid");
    }

    const auto& occupancy = state.elastic_fraction.empty() ? state.liquid_fraction : state.elastic_fraction;
    if (!occupancy.empty() && occupancy.size() != count)
        throw std::invalid_argument("Elastic fraction dimensions do not match the grid");

    std::vector<float> packed(3 * count, 0);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int i = y * width + x;
            const double fraction = occupancy.empty() ? 0 : occupancy[i];
            if (fraction <= 0)
                continue;

            double d = 0;
            double shear = 0;
            double trace = 0;
            if (logs)
            {
                d = .5 * (state.elastic_log_xx[i] - state.elastic_log_yy[i]);
                shear = state.elastic_log_xy[i];
                trace = .5 * (state.elastic_log_xx[i] + state.elastic_log_yy[i]);
            }

            // Imported history is external state; invalid values cannot be transported meaningfully.
            if (!std::isfinite(d) || !std::isfinite(shear) || !std::isfinite(trace))
                throw std::invalid_argument("Elastic history must be finite");

            packed[3 * i] = float(d);
            packed[3 * i + 1] = float(shear);
            packed[3 * i + 2] = float(trace);
        }
    }
    compute.write(history, packed);
    stress_valid = false;
    statistics_valid = false;
}

void ElasticityGPU::importReference(const Field& fraction)
{
    const auto coverage = compute.read(fraction);
    const auto displacement = compute.read(history);
    const int width = history.width;
    const int height = history.height;
    std::vector<float> packed(displacement.size(), 0.0f);

    // Version 9 checkpoints could be saved before exterior support was rebuilt.
    // Recreate its four affine extension bands only for this one-time migration.
    std::vector<double> extended(displacement.begin(), displacement.end());
    for (size_t i = 0; i < coverage.size(); ++i)
        extended[3 * i + 2] = coverage[i] > 1e-12f ? 0 : 255;
    auto next = extended;
    for (int band = 1; band <= 4; ++band)
    {
        next = extended;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int i = y * width + x;
                if (extended[3 * i + 2] < 255)
                    continue;

                bool adjacent = false;
                for (int sy = std::max(0, y - 1); sy <= std::min(height - 1, y + 1); ++sy)
                {
                    for (int sx = std::max(0, x - 1); sx <= std::min(width - 1, x + 1); ++sx)
                        adjacent |= extended[3 * (sy * width + sx) + 2] < band;
                }

                if (!adjacent)
                    continue;

                double count = 0;
                double sum_x = 0;
                double sum_y = 0;
                double square_x = 0;
                double square_y = 0;
                double cross = 0;
                std::array<double, 2> anchor{};
                std::array<double, 2> sum{};
                std::array<double, 2> moment_x{};
                std::array<double, 2> moment_y{};
                for (int sy = std::max(0, y - 2); sy <= std::min(height - 1, y + 2); ++sy)
                {
                    for (int sx = std::max(0, x - 2); sx <= std::min(width - 1, x + 2); ++sx)
                    {
                        const int sample = sy * width + sx;
                        if (extended[3 * sample + 2] >= band)
                            continue;

                        const double px = sx - x;
                        const double py = sy - y;
                        for (int channel = 0; channel < 2; ++channel)
                        {
                            if (count == 0)
                                anchor[channel] = extended[3 * sample + channel];

                            const double value = extended[3 * sample + channel] - anchor[channel];
                            sum[channel] += value;
                            moment_x[channel] += px * value;
                            moment_y[channel] += py * value;
                        }
                        ++count;
                        sum_x += px;
                        sum_y += py;
                        square_x += px * px;
                        square_y += py * py;
                        cross += px * py;
                    }
                }
                const double a = count * square_x - sum_x * sum_x;
                const double b = count * cross - sum_x * sum_y;
                const double c = count * square_y - sum_y * sum_y;
                const double determinant = a * c - b * b;
                for (int channel = 0; channel < 2; ++channel)
                {
                    double value = sum[channel] / count;
                    if (determinant > 0)
                    {
                        const double mx = count * moment_x[channel] - sum_x * sum[channel];
                        const double my = count * moment_y[channel] - sum_y * sum[channel];
                        value -= ((c * mx - b * my) * sum_x + (a * my - b * mx) * sum_y) / (count * determinant);
                    }
                    next[3 * i + channel] = anchor[channel] + value;
                }
                next[3 * i + 2] = band;
            }
        }
        extended.swap(next);
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int i = y * width + x;
            if (coverage[i] <= 0)
                continue;

            const int left = std::max(0, x - 1);
            const int right = std::min(width - 1, x + 1);
            const int top = std::max(0, y - 1);
            const int bottom = std::min(height - 1, y + 1);

            // The old cutoff left distant numerical remnants with no defined reference gradient.
            // Report that migration limit instead of inventing relaxed history for their mass.
            if (extended[3 * (y * width + left) + 2] == 255 || extended[3 * (y * width + right) + 2] == 255 ||
                extended[3 * (top * width + x) + 2] == 255 || extended[3 * (bottom * width + x) + 2] == 255)
            {
                throw std::runtime_error(
                    "Legacy elastic checkpoint has material without defined reference history; restart the scene");
            }

            auto dx = [&](int channel)
            {
                return (extended[3 * (y * width + right) + channel] - extended[3 * (y * width + left) + channel]) /
                       ((right - left) * parameters.value[0]);
            };
            auto dy = [&](int channel)
            {
                return (extended[3 * (bottom * width + x) + channel] - extended[3 * (top * width + x) + channel]) /
                       ((bottom - top) * parameters.value[0]);
            };
            const double a = 1 + dx(0);
            const double b = dy(0);
            const double c = dx(1);
            const double e = 1 + dy(1);
            const double determinant = a * e - b * c;
            if (!(determinant > 0) || !std::isfinite(determinant))
                throw std::runtime_error("Cannot import an inverted or non-finite elastic reference map");

            // Unit-determinant part and retained volume memory of B = inverse(A) inverse(A)^T.
            const double xx = (e * e + b * b) / determinant;
            const double xy = -(e * c + a * b) / determinant;
            const double yy = (c * c + a * a) / determinant;
            const double d = .5 * (xx - yy);
            const double magnitude = std::hypot(d, xy);
            const double scale = magnitude > 0 ? std::asinh(magnitude) / magnitude : 1;
            packed[3 * i] = float(d * scale);
            packed[3 * i + 1] = float(xy * scale);
            packed[3 * i + 2] = float(-std::log(determinant));
        }
    }
    compute.write(history, packed);
    stress_valid = false;
    statistics_valid = false;
}

void ElasticityGPU::download(FluidGridState<float>& state)
{
    const auto& occupancy = state.elastic_fraction.empty() ? state.liquid_fraction : state.elastic_fraction;
    const auto packed = compute.read(history);
    const size_t count = packed.size() / 3;
    state.elastic_log_xx.resize(count);
    state.elastic_log_xy.resize(count);
    state.elastic_log_yy.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        const float inverse = occupancy[i] > 0 ? 1.0f : 0.0f;
        state.elastic_log_xx[i] = (packed[3 * i + 2] + packed[3 * i]) * inverse;
        state.elastic_log_xy[i] = packed[3 * i + 1] * inverse;
        state.elastic_log_yy[i] = (packed[3 * i + 2] - packed[3 * i]) * inverse;
    }
}

void ElasticityGPU::deform(double dt, const Field& fraction, const Field& u, const Field& v)
{
    parameters.value[1] = float(dt);
    compute.run(deformation, {&next_history}, {{"history", &history}, {"fraction", &fraction}, {"u", &u}, {"v", &v}},
        parameters);
    history.swap(next_history);
    stress_valid = false;
    statistics_valid = false;
}

void ElasticityGPU::advect(
    double dt, bool horizontal, const Field& velocity, const Field& fraction, const Field& transfers,
    const Field* split_weights)
{
    parameters.grid[2] = horizontal ? 0 : 1;
    parameters.level[0] = split_weights ? 3.0f : 0.0f;
    parameters.value[1] = float(dt);
    if (horizontal)
    {
        compute.run(transport_x, {&next_history, &correction, &incoming},
            {{"history", &history}, {"fraction", &fraction}, {"velocity", &velocity}, {"transfers", &transfers},
                {"weights", split_weights ? split_weights : &fraction}},
            parameters);
    }
    else
    {
        compute.run(transport_y, {&next_history},
            {{"history", &history}, {"fraction", &fraction}, {"velocity", &velocity}, {"transfers", &transfers},
                {"correction", &correction}, {"incoming", &incoming}},
            parameters);
    }

    history.swap(next_history);
    stress_valid = false;
    statistics_valid = false;
}

void ElasticityGPU::rebuildStress(const Field& fraction)
{
    if (stress_valid)
        return;

    compute.run(stress, {&normal, &shear}, {{"history", &history}, {"fraction", &fraction}}, parameters,
        {unsigned(shear.width), unsigned(shear.height), 1});
    stress_valid = true;
    statistics_valid = false;
}

double ElasticityGPU::stepLimit(const Field& density)
{
    if (statistics_valid)
        return cached_limit;

    compute.run(reduce, {&partials}, {{"normal", &normal}, {"shear", &shear}, {"density", &density}}, parameters,
        {unsigned(partials.width) * 128, 1, 1});
    compute.run(finish, {&statistics}, {{"source", &partials}}, parameters, {128, 1, 1});
    const auto values = compute.read(statistics);
    if (values[2] != 0.0f)
        throw std::runtime_error("Elastic strain produced non-finite stress or stability state");

    cached_limit = std::numeric_limits<double>::infinity();
    if (values[0] > 0.0f)
        cached_limit = 0.4 * parameters.value[0] / std::sqrt(double(values[0]));

    // Geometric history transport and its kinematic source share the fluid transport CFL.
    // Complete force/projection steps need the elastic-wave bound, not the gas speed bound.
    statistics_valid = true;
    return cached_limit;
}

void ElasticityGPU::applyForces(double dt, Field& u, Field& v, Field& next_u, Field& next_v, const Field& density)
{
    parameters.value[1] = float(dt);
    compute.run(forces, {&next_u, &next_v},
        {{"normal", &normal}, {"shear", &shear}, {"density", &density}, {"u", &u}, {"v", &v}}, parameters,
        {unsigned(shear.width), unsigned(shear.height), 1});
    u.swap(next_u);
    v.swap(next_v);
    statistics_valid = false;
}

}
