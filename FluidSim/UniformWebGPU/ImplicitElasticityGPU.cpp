#include "ImplicitElasticityGPU.h"
#include <cmath>
#include <limits>

namespace fluid::webgpu
{
ImplicitElasticityGPU::ImplicitElasticityGPU(Compute& compute, int width, int height, float dx, float modulus)
    : compute(compute), multigrid(compute, width, height), reduction(compute, (width + 1) * (height + 1))
{
    parameters.grid = {width, height, 0, 0};
    parameters.value = {dx, 0, 0, modulus};
    auto cell = [&](int channels)
    {
        return compute.field(width, height, channels);
    };
    auto face = [&](int channels)
    {
        return compute.field(width + 1, height + 1, channels);
    };
    material = cell(4);
    corners = face(1);
    diagonal = face(4);
    rounding = face(2);
    coefficients = cell(4);
    stress = cell(4);
    corner_stress = face(2);
    for (Field* field : {&solution, &compensation, &r1, &r2, &next_r, &z, &lanczos, &product, &w1, &w2, &next_w})
        *field = face(3);
    pressure_rhs = cell(1);
    terms = face(4);
    for (Field* field : {&stats, &alpha, &beta})
        *field = compute.field(1, 1, 4);
    control = compute.field(3, 1, 4);
    next_control = compute.field(3, 1, 4);
    next_history = cell(3);

    compute.build(clear, "implicit elastic recurrence reset", 4, {},
        R"wgsl(
            fn execute()
            {
                write0(cell(), vec4f(0));
                write1(cell(), vec4f(0));
                write2(cell(), vec4f(0));
                write3(cell(), vec4f(0));
            }
        )wgsl");

    const std::string geometry = R"wgsl(
        fn ulp(value: f32) -> f32
        {
            let magnitude = abs(value);
            return bitcast<f32>(bitcast<u32>(magnitude) + 1u) - magnitude;
        }

        fn validU(p: vec2i) -> bool
        {
            return p.x > 0 && p.x < gridSize().x && p.y < gridSize().y;
        }

        fn validV(p: vec2i) -> bool
        {
            return p.y > 0 && p.y < gridSize().y && p.x < gridSize().x;
        }

        fn validCell(p: vec2i) -> bool
        {
            return all(p >= vec2i(0)) && all(p < gridSize());
        }

        fn mask(p: vec2i, value: vec3f) -> vec3f
        {
            return vec3f(select(0.0, value.x, validU(p)), select(0.0, value.y, validV(p)),
                select(0.0, value.z, validCell(p)));
        }
    )wgsl";

    compute.build(corner, "implicit elastic coverage", 1, {"fraction"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                var h = 0.0;
                if (all(p > vec2i(0)) && all(p < gridSize()))
                {
                    let f = vec4f(fraction(p).x, fraction(p - vec2i(1, 0)).x,
                        fraction(p - vec2i(0, 1)).x, fraction(p - 1).x);
                    let minimum = min(min(f.x, f.y), min(f.z, f.w));
                    if (minimum > 0.0)
                    {
                        h = 4.0 * minimum / dot(vec4f(minimum) / f, vec4f(1));
                    }
                }
                write0(p, vec4f(h));
            }
        )wgsl");

    // Weighted old deviator q and k = G f tr(B) define q_new = q_rotated + dt k E(v_new).
    compute.build(prepare, "implicit elastic constitutive coefficients", 1,
        {"history", "fraction", "corners", "u", "v"},
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
                var spin = 0.0;
                var weight = 0.0;
                for (var y = 0; y < 2; y++)
                {
                    for (var x = 0; x < 2; x++)
                    {
                        let c = p + vec2i(x, y);
                        let h = corners(c).x;
                        spin += h * ((u(c).x - u(c - vec2i(0, 1)).x) - (v(c).x - v(c - vec2i(1, 0)).x));
                        weight += h;
                    }
                }
                var angle = 0.0;
                if (weight > 0.0)
                {
                    angle = timeStep() * spin / (weight * cellSize());
                }
                let q = params.value.w * scale * s.xy;
                let rotated = vec2f(cos(angle) * q.x + sin(angle) * q.y,
                    -sin(angle) * q.x + cos(angle) * q.y);
                write0(p, vec4f(rotated, 2.0 * params.value.w * (positive + negative), f));
            }
        )wgsl");

    compute.build(build_diagonal, "implicit elastic diagonal", 1, {"material", "corners", "density"},
        R"wgsl(
            fn stiffness(p: vec2i) -> f32
            {
                return select(0.0, material(p).z, validCell(p));
            }

            fn shearDiagonal(p: vec2i, axis: vec2i) -> f32
            {
                var total = 0.0;
                let transverse = vec2i(axis.y, axis.x);
                for (var a = -1; a <= 1; a++)
                {
                    for (var b = -1; b <= 0; b++)
                    {
                        let c = p + a * axis + b * transverse;
                        if (!validCell(c) || material(c).w <= 0.0)
                        {
                            continue;
                        }
                        var h = 0.0;
                        if (a <= 0)
                        {
                            h += corners(p).x;
                        }
                        if (a >= 0)
                        {
                            h -= corners(p + axis).x;
                        }
                        let scaled = h / material(c).w;
                        total += material(c).z * scaled * scaled;
                    }
                }
                return total / 32.0;
            }

            fn execute()
            {
                let p = cell();
                let mass = 0.5 * cellSize() * cellSize() * vec2f(
                    dot(density(p).xy + density(p - vec2i(1, 0)).xy, vec2f(1)),
                    dot(density(p).xy + density(p - vec2i(0, 1)).xy, vec2f(1)));
                let normal = 0.5 * vec2f(stiffness(p) + stiffness(p - vec2i(1, 0)),
                    stiffness(p) + stiffness(p - vec2i(0, 1)));
                let shear = vec2f(shearDiagonal(p, vec2i(0, 1)), shearDiagonal(p, vec2i(1, 0)));
                write0(p, vec4f(mass + timeStep() * timeStep() * (normal + shear), mass));
            }
        )wgsl",
        geometry);

    // Use the mass Schur operator; the local stiffness response is added after its multigrid inverse.
    compute.build(weights, "implicit elastic Schur coefficients", 1, {"diagonal"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let d = cellSize() * cellSize();
                write0(p, vec4f(select(0.0, d / diagonal(p).z, p.x > 0),
                    select(0.0, d / diagonal(p + vec2i(1, 0)).z, p.x + 1 < gridSize().x),
                    select(0.0, d / diagonal(p).w, p.y > 0),
                    select(0.0, d / diagonal(p + vec2i(0, 1)).w, p.y + 1 < gridSize().y)));
            }
        )wgsl");

    compute.build(strain, "implicit elastic tangent strain", 1, {"source", "material", "corners"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let m = material(p);
                var e = vec2f(0);
                var uncertainty = vec2f(0);
                if (m.w > 0.0)
                {
                    let ux = source(p + vec2i(1, 0)).x - source(p).x;
                    let vy = source(p + vec2i(0, 1)).y - source(p).y;
                    e.x = 0.5 * (ux - vy) / cellSize();
                    if (mode() == 1)
                    {
                        uncertainty.x = 0.5 * (ulp(source(p + vec2i(1, 0)).x) + ulp(source(p).x) +
                            ulp(source(p + vec2i(0, 1)).y) + ulp(source(p).y) + ulp(ux) + ulp(vy) +
                            ulp(ux - vy)) / cellSize() + ulp(e.x);
                    }
                    for (var y = 0; y < 2; y++)
                    {
                        for (var x = 0; x < 2; x++)
                        {
                            let c = p + vec2i(x, y);
                            let uy = source(c).x - source(c - vec2i(0, 1)).x;
                            let vx = source(c).y - source(c - vec2i(1, 0)).y;
                            let weighted = corners(c).x * (uy + vx);
                            let divisor = 8.0 * m.w * cellSize();
                            let term = weighted / divisor;
                            e.y += term;
                            if (mode() == 1)
                            {
                                uncertainty.y += corners(c).x * (ulp(source(c).x) + ulp(source(c - vec2i(0, 1)).x) +
                                    ulp(source(c).y) + ulp(source(c - vec2i(1, 0)).y) + ulp(uy) + ulp(vx) +
                                    ulp(uy + vx)) / divisor + ulp(weighted) / divisor + ulp(term) + ulp(e.y);
                            }
                        }
                    }
                }
                var q = m.z * e;
                if (mode() == 1)
                {
                    let increment = timeStep() * q;
                    uncertainty = timeStep() * (m.z * uncertainty + vec2f(ulp(q.x), ulp(q.y))) +
                        vec2f(ulp(increment.x), ulp(increment.y));
                    q = m.xy + increment;
                    uncertainty += vec2f(ulp(q.x), ulp(q.y));
                }
                write0(p, vec4f(q, uncertainty));
            }
        )wgsl",
        geometry);

    compute.build(gather, "implicit elastic corner stress", 1, {"source", "material", "corners"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let h = corners(p).x;
                var shear = 0.0;
                var uncertainty = 0.0;
                if (h > 0.0)
                {
                    for (var y = -1; y <= 0; y++)
                    {
                        for (var x = -1; x <= 0; x++)
                        {
                            let c = p + vec2i(x, y);
                            let term = h * source(c).y / (4.0 * material(c).w);
                            shear += term;
                            if (mode() == 1)
                            {
                                uncertainty += (h * source(c).w + ulp(h * source(c).y)) /
                                    (4.0 * material(c).w) + ulp(term) + ulp(shear);
                            }
                        }
                    }
                }
                write0(p, vec4f(shear, uncertainty + ulp(shear), 0, 0));
            }
        )wgsl",
        geometry);

    const std::string force = R"wgsl(
        fn forceAt(p: vec2i) -> vec2f
        {
            return cellSize() * vec2f(
                (stress(p).x - stress(p - vec2i(1, 0)).x) + (shear(p + vec2i(0, 1)).x - shear(p).x),
                (stress(p - vec2i(0, 1)).x - stress(p).x) + (shear(p + vec2i(1, 0)).x - shear(p).x));
        }
    )wgsl";
    // [M + 2 dt^2 E^T H k E, B^T; B, 0], with B = -dx * face differences and lambda = dt * pressure.
    compute.build(matrix, "implicit elastic saddle operator", 1, {"source", "stress", "shear", "diagonal"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let s = source(p).xyz;
                let gradient = cellSize() * vec2f(s.z - source(p - vec2i(1, 0)).z,
                    s.z - source(p - vec2i(0, 1)).z);
                let velocity = diagonal(p).zw * s.xy - timeStep() * timeStep() * forceAt(p) + gradient;
                let divergence = cellSize() * ((s.x - source(p + vec2i(1, 0)).x) +
                    (s.y - source(p + vec2i(0, 1)).y));
                write0(p, vec4f(mask(p, vec3f(velocity, divergence)), 0));
            }
        )wgsl",
        geometry + force);

    compute.build(initial, "implicit elastic initial velocity", 1, {"u", "v"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let velocity = vec2f(u(p).x, v(p).x);
                write0(p, vec4f(mask(p, vec3f(velocity, 0)), 0));
            }
        )wgsl",
        geometry);

    compute.build(pressure_extract, "implicit elastic pressure residual", 1, {"source"},
        R"wgsl(
            fn execute()
            {
                write0(cell(), vec4f(source(cell()).z));
            }
        )wgsl");

    // The smooth constant-coefficient limit suggests dt^2 k / (2 dx^2). Center on both sides for symmetry.
    compute.build(pressure_response, "implicit elastic pressure response", 1,
        {"source", "correction", "material", "statistics"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let count = f32(gridSize().x * gridSize().y);
                let centered = fma(source(p).z, count, -statistics(vec2i(0)).x) / count;
                let elastic = 0.5 * timeStep() * timeStep() * material(p).z / (cellSize() * cellSize());
                write0(p, vec4f(correction(p).x + elastic * centered));
            }
        )wgsl");

    // Three half-weight sweeps give D^-1/2 [0.125 (T - 3I)^2 + 0.375 I] D^-1/2, T = D^-1/2 A D^-1/2.
    // This fixed polynomial is SPD. A 16x16 shared patch supplies two exact stencils for an 8x8 output tile.
    compute.build(precondition, "implicit elastic block preconditioner", 1,
        {"source", "diagonal", "correction", "statistics", "material", "corners"},
        R"wgsl(
            var<workgroup> velocity: array<vec2f, 256>;
            var<workgroup> intermediate: array<vec2f, 256>;
            var<workgroup> forcing: array<vec2f, 256>;
            var<workgroup> diagonal_mass: array<vec4f, 256>;
            var<workgroup> tangent: array<vec2f, 256>;
            var<workgroup> coverage: array<f32, 256>;
            var<workgroup> elastic_tile: atomic<u32>;

            fn index(p: vec2i) -> u32
            {
                return u32(p.y * 16 + p.x);
            }

            fn velocityAt(p: vec2i, sweep: i32) -> vec2f
            {
                if (sweep == 0)
                {
                    return velocity[index(p)];
                }
                return intermediate[index(p)];
            }

            fn normalAt(p: vec2i, sweep: i32) -> f32
            {
                let m = tangent[index(p)];
                if (m.x == 0.0)
                {
                    return 0.0;
                }
                let u = velocityAt(p + vec2i(1, 0), sweep).x - velocityAt(p, sweep).x;
                let v = velocityAt(p + vec2i(0, 1), sweep).y - velocityAt(p, sweep).y;
                return m.x * (0.5 * (u - v) / cellSize());
            }

            fn materialShear(p: vec2i, sweep: i32) -> f32
            {
                let m = tangent[index(p)];
                if (m.y <= 0.0)
                {
                    return 0.0;
                }
                var strain = 0.0;
                for (var y = 0; y < 2; y++)
                {
                    for (var x = 0; x < 2; x++)
                    {
                        let c = p + vec2i(x, y);
                        let u = velocityAt(c, sweep).x - velocityAt(c - vec2i(0, 1), sweep).x;
                        let v = velocityAt(c, sweep).y - velocityAt(c - vec2i(1, 0), sweep).y;
                        strain += coverage[index(c)] * (u + v) / (8.0 * m.y * cellSize());
                    }
                }
                return m.x * strain;
            }

            fn shearAt(p: vec2i, sweep: i32) -> f32
            {
                let h = coverage[index(p)];
                if (h <= 0.0)
                {
                    return 0.0;
                }
                var shear = 0.0;
                for (var y = -1; y <= 0; y++)
                {
                    for (var x = -1; x <= 0; x++)
                    {
                        let c = p + vec2i(x, y);
                        shear += h * materialShear(c, sweep) / (4.0 * tangent[index(c)].y);
                    }
                }
                return shear;
            }

            fn faceMask(p: vec2i) -> vec2<bool>
            {
                let horizontal = p.x > 0 && p.x < gridSize().x && p.y >= 0 && p.y < gridSize().y;
                let vertical = p.y > 0 && p.y < gridSize().y && p.x >= 0 && p.x < gridSize().x;
                return vec2<bool>(horizontal, vertical);
            }

            fn execute()
            {
                let p = cell();
                let local = p % vec2i(8);
                let lane = u32(local.y * 8 + local.x);
                let origin = p - local - vec2i(4);
                if (lane == 0u)
                {
                    atomicStore(&elastic_tile, 0u);
                }
                workgroupBarrier();
                let d = diagonal(p);

                // The output tile's tangent support is its 10x10 neighbourhood.
                // Test structural support: tiny stiffness can round away in mass + stiffness.
                for (var i = lane; i < 100u; i += 64u)
                {
                    let q = origin + vec2i(3) + vec2i(i32(i % 10u), i32(i / 10u));
                    if (validCell(q) && material(q).z > 0.0)
                    {
                        atomicStore(&elastic_tile, 1u);
                    }
                }
                workgroupBarrier();

                if (atomicLoad(&elastic_tile) != 0u)
                {
                    for (var i = lane; i < 256u; i += 64u)
                    {
                        let q = origin + vec2i(i32(i % 16u), i32(i / 16u));
                        let diagonal_value = diagonal(q);
                        let r = select(vec2f(0), source(q).xy, faceMask(q));
                        forcing[i] = r;
                        diagonal_mass[i] = diagonal_value;
                        velocity[i] = 0.5 * r / diagonal_value.xy;
                        tangent[i] = select(vec2f(0), material(q).zw, validCell(q));
                        let interior = q.x > 0 && q.y > 0 && q.x < gridSize().x && q.y < gridSize().y;
                        coverage[i] = select(0.0, corners(q).x, interior);
                    }
                }
                workgroupBarrier();

                // Each exact velocity stencil reaches two samples away. Shrink the halo after each sweep.
                for (var sweep = 0; sweep < 2; sweep++)
                {
                    let margin = 2 + 2 * sweep;
                    for (var i = lane; i < 256u; i += 64u)
                    {
                        let q = vec2i(i32(i % 16u), i32(i / 16u));
                        if (atomicLoad(&elastic_tile) != 0u && all(q >= vec2i(margin)) && all(q < vec2i(16 - margin)))
                        {
                            let value = velocityAt(q, sweep);
                            let normal = normalAt(q, sweep);
                            let shear = shearAt(q, sweep);
                            let force = cellSize() * vec2f(
                                (normal - normalAt(q - vec2i(1, 0), sweep)) + (shearAt(q + vec2i(0, 1), sweep) - shear),
                                (normalAt(q - vec2i(0, 1), sweep) - normal) + (shearAt(q + vec2i(1, 0), sweep) - shear));
                            let weights = diagonal_mass[i];
                            let applied = weights.zw * value - timeStep() * timeStep() * force;
                            let next = select(vec2f(0), value + 0.5 * (forcing[i] - applied) / weights.xy, faceMask(origin + q));
                            if (sweep == 0)
                            {
                                intermediate[i] = next;
                            }
                            else
                            {
                                velocity[i] = next;
                            }
                        }
                    }
                    workgroupBarrier();
                }

                if (all(p < gridSize() + vec2i(1)))
                {
                    var result = 0.875 * source(p).xy / d.xy;
                    if (atomicLoad(&elastic_tile) != 0u)
                    {
                        result = velocity[index(local + vec2i(4))];
                    }
                    let mean = statistics(vec2i(0)).x / f32(gridSize().x * gridSize().y);
                    write0(p, vec4f(mask(p, vec3f(result, correction(p).x - mean)), 0));
                }
            }
        )wgsl",
        geometry, true, {8, 8});

    compute.build(normalize, "implicit elastic Lanczos vector", 1, {"source", "control"},
        R"wgsl(
            fn execute()
            {
                let beta = control(vec2i(0)).x;
                var value = vec3f(0);
                if (beta > 0.0 && control(vec2i(1, 0)).w == 0.0)
                {
                    value = source(cell()).xyz / beta;
                }
                write0(cell(), vec4f(value, 0));
            }
        )wgsl");

    compute.build(recurrence, "implicit elastic Lanczos recurrence", 1,
        {"product", "previous", "current", "control", "alpha"},
        R"wgsl(
            fn execute()
            {
                let c = control(vec2i(0));
                var value = product(cell()).xyz;
                if (c.y > 0.0)
                {
                    value -= (c.x / c.y) * previous(cell()).xyz;
                }
                if (mode() == 1 && c.x > 0.0)
                {
                    value -= (alpha(vec2i(0)).x / c.x) * current(cell()).xyz;
                }
                write0(cell(), vec4f(value, 0));
            }
        )wgsl");
    compute.build(scalar_update, "implicit elastic MINRES rotations", 1, {"control", "alpha", "beta"},
        R"wgsl(
            fn execute()
            {
                if (mode() == 2)
                {
                    write0(vec2i(0), vec4f(0));
                    write0(vec2i(1, 0), vec4f(0));
                    write0(vec2i(2, 0), vec4f(0));
                    return;
                }
                let norm = beta(vec2i(0));
                let failed = norm.x < 0.0 || norm.z != 0.0 || norm.w != 0.0;
                let b = sqrt(max(0.0, norm.x));
                if (mode() == 0)
                {
                    write0(vec2i(0), vec4f(b, 0, 0, 0));
                    write0(vec2i(1, 0), vec4f(-1, 0, b, select(0.0, 1.0, failed)));
                    write0(vec2i(2, 0), vec4f(0));
                    return;
                }
                let c = control(vec2i(0));
                let r = control(vec2i(1, 0));
                let a = alpha(vec2i(0)).x;
                let delta = r.x * c.z + r.y * a;
                let gbar = r.y * c.z - r.x * a;
                let gamma = length(vec2f(gbar, b));
                var cosine = 1.0;
                var sine = 0.0;
                var inverse = 0.0;
                if (gamma > 0.0)
                {
                    cosine = gbar / gamma;
                    sine = b / gamma;
                    inverse = 1.0 / gamma;
                }
                write0(vec2i(0), vec4f(b, c.x, -r.x * b, r.y * b));
                let breakdown = failed || r.w != 0.0 || alpha(vec2i(0)).z != 0.0 || alpha(vec2i(0)).w != 0.0;
                write0(vec2i(1, 0), vec4f(cosine, sine, sine * r.z, select(0.0, 1.0, breakdown)));
                write0(vec2i(2, 0), vec4f(c.w, delta, inverse, cosine * r.z));
            }
        )wgsl");
    compute.build(update, "implicit elastic MINRES solution", 3, {"vector", "w1", "w2", "control"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let c = control(vec2i(2, 0));
                let w = (vector(p).xyz - c.x * w1(p).xyz - c.y * w2(p).xyz) * c.z;
                // These packed vectors are pointwise: read and update the same invocation's values.
                let i = 3u * (u32(p.y) * params.fields[1].x + u32(p.x));
                let current = vec3f(output1Data[i], output1Data[i + 1u], output1Data[i + 2u]);
                let error = vec3f(output2Data[i], output2Data[i + 1u], output2Data[i + 2u]);
                let increment = c.w * w - error;
                let next = current + increment;

                // Preserve small Krylov corrections on top of a large falling velocity.
                write0(p, vec4f(w, 0));
                write1(p, vec4f(next, 0));
                write2(p, vec4f((next - current) - increment, 0));
            }
        )wgsl");

    compute.build(reduction_terms, "implicit elastic residual statistics", 1,
        {"a", "b", "diagonal", "control", "rounding"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let value = a(p).xyz;
                if (!finite(value.x) || !finite(value.y) || !finite(value.z))
                {
                    write0(p, vec4f(0, 0, 1, 0));
                    return;
                }
                if (mode() == 0)
                {
                    write0(p, vec4f(dot(value, b(p).xyz), 0, 0, 0));
                    return;
                }
                if (mode() == 2)
                {
                    if (!finite(rounding(p).x) || !finite(rounding(p).y))
                    {
                        write0(p, vec4f(0, 0, 1, 0));
                        return;
                    }
                    let excess = max(vec2f(0), abs(value.xy) - rounding(p).xy);
                    write0(p, vec4f(dot(excess * excess / diagonal(p).zw, vec2f(1)), 0, 0, 0));
                    return;
                }
                let velocity_error = dot(value.xy * value.xy / diagonal(p).zw, vec2f(1));
                var divergence = 0.0;
                var excess = 0.0;
                if (validCell(p))
                {
                    divergence = abs(value.z) / (cellSize() * cellSize());
                    let allowance = (ulp(b(p).x) + ulp(b(p + vec2i(1, 0)).x) +
                        ulp(b(p).y) + ulp(b(p + vec2i(0, 1)).y)) / cellSize();
                    excess = max(0.0, divergence - tolerance() - allowance);
                }
                if (!finite(velocity_error) || !finite(divergence) || !finite(excess))
                {
                    write0(p, vec4f(0, 0, 1, 0));
                    return;
                }
                write0(p, vec4f(velocity_error, divergence, control(vec2i(1, 0)).w, excess));
            }
        )wgsl",
        geometry);
    compute.build(residual, "implicit elastic true residual", 2, {"source", "u", "v", "stress", "shear", "diagonal"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                let s = source(p).xyz;
                let gradient = cellSize() * vec2f(s.z - source(p - vec2i(1, 0)).z,
                    s.z - source(p - vec2i(0, 1)).z);
                // Form the physical increment directly, retaining small changes during fast translation.
                let inertia = diagonal(p).zw * (vec2f(u(p).x, v(p).x) - s.xy);
                let force = timeStep() * forceAt(p);
                let balance = inertia + force;
                let momentum = balance - gradient;
                let divergence = cellSize() * ((source(p + vec2i(1, 0)).x - s.x) +
                    (source(p + vec2i(0, 1)).y - s.y));
                write0(p, vec4f(mask(p, vec3f(momentum, divergence)), 0));

                // Propagate local stored-value ULPs through the absolute constitutive stencil.
                // Stiffness amplifies velocity quantization; a global epsilon misses that floor.
                let stress_error = timeStep() * cellSize() * vec2f(
                    stress(p).z + stress(p - vec2i(1, 0)).z + shear(p + vec2i(0, 1)).y + shear(p).y,
                    stress(p).z + stress(p - vec2i(0, 1)).z + shear(p + vec2i(1, 0)).y + shear(p).y);
                let pressure_error = cellSize() * vec2f(ulp(s.z) + ulp(source(p - vec2i(1, 0)).z),
                    ulp(s.z) + ulp(source(p - vec2i(0, 1)).z));
                let normal = vec2f(stress(p).x - stress(p - vec2i(1, 0)).x,
                    stress(p - vec2i(0, 1)).x - stress(p).x);
                let tangential = vec2f(shear(p + vec2i(0, 1)).x - shear(p).x,
                    shear(p + vec2i(1, 0)).x - shear(p).x);
                let stress_sum = normal + tangential;
                let force_rounding = timeStep() * (cellSize() * vec2f(
                    ulp(normal.x) + ulp(tangential.x) + ulp(stress_sum.x),
                    ulp(normal.y) + ulp(tangential.y) + ulp(stress_sum.y)) +
                    vec2f(ulp(cellSize() * stress_sum.x), ulp(cellSize() * stress_sum.y)));
                let error = diagonal(p).zw * vec2f(ulp(s.x) + ulp(u(p).x - s.x), ulp(s.y) + ulp(v(p).x - s.y)) +
                    stress_error + pressure_error + force_rounding +
                    vec2f(ulp(inertia.x) + ulp(force.x) + ulp(gradient.x) + ulp(balance.x) + ulp(momentum.x),
                        ulp(inertia.y) + ulp(force.y) + ulp(gradient.y) + ulp(balance.y) + ulp(momentum.y));
                write1(p, vec4f(error, 0, 0));
            }
        )wgsl",
        geometry + force);

    compute.build(commit, "implicit elastic velocity and history commit", 4,
        {"solution", "material", "stress", "history"},
        R"wgsl(
            fn execute()
            {
                let p = cell();
                if (p.y < gridSize().y)
                {
                    write0(p, vec4f(solution(p).x));
                }
                if (p.x < gridSize().x)
                {
                    write1(p, vec4f(solution(p).y));
                }
                if (!validCell(p))
                {
                    return;
                }
                let m = material(p);
                var s = vec3f(0);
                if (m.w > 0.0)
                {
                    s.z = history(p).z;
                    let q = m.xy + timeStep() * stress(p).xy;
                    let scale = max(abs(q.x), abs(q.y));
                    if (scale > 0.0)
                    {
                        let direction = q / scale;
                        let magnitude = length(direction);
                        let log_radius = log(scale) + log(magnitude) - log(params.value.w) - log(m.w) - s.z;
                        var radius = 0.0;
                        if (log_radius > 0.0)
                        {
                            radius = log_radius + log(1.0 + sqrt(1.0 + exp(-2.0 * log_radius)));
                        }
                        else
                        {
                            let r = exp(log_radius);
                            radius = asinh(r);
                            if (r < 1e-3)
                            {
                                radius = r * (1.0 - r * r / 6.0);
                            }
                        }
                        s = vec3f(radius * direction / magnitude, s.z);
                    }
                }
                write2(p, vec4f(s, 0));
                write3(p, vec4f(solution(p).z / timeStep()));
            }
        )wgsl",
        geometry);
}

void ImplicitElasticityGPU::apply(const Field& source, Field& destination)
{
    parameters.grid[2] = 0;
    compute.run(strain, {&stress}, {{"source", &source}, {"material", &material}, {"corners", &corners}}, parameters);
    compute.run(
        gather, {&corner_stress}, {{"source", &stress}, {"material", &material}, {"corners", &corners}}, parameters);
    compute.run(matrix, {&destination},
        {{"source", &source}, {"stress", &stress}, {"shear", &corner_stress}, {"diagonal", &diagonal}}, parameters);
}

void ImplicitElasticityGPU::trueResidual(Field& destination, const Field& u, const Field& v)
{
    parameters.grid[2] = 1;
    compute.run(strain, {&stress}, {{"source", &solution}, {"material", &material}, {"corners", &corners}}, parameters);
    compute.run(
        gather, {&corner_stress}, {{"source", &stress}, {"material", &material}, {"corners", &corners}}, parameters);
    compute.run(residual, {&destination, &rounding},
        {{"source", &solution}, {"u", &u}, {"v", &v}, {"stress", &stress}, {"shear", &corner_stress},
            {"diagonal", &diagonal}},
        parameters);
}

void ImplicitElasticityGPU::preconditionResidual()
{
    compute.run(pressure_extract, {&pressure_rhs}, {{"source", &r2}}, parameters);
    reduction.reduce(pressure_rhs, stats, 0);
    const auto& correction = multigrid.apply(pressure_rhs, stats);
    compute.run(pressure_response, {&pressure_rhs},
        {{"source", &r2}, {"correction", &correction}, {"material", &material}, {"statistics", &stats}}, parameters);
    reduction.reduce(pressure_rhs, stats, 0);
    compute.run(precondition, {&z},
        {{"source", &r2}, {"diagonal", &diagonal}, {"correction", &pressure_rhs}, {"statistics", &stats},
            {"material", &material}, {"corners", &corners}},
        parameters);
}

void ImplicitElasticityGPU::reduce(const Field& a, const Field& b, Field& output, int mode)
{
    parameters.grid[2] = mode;
    compute.run(reduction_terms, {&terms},
        {{"a", &a}, {"b", &b}, {"diagonal", &diagonal}, {"control", &control}, {"rounding", &rounding}}, parameters);
    reduction.reduce(terms, output, 3);
}

ImplicitElasticityGPU::Result ImplicitElasticityGPU::solve(double dt, double tolerance, int max_iterations,
    ElasticityGPU& elasticity, const Field& fraction, const Field& density, Field& u, Field& v, Field& pressure)
{
    if (dt <= 0.0)
        return {};

    parameters.value[1] = float(dt);
    parameters.value[2] = float(tolerance);
    parameters.grid[2] = 2;
    compute.run(scalar_update, {&control}, {{"control", &next_control}, {"alpha", &alpha}, {"beta", &beta}}, parameters,
        {1, 1, 1});
    compute.run(corner, {&corners}, {{"fraction", &fraction}}, parameters);
    compute.run(prepare, {&material},
        {{"history", &elasticity.history}, {"fraction", &fraction}, {"corners", &corners}, {"u", &u}, {"v", &v}},
        parameters);
    compute.run(build_diagonal, {&diagonal}, {{"material", &material}, {"corners", &corners}, {"density", &density}},
        parameters);
    compute.run(weights, {&coefficients}, {{"diagonal", &diagonal}}, parameters);
    multigrid.rebuild(coefficients);
    compute.run(initial, {&solution}, {{"u", &u}, {"v", &v}}, parameters);

    Result result;
    double initial_error = 0;
    double previous_error = std::numeric_limits<double>::infinity();
    double remaining_error = 0;
    int stalled = 0;
    const int limit = max_iterations >= 0 ? max_iterations : std::numeric_limits<int>::max();
    auto check = [&](const Field& residual_field)
    {
        reduce(residual_field, solution, stats, 1);
        const auto error = compute.read(stats);
        if (error[2] != 0.0f)
            throw std::runtime_error("Implicit elasticity Krylov solver broke down");
        if (!std::isfinite(error[0]) || !std::isfinite(error[1]) || !std::isfinite(error[3]))
            throw std::runtime_error("Implicit elasticity produced a non-finite residual");
        if (result.iterations == 0)
            initial_error = std::sqrt(error[0]);
        const double momentum = std::sqrt(error[0]);
        result.relative_residual = momentum / std::max(initial_error, 1e-30);
        result.divergence = error[1];
        const double absolute = 1e-6 * std::sqrt(double(parameters.grid[0]) * parameters.grid[1]);
        bool momentum_accepted = momentum <= std::max(1e-4 * initial_error, absolute);
        if (!momentum_accepted && error[3] == 0.0f)
        {
            reduce(residual_field, solution, stats, 2);
            const auto excess = compute.read(stats);
            momentum_accepted = excess[2] == 0.0f && excess[3] == 0.0f &&
                                std::sqrt(excess[0]) <= std::max(1e-4 * initial_error, absolute);
        }
        if (momentum_accepted && error[3] == 0.0f)
        {
            result.converged = error[1] <= tolerance && momentum <= std::max(1e-4 * initial_error, absolute);
            result.precision_limited = !result.converged;
            return true;
        }
        remaining_error = std::max(
            std::max(0.0, momentum / std::max(1e-4 * initial_error, absolute) - 1.0), double(error[3]) / tolerance);
        return false;
    };
    bool finished = false;
    while (!finished)
    {
        trueResidual(r2, u, v);
        if (check(r2))
            break;
        if (result.iterations >= limit)
            break;

        stalled = remaining_error < previous_error * 0.99 ? 0 : stalled + 1;
        previous_error = remaining_error;
        if (stalled >= 4)
            break;

        // Restart from an independently recomputed residual to control float Lanczos drift.
        compute.run(clear, {&r1, &w1, &w2, &compensation}, {}, parameters);
        preconditionResidual();
        reduce(r2, z, beta, 0);
        parameters.grid[2] = 0;
        compute.run(scalar_update, {&next_control}, {{"control", &control}, {"alpha", &alpha}, {"beta", &beta}},
            parameters, {1, 1, 1});
        control.swap(next_control);
        const int batch = std::min(64, limit - result.iterations);
        for (int iteration = 0; iteration < batch; ++iteration)
        {
            compute.run(normalize, {&lanczos}, {{"source", &z}, {"control", &control}}, parameters);
            apply(lanczos, product);
            parameters.grid[2] = 0;
            compute.run(recurrence, {&next_r},
                {{"product", &product}, {"previous", &r1}, {"current", &r2}, {"control", &control}, {"alpha", &alpha}},
                parameters);
            reduce(lanczos, next_r, alpha, 0);
            parameters.grid[2] = 1;
            compute.run(recurrence, {&next_r},
                {{"product", &product}, {"previous", &r1}, {"current", &r2}, {"control", &control}, {"alpha", &alpha}},
                parameters);
            r1.swap(r2);
            r2.swap(next_r);
            preconditionResidual();
            reduce(r2, z, beta, 0);
            parameters.grid[2] = 1;
            compute.run(scalar_update, {&next_control}, {{"control", &control}, {"alpha", &alpha}, {"beta", &beta}},
                parameters, {1, 1, 1});
            control.swap(next_control);
            compute.run(update, {&next_w, &solution, &compensation},
                {{"vector", &lanczos}, {"w1", &w1}, {"w2", &w2}, {"control", &control}}, parameters);
            w1.swap(w2);
            w2.swap(next_w);
            ++result.iterations;
            if ((iteration + 1) % 8 == 0 || result.iterations == limit)
            {
                trueResidual(next_r, u, v);
                if (check(next_r))
                {
                    finished = true;
                    break;
                }
            }
        }
    }
    if (result.converged || result.precision_limited)
    {
        parameters.grid[2] = 0;
        compute.run(
            strain, {&stress}, {{"source", &solution}, {"material", &material}, {"corners", &corners}}, parameters);
        compute.run(commit, {&u, &v, &next_history, &pressure},
            {{"solution", &solution}, {"material", &material}, {"stress", &stress}, {"history", &elasticity.history}},
            parameters, {unsigned(solution.width), unsigned(solution.height), 1});
        elasticity.history.swap(next_history);
        elasticity.stress_valid = false;
        elasticity.statistics_valid = false;
    }
    return result;
}
}
