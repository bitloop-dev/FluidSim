#pragma once
#include "Shared/WebGPU/FluidWebGPUCompute.h"

namespace fluid::webgpu
{
namespace viscosity_shaders
{
// Coefficients: cell mu, vertex mu, u-face mass, v-face mass. Writing the
// operator as a sum of squared normal/shear strains makes it symmetric positive.
inline const std::string geometry = R"wgsl(
    fn validFace(p: vec2i) -> vec2<bool>
    {
        return vec2<bool>(
            (p.x > 0 && p.x < gridSize().x && p.y < gridSize().y), (p.y > 0 && p.y < gridSize().y && p.x < gridSize().x));
    }

    fn diagonal(p: vec2i) -> vec2f
    {
        let c = coeff(p);
        let left = coeff(p - vec2i(1, 0));
        let down = coeff(p - vec2i(0, 1));
        return c.zw + timeStep() * vec2f(2.0 * (c.x + left.x) + c.y + coeff(p + vec2i(0, 1)).y,
                                       2.0 * (c.x + down.x) + c.y + coeff(p + vec2i(1, 0)).y);
    }
)wgsl";

inline const std::string stress = R"wgsl(
    fn shear(p: vec2i) -> f32
    {
        return coeff(p).y * ((sampleVelocity(p).x - sampleVelocity(p - vec2i(0, 1)).x) +
                                (sampleVelocity(p).y - sampleVelocity(p - vec2i(1, 0)).y));
    }

    fn stressProduct(p: vec2i) -> vec2f
    {
        let c = sampleVelocity(p);
        let a = coeff(p);
        let t = shear(p);
        let normal = 2.0 * vec2f(coeff(p - vec2i(1, 0)).x * (c.x - sampleVelocity(p - vec2i(1, 0)).x) +
                                     a.x * (c.x - sampleVelocity(p + vec2i(1, 0)).x),
                               coeff(p - vec2i(0, 1)).x * (c.y - sampleVelocity(p - vec2i(0, 1)).y) +
                                   a.x * (c.y - sampleVelocity(p + vec2i(0, 1)).y));

        return normal + vec2f(t - shear(p + vec2i(0, 1)), t - shear(p + vec2i(1, 0)));
    }
)wgsl";

inline const std::string operation = geometry + R"wgsl(
    fn sampleVelocity(p: vec2i) -> vec2f
    {
        return source(p).xy;
    }
)wgsl" + stress + R"wgsl(
    fn product(p: vec2i) -> vec2f
    {
        return select(vec2f(0), coeff(p).zw * sampleVelocity(p) + timeStep() * stressProduct(p), validFace(p));
    }
)wgsl";

inline const std::string driven_stress = geometry + R"wgsl(
    fn sampleVelocity(p: vec2i) -> vec2f
    {
        var value = source(p).xy;
        if (p.x == 0 || p.x == gridSize().x)
        {
            value.x = u(p).x;
        }
        if (p.y == 0 || p.y == gridSize().y)
        {
            value.y = v(p).x;
        }

        return value;
    }
)wgsl" + stress;

inline constexpr std::string_view control = R"wgsl(
    fn stopped() -> bool
    {
        return state(vec2i(0)).w != 0.0 || state(vec2i(1, 0)).y <= state(vec2i(1, 0)).x;
    }
)wgsl";
}

// Shader programs only. The solver owns fields, multigrid traversal and convergence.
struct ViscosityKernels
{
    static constexpr int lanes = 128;
    Kernel coarsen;
    Kernel coefficients;
    Kernel pack;
    Kernel unpack;
    Kernel apply;
    Kernel residual_pass;
    Kernel smooth;
    Kernel coarse_smooth;
    Kernel restrict_pass;
    Kernel prolong;
    Kernel first_reduce;
    Kernel finish_reduce;
    Kernel initialize;
    Kernel scalar;
    Kernel direction_pass;
    Kernel update;
    Kernel refresh;
    Kernel driven_residual;

    void build(Compute& compute)
    {
        using namespace viscosity_shaders;
        compute.build(coarsen, "viscosity coarse material", 1, {"fine"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    var value = vec2f(0);
                    var n = 0.0;

                    for (var y = 0; y < 2; y++)
                    {
                        for (var x = 0; x < 2; x++)
                        {
                            let q = 2 * p + vec2i(x, y);
                            if (all(q < fineSize()))
                            {
                                value += fine(q).xy;
                                n += 1.0;
                            }
                        }
                    }

                    write0(p, vec4f(value / n, 0, 0));
                }
            )wgsl");

        compute.build(coefficients, "viscosity coefficients", 1, {"material"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    let m = material(p);
                    var mu = 0.0;
                    if (all(p > vec2i(0)) && all(p < gridSize()))
                    {
                        mu = 0.25 * (m.y + material(p - vec2i(1, 0)).y + material(p - vec2i(0, 1)).y + material(p - vec2i(1, 1)).y);
                    }
                    let mass =
                        0.5 * cellSize() * cellSize() * (vec2f(m.x) + vec2f(material(p - vec2i(1, 0)).x, material(p - vec2i(0, 1)).x));
                    write0(p, vec4f(m.y, mu, mass));
                }
            )wgsl");

        compute.build(pack, "viscosity initial velocity and RHS", 2, {"u", "v", "coeff"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    let velocity = select(vec2f(0), vec2f(u(p).x, v(p).x), validFace(p));
                    write0(p, vec4f(velocity, 0, 0));
                    write1(p, vec4f(coeff(p).zw * velocity, 0, 0));
                }
            )wgsl",
            geometry);

        compute.build(unpack, "viscosity staggered result", 1, {"source"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    let coordinate = select(p.y, p.x, mode() == 0);
                    let extent = select(gridSize().y, gridSize().x, mode() == 0);

                    if (params.level.w != 0.0 && (coordinate == 0 || coordinate == extent))
                    {
                        return;
                    }

                    write0(p, vec4f(source(p)[u32(mode())]));
                }
            )wgsl");

        compute.build(apply, "viscosity matrix product", 1, {"source", "coeff", "state"},
            R"wgsl(
                fn execute()
                {
                    var value = vec2f(0);
                    if (!stopped())
                    {
                        value = product(cell());
                    }
                    write0(cell(), vec4f(value, 0, 0));
                }
            )wgsl",
            operation + std::string(control));

        compute.build(residual_pass, "viscosity true residual", 1, {"source", "rhs", "coeff"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    write0(p, vec4f(select(vec2f(0), rhs(p).xy - product(p), validFace(p)), 0, 0));
                }
            )wgsl",
            operation);

        // Evaluate the physical equation before subtracting large boundary forcing
        // from the homogeneous product. This avoids a cancellation floor at driven faces.
        compute.build(driven_residual, "viscosity driven true residual", 1, {"source", "u", "v", "coeff"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    let previous = vec2f(u(p).x, v(p).x);
                    let value = coeff(p).zw * (previous - sampleVelocity(p)) - timeStep() * stressProduct(p);

                    write0(p, vec4f(select(vec2f(0), value, validFace(p)), 0, 0));
                }
            )wgsl",
            driven_stress);

        compute.build(smooth, "viscosity symmetric smoother", 1, {"source", "rhs", "coeff", "state"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    var value = vec2f(0);
                    if (!stopped())
                    {
                        if (mode() == 1)
                        {
                            value = 0.5 * rhs(p).xy / diagonal(p);
                        }
                        else
                        {
                            value = source(p).xy + 0.5 * (rhs(p).xy - product(p)) / diagonal(p);
                        }
                    }

                    write0(p, vec4f(select(vec2f(0), value, validFace(p)), 0, 0));
                }
            )wgsl",
            operation + std::string(control));

        // Coarsest grids have at most130 MAC vector slots. Keep all16 symmetric
        // Jacobi sweeps in one workgroup rather than dispatching tiny global passes.
        const std::string shared = R"wgsl(
            var<workgroup> velocity: array<vec2f, 256>;
            var<workgroup> material: array<vec4f, 256>;
            fn index(p: vec2i) -> u32
            {
                let q = clamp(p, vec2i(0), gridSize());
                return u32(q.y * (gridSize().x + 1) + q.x);
            }

            fn source(p: vec2i) -> vec4f
            {
                return vec4f(velocity[index(p)], 0, 0);
            }

            fn coeff(p: vec2i) -> vec4f
            {
                return material[index(p)];
            }
        )wgsl";
        compute.build(coarse_smooth, "viscosity shared coarse solve", 1, {"rhs", "weights"},
            R"wgsl(
                fn execute()
                {
                    let lane = u32(cell().x);
                    let stride = u32(gridSize().x + 1);
                    let p = vec2i(i32(lane % stride), i32(lane / stride));

                    velocity[lane] = vec2f(0);
                    material[lane] = weights(p);
                    let b = rhs(p).xy;
                    workgroupBarrier();

                    for (var sweep = 0; sweep < 16; sweep++)
                    {
                        let value = select(vec2f(0), source(p).xy + 0.5 * (b - product(p)) / diagonal(p), validFace(p));
                        workgroupBarrier();
                        velocity[lane] = value;
                        workgroupBarrier();
                    }

                    if (lane < u32((gridSize().x + 1) * (gridSize().y + 1)))
                    {
                        write0(p, vec4f(velocity[lane], 0, 0));
                    }
                }
            )wgsl",
            shared + operation, true, {256, 1});

        // Bilinear interpolation uses face locations, not cell-centred vector locations.
        // The gather restriction below is its exact transpose, including tangential clamps.
        const std::string transfer = R"wgsl(
            fn offset(component: u32) -> vec2f
            {
                return select(vec2f(0, 0.5), vec2f(0.5, 0), component == 1u);
            }

            fn coarsePoint(p: vec2i, component: u32, size: vec2i) -> vec2i
            {
                return clamp(p, vec2i(0), size - select(vec2i(0, 1), vec2i(1, 0), component == 1u));
            }

            fn weight(fine: vec2i, coarse: vec2i, component: u32, size: vec2i) -> f32
            {
                let location = (vec2f(fine) + offset(component)) * 0.5 - offset(component);
                let base = vec2i(floor(location));
                let t = fract(location);
                var result = 0.0;

                for (var y = 0; y < 2; y++)
                {
                    for (var x = 0; x < 2; x++)
                    {
                        if (all(coarsePoint(base + vec2i(x, y), component, size) == coarse))
                        {
                            result += select(1.0 - t.x, t.x, x == 1) * select(1.0 - t.y, t.y, y == 1);
                        }
                    }
                }

                return result;
            }
        )wgsl";
        compute.build(restrict_pass, "viscosity transpose restriction", 1, {"source", "state"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    var value = vec2f(0);
                    let fine = sourceSize() - 1;

                    if (!stopped())
                    {
                        for (var component = 0u; component < 2u; component++)
                        {
                            if ((component == 0u && (p.x == 0 || p.x == gridSize().x || p.y == gridSize().y)) ||
                                (component == 1u && (p.y == 0 || p.y == gridSize().y || p.x == gridSize().x)))
                            {
                                continue;
                            }

                            let extent = select(vec2i(1, 2), vec2i(2, 1), component == 1u);
                            for (var y = -1; y <= extent.y; y++)
                            {
                                for (var x = -1; x <= extent.x; x++)
                                {
                                    let q = 2 * p + vec2i(x, y);

                                    if (any(q < vec2i(0)) || any(q > fine))
                                    {
                                        continue;
                                    }

                                    value[component] += weight(q, p, component, gridSize()) * source(q)[component];
                                }
                            }
                        }
                    }

                    write0(p, vec4f(value, 0, 0));
                }
            )wgsl",
            transfer + std::string(control));

        compute.build(prolong, "viscosity prolongation", 1, {"source", "coarse", "state"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    var value = vec2f(0);
                    if (!stopped())
                    {
                        value = source(p).xy;
                        for (var component = 0u; component < 2u; component++)
                        {
                            let location = (vec2f(p) + offset(component)) * 0.5 - offset(component);
                            let base = vec2i(floor(location));
                            let t = fract(location);

                            for (var y = 0; y < 2; y++)
                            {
                                for (var x = 0; x < 2; x++)
                                {
                                    value[component] += select(1.0 - t.x, t.x, x == 1) * select(1.0 - t.y, t.y, y == 1) *
                                                        coarse(coarsePoint(base + vec2i(x, y), component, coarseSize() - 1))[component];
                                }
                            }
                        }
                    }

                    let valid = vec2<bool>(
                        (p.x > 0 && p.x < gridSize().x && p.y < gridSize().y), (p.y > 0 && p.y < gridSize().y && p.x < gridSize().x));
                    write0(p, vec4f(select(vec2f(0), value, valid), 0, 0));
                }
            )wgsl",
            transfer + std::string(control));

        const std::string reductions = R"wgsl(
            var<workgroup> partial: array<vec4f, 128>;
            fn combine(a: vec4f, b: vec4f) -> vec4f
            {
                return vec4f(a.xy + b.xy, select(max(a.z, b.z), a.z + b.z, mode() == 0), max(a.w, b.w));
            }

            fn checked(value: vec4f) -> vec4f
            {
                if (!finite(value.x) || !finite(value.y) || !finite(value.z))
                {
                    return vec4f(0, 0, 0, 1);
                }

                return value;
            }

            fn groupReduction(lane: u32, value: vec4f) -> vec4f
            {
                partial[lane] = checked(value);
                workgroupBarrier();
                for (var stride = 64u; stride > 0u; stride /= 2u)
                {
                    if (lane < stride)
                    {
                        partial[lane] = combine(partial[lane], partial[lane + stride]);
                    }
                    workgroupBarrier();
                }

                return checked(partial[0]);
            }
        )wgsl";
        compute.build(first_reduce, "viscosity reduction partials", 1, {"source", "other", "coeff", "state"},
            R"wgsl(
                fn execute()
                {
                    let lane = u32(cell().x) % 128u;
                    let group = u32(cell().x) / 128u;
                    let count = u32(params.grid.x);
                    let groups = u32(params.grid.y);
                    let width = u32(sourceSize().x);
                    var total = vec4f(0);

                    for (var base = group * 512u; base < count; base += groups * 512u)
                    {
                        for (var offset = 0u; offset < 512u; offset += 128u)
                        {
                            let i = base + offset + lane;

                            if (i >= count)
                            {
                                continue;
                            }

                            let p = vec2i(i32(i % width), i32(i / width));
                            let a = source(p).xy;
                            let b = other(p).xy;
                            let mass = coeff(p).zw;
                            var value = vec4f(0);
                            if (mode() == 0)
                            {
                                value = vec4f(dot(a / mass, a), dot(mass * b, b), mass.x + mass.y, 0);
                            }
                            if (mode() == 1)
                            {
                                value = vec4f(dot(a, b), dot(a / mass, a), 0, 0);
                            }
                            if (mode() == 2)
                            {
                                value = vec4f(dot(a, b), 0, 0, 0);
                            }
                            if (mode() == 3)
                            {
                                value = vec4f(dot(a / mass, a), 0, select(0.0, 1.0, state(vec2i(1, 0)).y <= state(vec2i(1, 0)).x),
                                    state(vec2i(0)).w);
                                if (i == 0u)
                                {
                                    value.y = state(vec2i(1, 0)).x;
                                }
                            }

                            total = combine(total, checked(value));
                        }
                    }

                    let result = groupReduction(lane, total);
                    if (lane == 0u)
                    {
                        write0(vec2i(i32(group), 0), result);
                    }
                }
            )wgsl",
            reductions, true, {lanes, 1});

        compute.build(finish_reduce, "viscosity reduction finish", 1, {"source"},
            R"wgsl(
                fn execute()
                {
                    let lane = u32(cell().x);
                    var total = vec4f(0);
                    for (var i = lane; i < u32(params.grid.x); i += 128u)
                    {
                        total = combine(total, checked(source(vec2i(i32(i), 0))));
                    }

                    let result = groupReduction(lane, total);
                    if (lane == 0u)
                    {
                        write0(vec2i(0), result);
                    }
                }
            )wgsl",
            reductions, true, {lanes, 1});

        compute.build(initialize, "viscosity initialize CG", 1, {"stats"},
            R"wgsl(
                fn execute()
                {
                    let s = stats(vec2i(0));
                    write0(vec2i(0), vec4f(0, 0, 0, s.w));
                    write0(vec2i(1, 0), vec4f(params.value.z * params.value.z * s.y + 1e-12 * s.z, s.x, 0, s.y));
                }
            )wgsl");

        compute.build(scalar, "viscosity CG scalars", 1, {"state", "stats"},
            R"wgsl(
                fn execute()
                {
                    var a = state(vec2i(0));
                    var b = state(vec2i(1, 0));
                    let s = stats(vec2i(0));
                    if (!stopped())
                    {
                        a.w = max(a.w, s.w);
                        if (mode() == 1)
                        {
                            b.y = s.y;
                            if (b.y > b.x)
                            {
                                if (s.x <= 0.0 || !finite(s.x))
                                {
                                    a.w = 1;
                                }
                                else
                                {
                                    a.z = select(0.0, s.x / a.x, b.z > 0.0);
                                    a.x = s.x;
                                }
                            }
                        }
                        else
                        {
                            if (s.x <= 0.0 || !finite(s.x))
                            {
                                a.w = 1;
                            }
                            else
                            {
                                a.y = a.x / s.x;
                                b.z += 1.0;
                            }
                        }
                    }

                    write0(vec2i(0), a);
                    write0(vec2i(1, 0), b);
                }
            )wgsl",
            control);

        compute.build(direction_pass, "viscosity CG direction", 1, {"correction", "previous", "state"},
            R"wgsl(
                fn execute()
                {
                    var value = vec2f(0);
                    if (!stopped())
                    {
                        value = correction(cell()).xy;
                        if (state(vec2i(1, 0)).z > 0.0)
                        {
                            value += state(vec2i(0)).z * previous(cell()).xy;
                        }
                    }

                    write0(cell(), vec4f(value, 0, 0));
                }
            )wgsl",
            control);

        compute.build(update, "viscosity CG velocity and residual", 2,
            {"solution", "residual", "direction", "product", "state"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    var x = solution(p).xy;
                    var r = residual(p).xy;
                    if (!stopped())
                    {
                        let alpha = state(vec2i(0)).y;
                        x += alpha * direction(p).xy;
                        r -= alpha * product(p).xy;
                    }

                    write0(p, vec4f(x, 0, 0));
                    write1(p, vec4f(r, 0, 0));
                }
            )wgsl",
            control);

        compute.build(refresh, "viscosity refresh residual", 2, {"truth", "state", "stats"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    write0(p, truth(p));
                    if (all(p == vec2i(0)))
                    {
                        let b = state(vec2i(1, 0));
                        write1(vec2i(0), vec4f(0));
                        write1(vec2i(1, 0), vec4f(b.x, stats(vec2i(0)).x, 0, b.w));
                    }
                }
            )wgsl");
    }
};
}
