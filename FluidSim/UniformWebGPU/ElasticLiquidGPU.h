#pragma once
#include "Shared/WebGPU/FluidWebGPUCompute.h"
#include "FluidInterfaceReconstruction.h"

namespace fluid::webgpu
{

// One elastic ingredient, a miscible liquid phase and gas share a geometric partition.
// Nested PLIC cuts the elastic region inside the condensed polygon. Triple cells use
// one plane per interface; no contact-angle or separate solid-contact law is implied.
// A frozen largest-phase WY source and total outgoing CFL <= 1/3 keep split pools positive.
struct ElasticLiquidGPU
{
    Field fraction;
    Field next_fraction;
    Field interfaces;
    Kernel density;
    Kernel reconstruct;
    Kernel transfers;
    Kernel freeze;
    Kernel components_x;
    Kernel components_y;
    Kernel fractions;

    ElasticLiquidGPU(Compute& compute, int width, int height)
        : fraction(compute.field(width, height)),
          next_fraction(compute.field(width, height)),
          interfaces(compute.field(width, height, 4))
    {
        compute.build(density, "elastic liquid phase densities", 1,
            {"masses0", "masses1", "masses2", "masses3", "previous"},
            R"wgsl(
                fn phases(mass: vec4f, group: u32) -> vec4f
                {
                    let condensed = params.vectors[group];
                    let solid = params.vectors[group + 8u];
                    return vec4f(dot(mass, condensed), dot(mass, params.vectors[group + 4u]),
                        dot(mass, solid), dot(mass, condensed - solid)) / (cellSize() * cellSize());
                }

                fn execute()
                {
                    let p = cell();
                    var sum = phases(masses0(p), 0u);
                    if (params.grid.w > 1)
                    {
                        sum += phases(masses1(p), 1u);
                    }
                    if (params.grid.w > 2)
                    {
                        sum += phases(masses2(p), 2u);
                    }
                    if (params.grid.w > 3)
                    {
                        sum += phases(masses3(p), 3u);
                    }
                    if (mode() != 0)
                    {
                        sum += previous(p);
                    }
                    write0(p, sum);
                }
            )wgsl");

        const std::string polygon_geometry = R"wgsl(
            struct Polygon
            {
                vertices: array<vec2f, 8>,
                count: u32,
            }

            fn rectangle(lo: vec2f, hi: vec2f) -> Polygon
            {
                var result: Polygon;
                result.count = 4u;
                result.vertices[0] = lo;
                result.vertices[1] = vec2f(hi.x, lo.y);
                result.vertices[2] = hi;
                result.vertices[3] = vec2f(lo.x, hi.y);
                return result;
            }

            fn clip(input: Polygon, normal: vec2f, offset: f32) -> Polygon
            {
                var result: Polygon;
                if (input.count == 0u)
                {
                    return result;
                }
                var previous = input.vertices[input.count - 1u];
                var before = dot(normal, previous) - offset;
                for (var i = 0u; i < input.count; i++)
                {
                    let current = input.vertices[i];
                    let after = dot(normal, current) - offset;
                    if ((before <= 0.0) != (after <= 0.0))
                    {
                        result.vertices[result.count] = previous + (before / (before - after)) * (current - previous);
                        result.count++;
                    }
                    if (after <= 0.0)
                    {
                        result.vertices[result.count] = current;
                        result.count++;
                    }
                    previous = current;
                    before = after;
                }
                return result;
            }

            fn area(polygon: Polygon) -> f32
            {
                if (polygon.count < 3u)
                {
                    return 0.0;
                }

                // Triangulate relative to a vertex to avoid subtracting large areas for tiny slivers.
                let origin = polygon.vertices[0];
                var sum = 0.0;
                for (var i = 1u; i + 1u < polygon.count; i++)
                {
                    let a = polygon.vertices[i] - origin;
                    let b = polygon.vertices[i + 1u] - origin;
                    sum += a.x * b.y - a.y * b.x;
                }
                return 0.5 * abs(sum);
            }

            fn nestedOffset(normal: vec2f, outer: vec3f, solid: f32) -> f32
            {
                let polygon = clip(rectangle(vec2f(0), vec2f(1)), outer.xy, outer.z);
                var lo = min(normal.x, 0.0) + min(normal.y, 0.0);
                var hi = max(normal.x, 0.0) + max(normal.y, 0.0);
                for (var i = 0; i < 24; i++)
                {
                    let middle = 0.5 * (lo + hi);
                    if (middle == lo || middle == hi)
                    {
                        break;
                    }
                    if (area(clip(polygon, normal, middle)) < solid)
                    {
                        lo = middle;
                    }
                    else
                    {
                        hi = middle;
                    }
                }
                return 0.5 * (lo + hi);
            }
        )wgsl";

        compute.build(reconstruct, "elastic liquid nested interface", 1, {"liquid", "condensed", "outer", "u", "v"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    let c = clamp(condensed(p).x, 0.0, 1.0);
                    let s = clamp(liquid(p).x, 0.0, c);
                    if (s == 0.0 || s == c)
                    {
                        write0(p, vec4f(0));
                        return;
                    }
                    var g = reconstructInterface(p);
                    if (c < 1.0 && g.w == 0.0)
                    {
                        let outside = outer(p);
                        if (outside.w != 0.0)
                        {
                            // The existing outer fragment fallback is uniform within the donor cell.
                            g.z = plicOffset(g.xy, s / c);
                        }
                        else
                        {
                            g.z = nestedOffset(g.xy, outside.xyz, s);
                        }
                    }
                    write0(p, g);
                }
            )wgsl",
            std::string(geometry) + std::string(interface_reconstruction) + polygon_geometry);

        compute.build(transfers, "elastic liquid partitioned transfer", 1,
            {"condensed", "solid", "outer", "inner", "flow", "density"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    let axis = select(vec2i(0, 1), vec2i(1, 0), mode() == 0);
                    let coordinate = select(p.y, p.x, mode() == 0);
                    let extent = select(gridSize().y, gridSize().x, mode() == 0);
                    let q = flow(p).x * timeStep() / cellSize();
                    if (coordinate == 0 || coordinate == extent || q == 0.0)
                    {
                        write0(p, vec4f(0));
                        return;
                    }
                    let donor = select(p, p - axis, q > 0.0);
                    let source_c = condensed(donor).x;
                    let source_s = solid(donor).x;
                    let c = clamp(source_c, 0.0, 1.0);
                    let s = clamp(source_s, 0.0, c);
                    let strip = abs(q);
                    var lo = vec2f(0);
                    var hi = vec2f(1);
                    if (q > 0.0)
                    {
                        lo = vec2f(axis) * (1.0 - strip);
                    }
                    else
                    {
                        hi -= vec2f(axis) * (1.0 - strip);
                    }
                    let outside = outer(donor);
                    let inside = inner(donor);
                    var total = strip * c;
                    if (c > 0.0 && c < 1.0 && outside.w == 0.0)
                    {
                        total = plicArea(outside.xy, outside.z, lo, hi);
                    }
                    total = clamp(total, max(0.0, strip - (1.0 - c)), min(strip, c));
                    var elastic = 0.0;
                    if (s == c)
                    {
                        elastic = total;
                    }
                    else if (s > 0.0)
                    {
                        if (inside.w != 0.0)
                        {
                            elastic = total * (s / c);
                        }
                        else if (c == 1.0)
                        {
                            elastic = plicArea(inside.xy, inside.z, lo, hi);
                        }
                        else if (outside.w != 0.0)
                        {
                            elastic = c * plicArea(inside.xy, inside.z, lo, hi);
                        }
                        else
                        {
                            let polygon = clip(rectangle(lo, hi), outside.xy, outside.z);
                            elastic = area(clip(polygon, inside.xy, inside.z));
                        }
                    }

                    // Coupled bounds preserve a nonnegative partition, including tiny third phases.
                    elastic = clamp(elastic, max(0.0, total - (c - s)), min(total, s));

                    // Match ingredient ratios, including roundoff at pure-phase limits.
                    let fractions = vec3f(source_s, source_c - source_s, 1.0 - source_c);
                    let volume = vec3f(elastic, total - elastic, strip - total) * sign(q);
                    var ratio = vec3f(0);
                    for (var i = 0u; i < 3u; i++)
                    {
                        if (fractions[i] > 0.0)
                        {
                            ratio[i] = volume[i] / fractions[i];
                        }
                    }
                    let rho = density(donor);
                    write0(p, vec4f(total * sign(q), dot(rho.zwy, ratio), 0, elastic * sign(q)));
                }
            )wgsl",
            std::string(geometry) + polygon_geometry);

        compute.build(freeze, "elastic liquid split weights", 1, {"condensed", "solid", "density"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    let c = condensed(p).x;
                    let s = solid(p).x;
                    let f = vec3f(s, c - s, 1.0 - c);
                    var winner = 2u;
                    if (f.y > f[winner])
                    {
                        winner = 1u;
                    }
                    if (f.x > f[winner])
                    {
                        winner = 0u;
                    }
                    let rho = density(p).zwy;
                    write0(p, vec4f(select(0.0, 1.0, winner != 2u), rho[winner] / f[winner],
                        select(0.0, 1.0, winner == 0u), 0));
                }
            )wgsl");

        const std::string component_helpers = R"wgsl(
            fn fractions(p: vec2i) -> vec3f
            {
                let c = condensed(p).x;
                let s = solid(p).x;
                return vec3f(s, c - s, 1.0 - c);
            }

            fn shares(volume: vec3f, f: vec3f) -> vec4f
            {
                var ratio = vec3f(0);
                for (var i = 0u; i < 3u; i++)
                {
                    if (f[i] > 0.0)
                    {
                        ratio[i] = volume[i] / f[i];
                    }
                }
                return params.vectors[2] * ratio.x + (params.vectors[0] - params.vectors[2]) * ratio.y +
                    params.vectors[1] * ratio.z;
            }

            fn transfer(p: vec2i, axis: vec2i) -> vec4f
            {
                let coordinate = select(p.y, p.x, mode() == 0);
                let extent = select(gridSize().y, gridSize().x, mode() == 0);
                if (coordinate == 0 || coordinate == extent)
                {
                    return vec4f(0);
                }
                let q = flow(p).x * timeStep() / cellSize();
                let donor = select(p, p - axis, q > 0.0);
                let flux = transfers(p);
                return masses(donor) * shares(vec3f(flux.w, flux.x - flux.w, q - flux.x), fractions(donor));
            }

            fn transport(p: vec2i, weight: vec4f) -> vec4f
            {
                let axis = select(vec2i(0, 1), vec2i(1, 0), mode() == 0);
                let before = transfer(p, axis) - weight * (flow(p).x * timeStep() / cellSize());
                let after = transfer(p + axis, axis) - weight * (flow(p + axis).x * timeStep() / cellSize());
                return masses(p) + (before - after);
            }
        )wgsl";

        compute.build(components_x, "elastic liquid ingredients X", 2,
            {"condensed", "solid", "masses", "transfers", "flow", "weights"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    let w = weights(p);
                    let weight = masses(p) * shares(vec3f(w.z, w.x - w.z, 1.0 - w.x), fractions(p));
                    write0(p, transport(p, weight));
                    write1(p, weight);
                }
            )wgsl",
            component_helpers);
        compute.build(components_y, "elastic liquid ingredients Y", 1,
            {"condensed", "solid", "masses", "transfers", "flow", "corrections"},
            R"wgsl(
                fn execute()
                {
                    write0(cell(), transport(cell(), corrections(cell())));
                }
            )wgsl",
            component_helpers);
        compute.build(fractions, "elastic liquid volume update", 2,
            {"condensed", "solid", "transfers", "flow", "weights"},
            R"wgsl(
                fn execute()
                {
                    let p = cell();
                    let axis = select(vec2i(0, 1), vec2i(1, 0), mode() == 0);
                    let weight = weights(p).xz;
                    let before = transfers(p).xw - weight * (flow(p).x * timeStep() / cellSize());
                    let after = transfers(p + axis).xw - weight * (flow(p + axis).x * timeStep() / cellSize());
                    let next = vec2f(condensed(p).x, solid(p).x) + (before - after);
                    write0(p, vec4f(next.x));
                    write1(p, vec4f(next.y));
                }
            )wgsl");
    }
};

}
