#pragma once
#include <string_view>

namespace fluid::webgpu
{

// A complete red/black sweep. Each first-colour result is shared with its neighbours.
inline constexpr std::string_view shared_red_black = R"wgsl(
var<workgroup> first_color_values: array<f32, 324>;

fn firstColor(point: vec2i) -> f32 {
    let p = clamp(point, vec2i(0), gridSize() - 1);
    let a = weights(p);
    let diagonal = (a.x + a.y) + (a.z + a.w);
    var value = rhsAt(p);
    if (mode() < 2) {
        value += (a.x * source(p - vec2i(1, 0)).x + a.y * source(p + vec2i(1, 0)).x) +
                 (a.z * source(p - vec2i(0, 1)).x + a.w * source(p + vec2i(0, 1)).x);
    }
    if (diagonal > 0.0) { return value / diagonal; }
    return 0.0;
}

fn execute() {
    let p = cell();
    let local = p % vec2i(16);
    let base = p - local;
    let lane = local.y * 16 + local.x;
    if (skipSharedCycle(u32(lane))) {
        if (all(p < gridSize())) { write0(p, vec4f(0)); }
        return;
    }

    // One-cell halo supplies the first-colour neighbours of every output cell.
    for (var i = lane; i < 324; i += 256) {
        let q = base + vec2i(i % 18, i / 18) - vec2i(1);
        if (((q.x + q.y) & 1) == (mode() & 1)) {
            first_color_values[i] = firstColor(q);
        } else {
            first_color_values[i] = 0.0;
        }
    }
    workgroupBarrier();
    if (any(p >= gridSize())) { return; }

    let index = (local.y + 1) * 18 + local.x + 1;
    if (((p.x + p.y) & 1) == (mode() & 1)) {
        write0(p, vec4f(first_color_values[index]));
        return;
    }
    let a = weights(p);
    let diagonal = (a.x + a.y) + (a.z + a.w);
    let value = rhsAt(p) +
        (a.x * first_color_values[index - 1] + a.y * first_color_values[index + 1]) +
        (a.z * first_color_values[index - 18] + a.w * first_color_values[index + 18]);
    var result = 0.0;
    if (diagonal > 0.0) { result = value / diagonal; }
    write0(p, vec4f(result));
}
)wgsl";

// Two original polynomial sweeps, using 11,200 bytes of workgroup storage.
inline constexpr std::string_view paired_polynomial = R"wgsl(
var<workgroup> tile_weights: array<vec4f, 400>;
var<workgroup> tile_rhs: array<f32, 400>;
var<workgroup> old_values: array<f32, 400>;
var<workgroup> first_values: array<f32, 400>;

fn execute() {
    let p = cell();
    let local = p % vec2i(16);
    let base = p - local;
    let lane = local.y * 16 + local.x;
    if (skipSharedCycle(u32(lane))) {
        if (all(p < gridSize())) { write0(p, vec4f(0)); }
        return;
    }

    // Two-cell halo preserves both global stencils, including clamped wall samples.
    for (var i = lane; i < 400; i += 256) {
        let q = clamp(base + vec2i(i % 20, i / 20) - vec2i(2), vec2i(0), gridSize() - 1);
        tile_weights[i] = weights(q);
        tile_rhs[i] = rhsAt(q);
        old_values[i] = 0.0;
        if (mode() == 0) { old_values[i] = source(q).x; }
    }
    workgroupBarrier();

    // The first sweep updates the output region and its one-cell halo.
    for (var i = lane; i < 324; i += 256) {
        let location = vec2i(i % 18, i / 18) + vec2i(1);
        let clamped = clamp(base + location - vec2i(2), vec2i(0), gridSize() - 1);
        let sample = clamped - base + vec2i(2);
        let index = sample.y * 20 + sample.x;
        let a = tile_weights[index];
        let diagonal = (a.x + a.y) + (a.z + a.w);
        var value = tile_rhs[index];
        var result = 0.0;
        if (mode() == 0) {
            let center = old_values[index];
            let product = (a.x * (center - old_values[index - 1]) + a.y * (center - old_values[index + 1])) +
                          (a.z * (center - old_values[index - 20]) + a.w * (center - old_values[index + 20]));
            value -= product;
            result = center;
        }
        if (diagonal > 0.0) { result += params.value.w * value / diagonal; }
        first_values[location.y * 20 + location.x] = result;
    }
    workgroupBarrier();
    if (any(p >= gridSize())) { return; }

    let index = (local.y + 2) * 20 + local.x + 2;
    let a = tile_weights[index];
    let center = first_values[index];
    let diagonal = (a.x + a.y) + (a.z + a.w);
    let product = (a.x * (center - first_values[index - 1]) + a.y * (center - first_values[index + 1])) +
                  (a.z * (center - first_values[index - 20]) + a.w * (center - first_values[index + 20]));
    let value = tile_rhs[index] - product;
    var result = center;
    if (diagonal > 0.0) { result += params.vectors[0].x * value / diagonal; }
    write0(p, vec4f(result));
}
)wgsl";

}
