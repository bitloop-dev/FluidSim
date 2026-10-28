#pragma once
#include <string_view>

namespace fluid::webgpu {

// Shared MYC/PLIC geometry; the CPU reference is FluidInterfaceGeometry.h.
inline constexpr std::string_view geometry = R"wgsl(
fn plicNormal(s: array<f32, 9>) -> vec2f {
    let left = s[0] + s[3] + s[6];
    let right = s[2] + s[5] + s[8];
    let top = s[0] + s[1] + s[2];
    let bottom = s[6] + s[7] + s[8];
    let centered = 0.5 * vec2f(left - right, top - bottom);
    let youngs = vec2f(left + s[3] - right - s[5], top + s[1] - bottom - s[7]);
    var normal = centered;
    if (all(centered == vec2f(0))) {
        normal = youngs;
    } else if (abs(centered.x) <= abs(centered.y)) {
        normal.y = select(-1.0, 1.0, centered.y > 0.0);
        if (abs(youngs.x) > abs(centered.x * youngs.y)) { normal = youngs; }
    } else {
        normal.x = select(-1.0, 1.0, centered.x > 0.0);
        if (abs(youngs.y) > abs(centered.y * youngs.x)) { normal = youngs; }
    }
    let magnitude = max(abs(normal.x), abs(normal.y));
    if (magnitude == 0.0) { return vec2f(1, 0); }
    return normal / magnitude;
}

fn plicOffset(normal: vec2f, fraction: f32) -> f32 {
    let a = min(abs(normal.x), abs(normal.y));
    let b = max(abs(normal.x), abs(normal.y));
    let triangle = 0.5 * a / b;
    var cut: f32;
    if (fraction < triangle) {
        cut = sqrt(2.0 * a * b * fraction);
    } else if (fraction <= 1.0 - triangle) {
        cut = b * fraction + 0.5 * a;
    } else {
        cut = a + b - sqrt(2.0 * a * b * (1.0 - fraction));
    }
    return min(normal.x, 0.0) + min(normal.y, 0.0) + cut;
}

fn plicArea(normal: vec2f, offset: f32, lo: vec2f, hi: vec2f) -> f32 {
    let size = hi - lo;
    let total = size.x * size.y;
    let cut = offset - dot(normal, select(hi, lo, normal >= vec2f(0)));
    let span = abs(normal) * size;
    let a = min(span.x, span.y);
    let b = max(span.x, span.y);
    if (cut <= 0.0) { return 0.0; }
    if (cut >= a + b) { return total; }
    if (a == 0.0) { return total * cut / b; }
    // Divide before multiplying so a vanishing triangle cannot underflow into 0/0.
    if (cut < a) { return total * 0.5 * (cut / a) * (cut / b); }
    if (cut < b) { return total * (cut - 0.5 * a) / b; }
    let dry = a + b - cut;
    return total * (1.0 - 0.5 * (dry / a) * (dry / b));
}

fn slopeFactor(center: f32, left: f32, right: f32) -> f32 {
    let a = center - left;
    let b = right - center;
    let slope = 0.5 * (right - left);
    if (slope == 0.0) { return 1.0; }
    if (a * b <= 0.0) { return 0.0; }
    return min(1.0, 2.0 * min(abs(a), abs(b)) / abs(slope));
}

fn cornerFactor(center: f32, neighbours: vec4f, limiter: vec2f) -> f32 {
    let excursion = 0.25 * dot(abs(neighbours.yw - neighbours.xz), limiter);
    if (excursion == 0.0) { return 1.0; }
    let low = min(center, min(min(neighbours.x, neighbours.y), min(neighbours.z, neighbours.w)));
    let high = max(center, max(max(neighbours.x, neighbours.y), max(neighbours.z, neighbours.w)));
    return min(1.0, min(center - low, high - center) / excursion);
}
)wgsl";

} // namespace fluid::webgpu
