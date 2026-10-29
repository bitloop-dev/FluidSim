#pragma once
#include <string_view>
#include <bitloop/core/types.h>
#include <algorithm>
#include <array>
#include <cmath>

inline std::array<float, 4> mouseAttraction(bl::DVec2 position, const std::array<float, 4>& settings)
{
    const double dx = settings[0] - position.x;
    const double dy = settings[1] - position.y;
    const double distance = std::sqrt(dx * dx + dy * dy);
    const double scale = settings[2] / std::max(distance, double(settings[3]));
    return {float(scale * dx), float(scale * dy), 0.f, 0.f};
}

// Scene-owned example. Replace this formula to generate a different acceleration field.
inline constexpr std::string_view mouse_attraction_shader = R"wgsl(
fn fieldValue(position: vec2f) -> vec4f {
    let settings = fieldParameter(0u);
    let attractor_position = settings.xy;
    let strength = settings.z;
    let softening_radius = settings.w;

    let offset = attractor_position - position;
    let distance = length(offset);

    // Constant strength outside the radius, falling smoothly to zero at the target.
    return vec4f(strength * offset / max(distance, softening_radius), 0, 0);
}
)wgsl";
