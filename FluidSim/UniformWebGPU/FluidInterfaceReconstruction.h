#pragma once
#include "Shared/WebGPU/FluidWebGPUShaders.h"

namespace fluid::webgpu
{

// Same MYC reconstruction and wall-fragment fallback for any selected occupancy field.
inline constexpr std::string_view interface_reconstruction = R"wgsl(
    fn blockedFragment(p: vec2i, normal: vec2f, offset: f32) -> bool
    {
        let flow = vec4f(-u(p).x, u(p + vec2i(1, 0)).x, -v(p).x, v(p + vec2i(0, 1)).x);
        var outflow = 0.0;
        for (var face = 0; face < 4; face++)
        {
            let strip = max(flow[face], 0.0) * timeStep() / cellSize();

            if (strip == 0.0)
            {
                continue;
            }

            outflow += strip;
            let axis = select(vec2f(0, 1), vec2f(1, 0), face < 2);
            var lo = vec2f(0);
            var hi = vec2f(1);
            if ((face % 2) == 1)
            {
                lo = axis * (1.0 - strip);
            }
            else
            {
                hi -= axis * (1.0 - strip);
            }

            if (plicArea(normal, offset, lo, hi) > 0.0)
            {
                return false;
            }
        }

        return outflow > 0.0;
    }

    fn reconstructInterface(p: vec2i) -> vec4f
    {
        let f = liquid(p).x;

        if (f <= 0.0 || f >= 1.0)
        {
            return vec4f(0);
        }

        var samples: array<f32, 9>;
        var low = 1.0;
        var high = 0.0;

        for (var y = 0; y < 3; y++)
        {
            for (var x = 0; x < 3; x++)
            {
                let sample = liquid(p + vec2i(x - 1, y - 1)).x;
                samples[y * 3 + x] = sample;
                low = min(low, sample);
                high = max(high, sample);
            }
        }

        let wall = p.x == 0 || p.y == 0 || p.x + 1 == gridSize().x || p.y + 1 == gridSize().y;

        // At a wall the clamped stencil can repeatedly pin an unresolved fragment
        // against the solid. A resolved interface tip may also lack a pure interior
        // in the MYC stencil, so check one surrounding ring before falling back.
        if (wall && (low > 1e-6 || high < 1.0 - 1e-6))
        {
            for (var y = -2; y <= 2; y++)
            {
                for (var x = -2; x <= 2; x++)
                {
                    if (abs(x) != 2 && abs(y) != 2)
                    {
                        continue;
                    }

                    let sample = liquid(p + vec2i(x, y)).x;
                    low = min(low, sample);
                    high = max(high, sample);
                }
            }
        }

        // One phase has a resolved interior and the other does not. Uniform mixed
        // regions are not fragments. Ignore roundoff in otherwise pure cells.
        let normal = plicNormal(samples);
        let offset = plicOffset(normal, f);
        var fallback = false;
        if (wall && ((low <= 1e-6) != (high >= 1.0 - 1e-6)))
        {
            // Only replace geometry that cannot export its unresolved phase through
            // any outgoing face. Freely travelling small features retain PLIC.
            let phase_sign = select(-1.0, 1.0, low <= 1e-6);
            fallback = blockedFragment(p, phase_sign * normal, phase_sign * offset);
        }

        return vec4f(normal, offset, select(0.0, 1.0, fallback));
    }
)wgsl";

}
