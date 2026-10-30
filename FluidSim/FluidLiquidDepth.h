#pragma once
#include "FluidGridGPUInternal.h"
#include <bitloop/core/debug.h>
#include <bitloop/nanovgx/nano_shader_surface.h>
#include <algorithm>
#include <array>
#include <string>

namespace fluid {

// Render-only approximate distance to gas. Jump flooding propagates the nearest
// gas-cell centre across the image in logarithmically many fullscreen passes.
// RGBA8 packs two 16-bit coordinates, so this also works without float render
// targets in WebGL2. All GL work, including destroy(), stays on the render thread.
class FluidLiquidDepth {
public:
    FluidLiquidDepth() {
        const std::string common=R"(#version 300 es
            precision highp float;
            precision highp int;
            uniform vec2 u_grid_size;
            in vec2 v_uv;
            out vec4 outColor;
            vec4 encodeSeed(vec2 point) {
                uvec2 q=uvec2(round(point/u_grid_size*65534.0));
                return vec4(q.x&255u,q.x>>8u,q.y&255u,q.y>>8u)/255.0;
            }
            vec2 decodeSeed(vec4 encoded) {
                vec4 bytes=round(encoded*255.0);
                return vec2(bytes.x+256.0*bytes.y,bytes.z+256.0*bytes.w)/65534.0*u_grid_size;
            }
        )";
        seed.setFragmentSource(common+R"(
            uniform sampler2D u_liquid;
            void main() {
                ivec2 cell=ivec2(gl_FragCoord.xy);
                // The 50% phase contour is an appearance estimate. Tiny gas
                // roundoff in a full liquid cell must not illuminate its bulk.
                outColor=texelFetch(u_liquid,cell,0).r<0.5
                    ? encodeSeed(vec2(cell)+0.5) : vec4(1.0);
            }
        )");
        const auto propagation=common+R"(
            uniform sampler2D u_seeds;
            uniform float u_jump;
            void main() {
                ivec2 cell=ivec2(gl_FragCoord.xy);
                int step=int(u_jump);
                float best=1e30;
                vec4 nearest=vec4(1.0);
                for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
                    ivec2 q=cell+ivec2(x,y)*step;
                    // The domain walls are not gas sources.
                    if(any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,ivec2(u_grid_size)))) continue;
                    vec4 candidate=texelFetch(u_seeds,q,0);
                    if(all(equal(candidate,vec4(1.0)))) continue;
                    vec2 delta=decodeSeed(candidate)-(vec2(cell)+0.5);
                    float distanceSquared=dot(delta,delta);
                    if(distanceSquared<best) { best=distanceSquared;nearest=candidate; }
                }
                outColor=nearest;
            }
        )";
        for(auto& pass:jump) pass.setFragmentSource(propagation);
    }

    std::uint32_t texture(int width,int height,std::uint32_t liquid) const {
        // Blending, dithering and sRGB conversion would corrupt packed coordinates.
        gpu_detail::ComputeState guard;
        seed.render(width,height,[&](const bl::ShaderSurface& shader) {
            shader.bindTexture2D("u_liquid",liquid);
            shader.setUniform2f("u_grid_size",float(width),float(height));
        });
        auto input=seed.texture();
        int pass=0;
        auto propagate=[&](int step) {
            auto& output=jump[pass++%2];
            output.render(width,height,[&](const bl::ShaderSurface& shader) {
                shader.bindTexture2D("u_seeds",input);
                shader.setUniform2f("u_grid_size",float(width),float(height));
                shader.setUniform1f("u_jump",float(step));
            });
            input=output.texture();
        };
        int step=1;
        while(step<std::max(width,height)) step*=2;
        for(step/=2;step>0;step/=2) propagate(step);
        propagate(1); // One local refinement reduces jump-flood approximation error.
        return input;
    }

    void destroy() {
        seed.destroy();
        for(auto& pass:jump) pass.destroy();
    }

private:
    mutable bl::ShaderSurface seed;
    mutable std::array<bl::ShaderSurface,2> jump;
};

} // namespace fluid
