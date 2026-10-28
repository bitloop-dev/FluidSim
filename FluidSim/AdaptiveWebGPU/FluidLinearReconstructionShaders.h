#pragma once
#include <string_view>

namespace fluid::webgpu {

// Mirrors FluidLinearReconstruction.h; each caller supplies reconstructionSample(stencil,n,side).
inline constexpr std::string_view linearReconstruction=R"wgsl(
fn materialLimiter(stencil:vec3i,count:i32)->vec2f {
    var limited=vec2f(1);
    for(var n=0;n<count;n++) {
        let c=reconstructionSample(stencil,n,0);
        limited=min(limited,vec2f(slopeFactor(c,reconstructionSample(stencil,n,1),reconstructionSample(stencil,n,2)),
                                 slopeFactor(c,reconstructionSample(stencil,n,3),reconstructionSample(stencil,n,4))));
    }
    var corner=1.0;
    for(var n=0;n<count;n++) {
        corner=min(corner,cornerFactor(reconstructionSample(stencil,n,0),
            vec4f(reconstructionSample(stencil,n,1),reconstructionSample(stencil,n,2),reconstructionSample(stencil,n,3),reconstructionSample(stencil,n,4)),limited));
    }
    return limited*corner;
}
fn materialSlope(stencil:vec3i,n:i32,limited:vec2f)->vec2f {
    return .5*vec2f(reconstructionSample(stencil,n,2)-reconstructionSample(stencil,n,1),
                    reconstructionSample(stencil,n,4)-reconstructionSample(stencil,n,3))*limited;
}
fn momentumSlopes(stencil:vec3i,ingredients:i32,limited:vec2f)->mat2x2f {
    var rho:array<f32,5>;
    for(var n=0;n<ingredients;n++) { for(var side=0;side<5;side++) { rho[side]+=reconstructionSample(stencil,n,side); } }
    let densitySlope=.5*vec2f(rho[2]-rho[1],rho[4]-rho[3])*limited;
    var result:mat2x2f;
    for(var d=0;d<2;d++) {
        let velocity=reconstructionSample(stencil,ingredients+d,0)/rho[0];
        var low=velocity;var high=velocity;var residual:array<f32,5>;
        for(var side=1;side<5;side++) {
            let momentum=reconstructionSample(stencil,ingredients+d,side);
            residual[side]=momentum-velocity*rho[side];
            low=min(low,momentum/rho[side]);high=max(high,momentum/rho[side]);
        }
        var factor=vec2f(slopeFactor(0.,residual[1],residual[2]),slopeFactor(0.,residual[3],residual[4]));
        factor*=cornerFactor(0.,vec4f(residual[1],residual[2],residual[3],residual[4]),factor);
        let slope=.5*vec2f(residual[2]-residual[1],residual[4]-residual[3])*factor;
        var bound=1.0;
        // Keep velocity within neighbour extrema, with no feedback into material slopes.
        for(var y=0;y<2;y++) { for(var x=0;x<2;x++) {
            let delta=vec2f(f32(x),f32(y))-.5;
            let density=rho[0]+dot(delta,densitySlope);let excursion=dot(delta,slope);
            if(excursion>0.) { bound=min(bound,(high-velocity)*density/excursion); }
            else if(excursion<0.) { bound=min(bound,(low-velocity)*density/excursion); }
        } }
        result[d]=velocity*densitySlope+bound*slope;
    }
    return result;
}
)wgsl";
} // namespace fluid::webgpu
