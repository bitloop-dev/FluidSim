#pragma once
#include "FluidWebGPUCompute.h"

namespace fluid::webgpu
{

struct TransportFusionKernels
{
    Kernel fraction_x, fraction_y;

    void build(Compute& compute, std::string_view component_helpers)
    {
        const std::string fraction_helper = R"wgsl(
fn nextFraction(p: vec2i) -> f32 {
    let axis = select(vec2i(0, 1), vec2i(1, 0), mode() == 0);
    let weight = weights(p).x;
    let before = transfers(p).x - weight * (flow(p).x * timeStep() / cellSize());
    let after = transfers(p + axis).x - weight * (flow(p + axis).x * timeStep() / cellSize());
    return liquid(p).x + (before - after);
}
)wgsl";
        const std::string fraction_helpers = std::string(component_helpers) + fraction_helper;
        // Keep adjacent row-major cells together in each 64-thread workgroup.
        compute.build(fraction_x, "fluid ingredients and fraction X", 3,
            {"liquid", "masses", "transfers", "flow", "weights"}, R"wgsl(
fn execute() {
    let p = cell();
    let f = liquid(p).x;
    let weight = masses(p) * select(params.vectors[1], params.vectors[0], f > 0.5) /
        select(1.0 - f, f, f > 0.5);
    write0(p, transport(p, weight));
    write1(p, weight);
    write2(p, vec4f(nextFraction(p)));
}
)wgsl", fraction_helpers, false, {32, 2});
        compute.build(fraction_y, "fluid ingredients and fraction Y", 2,
            {"liquid", "masses", "transfers", "flow", "corrections", "weights"}, R"wgsl(
fn execute() {
    let p = cell();
    write0(p, transport(p, corrections(p)));
    write1(p, vec4f(nextFraction(p)));
}
)wgsl", fraction_helpers, false, {32, 2});

    }
};

} // namespace fluid::webgpu
