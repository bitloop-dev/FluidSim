#pragma once
#include "FluidWebGPUCompute.h"

namespace fluid::webgpu
{

// Average independently supplied cell accelerations onto the staggered velocity faces.
struct AccelerationKernels
{
    Kernel apply;

    explicit AccelerationKernels(Compute& compute)
    {
        compute.build(apply, "fluid apply acceleration", 1, {"velocity", "acceleration"}, R"wgsl(
fn execute() {
    let face = cell();
    let horizontal = mode() == 0;
    let axis = select(vec2i(0, 1), vec2i(1, 0), horizontal);
    let coordinate = select(face.y, face.x, horizontal);
    let extent = select(gridSize().y, gridSize().x, horizontal);

    if (coordinate == 0 || coordinate == extent) {
        write0(face, vec4f(0));
        return;
    }

    let average = 0.5 * (acceleration(face - axis).xy + acceleration(face).xy);
    let component = select(average.y, average.x, horizontal);
    let updated_velocity = velocity(face).x + timeStep() * component;
    write0(face, vec4f(updated_velocity));
}
)wgsl");
    }
};

}
