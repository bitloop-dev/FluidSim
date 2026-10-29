#pragma once
#include "FluidField.h"
#include "UniformWebGPU/FluidWebGPUCompute.h"

namespace fluid::detail
{
// Backend access stays private to the project; the public field API exposes no GPU handles.
struct FluidFieldAccess
{
    static webgpu::Compute& commands(const FluidWebGPUContext& context)
    {
        return *context.commands;
    }
    static const webgpu::Field& gpuField(const FluidField& field);
    static bool usesContext(const FluidField& field, const FluidWebGPUContext& context);
};
}
