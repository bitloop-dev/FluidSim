#pragma once
#include "Shared/WebGPU/FluidWebGPUCompute.h"
#include <memory>

namespace fluid::webgpu
{

// Backward Euler for the full symmetric viscous stress on a free-slip MAC grid.
// Normal boundary velocities are zero, or prescribed by the caller when explicitly enabled.
// The caller supplies cell-centred (density, dynamic viscosity); no history is retained.
class ViscositySolverGPU
{
public:
    ViscositySolverGPU(Compute& compute, int width, int height);
    ~ViscositySolverGPU();

    // Returns dispatched CG slots, including unused slots in the final four-slot batch.
    // Negative budget runs to convergence; an explicit nonnegative budget is for bounded callers.
    int solve(Field& u, Field& v, const Field& material, double dt, double cell_size, double relative_tolerance = 1e-4,
        int max_iterations = -1, bool prescribed_boundary_velocity = false);
    double lastRelativeResidual() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}
