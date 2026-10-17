#pragma once
#include "FluidGridTypes.h"
#include <bitloop/core/types.h>
#include <cstdint>
#include <memory>

namespace fluid
{

class FluidGridGPU
{
public:
    using Scalar = float;
    using State = FluidGridState<Scalar>;
    using ProjectionResult = FluidProjectionResult;
    static constexpr bool requires_render_thread = true;

    int width;
    int height;
    double cell_size;
    bl::DVec2 origin = {};
    double air_density = 0.1;
    double water_density = 100.0;
    double gravity_y = 100.0;

    enum class PressurePreconditioner { Diagonal, Multigrid };
    PressurePreconditioner pressure_preconditioner = PressurePreconditioner::Multigrid;

    FluidGridGPU(int width, int height, double cell_size);
    ~FluidGridGPU();
    FluidGridGPU(const FluidGridGPU&) = delete;
    FluidGridGPU& operator=(const FluidGridGPU&) = delete;

    // call upload once before stepping; empty fields start at zero
    // GPU operations and destruction require the owning GL context; construction itself needs none
    void upload(const State& state);
    State download() const;
    void destroy();
    std::uint32_t waterTexture() const;

    ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-3);
    void advect(double dt);
    void applyGravity(double dt);
    ProjectionResult project(double dt, int max_iterations = 200, double divergence_tolerance = 1e-3);
    double maxDivergence() const;

    constexpr int uIndex(int x, int y) const { return y * (width + 1) + x; }
    constexpr int vIndex(int x, int y) const { return y * width + x; }

private:
    struct Impl;
    std::unique_ptr<Impl> gpu;
};

}
