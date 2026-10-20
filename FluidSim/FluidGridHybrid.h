#pragma once
#include "FluidGridGPU.h"
#include "PressureMultigrid.h"

namespace fluid
{

// Opt-in experiment: GPU transport/fine pressure, CPU whole-domain coarse correction.
class FluidGridHybrid
{
public:
    using Scalar = FluidGridGPU::Scalar;
    using State = FluidGridGPU::State;
    using ProjectionResult = FluidGridGPU::ProjectionResult;
    using PressurePreconditioner = FluidGridGPU::PressurePreconditioner;
    using Edit = FluidGridEdit<FluidGridHybrid>;
    static constexpr bool requires_render_thread = true;

    int width, height;
    double cell_size;
    bl::DVec2 origin = {};
    ElementCatalogue elements;
    FluidMixture liquid_mixture, gas_mixture;
    double gravity_y = 100.0;
    PressurePreconditioner pressure_preconditioner = PressurePreconditioner::Multigrid;

    // Set before upload(): zero keeps the original GPU path; positive cutoffs are at least 16.
    int cpu_pressure_cells = 1024;

    FluidGridHybrid(int width, int height, double cell_size);
    void upload(const State& state);
    State download() const { return gpu.download(); }
    Edit beginEdit() { return Edit(*this); }
    void destroy();
    ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-3);
    void advect(double dt);
    void applyGravity(double dt);
    ProjectionResult project(double dt, int max_iterations = 200, double divergence_tolerance = 1e-3);
    double maxDivergence() const { return gpu.maxDivergence(); }
    std::uint32_t liquidFractionTexture() const { return gpu.liquidFractionTexture(); }
    std::uint32_t colorTexture() const { return gpu.colorTexture(); }
    constexpr int uIndex(int x, int y) const { return y * (width + 1) + x; }
    constexpr int vIndex(int x, int y) const { return y * width + x; }

private:
    friend Edit;
    std::unique_ptr<PressureMultigrid> cpu;
    FluidGridGPU gpu;
    void configure();
    void commitEdits(detail::CellEditBatch& batch);
};

}
