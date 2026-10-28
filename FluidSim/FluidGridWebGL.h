#pragma once
#include "FluidGridTypes.h"
#include "FluidGridEdit.h"
#include "ElementCatalogue.h"
#include "FluidMixture.h"
#include <bitloop/core/types.h>
#include <cstdint>
#include <memory>

namespace fluid
{

class FluidGridWebGL
{
public:
    using Scalar = float;
    using State = FluidGridState<Scalar>;
    using ProjectionResult = FluidProjectionResult;
    using Edit = FluidGridEdit<FluidGridWebGL>;
    static constexpr bool requires_render_thread = true;

    int width;
    int height;
    double cell_size;
    bl::DVec2 origin = {};
    ElementCatalogue elements;
    FluidMixture liquid_mixture;
    FluidMixture gas_mixture;
    double gravity_y = 100.0;

    enum class PressurePreconditioner { Diagonal, Multigrid };
    PressurePreconditioner pressure_preconditioner = PressurePreconditioner::Multigrid;

    // CPU coarse-pressure cutoff in cells, set before upload(); zero keeps every level on the GPU.
    int cpu_pressure_cells = 1024;

    FluidGridWebGL(int width, int height, double cell_size);
    ~FluidGridWebGL();
    FluidGridWebGL(const FluidGridWebGL&) = delete;
    FluidGridWebGL& operator=(const FluidGridWebGL&) = delete;

    // configure the catalogue and mixtures, then upload per-cell masses before stepping
    // empty fields start at zero; mixture membership/order stays fixed until the next upload
    // GPU operations and destruction require the owning GL context; construction itself needs none
    void upload(const State& state);
    State download() const;
    // Staging needs no GL context; commit requires the owning context after upload().
    Edit beginEdit() { return Edit(*this); }
    void destroy();
    std::uint32_t liquidFractionTexture() const;
    std::uint32_t colorTexture() const;

    ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-3);
    void advect(double dt);
    void applyGravity(double dt);
    ProjectionResult project(double dt, int max_iterations = 200, double divergence_tolerance = 1e-3);
    double maxDivergence() const;

    constexpr int uIndex(int x, int y) const { return y * (width + 1) + x; }
    constexpr int vIndex(int x, int y) const { return y * width + x; }

private:
    friend Edit;
    void commitEdits(detail::CellEditBatch& batch);

    struct Impl;
    std::unique_ptr<Impl> gpu;
};

}
