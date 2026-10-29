#pragma once
#include "FluidGridTypes.h"
#include "FluidGridEdit.h"
#include "ElementCatalogue.h"
#include "FluidMixture.h"
#include "FluidCheckpoint.h"
#include "FluidField.h"
#include <bitloop/core/types.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <array>
#include <span>
#include <string_view>

namespace fluid
{

class FluidGridGPU
{
public:
    using Scalar = float;
    using State = FluidGridState<Scalar>;
    using ProjectionResult = FluidProjectionResult;
    using Edit = FluidGridEdit<FluidGridGPU>;
    static constexpr bool requires_render_thread = false;

    int width;
    int height;
    double cell_size;
    bl::DVec2 origin = {};
    ElementCatalogue elements;
    FluidMixture liquid_mixture;
    FluidMixture gas_mixture;
    double gravity_y = 100.0;
    double gas_damping_rate = 0.0;
    double getSurfaceTension() const { return surface_tension; }
    void setSurfaceTension(double coefficient);
    double capillaryStepLimit() const;
    void setAccelerationField(const FluidField& field);
    void clearAccelerationField();
    void setWebGPUContext(const FluidWebGPUContext& context);
    FluidWebGPUContext getWebGPUContext();

    enum class PressurePreconditioner { Diagonal, Multigrid };
    PressurePreconditioner pressure_preconditioner = PressurePreconditioner::Multigrid;

    FluidGridGPU(int width, int height, double cell_size);
    ~FluidGridGPU();
    FluidGridGPU(const FluidGridGPU&) = delete;
    FluidGridGPU& operator=(const FluidGridGPU&) = delete;

    // configure the catalogue and mixtures, then upload per-cell masses before stepping
    // empty fields start at zero; mixture membership/order stays fixed until the next upload
    // Upload, stepping, readback and destruction stay on the device's owning thread.
    void upload(const State& state);
    State download() const;
    bool saveCheckpoint(FluidCheckpoint& checkpoint) const;
    bool loadCheckpoint(const FluidCheckpoint& checkpoint);
    // Staging needs no device; commit requires the owning thread after upload().
    Edit beginEdit() { return Edit(*this); }
    void destroy();

    // Optional completed-work stage timings add GPU synchronization.
    ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-3, FluidStepTimings* timings = nullptr);
    void advect(double dt);
    void dampGasVelocity(double dt);
    void applyGravity(double dt);
    ProjectionResult project(double dt, int max_iterations = 200, double divergence_tolerance = 1e-3);
    double maxDivergence() const;

    std::vector<float> colorPixels() const;
    void writeColorPixels(std::vector<float>& pixels) const;
    void synchronize() const;

    constexpr int uIndex(int x, int y) const { return y * (width + 1) + x; }
    constexpr int vIndex(int x, int y) const { return y * width + x; }

private:
    friend Edit;
    friend class FluidGridWebGPU;
    void commitEdits(detail::CellEditBatch& batch);

    struct Impl;
    std::optional<FluidWebGPUContext> context;
    std::unique_ptr<Impl> gpu;
    double surface_tension = 0.;
};

}
