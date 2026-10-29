#pragma once
#include "FluidGridBase.h"
#include <memory>

namespace fluid
{
class FluidGridGPU;

// Common API over the native float uniform WebGPU engine. Device operations,
// including colour observation and destruction, stay on its simulation owner.
class FluidGridWebGPU final : public FluidGridBase
{
public:
    FluidGridWebGPU(int width, int height, double cell_size);
    ~FluidGridWebGPU() override;
    int getWidth() const override;
    int getHeight() const override;
    double getCellSize() const override;
    bl::DVec2 getOrigin() const override;
    void setOrigin(bl::DVec2 origin) override;
    bool requiresRenderThread() const override { return false; }
    void configureMaterials(ElementCatalogue elements, FluidMixture liquid, FluidMixture gas) override;
    const ElementCatalogue& getElements() const override;
    const FluidMixture& getLiquidMixture() const override;
    const FluidMixture& getGasMixture() const override;
    double getGravityY() const override;
    void setGravityY(double acceleration) override;
    double getSurfaceTension() const override;
    void setSurfaceTension(double coefficient) override;
    void setAccelerationField(const FluidField& field) override;
    void clearAccelerationField() override;
    void setWebGPUContext(const FluidWebGPUContext& context) override;
    FluidWebGPUContext getWebGPUContext() override;
    void upload(const State& state) override;
    State download() const override;
    bool saveCheckpoint(FluidCheckpoint& checkpoint) const override;
    bool loadCheckpoint(const FluidCheckpoint& checkpoint) override;
    void destroy() override;
    ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6, StepTimings* timings = nullptr) override;
    double maxDivergence() const override;
    void writeColorPixels(std::vector<float>& pixels) const override;

private:
    void requireUploaded() const;
    void commitEdits(detail::CellEditBatch& batch) override;
    std::unique_ptr<FluidGridGPU> engine;
};
}
