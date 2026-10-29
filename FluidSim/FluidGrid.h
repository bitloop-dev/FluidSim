#pragma once
#include "FluidGridBase.h"
#include "FluidAdaptiveSettings.h"
#include <memory>

namespace fluid
{

enum class FluidDevice { CPU, GPU };
enum class AdaptiveMode { DISABLED = 0, MAX_2 = 2, MAX_4 = 4, MAX_8 = 8, MAX_16 = 16, MAX_32 = 32 };

class FluidGridAdaptiveCPU;
class FluidGridAdaptiveWebGPU;

// Public API over one selected backend. Configure mode before initialization.
class FluidGrid final : public FluidGridBase
{
public:
    FluidGrid(int width, int height, double cell_size);
    ~FluidGrid() override;
    FluidGrid(const FluidGrid&) = delete;
    FluidGrid& operator=(const FluidGrid&) = delete;
    FluidGrid(FluidGrid&&) noexcept = default;
    FluidGrid& operator=(FluidGrid&&) noexcept = default;

    void setDevice(FluidDevice device);
    FluidDevice getDevice() const { return device; }
    void setAdaptiveMode(AdaptiveMode mode);
    AdaptiveMode getAdaptiveMode() const { return adaptive_mode; }
    void setAdaptiveSettings(const FluidAdaptiveSettings& settings);
    const FluidAdaptiveSettings& getAdaptiveSettings() const { return adaptive_settings; }

    int getWidth() const override;
    int getHeight() const override;
    double getCellSize() const override;
    bl::DVec2 getOrigin() const override;
    void setOrigin(bl::DVec2 origin) override;
    bool requiresRenderThread() const override;
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
    void selectBackend(FluidDevice device, AdaptiveMode mode);
    void commitEdits(detail::CellEditBatch& batch) override;

    std::unique_ptr<FluidGridBase> backend;
    FluidGridAdaptiveCPU* adaptive_backend = nullptr; // Non-owning configuration dispatch.
    FluidGridAdaptiveWebGPU* adaptive_webgpu_backend = nullptr;

    FluidDevice device = FluidDevice::CPU;
    AdaptiveMode adaptive_mode = AdaptiveMode::DISABLED;
    FluidAdaptiveSettings adaptive_settings;
    bool initialized = false;
};

}
