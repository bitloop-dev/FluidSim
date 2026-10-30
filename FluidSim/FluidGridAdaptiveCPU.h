#pragma once
#include "FluidGridBase.h"
#include "FluidAdaptiveSettings.h"
#include <memory>

namespace fluid
{
class FluidGridAdaptive;
class AmrGasRuntime;

// Common API over one native AMR hierarchy. No persistent finest-grid state.
class FluidGridAdaptiveCPU final : public FluidGridBase
{
public:
    FluidGridAdaptiveCPU(int width, int height, double cell_size, int max_cell_scale = 4,
                         FluidAdaptiveSettings settings = {});
    ~FluidGridAdaptiveCPU() override;
    int getWidth() const override { return width; }
    int getHeight() const override { return height; }
    double getCellSize() const override { return cell_size; }
    bl::DVec2 getOrigin() const override { return origin; }
    void setOrigin(bl::DVec2 value) override { origin = value; }
    bool requiresRenderThread() const override { return false; }
    void configureMaterials(ElementCatalogue elements, FluidMixture liquid, FluidMixture gas) override;
    const ElementCatalogue& getElements() const override { return elements; }
    const FluidMixture& getLiquidMixture() const override { return liquid; }
    const FluidMixture& getGasMixture() const override { return gas; }
    double getGravityY() const override { return gravity_y; }
    void setGravityY(double value) override { gravity_y = value; }
    void upload(const State& state) override;
    State download() const override;
    void destroy() override;
    ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6, StepTimings* timings = nullptr) override;
    double maxDivergence() const override;
    void writeColorPixels(std::vector<float>& pixels, bool liquid_fraction_alpha = false) const override;

    void setAdaptiveSettings(const FluidAdaptiveSettings& value);
    const FluidAdaptiveSettings& getAdaptiveSettings() const { return settings; }
    // Backend-specific inspection for learning/tests; observation never advances physics.
    const FluidGridAdaptive& native() const;
    double simulationTime() const { return simulation_time; }
    int refinementChecks() const { return refinement_checks; }

private:
    void commitEdits(detail::CellEditBatch& batch) override;
    void refineForEdits(const detail::CellEditBatch& batch);
    int width, height, max_cell_scale;
    double cell_size, gravity_y = 100.;
    bl::DVec2 origin{};
    ElementCatalogue elements;
    FluidMixture liquid, gas;
    FluidAdaptiveSettings settings;
    FluidCoarseningPreview selector;
    double simulation_time = 0., next_refinement_time = 0.;
    double projection_tolerance = 1e-6;
    int refinement_checks = 0;
    std::unique_ptr<AmrGasRuntime> runtime;
    std::unique_ptr<FluidGridAdaptive> engine;
};
}
