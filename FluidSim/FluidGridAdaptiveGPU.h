#pragma once
#include "FluidGridAdaptive.h"
#include "FluidWebGPUCompute.h"
#include "FluidMaterialPixels.h"

namespace fluid {

// Conserved adaptive state lives only in these native-resolution GPU buffers.
class FluidGridAdaptiveGPU {
public:
    using State = FluidGridAdaptive::State;
    struct Level {
        AmrGeometry2D geometry;
        std::vector<AmrBox2D> boxes;
        AmrFieldLayout2D qLayout, cellLayout, pressureLayout;
        std::array<AmrFieldLayout2D, 2> faceLayout;
        webgpu::Field conserved, pressure, active; // Pressure stores high/low planes separated by width*height.
        std::array<webgpu::Field, 2> velocity;
        struct Storage;
        std::unique_ptr<Storage> storage;
        Level();
        ~Level();
    };
    FluidGridAdaptiveGPU(int width, int height, double cellSize, ElementCatalogue catalogue, FluidMixture gasMixture, GasLayout layout, int maxGridSize = 64);
    FluidGridAdaptiveGPU(int width, int height, double cellSize, ElementCatalogue catalogue, FluidMixture liquidMixture, FluidMixture gasMixture, GasLayout layout,
              int maxGridSize = 64);
    ~FluidGridAdaptiveGPU();
    FluidGridAdaptiveGPU(const FluidGridAdaptiveGPU&) = delete;
    FluidGridAdaptiveGPU& operator=(const FluidGridAdaptiveGPU&) = delete;
    void upload(const State& state, double tolerance = 1.e-4);
    bool regrid(GasLayout layout);
    // Returns completed work; optional stage timings add a transport completion wait.
    // Without measureTimings, transportMs/projectionMs are -1 for a nonzero step.
    GasStepResult step(double dt, double tolerance, double damping = 0., std::array<double, 2> acceleration = {}, bool measureTimings = false);
    FluidProjectionResult project(double tolerance, double dt = 0., std::array<double, 2> acceleration = {});
    void transport(double dt);
    double stableTimeStep() const;
    double maxDivergence();
    GasTotals totals() const;
    std::vector<float> colorPixels() const;
    FluidMaterialPixels materialPixels() const;
    void synchronize() const;
    webgpu::Compute& commands() const;
    webgpu::Compute& compute() const { return commands(); }
    const std::vector<std::unique_ptr<Level>>& levels() const;
    const ElementCatalogue& elements() const;
    const FluidMixture& gasMixture() const;
    const FluidMixture& materialMixture() const;
    int liquidIngredientCount() const;
    int ingredientCount() const;
    const GasLayout& layout() const;
    long long activeCellCount() const;
    long long allocatedCellCount() const;
    int finestWidth() const;
    int finestHeight() const;
    double finestCellSize() const;
    double pressureInterval() const;
    std::array<double, 2> acceleration() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace fluid
