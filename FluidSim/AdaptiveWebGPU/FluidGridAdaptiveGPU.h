#pragma once
#include "../Adaptive/FluidGridAdaptive.h"
#include "../FluidGridEdit.h"
#include "../UniformWebGPU/FluidWebGPUCompute.h"

namespace fluid {

struct FluidAdaptiveGPUProjectionFailure : std::runtime_error {
    int iterations;
    FluidAdaptiveGPUProjectionFailure(int count, const std::string& message)
        : std::runtime_error(message), iterations(count) {}
};

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
    FluidGridAdaptiveGPU(int width, int height, double cellSize, SubstanceCatalogue catalogue, FluidMixture gasMixture, GasLayout layout, int maxGridSize = 64);
    FluidGridAdaptiveGPU(int width, int height, double cellSize, SubstanceCatalogue catalogue, FluidMixture liquidMixture, FluidMixture gasMixture, GasLayout layout,
              int maxGridSize = 64);
    ~FluidGridAdaptiveGPU();
    FluidGridAdaptiveGPU(const FluidGridAdaptiveGPU&) = delete;
    FluidGridAdaptiveGPU& operator=(const FluidGridAdaptiveGPU&) = delete;
    void upload(const State& state, double tolerance = 1.e-4);
    bool regrid(GasLayout layout);
    // Returns completed work. Optional stage timings add GPU completion waits;
    // transportMs/projectionMs are -1 without measurement for a nonzero step.
    GasStepResult step(double dt, double tolerance, double damping = 0., std::array<double, 2> acceleration = {}, int maxIterations = -1, bool measureTimings = false);
    FluidProjectionResult project(double tolerance, double dt = 0., std::array<double, 2> acceleration = {}, int maxIterations = -1);
    void transport(double dt);
    double stableTimeStep() const;
    double maxDivergence();
    GasTotals totals() const;
    std::vector<float> colorPixels() const;
    void writeColorPixels(std::vector<float>& pixels, bool liquid_fraction_alpha = false) const;
    // Common adapter supplies validated, unique edits at finest native cells.
    void replaceMaterials(std::span<const detail::ResolvedCellEdit> cells);
    void synchronize() const;
    webgpu::Compute& commands() const;
    webgpu::Compute& compute() const { return commands(); }
    const std::vector<std::unique_ptr<Level>>& levels() const;
    const SubstanceCatalogue& substances() const;
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
