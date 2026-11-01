#pragma once
#include "Shared/Adaptive/FluidCoarseningPreview.h"
#include <memory>

namespace fluid
{
class FluidGridAdaptiveGPU;

// The same bounded native-cell assessment as FluidCoarseningPreview. Physical
// samples and quiet history remain on the GPU; the host clusters compact tiles.
class FluidCoarseningGPU
{
public:
    using Tolerances = FluidCoarseningPreview::Tolerances;
    using Stability = FluidCoarseningPreview::Stability;
    using Statistics = FluidCoarseningPreview::Statistics;
    using Trace = FluidCoarseningPreview::Trace;
    FluidCoarseningGPU();
    ~FluidCoarseningGPU();
    FluidCoarseningGPU(const FluidCoarseningGPU&) = delete;
    FluidCoarseningGPU& operator=(const FluidCoarseningGPU&) = delete;
    void update(const FluidGridAdaptiveGPU& grid, const Tolerances& tolerance, const Stability& stability, double time,
        double lookahead = 0.0, int maxCellScale = 4, Trace* trace = nullptr);
    AmrLayout proposeLayout(const FluidGridAdaptiveGPU& grid, Trace* trace = nullptr) const;
    void reset();

    // Optional render output, never read by assessment or layout construction.
    void downloadLabels();
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> labels;
    std::array<int, FluidCoarseningPreview::LabelCount> counts{};
    Statistics statistics;
    std::uint64_t revision = 0;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace fluid
