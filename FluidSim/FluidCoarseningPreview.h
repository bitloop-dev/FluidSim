#pragma once
#include "FluidGrid.h"
#include <array>
#include <cstdint>

namespace fluid
{
class AmrGas;
struct GasLayout;
class FluidCellView;
// Merge/refinement assessment and layout proposals. Resolved neighbours can request finer
// cells, but cannot reveal detail already discarded. Never applies a layout.
class FluidCoarseningPreview
{
public:
    enum Label : std::uint8_t { Coarse, Unassessed, LiquidBuffer, Detail, Candidate, Refine, Approaching, LabelCount };
    struct Tolerances
    {
        double velocity = 0.1;          // world distance / second
        double pressure_gradient = 0.1; // pressure / world distance
        double density = 0.02;          // relative to group mean
        double ingredient = 0.02;       // absolute cell-volume fraction
        bool operator==(const Tolerances&) const = default;
    };
    struct Error
    {
        double velocity = 0, pressure_gradient = 0, density = 0, ingredient = 0;
        double score(const Tolerances& tolerance) const;
    };
    struct Stability
    {
        double quiet_ratio = 0.5;   // fraction of each detail tolerance, in (0,1]
        double quiet_seconds = 0.24;
        bool operator==(const Stability&) const = default;
    };
    struct Statistics
    {
        int raw_changes = 0, decision_changes = 0;
        int waiting = 0, held = 0;
        bool operator==(const Statistics&) const = default;
    };
    // Optional observation of the actual selection pass, before regrid. No
    // trace storage or extra error estimates are needed when the pointer is null.
    struct Trace
    {
        struct Cell {
            Error error;
            int observed_scale = 0;
            std::uint8_t source = Unassessed, raw = Unassessed, decision = Unassessed;
            bool evaluated = false, quiet = false, requested = false;
            double quiet_since = -1;
        };
        struct Level {
            int scale = 0, width = 0, height = 0;
            std::vector<Cell> cells;
            // Tile masks after request padding/quantization, then parent nesting.
            int tile = 0, tiles_x = 0, tiles_y = 0;
            std::vector<std::uint8_t> padded, nested;
        };
        std::vector<Level> levels;
        int fallback_scale = 0; // nonzero when broad coverage drops coarser levels
    };
    // Requires a gas-only 2x2 group. Material error includes actual loss to a flat
    // mean and neighbour-predicted variation; it is not an accumulated transport-error bound.
    static Error estimate(const FluidGrid& block, int x, int y);
    // Time is simulated seconds. Equal timestamps do not age decisions. Reset
    // after replacing state or a gap in observation; control changes reset history; changed resolution resets only affected groups.
    void update(const FluidGrid& grid, const Tolerances& tolerance, const Stability& stability, double time, double lookahead = 0.0);
#ifdef FLUIDSIM_HAS_AMR
    // Native material and velocity use the bounded regrid reconstruction;
    // independent face residuals and neighbours also protect resolved detail.
    static Error estimate(const AmrGas& grid, int x, int y, int scale = 2);
    void update(const AmrGas& grid, const Tolerances& tolerance, const Stability& stability, double time, double lookahead = 0.0, int maxCellScale = 4, Trace* trace = nullptr);
    // Buffer current detail and cluster decisions. Quiet retention keeps its
    // own cells without creating further padding; caller applies the layout.
    // When tracing, pass the same Trace filled by the preceding update.
    GasLayout proposeLayout(const AmrGas& grid, Trace* trace = nullptr) const;
#endif
    void reset();

    int width = 0, height = 0; // display labels, one per aligned 2x2 finest-cell group
    std::vector<std::uint8_t> labels;
    std::array<int, LabelCount> counts{};
    Statistics statistics;
    std::uint64_t revision = 0;

private:
    struct History
    {
        double quiet_since = -1;
        std::uint8_t decision = Unassessed, raw = Unassessed;
        bool quiet = false;
        bool coarse = false;
    };
    struct AssessmentLevel {
        int scale=2, width=0, height=0;
        std::vector<std::uint8_t> labels;
        std::vector<History> history;
        std::vector<int> swept;
        Statistics statistics;
    };
    std::vector<AssessmentLevel> assessments;
    // Exact previous resolution per 2x2 footprint, shared by every assessment size.
    std::vector<std::uint8_t> observed_scales;
    const void* observed_grid = nullptr;
    Tolerances observed_tolerances;
    Stability observed_stability;
    double observed_cell_size = 0, observed_lookahead = 0, last_time = 0;
    void updateView(const FluidCellView& grid, const Tolerances& tolerance, const Stability& stability, double time, double lookahead, int maxCellScale, Trace* trace = nullptr);
    static void stabilise(AssessmentLevel& level, const Stability& stability, double time, bool fresh);
    static void markApproaching(AssessmentLevel& level, const FluidCellView& grid, const Tolerances& tolerance, double lookahead);
};
}
