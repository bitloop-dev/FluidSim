#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace fluid
{
struct AmrLayout;
class FluidCellView;

// Merge/refinement assessment and layout proposals. Resolved neighbours can request finer
// cells, but cannot reveal detail already discarded. Never applies a layout.
class FluidCoarseningPreview
{
public:
    enum Label : std::uint8_t
    {
        Coarse,
        Unassessed,
        LiquidBuffer,
        Detail,
        Candidate,
        Refine,
        Approaching,
        LabelCount
    };

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
        double velocity = 0;
        double pressure_gradient = 0;
        double density = 0;
        double ingredient = 0;
        double score(const Tolerances& tolerance) const;
    };

    struct Stability
    {
        double quiet_ratio = 0.5; // fraction of each detail tolerance, in (0,1]
        double quiet_seconds = 0.24;
        bool operator==(const Stability&) const = default;
    };

    struct Statistics
    {
        int raw_changes = 0;
        int decision_changes = 0;
        int waiting = 0;
        int held = 0;
        bool operator==(const Statistics&) const = default;
    };

    struct Requests
    {
        int scale = 2;
        std::vector<std::uint8_t> tiles; // padded requests, tile width=max(16,scale)
    };

    // Optional observation of the actual selection pass, before regrid. No
    // trace storage or extra error estimates are needed when the pointer is null.
    struct Trace
    {
        struct Cell
        {
            Error error;
            int observed_scale = 0;
            std::uint8_t source = Unassessed;
            std::uint8_t raw = Unassessed;
            std::uint8_t decision = Unassessed;
            bool evaluated = false;
            bool quiet = false;
            bool requested = false;
            double quiet_since = -1;
        };

        struct Level
        {
            int scale = 0;
            int width = 0;
            int height = 0;
            std::vector<Cell> cells;

            // Tile masks after request padding/quantization, then parent nesting.
            int tile = 0;
            int tiles_x = 0;
            int tiles_y = 0;
            std::vector<std::uint8_t> padded;
            std::vector<std::uint8_t> nested;
        };

        std::vector<Level> levels;
        int fallback_scale = 0; // nonzero when broad coverage drops coarser levels
    };

    // Requires a gas-only group. Uniform previews retain the 2x2 flat-mean
    // contract; native material and momentum use bounded regrid reconstruction.
    static Error estimate(const FluidCellView& grid, int x, int y, int scale = 2);

    // Time is simulated seconds. Equal timestamps do not age decisions. Reset
    // after replacing state or a gap in observation; control changes reset history;
    // changed resolution resets only affected groups. Uniform previews use scale 2.
    void update(const FluidCellView& grid, const Tolerances& tolerance, const Stability& stability, double time,
        double lookahead = 0.0, int maxCellScale = 4, Trace* trace = nullptr);

    // Buffer current detail and cluster decisions. Quiet retention keeps its
    // own cells without creating further padding; caller applies the layout.
    // When tracing, pass the same Trace filled by the preceding update.
    AmrLayout proposeLayout(int finestWidth, int finestHeight, Trace* trace = nullptr) const;
    static AmrLayout clusterRequests(
        int width, int height, const std::vector<Requests>& requests, Trace* trace = nullptr);
    void reset();

    int width = 0;
    int height = 0; // display labels, one per aligned 2x2 finest-cell group
    std::vector<std::uint8_t> labels;
    std::array<int, LabelCount> counts{};
    Statistics statistics;
    std::uint64_t revision = 0;

private:
    struct History
    {
        double quiet_since = -1;
        std::uint8_t decision = Unassessed;
        std::uint8_t raw = Unassessed;
        bool quiet = false;
        bool coarse = false;
    };

    struct AssessmentLevel
    {
        int scale = 2;
        int width = 0;
        int height = 0;
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
    double observed_cell_size = 0;
    double observed_lookahead = 0;
    double last_time = 0;

    static void stabilise(AssessmentLevel& level, const Stability& stability, double time, bool fresh);
    static void markApproaching(
        AssessmentLevel& level, const FluidCellView& grid, const Tolerances& tolerance, double lookahead);
};
}
