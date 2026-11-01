#pragma once
#include "FluidMixture.h"
#include <bitloop/core/types.h>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace fluid
{

namespace detail
{
struct CellEdit
{
    int x;
    int y;
    Phase phase;
    SubstanceId substance;
};

struct ResolvedCellEdit
{
    int x;
    int y;
    size_t component;
    double mass;
};

struct CellEditBatch
{
    std::vector<CellEdit> changes;
    std::vector<ResolvedCellEdit> cells;
    std::optional<bl::DVec2> velocity;

    void prepare(int width, int height, double cell_volume, const FluidMixture& liquid, const FluidMixture& gas)
    {
        cells.clear();
        cells.reserve(changes.size());
        size_t component = 0;
        double mass = 0;

        for (size_t i = 0; i < changes.size(); ++i)
        {
            const auto& change = changes[i];

            // A brush can cross the grid edge; reject the entire batch before writing anything.
            if (change.x < 0 || change.x >= width || change.y < 0 || change.y >= height)
                throw std::out_of_range("Fluid grid edit cell is outside the grid");

            if (i == 0 || change.phase != changes[i - 1].phase || change.substance != changes[i - 1].substance)
            {
                const auto& mixture = change.phase == Phase::Liquid ? liquid : gas;
                const auto found = std::find_if(mixture.components.begin(), mixture.components.end(),
                    [&](const MixtureComponent& candidate)
                {
                    return candidate.substance == change.substance;
                });

                if (found == mixture.components.end())
                    throw std::invalid_argument("Fluid grid edit substance is not configured in the selected phase");

                component = size_t(found - mixture.components.begin()) +
                            (change.phase == Phase::Liquid ? 0 : liquid.components.size());
                mass = found->density * cell_volume;
            }

            cells.push_back({change.x, change.y, component, mass});
        }

        const auto before = [](const ResolvedCellEdit& a, const ResolvedCellEdit& b)
        {
            return a.y < b.y || (a.y == b.y && a.x < b.x);
        };

        // Raster-order drawing already has the order needed for sparse texture uploads.
        if (!std::is_sorted(cells.begin(), cells.end(), before))
            std::stable_sort(cells.begin(), cells.end(), before);

        size_t kept = 0;

        for (size_t first = 0; first < cells.size();)
        {
            size_t last = first + 1;
            while (last < cells.size() && cells[last].x == cells[first].x && cells[last].y == cells[first].y)
                ++last;
            cells[kept++] = cells[last - 1];
            first = last;
        }

        cells.resize(kept);
    }
};
}

// Staged material replacements, with no grid readback. The grid must outlive the session.
// Only commit() touches the grid and requires the GPU's owning graphics context.
template <class Grid> class FluidGridEdit
{
    friend Grid;
    Grid* grid;
    detail::CellEditBatch batch;

    explicit FluidGridEdit(Grid& grid) : grid(&grid) {}

public:
    FluidGridEdit(const FluidGridEdit&) = delete;
    FluidGridEdit& operator=(const FluidGridEdit&) = delete;
    FluidGridEdit(FluidGridEdit&&) = default;
    FluidGridEdit& operator=(FluidGridEdit&&) = default;

    void reserve(size_t count)
    {
        batch.changes.reserve(count);
        batch.cells.reserve(count);
    }

    size_t size() const { return batch.changes.size(); }

    bool empty() const { return batch.changes.empty(); }

    void clear()
    {
        batch.changes.clear();
        batch.cells.clear();
        batch.velocity.reset();
    }

    void setCell(int x, int y, Phase phase, SubstanceId substance)
    {
        batch.changes.push_back({x, y, phase, substance});
    }

    // The last write to a cell wins. Successful commits clear commands and retain capacity.
    // Dropping or clearing a session discards uncommitted edits; commit never runs implicitly.
    // Replaces existing cell volume; does not prescribe flow through a boundary.
    // Optional velocity replaces the edited material's momentum (uniform WebGPU).
    // Without it, face velocities are retained. step() projects supplied momentum before transport.
    void commit(std::optional<bl::DVec2> velocity = std::nullopt)
    {
        if (empty())
            return;

        batch.velocity = velocity;
        grid->commitEdits(batch);
        clear();
    }
};

}
