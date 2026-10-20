#pragma once
#include "FluidMixture.h"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fluid
{

enum class Phase { Liquid, Gas };

namespace detail
{
struct CellEdit
{
    int x, y;
    Phase phase;
    ElementId element;
};

struct ResolvedCellEdit
{
    int x, y;
    size_t component;
    double mass;
};

struct CellEditBatch
{
    std::vector<CellEdit> changes;
    std::vector<ResolvedCellEdit> cells;

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

            if (i == 0 || change.phase != changes[i - 1].phase || change.element != changes[i - 1].element)
            {
                const auto& mixture = change.phase == Phase::Liquid ? liquid : gas;
                const auto found = std::find_if(mixture.components.begin(), mixture.components.end(),
                    [&](const MixtureComponent& candidate) { return candidate.element == change.element; });
                if (found == mixture.components.end())
                    throw std::invalid_argument("Fluid grid edit element is not configured in the selected phase");
                component = size_t(found - mixture.components.begin()) + (change.phase == Phase::Liquid ? 0 : liquid.components.size());
                mass = found->density * cell_volume;
            }
            cells.push_back({change.x, change.y, component, mass});
        }

        const auto before = [](const ResolvedCellEdit& a, const ResolvedCellEdit& b) {
            return a.y < b.y || (a.y == b.y && a.x < b.x);
        };
        // Raster-order drawing already has the order needed for sparse texture uploads.
        if (!std::is_sorted(cells.begin(), cells.end(), before))
            std::stable_sort(cells.begin(), cells.end(), before);

        size_t kept = 0;
        for (size_t first = 0; first < cells.size();)
        {
            size_t last = first + 1;
            while (last < cells.size() && cells[last].x == cells[first].x && cells[last].y == cells[first].y) ++last;
            cells[kept++] = cells[last - 1];
            first = last;
        }
        cells.resize(kept);
    }
};
}

// Staged material replacements, with no grid readback. The grid must outlive the session.
// Only commit() touches the grid and requires the GPU's owning graphics context.
template<class Grid>
class FluidGridEdit
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

    void reserve(size_t count) { batch.changes.reserve(count); batch.cells.reserve(count); }
    size_t size() const { return batch.changes.size(); }
    bool empty() const { return batch.changes.empty(); }
    void clear() { batch.changes.clear(); batch.cells.clear(); }

    void setCell(int x, int y, Phase phase, ElementId element)
    {
        batch.changes.push_back({x, y, phase, element});
    }

    // The last write to a cell wins. Successful commits clear commands and retain capacity.
    // Dropping or clearing a session discards uncommitted edits; commit never runs implicitly.
    void commit()
    {
        if (empty()) return;
        grid->commitEdits(batch);
        clear();
    }
};

}
