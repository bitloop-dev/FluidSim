#pragma once
#include "FluidGrid.h"

namespace fluid::test
{

// Legacy two-phase fixtures explicitly provide ingredients, just like a scene does.
template<class Grid>
void uploadTestMaterials(Grid& grid, typename Grid::State state)
{
    using Scalar = typename Grid::Scalar;
    if (grid.elements.size() == 0)
    {
        const auto water = grid.elements.add({"Water", {0.0f, 0.0f, 1.0f}});
        const auto air = grid.elements.add({"Air", {0.0f, 0.0f, 0.0f}});
        grid.liquid_mixture.components = {{water, 100.0}};
        grid.gas_mixture.components = {{air, 0.1}};
    }
    if (state.liquid_mass.empty() && state.gas_mass.empty())
    {
        const size_t count = state.liquid_fraction.size();
        const double volume = grid.cell_size * grid.cell_size;
        state.liquid_mass.assign(1, std::vector<Scalar>(count));
        state.gas_mass.assign(1, std::vector<Scalar>(count));
        for (size_t i = 0; i < count; ++i)
        {
            state.liquid_mass[0][i] = Scalar(100.0 * state.liquid_fraction[i] * volume);
            state.gas_mass[0][i] = Scalar(0.1 * (1.0 - state.liquid_fraction[i]) * volume);
        }
    }
    grid.upload(state);
}

// Initialize a fresh test grid from its existing fraction and velocity fields.
inline FluidGrid::State initializeTestMaterials(
    FluidGrid& grid, double liquid_density = 100.0, double gas_density = 0.1)
{
    const auto water = grid.elements.add({"Water", {0.0f, 0.0f, 1.0f}});
    const auto air = grid.elements.add({"Air", {0.0f, 0.0f, 0.0f}});
    grid.liquid_mixture.components = {{water, liquid_density}};
    grid.gas_mixture.components = {{air, gas_density}};

    auto state = grid.download();
    const size_t count = state.liquid_fraction.size();
    const double cell_volume = grid.cell_size * grid.cell_size;
    state.liquid_mass.assign(1, std::vector<double>(count));
    state.gas_mass.assign(1, std::vector<double>(count));
    for (size_t i = 0; i < count; ++i)
    {
        const double fraction = state.liquid_fraction[i];
        // Mass equals density times the volume occupied by that phase.
        state.liquid_mass[0][i] = liquid_density * fraction * cell_volume;
        state.gas_mass[0][i] = gas_density * (1.0 - fraction) * cell_volume;
    }
    grid.upload(state);
    return state;
}

}
