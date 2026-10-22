#pragma once
#include "FluidGrid.h"
#include <cstdint>

namespace fluid
{

// CPU observer only. A candidate is a modelling proposal; physics still runs everywhere.
class FluidAmbientMask
{
public:
    enum Cell : std::uint8_t { Material, Disturbed, Buffer, Waiting, Isolated, Candidate };
    enum Edge { Left = 1, Right = 2, Top = 4, Bottom = 8 };
    struct Settings
    {
        // Whole-cell ingredient volume fractions, in the grid's mixture order.
        std::vector<double> liquid_fractions, gas_fractions;
        // One explicitly equivalent gas group: compare total volume, ignoring its colour split.
        // Other ingredients are still compared individually. This never changes transport.
        std::vector<ElementId> gas_tracer_elements;
        double pressure = 0; // at world (0, 0), in the same gauge as grid.pressure
        bl::DVec2 pressure_gradient{};
        float speed_tolerance = 2;
        float pressure_tolerance = 1;
        float acceleration_tolerance = 5; // gradient of pressure departure / density
        float composition_tolerance = 0.001f;
        float quiet_seconds = 0.2f;
        int halo_cells = 2;
        int reservoir_edges = 0; // explicit connections, independent of gravity
        bool operator==(const Settings&) const = default;
    };
    struct Statistics
    {
        int cells = 0, material = 0, disturbed = 0, buffer = 0, waiting = 0, isolated = 0, candidates = 0;
        int speed = 0, pressure = 0, acceleration = 0; // overlapping reasons above full tolerance, matching material only
    };

    void observe(const FluidGrid& grid, double dt, const Settings& settings);
    void reset() { valid = false; }
    int width = 0, height = 0;
    std::uint64_t revision = 0;
    std::vector<std::uint8_t> cells;
    Statistics statistics;

private:
    bool valid = false;
    Settings previous_settings;
    double spacing = 0;
    bl::DVec2 origin{};
    std::vector<double> pressure_error, density;
    std::vector<float> quiet, material_error, activity;
    std::vector<std::uint8_t> disturbed, horizontal, neighbourhood;
    std::vector<int> frontier;
};

}
