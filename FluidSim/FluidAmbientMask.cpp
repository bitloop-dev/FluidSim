#include "FluidAmbientMask.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fluid
{

void FluidAmbientMask::observe(const FluidGrid& grid, double dt, const Settings& s)
{
    // A mismatched catalogue would otherwise index missing reference ingredients.
    if (s.liquid_fractions.size() != grid.liquid_mass.size() || s.gas_fractions.size() != grid.gas_mass.size())
        throw std::invalid_argument("Ambient mask fractions must match both grid mixtures");
    const bool restart = !valid || dt <= 0 || width != grid.width || height != grid.height ||
        spacing != grid.cell_size || origin != grid.origin || previous_settings != s;
    width = grid.width; height = grid.height; spacing = grid.cell_size; origin = grid.origin;
    const int count = width * height;
    cells.resize(count); quiet.resize(count); pressure_error.resize(count); density.resize(count);
    material_error.resize(count); activity.resize(count); disturbed.resize(count);
    horizontal.resize(count); neighbourhood.resize(count);
    if (restart) std::fill(quiet.begin(), quiet.end(), 0.0f);
    const double elapsed = std::max(dt, 0.0), area = spacing * spacing;
    int speed_count = 0, pressure_count = 0, acceleration_count = 0;
    double ambient_liquid = 0;
    for (double fraction : s.liquid_fractions) ambient_liquid += fraction;

    for (int y = 0; y < height; ++y)
    for (int x = 0, i = y * width; x < width; ++x, ++i)
    {
        double composition = 0, mass = 0;
        auto compare = [&](const auto& fields, const auto& components, const auto& reference, bool gas) {
            double tracer_difference = 0;
            for (size_t k = 0; k < fields.size(); ++k) {
                mass += fields[k][i];
                const double difference = fields[k][i] / (components[k].density * area) - reference[k];
                if (gas && std::ranges::find(s.gas_tracer_elements, components[k].element) != s.gas_tracer_elements.end())
                    tracer_difference += difference;
                else
                    composition += std::abs(difference);
            }
            composition += std::abs(tracer_difference);
        };
        compare(grid.liquid_mass, grid.liquid_mixture.components, s.liquid_fractions, false);
        compare(grid.gas_mass, grid.gas_mixture.components, s.gas_fractions, true);
        material_error[i] = float(std::max(composition, std::abs(grid.liquid_fraction[i] - ambient_liquid)));
        density[i] = mass / area;
        const double background = s.pressure + s.pressure_gradient.x * (origin.x + (x + 0.5) * spacing)
            + s.pressure_gradient.y * (origin.y + (y + 0.5) * spacing);
        pressure_error[i] = grid.pressure[i] - background;
        const double u0 = grid.u[grid.uIndex(x, y)], u1 = grid.u[grid.uIndex(x + 1, y)];
        const double v0 = grid.v[i], v1 = grid.v[i + width];
        // Opposing face velocities must not cancel into apparently still fluid.
        const double speed = std::sqrt(0.5 * (u0 * u0 + u1 * u1 + v0 * v0 + v1 * v1));
        if (material_error[i] < s.composition_tolerance) {
            speed_count += speed >= s.speed_tolerance;
            pressure_count += std::abs(pressure_error[i]) >= s.pressure_tolerance;
        }
        activity[i] = float(std::max(speed / s.speed_tolerance, std::abs(pressure_error[i]) / s.pressure_tolerance));
    }
    for (int y = 0; y < height; ++y)
    for (int x = 0, i = y * width; x < width; ++x, ++i)
    {
        double acceleration = 0;
        auto face = [&](int n) {
            const double face_density = 0.5 * (density[i] + density[n]);
            // Uninitialised/empty diagnostic cells have no pressure-bearing mass.
            if (face_density > 0)
                acceleration = std::max(acceleration, std::abs(pressure_error[n] - pressure_error[i]) / (face_density * spacing));
        };
        if (x > 0) face(i - 1);
        if (x + 1 < width) face(i + 1);
        if (y > 0) face(i - width);
        if (y + 1 < height) face(i + width);
        acceleration_count += material_error[i] < s.composition_tolerance && acceleration >= s.acceleration_tolerance;
        activity[i] = std::max(activity[i], float(acceleration / s.acceleration_tolerance));
        // Classify locally before buffering. Buffer/waiting/connectivity never change this history.
        // Start conservatively; thereafter wake at full tolerance and clear below half.
        const float threshold = restart || disturbed[i] ? 0.5f : 1.0f;
        disturbed[i] = activity[i] >= threshold;
        neighbourhood[i] = disturbed[i] || material_error[i] >= s.composition_tolerance;
    }
    // Separable square buffer; no orientation has special status.
    for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
        std::uint8_t maximum = 0;
        for (int nx = std::max(0, x - s.halo_cells); nx <= std::min(width - 1, x + s.halo_cells); ++nx)
            maximum = std::max(maximum, neighbourhood[y * width + nx]);
        horizontal[y * width + x] = maximum;
    }
    for (int y = 0; y < height; ++y)
    for (int x = 0, i = y * width; x < width; ++x, ++i) {
        std::uint8_t maximum = 0;
        for (int ny = std::max(0, y - s.halo_cells); ny <= std::min(height - 1, y + s.halo_cells); ++ny)
            maximum = std::max(maximum, horizontal[ny * width + x]);
        const bool compatible = material_error[i] < s.composition_tolerance;
        const bool undisturbed = maximum == 0;
        quiet[i] = undisturbed ? float(quiet[i] + elapsed) : 0;
        cells[i] = !compatible ? Material : disturbed[i] ? Disturbed :
            !undisturbed ? Buffer : quiet[i] < s.quiet_seconds ? Waiting : Isolated;
    }

    // A quiet island is shown separately from candidates connected to a prescribed reservoir.
    frontier.clear();
    auto visit = [&](int i) {
        if (cells[i] == Isolated) { cells[i] = Candidate; frontier.push_back(i); }
    };
    for (int y = 0; y < height; ++y) {
        if (s.reservoir_edges & Left) visit(y * width);
        if (s.reservoir_edges & Right) visit(y * width + width - 1);
    }
    for (int x = 0; x < width; ++x) {
        if (s.reservoir_edges & Top) visit(x);
        if (s.reservoir_edges & Bottom) visit((height - 1) * width + x);
    }
    for (size_t head = 0; head < frontier.size(); ++head) {
        const int i = frontier[head], x = i % width, y = i / width;
        if (x > 0) visit(i - 1);
        if (x + 1 < width) visit(i + 1);
        if (y > 0) visit(i - width);
        if (y + 1 < height) visit(i + width);
    }
    int counts[6] = {};
    for (auto cell : cells) ++counts[cell];
    statistics = {count, counts[Material], counts[Disturbed], counts[Buffer], counts[Waiting], counts[Isolated], counts[Candidate],
        speed_count, pressure_count, acceleration_count};
    previous_settings = s;
    valid = true;
    ++revision;
}

}
