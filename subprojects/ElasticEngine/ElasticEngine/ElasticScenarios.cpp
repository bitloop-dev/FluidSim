#include "ElasticEngine.h"
#include <cmath>
#include <optional>

SIM_BEG;

void setupTwoBalls(ElasticEngine& engine, double breaking_strain)
{
    const double spacing = 10.0;
    const double row_height = spacing * std::sqrt(3.0) / 2.0;

    auto add_ball = [&](DVec2 center, double radius, DVec2 velocity, double strain_limit)
    {
        // Clip an equilateral triangular lattice to a circular patch.
        const int extent = static_cast<int>(std::ceil(radius / row_height));
        const int width = 2 * extent + 1;
        std::vector<std::optional<NodeId>> lattice(width * width);

        for (int row = 0; row < width; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                DVec2 offset{spacing * (col - extent + 0.5 * (row - extent)), row_height * (row - extent)};

                if (offset.mag2() > radius * radius)
                    continue;

                NodeId id = engine.addNode(center + offset, velocity);
                lattice[row * width + col] = id;

                // These three neighbours already exist, so each spring is added once.
                for (IVec2 neighbor : {IVec2{col - 1, row}, IVec2{col, row - 1}, IVec2{col + 1, row - 1}})
                {
                    if (neighbor.x < 0 || neighbor.x >= width || neighbor.y < 0)
                        continue;

                    const auto& other = lattice[neighbor.y * width + neighbor.x];
                    if (other)
                        engine.addBond(*other, id, spacing, 1.0, 0.99, strain_limit);
                }
            }
        }
    };

    add_ball({75.0, 0.0}, 65.0, {}, breaking_strain);

    // Velocity is in world units per step in the current playground.
    add_ball({-125.0, 0.0}, 28.0, {2.0, 0.0}, std::numeric_limits<double>::infinity());
}

void setupCupDrop(ElasticEngine& engine, double stiffness, double breaking_strain)
{
    // Explicit boundaries stay solid when the material loses its stiffness.
    for (DVec2 position : {DVec2{-220.0, -180.0}, DVec2{-220.0, 200.0}, DVec2{220.0, 200.0}, DVec2{220.0, -180.0}})
        engine.addNode(position, {}, true);

    for (NodeId id = 0; id < 3; ++id)
        engine.addBoundary(id, id + 1);

    // Stagger alternate rows to keep the patch rectangular and every triangle equilateral.
    constexpr int columns = 32;
    constexpr int rows = 32;
    constexpr double spacing = 10.0;
    const double row_height = spacing * std::sqrt(3.0) / 2.0;
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < columns; ++col)
        {
            double x = spacing * (col + 0.5 * (row % 2) - 15.75);
            NodeId id = engine.addNode({x, -20.0 - row_height * row});
            if (col > 0)
                engine.addBond(id - 1, id, spacing, stiffness, 0.99, breaking_strain);

            if (row == 0)
                continue;

            for (int previous_col : {col, col + (row % 2 == 0 ? -1 : 1)})
            {
                if (previous_col >= 0 && previous_col < columns)
                {
                    NodeId other = 4 + (row - 1) * columns + previous_col;
                    engine.addBond(other, id, spacing, stiffness, 0.99, breaking_strain);
                }
            }
        }
    }
}

SIM_END;
