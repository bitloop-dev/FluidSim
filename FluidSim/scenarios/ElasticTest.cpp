#include "ElasticTest.h"

SIM_BEG;

ElasticTest::ElasticTest(const Config& config) : Base(config.settings, config)
{
    // Abstract materials have no chemical formula; this experiment owns its catalogue.
    fluid::SubstanceCatalogue substances;
    test_substance = substances.add({"Test material", {0.2f, 0.8f, 0.6f}});
    atmosphere_substance = substances.add({"Atmosphere", {0.0f, 0.0f, 0.0f}});

    // Stiffness belongs to the material; the scene supplies its initial deformation.
    fluid::FluidMixture material{{{test_substance, 10.0, 1000.0, 20000.0}}};
    fluid::FluidMixture atmosphere{{{atmosphere_substance, 0.01}}};
    fluid.configureMaterials(std::move(substances), std::move(material), std::move(atmosphere));
}

FluidGrid::State ElasticTest::initialFluidState() const
{
    const int width = fluid.getWidth();
    const int height = fluid.getHeight();
    const size_t cells = size_t(width) * height;
    const double cell_area = fluid.getCellSize() * fluid.getCellSize();
    const double material_density = fluid.getLiquidMixture().components.front().density;
    const double atmosphere_density = fluid.getGasMixture().components.front().density;

    // Stretch an initially relaxed square without changing its area.
    constexpr double stretch = 1.15;
    const double side = 0.3 * std::min(width, height);
    const double left = 0.5 * (width - side * stretch);
    const double top = 0.5 * (height - side / stretch);
    const auto overlap = [](double cell, double start, double length)
    {
        return std::max(0.0, std::min(cell + 1.0, start + length) - std::max(cell, start));
    };

    FluidGrid::State initial;
    initial.liquid_fraction.resize(cells);
    initial.liquid_mass.assign(1, std::vector<double>(cells));
    initial.gas_mass.assign(1, std::vector<double>(cells));
    initial.elastic_log_xx.assign(cells, 2.0 * std::log(stretch));
    initial.elastic_log_xy.assign(cells, 0.0);
    initial.elastic_log_yy.assign(cells, -2.0 * std::log(stretch));

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const size_t i = size_t(y) * width + x;

            // Exact coverage preserves the block's area at partial edge cells.
            const double fraction = overlap(x, left, side * stretch) * overlap(y, top, side / stretch);
            initial.liquid_fraction[i] = fraction;
            initial.liquid_mass[0][i] = material_density * fraction * cell_area;
            initial.gas_mass[0][i] = atmosphere_density * (1.0 - fraction) * cell_area;
        }
    }

    // Release from rest. The stored deformation supplies the restoring force.
    return initial;
}

JSON::json ElasticTest::captureScenarioState() const
{
    return {{"substances", {test_substance, atmosphere_substance}}};
}

void ElasticTest::validateScenarioState(const JSON::json& state, const FluidGrid& restored) const
{
    const auto ids = state.at("substances").get<std::array<fluid::SubstanceId, 2>>();
    for (const auto id : ids)
    {
        if (id >= restored.getSubstances().size())
            throw std::runtime_error("Invalid saved ElasticTest substance");
    }

    // Restart expects exactly one test ingredient and one atmosphere ingredient.
    const auto& liquid = restored.getLiquidMixture().components;
    const auto& gas = restored.getGasMixture().components;
    if (liquid.size() != 1 || gas.size() != 1 || liquid[0].substance != ids[0] || gas[0].substance != ids[1])
        throw std::runtime_error("Saved ElasticTest materials do not match the scenario");
}

void ElasticTest::restoreScenarioState(const JSON::json& state)
{
    const auto ids = state.at("substances").get<std::array<fluid::SubstanceId, 2>>();
    test_substance = ids[0];
    atmosphere_substance = ids[1];
}

SIM_END;
