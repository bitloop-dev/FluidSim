#include "ElasticTest.h"
#include "PerlinAtmosphere.h"

SIM_BEG;

ElasticTest::ElasticTest(const Config& config) : Base(config.settings, config)
{
    fluid::SubstanceCatalogue substances;
    const auto hydrogen = substances.addElement({"Hydrogen", "H", 1.0080});
    const auto helium = substances.addElement({"Helium", "He", 4.0026});
    hydrogen_substance = substances.add({"Hydrogen", {0.0f, 0.0f, 0.0f}, {{hydrogen, 2}}});
    helium_substance = substances.add({"Helium", {1.0f, 0.6f, 1.0f}, {{helium, 1}}});

    // Match the other scenarios' diagnostic atmosphere without a liquid catalogue.
    fluid::FluidMixture gas{{{hydrogen_substance, 0.1}, {helium_substance, 0.1}}};
    fluid.configureMaterials(std::move(substances), {}, std::move(gas));
}

FluidGrid::State ElasticTest::initialFluidState() const
{
    const int width = fluid.getWidth();
    const int height = fluid.getHeight();
    const size_t cell_count = size_t(width) * height;
    const double cell_area = fluid.getCellSize() * fluid.getCellSize();

    FluidGrid::State initial;
    initial.liquid_fraction.assign(cell_count, 0.0);
    initial.gas_mass.assign(fluid.getGasMixture().components.size(), std::vector<double>(cell_count, 0.0));

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const double cloud_share = perlinCloudShare(x, y, width);
            for (size_t k = 0; k < fluid.getGasMixture().components.size(); ++k)
            {
                const auto& component = fluid.getGasMixture().components[k];
                const double share = component.substance == helium_substance ? cloud_share : 1.0 - cloud_share;
                initial.gas_mass[k][size_t(y) * width + x] = component.density * cell_area * share;
            }
        }
    }

    return initial;
}

JSON::json ElasticTest::captureScenarioState() const
{
    return {{"substances", {hydrogen_substance, helium_substance}}};
}

void ElasticTest::validateScenarioState(const JSON::json& state, const FluidGrid& restored) const
{
    const auto ids = state.at("substances").get<std::array<fluid::SubstanceId, 2>>();
    for (const auto id : ids)
    {
        if (id >= restored.getSubstances().size())
            throw std::runtime_error("Invalid saved ElasticTest substance");
    }
}

void ElasticTest::restoreScenarioState(const JSON::json& state)
{
    const auto ids = state.at("substances").get<std::array<fluid::SubstanceId, 2>>();
    hydrogen_substance = ids[0];
    helium_substance = ids[1];
}

SIM_END;
