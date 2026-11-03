#include "ElasticTest.h"

SIM_BEG;

ElasticTest::ElasticTest(const Config& config) : Base(config.settings, config)
{
    material_setup = config.with_pool ? poolMaterials : blockMaterials;
    auto materials = material_setup();
    fluid.configureMaterials(std::move(materials.substances), std::move(materials.liquid), std::move(materials.gas));
}

FluidScenePreset::Materials ElasticTest::blockMaterials()
{
    // Abstract materials have no chemical formula; this experiment owns its catalogue.
    fluid::SubstanceCatalogue substances;
    const auto test_substance = substances.add({"Test material", {0.2f, 0.8f, 0.6f}});
    const auto atmosphere_substance = substances.add({"Atmosphere", {0.0f, 0.0f, 0.0f}});

    // Stiffness belongs to the material; the scene supplies its initial deformation.
    fluid::FluidMixture material{{{test_substance, 10.0, 1000.0, 50000.0}}};
    fluid::FluidMixture atmosphere{{{atmosphere_substance, 0.01}}};
    return {std::move(substances), std::move(material), std::move(atmosphere)};
}

FluidScenePreset::Materials ElasticTest::poolMaterials()
{
    auto materials = blockMaterials();
    const auto water = materials.substances.add({"Water", {0.1f, 0.4f, 1.0f}});

    // Water shares the condensed density scale, but carries no elastic stress or history.
    materials.liquid.components.push_back({water, 10.0});
    return materials;
}

FluidGrid::State ElasticTest::initialFluidState() const
{
    const int width = fluid.getWidth();
    const int height = fluid.getHeight();
    const size_t cells = size_t(width) * height;
    const double cell_area = fluid.getCellSize() * fluid.getCellSize();
    const auto& liquid = fluid.getLiquidMixture();
    const size_t elastic = liquid.elasticComponentIndex();
    const bool with_pool = liquid.components.size() == 2;
    const size_t water = with_pool ? 1 - elastic : 0;
    const double material_density = liquid.components[elastic].density;
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
    initial.liquid_mass.assign(liquid.components.size(), std::vector<double>(cells));
    initial.gas_mass.assign(1, std::vector<double>(cells));
    initial.elastic_log_xx.assign(cells, 2.0 * std::log(stretch));
    initial.elastic_log_xy.assign(cells, 0.0);
    initial.elastic_log_yy.assign(cells, -2.0 * std::log(stretch));
    if (with_pool)
        initial.elastic_fraction.resize(cells);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const size_t i = size_t(y) * width + x;

            // Exact coverage preserves the block's area at partial edge cells.
            const double fraction = overlap(x, left, side * stretch) * overlap(y, top, side / stretch);
            initial.liquid_fraction[i] = fraction;
            initial.liquid_mass[elastic][i] = material_density * fraction * cell_area;

            if (with_pool)
            {
                // The bottom quarter is exactly 120 rows in the 720 x 480 preset.
                const double water_fraction = overlap(y, 0.75 * height, 0.25 * height);
                initial.liquid_fraction[i] += water_fraction;
                initial.elastic_fraction[i] = fraction;
                initial.liquid_mass[water][i] = liquid.components[water].density * water_fraction * cell_area;
                if (fraction == 0.0)
                {
                    initial.elastic_log_xx[i] = 0.0;
                    initial.elastic_log_yy[i] = 0.0;
                }
            }

            initial.gas_mass[0][i] = atmosphere_density * (1.0 - initial.liquid_fraction[i]) * cell_area;
        }
    }

    // Release from rest. The stored deformation supplies the restoring force.
    return initial;
}

JSON::json ElasticTest::captureScenarioState() const
{
    const auto& liquid = fluid.getLiquidMixture();
    const size_t elastic = liquid.elasticComponentIndex();
    JSON::json state{
        {"substances", {liquid.components[elastic].substance, fluid.getGasMixture().components.front().substance}}};
    if (liquid.components.size() == 2)
        state["water_substance"] = liquid.components[1 - elastic].substance;

    return state;
}

void ElasticTest::validateScenarioState(const JSON::json& state, const FluidGrid& restored) const
{
    const auto ids = state.at("substances").get<std::array<fluid::SubstanceId, 2>>();
    for (const auto id : ids)
    {
        if (id >= restored.getSubstances().size())
            throw std::runtime_error("Invalid saved ElasticTest substance");
    }

    // Older snapshots contain only the block and atmosphere IDs.
    const auto& liquid = restored.getLiquidMixture().components;
    const auto& gas = restored.getGasMixture().components;
    const bool with_pool = state.contains("water_substance");
    const size_t elastic = restored.getLiquidMixture().elasticComponentIndex();
    if (liquid.size() != (with_pool ? 2 : 1) || gas.size() != 1 || elastic >= liquid.size() ||
        liquid[elastic].substance != ids[0] || gas[0].substance != ids[1] || ids[0] == ids[1])
    {
        throw std::runtime_error("Saved ElasticTest materials do not match the scenario");
    }

    if (with_pool)
    {
        const auto water = state.at("water_substance").get<fluid::SubstanceId>();
        if (water >= restored.getSubstances().size() || water == ids[0] || water == ids[1] ||
            liquid[1 - elastic].substance != water || liquid[1 - elastic].shear_modulus != 0.0)
        {
            throw std::runtime_error("Invalid saved ElasticTest water material");
        }
    }
}

void ElasticTest::restoreScenarioState(const JSON::json&)
{
    material_setup = fluid.getLiquidMixture().components.size() == 2 ? poolMaterials : blockMaterials;
}

SIM_END;
