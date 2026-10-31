#include "SplashCubes.h"
#include "PerlinAtmosphere.h"
#include "ChemicalMaterials.h"

SIM_BEG;

SplashCubes::SplashCubes(const Config& config) : Base(config.settings, config)
{
    ChemicalMaterials materials({0.f, 0.f, 0.f}, {1.f, 0.6f, 1.f});
    hydrogen_substance = materials.hydrogen;
    water_substance = materials.water;
    glycerol_substance = materials.glycerol;
    helium_substance = materials.helium;
    fluid.configureMaterials(std::move(materials.substances), std::move(materials.liquid), std::move(materials.gas));
}

FluidGrid::State SplashCubes::initialFluidState() const
{
    const int width = fluid.getWidth();
    const int height = fluid.getHeight();
    const size_t cells = size_t(width) * height;
    const double cell_volume = fluid.getCellSize() * fluid.getCellSize();

    // Two 2D squares centred as a pair, with a small gap between them.
    // Cell-coordinate geometry keeps the proportions when resolution changes.
    const double side = 0.30 * std::min(width, height);
    const double gap = 0.06 * std::min(width, height);
    const double top = 0.9 * (height - side);
    const double red_left = 0.5 * width - 0.5 * gap - side;
    const double blue_left = 0.5 * width + 0.5 * gap;
    const auto overlap = [](double cell, double start, double length)
    {
        return std::max(0.0, std::min(cell + 1.0, start + length) - std::max(cell, start));
    };

    FluidGrid::State initial;
    initial.liquid_fraction.resize(cells);
    initial.liquid_mass.assign(fluid.getLiquidMixture().components.size(), std::vector<double>(cells));
    initial.gas_mass.assign(fluid.getGasMixture().components.size(), std::vector<double>(cells));

    for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
    {
        const int i = y * width + x;
        // Exact cell coverage avoids jagged/size-dependent initial volumes.
        const double vertical = overlap(y, top, side);
        const double red = vertical * overlap(x, red_left, side);
        const double blue = vertical * overlap(x, blue_left, side);
        const double fraction = red + blue;
        initial.liquid_fraction[i] = fraction;

        for (size_t k = 0; k < initial.liquid_mass.size(); ++k)
        {
            const auto& component = fluid.getLiquidMixture().components[k];
            double coverage = 0.0;
            if (component.substance == glycerol_substance) coverage = red;
            else if (component.substance == water_substance) coverage = blue;
            initial.liquid_mass[k][i] = component.density * coverage * cell_volume;
        }

        const double cloud = perlinCloudShare(x, y, width);
        for (size_t k = 0; k < initial.gas_mass.size(); ++k)
        {
            const auto& component = fluid.getGasMixture().components[k];
            double share = 0.0;
            if (component.substance == helium_substance) share = cloud;
            else if (component.substance == hydrogen_substance) share = 1.0 - cloud;
            initial.gas_mass[k][i] = component.density * (1.0 - fraction) * cell_volume * share;
        }
    }
    // Empty velocity arrays upload as zero: gravity supplies all initial motion.
    return initial;
}

JSON::json SplashCubes::captureScenarioState() const
{
    return {{"substances", {hydrogen_substance, water_substance, glycerol_substance, helium_substance}}};
}

void SplashCubes::validateScenarioState(const JSON::json& state, const FluidGrid& restored) const
{
    const auto ids = state.at(state.contains("substances") ? "substances" : "elements").get<std::array<fluid::SubstanceId, 4>>();
    for (const auto id : ids)
        if (id >= restored.getSubstances().size())
            throw std::runtime_error("Invalid saved SplashCubes substance");
}

void SplashCubes::restoreScenarioState(const JSON::json& state)
{
    const auto ids = state.at(state.contains("substances") ? "substances" : "elements").get<std::array<fluid::SubstanceId, 4>>();
    hydrogen_substance = ids[0];
    water_substance = ids[1];
    glycerol_substance = ids[2];
    helium_substance = ids[3];
}

SIM_END;
