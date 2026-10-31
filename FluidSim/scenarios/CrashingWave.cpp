#include "CrashingWave.h"
#include "PerlinAtmosphere.h"
#include "ChemicalMaterials.h"

SIM_BEG;

CrashingWave::CrashingWave(const Config& config) : Base(config.settings, config)
{
    ChemicalMaterials materials({ 0.f, 0.f, 0.f }, { 1.f, 0.6f, 1.f });
    hydrogen_substance = materials.hydrogen;
    water_substance = materials.water;
    glycerol_substance = materials.glycerol;
    helium_substance = materials.helium;
    fluid.configureMaterials(std::move(materials.substances), std::move(materials.liquid), std::move(materials.gas));
}

FluidGrid::State CrashingWave::initialFluidState() const
{
    constexpr double amplitude = 0.04;
    constexpr double pi = 3.14159265358979323846;
    const size_t cell_count = size_t(fluid.getWidth()) * fluid.getHeight();

    using Scalar = FluidGrid::Scalar;
    FluidGrid::State initial;
    initial.liquid_fraction.resize(cell_count);

    const double cell_volume = fluid.getCellSize() * fluid.getCellSize();

    initial.liquid_mass.assign(fluid.getLiquidMixture().components.size(),
        std::vector<Scalar>(cell_count, 0.0));

    initial.gas_mass.assign(fluid.getGasMixture().components.size(),
        std::vector<Scalar>(cell_count, 0.0));

    for (int y = 0; y < fluid.getHeight(); ++y)
    {
        for (int x = 0; x < fluid.getWidth(); ++x)
        {
            // the experiment starts with a sloped, rippled surface
            const double surface =
                x + fluid.getHeight() * (0.5 + amplitude * std::cos(pi * (x*3 + 0.5) / fluid.getWidth()));

            // measure how much of this cell lies below the liquid surface
            const int i = y * fluid.getWidth() + x;
            const double fraction = std::clamp(y + 1.0 - surface, 0.0, 1.0);

            initial.liquid_fraction[i] = Scalar(fraction);

            // Fill the left quarter's liquid with glycerol; use water elsewhere.
            const double red_share = x < fluid.getWidth() / 4 ? 1.0 : 0.0;

            for (size_t k = 0; k < fluid.getLiquidMixture().components.size(); ++k)
            {
                const auto& component = fluid.getLiquidMixture().components[k];

                double share = 0.0;

                if (component.substance == water_substance)
                    share = 1.0 - red_share;
                else if (component.substance == glycerol_substance)
                    share = red_share;

                // Split the existing liquid volume between its ingredients.
                initial.liquid_mass[k][i] =
                    Scalar(component.density * fraction * cell_volume * share);
            }

            const double cloud_share = perlinCloudShare(x, y, fluid.getWidth());

            for (size_t k = 0; k < fluid.getGasMixture().components.size(); ++k)
            {
                const auto& component = fluid.getGasMixture().components[k];

                double share = 0.0;
                if (component.substance == hydrogen_substance)
                    share = 1.0 - cloud_share;
                else if (component.substance == helium_substance)
                    share = cloud_share;

                // Helium replaces hydrogen within the existing gas volume.
                initial.gas_mass[k][i] =
                    Scalar(component.density * (1.0 - fraction) * cell_volume * share);
            }
        }
    }

    return initial;
}

JSON::json CrashingWave::captureScenarioState() const
{
    return {{"substances", {hydrogen_substance, water_substance, glycerol_substance, helium_substance}}};
}

void CrashingWave::validateScenarioState(const JSON::json& state, const FluidGrid& restored) const
{
    const auto ids = state.at(state.contains("substances") ? "substances" : "elements").get<std::array<fluid::SubstanceId, 4>>();
    for (const auto id : ids)
        if (id >= restored.getSubstances().size())
            throw std::runtime_error("Invalid saved CrashingWave substance");
}

void CrashingWave::restoreScenarioState(const JSON::json& state)
{
    const auto ids = state.at(state.contains("substances") ? "substances" : "elements").get<std::array<fluid::SubstanceId, 4>>();
    hydrogen_substance = ids[0];
    water_substance = ids[1];
    glycerol_substance = ids[2];
    helium_substance = ids[3];
}

SIM_END;
