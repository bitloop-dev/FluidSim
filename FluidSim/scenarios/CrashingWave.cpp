#include "CrashingWave.h"
#include "PerlinAtmosphere.h"

SIM_BEG;

CrashingWave::CrashingWave(const Config& config) : Base(config.settings, config)
{
    fluid::ElementCatalogue elements;
    fluid::FluidMixture liquid_mixture, gas_mixture;

    air_element = elements.add({
        "Air", {0.015f, 0.025f, 0.06f}
    });

    water_element = elements.add({
        "Water", {0.1f, 0.35f, 1.0f}
    });

    red_element = elements.add({
        "Red fluid", {0.95f, 0.25f, 0.15f}
    });

    cloud_element = elements.add({
        "Cloud", {0.0f, 1.0f, 1.0f}
    });

    liquid_mixture.components = {
        {water_element, 10.0},
        {red_element, 30.0}
    };

    const double atmosphere_density = 0.1;
    gas_mixture.components = {
        {air_element, atmosphere_density},
        {cloud_element, atmosphere_density}
    };
    fluid.configureMaterials(std::move(elements), std::move(liquid_mixture), std::move(gas_mixture));
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

            // Fill the left quarter's liquid with Red; use Water elsewhere.
            const double red_share = x < fluid.getWidth() / 4 ? 1.0 : 0.0;

            for (size_t k = 0; k < fluid.getLiquidMixture().components.size(); ++k)
            {
                const auto& component = fluid.getLiquidMixture().components[k];

                double share = 0.0;

                if (component.element == water_element)
                    share = 1.0 - red_share;
                else if (component.element == red_element)
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
                if (component.element == air_element)
                    share = 1.0 - cloud_share;
                else if (component.element == cloud_element)
                    share = cloud_share;

                // Clouds replace air within the existing gas volume.
                initial.gas_mass[k][i] =
                    Scalar(component.density * (1.0 - fraction) * cell_volume * share);
            }
        }
    }

    return initial;
}

JSON::json CrashingWave::captureScenarioState() const
{
    return {{"elements", {air_element, water_element, red_element, cloud_element}}};
}

void CrashingWave::validateScenarioState(const JSON::json& state, const FluidGrid& restored) const
{
    const auto ids = state.at("elements").get<std::array<fluid::ElementId, 4>>();
    for (const auto id : ids)
        if (id >= restored.getElements().size())
            throw std::runtime_error("Invalid saved CrashingWave element");
}

void CrashingWave::restoreScenarioState(const JSON::json& state)
{
    const auto ids = state.at("elements").get<std::array<fluid::ElementId, 4>>();
    air_element = ids[0];
    water_element = ids[1];
    red_element = ids[2];
    cloud_element = ids[3];
}

SIM_END;
