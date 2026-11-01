#include "HoneyTap.h"
#include "ChemicalMaterials.h"
#include "PerlinAtmosphere.h"

SIM_BEG;

HoneyTap::HoneyTap(const Config& config)
    : Base(config.settings, config), tap_width_percent(config.tap_width_percent), tap_speed(config.tap_speed)
{
    // Match CrashingWave's gas definitions and diagnostic colours.
    ChemicalMaterials materials({0.0f, 0.0f, 0.0f}, {1.0f, 0.6f, 1.0f});
    hydrogen_substance = materials.hydrogen;
    helium_substance = materials.helium;

    // Honey is an effective bulk material here, not a molecule with a single formula.
    honey_substance = materials.substances.add({"Honey", {1.0f, 0.58f, 0.08f}});
    fluid::FluidMixture honey{{{honey_substance, 14.0}}};
    fluid.configureMaterials(std::move(materials.substances), std::move(honey), std::move(materials.gas));
}

FluidGrid::State HoneyTap::initialFluidState() const
{
    if (fluid.getDevice() != fluid::FluidDevice::GPU || fluid.getAdaptiveMode() != fluid::AdaptiveMode::DISABLED)
        throw std::runtime_error("HoneyTap boundary inflow requires uniform WebGPU");

    const int width = fluid.getWidth();
    const int height = fluid.getHeight();
    if (width < 3)
        throw std::runtime_error("HoneyTap needs at least three columns for its inlet and two ceiling vents");

    const size_t cell_count = size_t(width) * height;
    const double cell_area = fluid.getCellSize() * fluid.getCellSize();
    FluidGrid::State initial;
    initial.liquid_fraction.assign(cell_count, 0.0);
    initial.liquid_mass.assign(fluid.getLiquidMixture().components.size(), std::vector<double>(cell_count, 0.0));
    initial.gas_mass.assign(fluid.getGasMixture().components.size(), std::vector<double>(cell_count, 0.0));

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const double helium_share = perlinCloudShare(x, y, width);
            for (size_t k = 0; k < fluid.getGasMixture().components.size(); ++k)
            {
                const auto& component = fluid.getGasMixture().components[k];
                const double share = component.substance == helium_substance ? helium_share : 1.0 - helium_share;
                initial.gas_mass[k][size_t(y) * width + x] = component.density * cell_area * share;
            }
        }
    }

    return initial;
}

void HoneyTap::UI::sidebar()
{
    if (ImGui::CollapsingHeaderBox("Honey tap", true))
    {
        bl_scoped(tap_width_percent, tap_speed);

        ImGui::SliderDouble("Tap width", &tap_width_percent, 0.1, 100.0, "%.2f%%", ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip(
                "Nozzle width as a percentage of the grid width, rounded down to whole cells (minimum 1).\n"
                "Capped at grid width minus 2, leaving at least one ceiling vent face on each side.\n"
                "Applies live. The falling stream can stretch and narrow after leaving the nozzle.");
        }

        ImGui::SliderDouble("Inlet downward speed", &tap_speed, 0.0, 1000.0, "%.1f", ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip(
                "Downward injection velocity in scene units per simulated second. Applies live.\n"
                "The two ceiling vents remove the same volume rate, carrying the material that reaches them.\n"
                "0 closes both the inlet and vents.");
        }

        ImGui::TextDisabled("At least one ceiling vent face is reserved on each side.");
        ImGui::TextDisabled("Tap controls are preserved by restart and physics preset changes.");
        ImGui::EndCollapsingHeaderBox();
    }

    Base::UI::sidebar();
}

void HoneyTap::beforeFluidFrame(double)
{
    const int width = fluid.getWidth();
    const int tap_width = std::clamp(int(width * tap_width_percent / 100.0), 1, width - 2);
    const int left = (width - tap_width) / 2;

    fluid::BoundaryFlows flows;
    if (tap_speed > 0.0)
    {
        const double vent_speed = tap_speed * tap_width / (width - tap_width);
        flows.inflows = {
            {{fluid::BoundarySide::Top, left, tap_width}, fluid::Phase::Liquid, honey_substance, tap_speed}};
        flows.outflows = {{{fluid::BoundarySide::Top, 0, left}, vent_speed},
            {{fluid::BoundarySide::Top, left + tap_width, width - left - tap_width}, vent_speed}};
    }

    // The grid skips unchanged configurations, including after an exact checkpoint resume.
    fluid.setBoundaryFlows(flows);
}

JSON::json HoneyTap::captureScenarioState() const
{
    return {{"substances", {hydrogen_substance, helium_substance, honey_substance}},
        {"tap_width_percent", tap_width_percent}, {"tap_speed", tap_speed}};
}

void HoneyTap::validateScenarioState(const JSON::json& state, const FluidGrid& restored) const
{
    if (restored.getWidth() < 3)
        throw std::runtime_error("HoneyTap needs at least three columns for its inlet and two ceiling vents");

    const auto ids = state.at("substances").get<std::array<fluid::SubstanceId, 3>>();
    for (const auto id : ids)
    {
        if (id >= restored.getSubstances().size())
            throw std::runtime_error("Invalid saved HoneyTap substance");
    }

    const auto& liquid = restored.getLiquidMixture().components;
    if (liquid.size() != 1 || liquid[0].substance != ids[2])
        throw std::runtime_error("Saved HoneyTap inlet is missing its honey ingredient");

    const double width_percent = state.value("tap_width_percent", 100.0 / 24.0);
    const double speed = state.value("tap_speed", 20.0);

    // Snapshot controls are external input and define the boundary opening and speed.
    if (!std::isfinite(width_percent) || width_percent <= 0.0 || width_percent > 100.0 || !std::isfinite(speed) ||
        speed < 0.0 || speed > 1000.0)
    {
        throw std::runtime_error("Invalid saved HoneyTap source controls");
    }
}

void HoneyTap::restoreScenarioState(const JSON::json& state)
{
    const auto ids = state.at("substances").get<std::array<fluid::SubstanceId, 3>>();
    hydrogen_substance = ids[0];
    helium_substance = ids[1];
    honey_substance = ids[2];
    tap_width_percent = state.value("tap_width_percent", 100.0 / 24.0);
    tap_speed = state.value("tap_speed", 20.0);
}

SIM_END;
