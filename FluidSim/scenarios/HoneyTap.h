#pragma once
#include "FluidScene.h"

SIM_BEG;

struct HoneyTap : FluidScene<HoneyTap>
{
    using Base = FluidScene<HoneyTap>;
    static constexpr const char* scenario_id = "honey-tap";

    inline static const std::array presets{
        FluidScenePreset{"Default (512 x 420)", {.grid_w = 512,
                                                    .grid_h = 420,
                                                    .gravity_y = 100.0,
                                                    .surface_tension = 2000.0,
                                                    .liquid_viscosities = {300000.0}}}};

    struct Config : Base::Config
    {
        FluidSceneSettings settings = presets.front().settings;
        double tap_width_percent = 100.0 / 24.0;
        double tap_speed = 20.0;
    };

    explicit HoneyTap(const Config& config);

    std::string name() const override { return "Honey Tap"; }

    fluid::SubstanceId hydrogen_substance{}, helium_substance{}, honey_substance{};
    double tap_width_percent;
    double tap_speed;

    struct UI : Base::UI
    {
        using Base::UI::UI;
        void sidebar() override;
    };

    FluidGrid::State initialFluidState() const override;
    void beforeFluidFrame(double dt) override;
    JSON::json captureScenarioState() const override;
    void validateScenarioState(const JSON::json& state, const FluidGrid& restored) const override;
    void restoreScenarioState(const JSON::json& state) override;
};

SIM_END;
