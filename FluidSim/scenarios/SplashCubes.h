#pragma once
#include "../FluidScene.h"

SIM_BEG;

struct SplashCubes : FluidScene<SplashCubes>
{
    using Base = FluidScene<SplashCubes>;
    static constexpr const char* scenario_id = "splash-cubes";

    inline static constexpr std::array presets{
        FluidScenePreset{"Default (1024 x 720)", {
            .grid_w = 1024,
            .grid_h = 720,
            .gravity_y = 500.0,
            .surface_tension = 5000.0
        }}
    };

    struct Config : Base::Config
    {
        FluidSceneSettings settings = presets.front().settings;
    };

    explicit SplashCubes(const Config& config);
    std::string name() const override { return "Splash Cubes"; }

    fluid::SubstanceId hydrogen_substance{};
    fluid::SubstanceId water_substance{};
    fluid::SubstanceId glycerol_substance{};
    fluid::SubstanceId helium_substance{};

    FluidGrid::State initialFluidState() const override;
    JSON::json captureScenarioState() const override;
    void validateScenarioState(const JSON::json& state, const FluidGrid& restored) const override;
    void restoreScenarioState(const JSON::json& state) override;
};

SIM_END;
