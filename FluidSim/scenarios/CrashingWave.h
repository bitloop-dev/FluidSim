#pragma once
#include "../FluidScene.h"

SIM_BEG;

struct CrashingWave : FluidScene<CrashingWave>
{
    using Base = FluidScene<CrashingWave>;
    static constexpr const char* scenario_id = "crashing-wave";

    struct Config : Base::Config
    {
        FluidSceneSettings settings{
            .grid_w = 1024,
            .grid_h = 720,
            .gravity_y = 500.0,
            .surface_tension = 20000.0
        };
    };

    explicit CrashingWave(const Config& config);
    std::string name() const override { return "Crashing Wave"; }

    fluid::ElementId air_element{};
    fluid::ElementId water_element{};
    fluid::ElementId red_element{};
    fluid::ElementId cloud_element{};

    FluidGrid::State initialFluidState() const override;
    JSON::json captureScenarioState() const override;
    void validateScenarioState(const JSON::json& state, const FluidGrid& restored) const override;
    void restoreScenarioState(const JSON::json& state) override;
};

SIM_END;
