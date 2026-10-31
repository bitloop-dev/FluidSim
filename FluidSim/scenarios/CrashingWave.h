#pragma once
#include "../FluidScene.h"

SIM_BEG;

struct CrashingWave : FluidScene<CrashingWave>
{
    using Base = FluidScene<CrashingWave>;
    static constexpr const char* scenario_id = "crashing-wave";

    inline static constexpr std::array presets{
        FluidScenePreset{"Default (1024 x 720)", {
            .grid_w = 1024,
            .grid_h = 720,
            .gravity_y = 500.0,
            .surface_tension = 20000.0
        }},
        FluidScenePreset{"Realtime (512 x 420)", {
            .grid_w = 512,
            .grid_h = 420,
            .frame_time = 0.02,
            .projection_tolerance = 0.1,
            .pressure_recovery_frames = 1,
            .gravity_y = 100.0,
            .surface_tension = 2000.0
        }}
    };

    struct Config : Base::Config
    {
        FluidSceneSettings settings = presets.front().settings;
    };

    explicit CrashingWave(const Config& config);
    std::string name() const override { return "Crashing Wave"; }

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
