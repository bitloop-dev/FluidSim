#pragma once
#include "FluidScene.h"

SIM_BEG;

struct ElasticTest : FluidScene<ElasticTest>
{
    using Base = FluidScene<ElasticTest>;
    static constexpr const char* scenario_id = "elastic-test";

    inline static const std::array presets{
        FluidScenePreset{"Default (512 x 400)", 
        {
            .grid_w = 512,
            .grid_h = 400,
            .fluid_device = fluid::FluidDevice::GPU,
            .frame_time = 0.04,
            .projection_tolerance = 1e-5,
            .gravity_y = 10.0,
            .liquid_viscosities = {1000.0}}
        },
        FluidScenePreset{"High Gravity (720 x 480)", 
        {
            .grid_w = 720,
            .grid_h = 480,
            .fluid_device = fluid::FluidDevice::GPU,
            .frame_time = 0.04,
            .projection_tolerance = 1e-5,
            .gravity_y = 100.0,
            .liquid_viscosities = {1000.0}}
        }
    };

    struct Config : Base::Config
    {
        FluidSceneSettings settings = presets.front().settings;
    };

    explicit ElasticTest(const Config& config);

    std::string name() const override { return "Elastic Test"; }

    fluid::SubstanceId test_substance{};
    fluid::SubstanceId atmosphere_substance{};

    FluidGrid::State initialFluidState() const override;
    JSON::json captureScenarioState() const override;
    void validateScenarioState(const JSON::json& state, const FluidGrid& restored) const override;
    void restoreScenarioState(const JSON::json& state) override;
};

SIM_END;
