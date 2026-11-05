#pragma once
#include "FluidScene.h"

SIM_BEG;

struct ElasticTest : FluidScene<ElasticTest>
{
    using Base = FluidScene<ElasticTest>;
    static constexpr const char* scenario_id = "elastic-test";

    inline static const std::array presets{FluidScenePreset{"Default (512 x 420)", {.grid_w = 512, .grid_h = 420}}};

    struct Config : Base::Config
    {
        FluidSceneSettings settings = presets.front().settings;
    };

    explicit ElasticTest(const Config& config);

    std::string name() const override
    {
        return "Elastic Test";
    }

    fluid::SubstanceId hydrogen_substance{};
    fluid::SubstanceId helium_substance{};

    FluidGrid::State initialFluidState() const override;
    JSON::json captureScenarioState() const override;
    void validateScenarioState(const JSON::json& state, const FluidGrid& restored) const override;
    void restoreScenarioState(const JSON::json& state) override;
};

SIM_END;
