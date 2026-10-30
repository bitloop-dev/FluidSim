#include "FluidSim.h"
#include "scenarios/CrashingWave.h"
#include "scenarios/SplashCubes.h"

SIM_BEG;

using namespace bl;

void FluidSim_Project::UI::sidebar()
{
    bl_scoped(viewport_count, scenario);

    ImGui::Combo("Scenario (next start)", &scenario, "Splash Cubes\0Crashing Wave\0");
    ImGui::SliderInt("Viewport Count", &viewport_count, 1, 8);
}

void FluidSim_Project::projectPrepare(Layout& layout)
{
    const auto add_scene = [&](auto scene)
    {
        scene.setLaunchConfigParser([](auto& config, const JSON::json& values)
        {
            for (const auto& [key, value] : values.items())
            {
                if (key == "resume_image")
                {
                    config.resume_image = value.get<std::string>();
                    continue;
                }
                if (key != "checkpoint_brotli_quality")
                    throw std::invalid_argument("Unknown FluidSim config field: " + key);
                if (!value.is_number_integer() || value < -1 || value > 11)
                    throw std::invalid_argument("checkpoint_brotli_quality must be -1 (off) or an integer 0..11");
                config.checkpoint_brotli_quality = value.get<int>();
            }
        });
        layout << scene;
    };
    for (int i = 0; i < viewport_count; ++i)
    {
        const std::string id = "scene-" + std::to_string(i);
        if (scenario == 0)
            add_scene(create<SplashCubes>(id));
        else
            add_scene(create<CrashingWave>(id));
    }
}

SIM_END;
