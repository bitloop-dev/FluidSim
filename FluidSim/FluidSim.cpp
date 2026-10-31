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

    if (ImGui::Button("Restart All Simulations"))
        bl_schedule([](FluidSim_Project& project) { project.restartAllSimulations(); });

    if (ImPlot::BeginPlot("Scene frame times (10-frame MA)", ImVec2(-1, 320)))
    {
        ImPlot::SetupAxes("Elapsed time (s)", "Frame time (ms)",
            ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Once);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        const auto plot_scene = [](const auto& scene, int colour_index)
        {
            const auto& history = scene.frame_time_history;
            if (history.count == 0) return;
            const std::string label = scene.id() + " (" + scene.name() + ")";
            const int offset = history.count == int(history.times.size()) ? history.next : 0;
            ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(colour_index, ImPlotColormap_Dark), 2.f);
            ImPlot::PlotLine(label.c_str(), history.times.data(), history.averages.data(),
                history.count, ImPlotLineFlags_None, offset);
        };
        int colour_index = 0;
        for (const SceneBase* scene : __target.scenes())
        {
            if (const auto* cubes = dynamic_cast<const SplashCubes*>(scene))
                plot_scene(*cubes, colour_index);
            else if (const auto* wave = dynamic_cast<const CrashingWave*>(scene))
                plot_scene(*wave, colour_index);
            ++colour_index;
        }
        ImPlot::EndPlot();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Each line is a scene's 10-frame mean (available samples during startup).\n"
            "Use Restart All Simulations to align their elapsed times.\n"
            "Update order rotates each frame to share first-scene overhead.\n"
            "Scene update + render CPU submission; UI/presentation and GPU draw completion excluded.");
}

void FluidSim_Project::restartAllSimulations()
{
    // One owner-scheduled action resets every active scene before any advances again.
    for (SceneBase* scene : scenes())
    {
        if (auto* cubes = dynamic_cast<SplashCubes*>(scene))
            cubes->restartSimulation();
        else if (auto* wave = dynamic_cast<CrashingWave*>(scene))
            wave->restartSimulation();
    }
}

void FluidSim_Project::projectPrepare(Layout& layout)
{
    // Independent comparison scenes share the post-render warm-up cost fairly.
    setSceneUpdateOrder(SceneUpdateOrder::Rotating);

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
