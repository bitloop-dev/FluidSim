#include "ElasticEngine.h"
#include <stdexcept>

SIM_BEG;

using namespace bl;

/// ─────────────────────── project setup ───────────────────────

void ElasticEngine_Project::UI::sidebar()
{
    bl_scoped(viewport_count);

    ImGui::SliderInt("Viewport Count", &viewport_count, 1, 8);
}

void ElasticEngine_Project::projectPrepare(Layout& layout)
{
    for (int i = 0; i < viewport_count; i++)
        layout << create<ElasticEngine_Scene>();
}

/// ────────────────────────── scene ──────────────────────────

void ElasticEngine_Scene::UI::sidebar()
{
    if (ImGui::CollapsingHeaderBox("Experiment", true))
    {
        bl_pull_temp(scenario);
        int selected = static_cast<int>(scenario);
        if (ImGui::Combo("Scenario", &selected, "Two Balls\0Cup Drop\0"))
        {
            bl_schedule([selected](ElasticEngine_Scene& scene)
            {
                scene.selectScenario(static_cast<ElasticScenario>(selected));
            });
        }

        bl_pull_temp(stiffness);
        if (ImGui::SliderDouble("Stiffness", &stiffness, 0.0, 100.0, "%.3f",
                ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp))
        {
            bl_schedule([stiffness](ElasticEngine_Scene& scene)
            {
                scene.setStiffness(stiffness);
            });
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Zero releases material bonds on the next step. Restart rebuilds them; raising stiffness does not.\n"
                              "Fluid pressure is not simulated here.");

        bl_scoped(gravity);
        ImGui::SliderDouble("Gravity", &gravity, 0.0, 0.1, "%.3f", ImGuiSliderFlags_AlwaysClamp);

        if (ImGui::Button("Restart scenario"))
        {
            bl_schedule([](ElasticEngine_Scene& scene)
            {
                scene.restartScenario();
            });
        }

        bl_view(simulation_error);
        if (!simulation_error.empty())
            ImGui::TextWrapped("%s", simulation_error.c_str());

        ImGui::EndCollapsingHeaderBox();
    }

    if (ImGui::CollapsingHeaderBox("View", true))
    {
        bl_scoped(camera);

        camera.populateUI();
        ImGui::EndCollapsingHeaderBox();
    }
}

void ElasticEngine_Scene::selectScenario(ElasticScenario selected)
{
    scenario = selected;
    stiffness = scenario == ElasticScenario::CupDrop ? 30.0 : 1.0;
    gravity = scenario == ElasticScenario::CupDrop ? 0.02 : 0.0;
    restartScenario();
}

void ElasticEngine_Scene::restartScenario()
{
    elastic_engine = {};
    elastic_engine.setMinimumNodeDistance(minimum_node_distance);
    if (scenario == ElasticScenario::TwoBalls)
        setupTwoBalls(elastic_engine, breaking_strain);
    else
        setupCupDrop(elastic_engine, stiffness, breaking_strain);

    setStiffness(stiffness);
    frame_scenario = true;
    requestRedraw(true);
}

void ElasticEngine_Scene::setStiffness(double value)
{
    stiffness = value;
    simulation_error.clear();
    for (std::size_t index = 0; index < elastic_engine.getBonds().size(); ++index)
    {
        auto& bond = elastic_engine.getBond(index);
        if (!bond.boundary)
            bond.stiffness = value;
    }
}

void ElasticEngine_Scene::frameScenario(Viewport* ctx)
{
    camera.setOriginViewportAnchor(Anchor::CENTER);
    if (scenario == ElasticScenario::TwoBalls)
        camera.focusWorldRect(*ctx, -175.0, -90.0, 175.0, 90.0);
    else
        camera.focusWorldRect(*ctx, -270.0, -330.0, 270.0, 240.0);

    camera.uiSetCurrentAsDefault();
    frame_scenario = false;
}

void ElasticEngine_Scene::sceneStart()
{
    selectScenario(scenario);
}

void ElasticEngine_Scene::sceneMounted(Viewport* ctx)
{
    frameScenario(ctx);

    /// Configure default navigator.
    navigator.setTarget(camera);
    navigator.setDirectCameraPanning(true);
}

void ElasticEngine_Scene::sceneUnmounted([[maybe_unused]] Viewport* ctx)
{
    /// Release state owned specifically for this Viewport.
}

void ElasticEngine_Scene::sceneDestroy()
{
    /// Release dynamically allocated Scene resources.
}

void ElasticEngine_Scene::sceneProcess([[maybe_unused]] double dt)
{
    /// Process the Scene once per frame, not once per viewport.
    if (simulation_error.empty())
    {
        try
        {
            elastic_engine.advance(1.0, {0.0, gravity});
        }
        catch (const std::runtime_error& error)
        {
            simulation_error = error.what();
        }
    }

    // redraw the scene every frame
    requestRedraw(true);
}

void ElasticEngine_Scene::viewportProcess(
    Viewport* ctx,
    [[maybe_unused]] double dt)
{
    if (frame_scenario)
        frameScenario(ctx);
}

void ElasticEngine_Scene::viewportDraw(Viewport* ctx) const
{
    /// Draw the Scene on this viewport without modifying simulation state.
    ctx->transform(camera);
    ctx->drawWorldAxis();

    elastic_engine.draw(ctx);
}

void ElasticEngine_Scene::onEvent(Event e)
{
    if (e.ownedBy(this))
        navigator.handleWorldNavigation(e, true);
}

// void ElasticEngine_Scene::onPointerDown(PointerEvent e) {}
// void ElasticEngine_Scene::onPointerUp(PointerEvent e) {}
// void ElasticEngine_Scene::onPointerMove(PointerEvent e) {}
// void ElasticEngine_Scene::onWheel(PointerEvent e) {}
// void ElasticEngine_Scene::onKeyDown(KeyEvent e) {}
// void ElasticEngine_Scene::onKeyUp(KeyEvent e) {}

#ifdef BITLOOP_DEV_MODE
void ElasticEngine_Scene::traceFrame([[maybe_unused]] SimTraceFrame& frame) const
{
}

bool ElasticEngine_Scene::captureTraceSnapshot([[maybe_unused]] std::string_view route, [[maybe_unused]] SimSnapshotImage& image) const
{
    return false;
}

bool ElasticEngine_Scene::saveCheckpoint([[maybe_unused]] SimCheckpointWriter& writer) const
{
    return false;
}

bool ElasticEngine_Scene::loadCheckpoint([[maybe_unused]] const SimCheckpointReader& reader)
{
    return false;
}
#endif

SIM_END;
