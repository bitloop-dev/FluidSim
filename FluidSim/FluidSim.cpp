#include "FluidSim.h"

SIM_BEG;

using namespace bl;

/// ─────────────────────── project setup ───────────────────────

void FluidSim_Project::UI::sidebar()
{
    bl_scoped(viewport_count);

    ImGui::SliderInt("Viewport Count", &viewport_count, 1, 8);
}

void FluidSim_Project::projectPrepare(Layout& layout)
{
    for (int i = 0; i < viewport_count; i++)
        layout << create<FluidSim_Scene>();
}

/// ────────────────────────── scene ──────────────────────────

void FluidSim_Scene::UI::sidebar()
{
    if (ImGui::CollapsingHeaderBox("View", true))
    {
        bl_scoped(camera);

        camera.populateUI();
        ImGui::EndCollapsingHeaderBox();
    }

}

void FluidSim_Scene::initializePool(double amplitude)
{
    constexpr double pi = 3.14159265358979323846;
    decltype(fluid)::State initial;
    initial.water_fraction.resize(size_t(fluid.width) * fluid.height);

    for (int y = 0; y < fluid.height; ++y)
    {
        for (int x = 0; x < fluid.width; ++x)
        {
            // the experiment starts with a sloped, rippled surface
            const double surface =
                x + fluid.height * (0.5 + amplitude * std::cos(pi * (x*3 + 0.5) / fluid.width));

            // measure how much of this cell lies below the water surface
            initial.water_fraction[y * fluid.width + x] =
                decltype(fluid)::Scalar(std::clamp(y + 1.0 - surface, 0.0, 1.0));
        }
    }
    withFluidContext([&]() { fluid.upload(initial); });
}

void FluidSim_Scene::sceneStart()
{
    initializePool();
    surface.setFragmentSource(fluid_shader_src);
}

void FluidSim_Scene::sceneMounted([[maybe_unused]] Viewport* ctx)
{
    /// Initialize viewport.
    camera.setOriginViewportAnchor(Anchor::CENTER);
    camera.focusWorldRect(*ctx, 0.0, 0.0, world_w, world_h);

    /// Enable reverting to the current camera view.
    camera.uiSetCurrentAsDefault();

    /// Configure default navigator.
    navigator.setTarget(camera);
    navigator.setDirectCameraPanning(true);
}

void FluidSim_Scene::sceneUnmounted([[maybe_unused]] Viewport* ctx)
{
    /// Release state owned specifically for this Viewport.
}

void FluidSim_Scene::sceneDestroy()
{
    renderThreadQueue().post([this]()
    {
        fluid.destroy();
        fluid_display.destroy();
        surface.destroy();
#ifdef BITLOOP_DEV_MODE
        snapshot_preprocessor.destroyGL();
#endif
    });
}

void FluidSim_Scene::sceneProcess(
    [[maybe_unused]] double dt
)
{
    // redraw the scene every frame
    requestRedraw(true);
}

void FluidSim_Scene::viewportProcess(
    [[maybe_unused]] Viewport* ctx,
    double dt)
{
    if (simulation_failed || dt <= 0.0)
        return;

    // Equal substeps avoid tiny leftover timesteps that spoil pressure reuse.
    const double frame_time = 0.05;// std::min(dt, 0.05);
    const int step_count = int(std::ceil(frame_time * 120.0));
    const double step = frame_time / step_count;

    withFluidContext([&]()
    {
        for (int substep = 0; substep < step_count; ++substep)
        {
            // use one float-compatible tolerance for backend comparisons
            const auto result = fluid.step(step, 2000, 1e-3);

            if (!result.converged)
            {
                blPrint() << "Projection failed; max divergence: "
                    << result.max_divergence;

                simulation_failed = true;
                return;
            }
        }
    });
}

void FluidSim_Scene::renderFluidSurface() const
{
    surface.render(fluid.width, fluid.height, [&](const ShaderSurface& shader)
    {
        shader.bindTexture2D("u_water", fluid_display.texture(fluid));
    });
}

void FluidSim_Scene::viewportDraw(Viewport* ctx) const
{
    /// Draw the Scene on this viewport without modifying simulation state.
    ctx->transform(camera);
    renderFluidSurface();

    DQuad bounds{
        fluid.origin.x,
        fluid.origin.y,
        fluid.origin.x + fluid.width * fluid.cell_size,
        fluid.origin.y + fluid.height * fluid.cell_size
    };

    ctx->worldMode();
    ctx->drawShaderSurface(surface, bounds);
    ctx->drawWorldAxis();
}

void FluidSim_Scene::onEvent(Event e)
{
    if (e.ownedBy(this))
        navigator.handleWorldNavigation(e, true);
}

// void FluidSim_Scene::onPointerDown(PointerEvent e) {}
// void FluidSim_Scene::onPointerUp(PointerEvent e) {}
// void FluidSim_Scene::onPointerMove(PointerEvent e) {}
// void FluidSim_Scene::onWheel(PointerEvent e) {}
// void FluidSim_Scene::onKeyDown(KeyEvent e) {}
// void FluidSim_Scene::onKeyUp(KeyEvent e) {}

#ifdef BITLOOP_DEV_MODE
void FluidSim_Scene::traceFrame([[maybe_unused]] SimTraceFrame& frame) const
{
}

bool FluidSim_Scene::captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const
{
    if (route != "grid")
        return false;

    renderFluidSurface();
    image.resolution = IVec2(fluid.width, fluid.height);
    CapturePreprocessParams params;
    params.src_resolution = params.dst_resolution = image.resolution;
    // texture row zero already represents the top grid row
    params.flip_y = false;
    return snapshot_preprocessor.preprocessToFrame(surface.texture(), params, image.pixels);
}

bool FluidSim_Scene::saveCheckpoint(SimCheckpointWriter& writer) const
{
    (void)writer;
    return false;
}

bool FluidSim_Scene::loadCheckpoint(const SimCheckpointReader& reader)
{
    (void)reader;
    return false;
}
#endif

SIM_END;
