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

void FluidSim_Scene::uploadWaterTexture() const
{
    // Convert water fractions to display bytes without changing the simulation state.
    water_pixels.resize(fluid.water_fraction.size());

    for (size_t i = 0; i < fluid.water_fraction.size(); ++i)
    {
        double value = std::clamp(fluid.water_fraction[i], 0.0, 1.0);
        water_pixels[i] = static_cast<unsigned char>(value * 255.0 + 0.5);
    }

    // restore previous GL state on exit and allow tightly packed single-channel rows
    GLStateGuard guard;
    glActiveTexture(GL_TEXTURE0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (water_texture == 0)
    {
        // create the texture on first use
        glGenTextures(1, &water_texture);
        glBindTexture(GL_TEXTURE_2D, water_texture);

        // blend neighbouring cell values and prevent sampling across opposite edges
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // allocate one normalized byte per cell for the fixed-size fluid grid
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R8,
            fluid.width, fluid.height, 0,
            GL_RED, GL_UNSIGNED_BYTE, nullptr);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, water_texture);
    }

    // Replace the texture contents with the current water fractions without reallocating.
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0,
        fluid.width, fluid.height,
        GL_RED, GL_UNSIGNED_BYTE, water_pixels.data());
}

void FluidSim_Scene::initializePool(double amplitude)
{
    constexpr double pi = 3.14159265358979323846;

    for (int y = 0; y < fluid.height; ++y)
    {
        for (int x = 0; x < fluid.width; ++x)
        {
            // the experiment starts with a sloped, rippled surface
            const double surface =
                x + fluid.height * (0.5 + amplitude * std::cos(pi * (x*3 + 0.5) / fluid.width));

            // measure how much of this cell lies below the water surface
            fluid.water_fraction[y * fluid.width + x] =
                std::clamp(y + 1.0 - surface, 0.0, 1.0);
        }
    }
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
        if (water_texture)
        {
            glDeleteTextures(1, &water_texture);
            water_texture = 0;
        }

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

    for (int substep = 0; substep < step_count; ++substep) 
    {
        const auto result = fluid.step(step, 2000, 1e-8);

        if (!result.converged) 
        {
            blPrint() << "Projection failed; max divergence: "
                << result.max_divergence;

            simulation_failed = true;
            return;
        }
    }
}

void FluidSim_Scene::renderFluidSurface() const
{
    uploadWaterTexture();
    surface.render(fluid.width, fluid.height, [&](const ShaderSurface& shader)
    {
        shader.bindTexture2D("u_water", water_texture);
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
