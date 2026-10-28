#include "FluidSim.h"
#include <cmath>
#include <cstdint>

#include "noise.h"

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
    const size_t cell_count = size_t(fluid.getWidth()) * fluid.getHeight();

    using FluidGridType = decltype(fluid);
    using Scalar = FluidGridType::Scalar;

    FluidGridType::State initial;
    initial.liquid_fraction.resize(cell_count);

    const double cell_volume = fluid.getCellSize() * fluid.getCellSize();
    const double cloud_noise_scale = 6.0 / fluid.getWidth();

    initial.liquid_mass.assign(fluid.getLiquidMixture().components.size(),
        std::vector<Scalar>(cell_count, 0.0));

    initial.gas_mass.assign(fluid.getGasMixture().components.size(),
        std::vector<Scalar>(cell_count, 0.0));

    for (int y = 0; y < fluid.getHeight(); ++y)
    {
        for (int x = 0; x < fluid.getWidth(); ++x)
        {
            // the experiment starts with a sloped, rippled surface
            const double surface =
                x + fluid.getHeight() * (0.5 + amplitude * std::cos(pi * (x*3 + 0.5) / fluid.getWidth()));

            // measure how much of this cell lies below the liquid surface
            const int i = y * fluid.getWidth() + x;
            const double fraction = std::clamp(y + 1.0 - surface, 0.0, 1.0);

            initial.liquid_fraction[i] = Scalar(fraction);

            // Fill the left quarter's liquid with Red; use Water elsewhere.
            const double red_share = x < fluid.getWidth() / 4 ? 1.0 : 0.0;

            for (size_t k = 0; k < fluid.getLiquidMixture().components.size(); ++k)
            {
                const auto& component = fluid.getLiquidMixture().components[k];

                double share = 0.0;

                if (component.element == water_element)
                    share = 1.0 - red_share;
                else if (component.element == red_element)
                    share = red_share;

                // Split the existing liquid volume between its ingredients.
                initial.liquid_mass[k][i] =
                    Scalar(component.density * fraction * cell_volume * share);
            }

            // Layer three Perlin scales to give broad clouds some smaller detail.
            const double nx = (x + 0.5) * cloud_noise_scale;
            const double ny = (y + 0.5) * cloud_noise_scale;
            const double cloud_noise = (
                perlinNoise(nx, ny) +
                0.5 * perlinNoise(2.0 * nx, 2.0 * ny) +
                0.25 * perlinNoise(4.0 * nx, 4.0 * ny)) / 1.75;

            const double cloud_share = std::clamp(0.15 + 2.5 * cloud_noise, 0.0, 1.0);

            for (size_t k = 0; k < fluid.getGasMixture().components.size(); ++k)
            {
                const auto& component = fluid.getGasMixture().components[k];

                double share = 0.0;
                if (component.element == air_element)
                    share = 1.0 - cloud_share;
                else if (component.element == cloud_element)
                    share = cloud_share;

                // Clouds replace air within the existing gas volume.
                initial.gas_mass[k][i] =
                    Scalar(component.density * (1.0 - fraction) * cell_volume * share);
            }
        }
    }

    try
    {
        withFluidContext([&]()
        {
            fluid.upload(initial);
            fluid.writeColorPixels(display_pixels);
        });
    }
    catch (const std::runtime_error& error)
    {
        simulation_failed = true;
        blPrint() << error.what();
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
    withFluidContext([&]() { fluid.destroy(); });
    renderThreadQueue().post([this]()
    {
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
    const int steps_per_frame = 40;
    //const double play_speed = 0.8;

    //const int step_count = int(play_speed / frame_time);
    const int step_count = int(std::ceil(frame_time * double(steps_per_frame)));
    const double step = frame_time / step_count;

    ///const double previous_time = simulation_time;
    simulation_time += frame_time;
    real_time += dt;

    try
    {
        withFluidContext([&]()
        {
            // Spawn water by replacing gas; the ordinary fluid step handles the fall.
            ///constexpr double drops_per_second = 20.0;
            ///constexpr int drop_radius = 2;
            ///auto rain = fluid.beginEdit();
            ///for (int drop = int(previous_time * drops_per_second); drop < int(simulation_time * drops_per_second); ++drop)
            ///{
            ///    const int center_x = int(random(drop_radius + 1, fluid.getWidth() - drop_radius - 1));
            ///    for (int y = -drop_radius; y <= drop_radius; ++y)
            ///    for (int x = -drop_radius; x <= drop_radius; ++x)
            ///        if (x * x + y * y <= drop_radius * drop_radius)
            ///            rain.setCell(center_x + x, drop_radius + 2 + y, fluid::Phase::Liquid, water_element);
            ///}
            ///rain.commit();

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

            // Observe on the simulation/device owner; render only the derived image.
            fluid.writeColorPixels(display_pixels);
        });
    }
    catch (const std::runtime_error& error)
    {
        simulation_failed = true;
        blPrint() << error.what();
    }
}

void FluidSim_Scene::renderFluidSurface() const
{
    // Device initialization can fail before the first derived image is ready.
    if (display_pixels.empty()) return;
    surface.render(fluid.getWidth(), fluid.getHeight(), [&](const ShaderSurface& shader)
    {
        shader.bindTexture2D("u_color", fluid_display.texture(fluid.getWidth(), fluid.getHeight(), display_pixels));
    });
}

void FluidSim_Scene::viewportDraw(Viewport* ctx) const
{
    /// Draw the Scene on this viewport without modifying simulation state.
    ctx->transform(camera);
    renderFluidSurface();

    DQuad bounds{
        fluid.getOrigin().x,
        fluid.getOrigin().y,
        fluid.getOrigin().x + fluid.getWidth() * fluid.getCellSize(),
        fluid.getOrigin().y + fluid.getHeight() * fluid.getCellSize()
    };

    ctx->worldMode();
    ctx->drawShaderSurface(surface, bounds);
    ctx->drawWorldAxis(0.3, 0.0);

    ctx->worldHudMode();
    ctx->setFontSize(12.0);
    ctx->fillText<double>("Sim Time: " + std::to_string(simulation_time), {10.0, 10.0});
    ctx->fillText<double>("Real Time: " + std::to_string(real_time), {10.0, 30.0});
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
    if (route != "grid" || display_pixels.empty())
        return false;

    renderFluidSurface();
    image.resolution = IVec2(fluid.getWidth(), fluid.getHeight());
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
