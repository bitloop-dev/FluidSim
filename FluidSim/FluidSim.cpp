#include "FluidSim.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>

SIM_BEG;

using namespace bl;

namespace
{
double perlinNoise(double x, double y)
{
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const double fx = x - ix, fy = y - iy;

    auto gradient = [](int gx, int gy, double dx, double dy)
    {
        // Hash each lattice corner to a repeatable gradient direction.
        std::uint32_t hash = std::uint32_t(gx) * 0x9e3779b9u ^ std::uint32_t(gy) * 0x85ebca6bu ^ 0x12345678u;
        hash ^= hash >> 16;
        hash *= 0x7feb352du;
        hash ^= hash >> 15;
        hash *= 0x846ca68bu;
        hash ^= hash >> 16;
        constexpr double diagonal = 0.7071067811865476;
        constexpr double directions[8][2] = {
            {1, 0}, {-1, 0}, {0, 1}, {0, -1},
            {diagonal, diagonal}, {-diagonal, diagonal},
            {diagonal, -diagonal}, {-diagonal, -diagonal}
        };
        const auto& direction = directions[hash & 7u];
        return direction[0] * dx + direction[1] * dy;
    };

    // Quintic interpolation joins neighbouring cells with smooth derivatives.
    auto fade = [](double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); };
    const double sx = fade(fx), sy = fade(fy);
    const double top = std::lerp(gradient(ix, iy, fx, fy), gradient(ix + 1, iy, fx - 1.0, fy), sx);
    const double bottom = std::lerp(gradient(ix, iy + 1, fx, fy - 1.0), gradient(ix + 1, iy + 1, fx - 1.0, fy - 1.0), sx);
    return std::lerp(top, bottom, sy);
}
}

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
    if (ImGui::Button("Restart scene"))
        bl_schedule([](FluidSim_Scene& scene) { scene.restartSimulation(); });

    if (ImGui::CollapsingHeaderBox("View", true))
    {
        bl_scoped(camera);

        camera.populateUI();
        ImGui::EndCollapsingHeaderBox();
    }
    if (ImGui::CollapsingHeaderBox("Atmosphere", true))
    {
        bl_scoped(atmosphere_damping);
        bl_view(last_physics_ms);
        ImGui::Text("Physics: %.2f ms / frame", last_physics_ms);
        if constexpr (!decltype(FluidSim_Scene::fluid)::requires_render_thread)
        {
            ImGui::SliderFloat("Gas damping (per second)", &atmosphere_damping, 0.0f, 2.0f, "%.2f");
            ImGui::TextWrapped("Zero disables damping. Applies to gas-only velocities in every scenario, with or without the overlay.");
        }
        else ImGui::TextUnformatted("Gas damping currently requires the CPU backend.");
        ImGui::EndCollapsingHeaderBox();
    }
    if constexpr (!decltype(FluidSim_Scene::fluid)::requires_render_thread)
    {
        if (ImGui::CollapsingHeaderBox("Ambient eligibility mask", true))
        {
            bl_scoped(ambient_mask_enabled, ambient_mask_settings, mask_opacity);
            bl_view(ambient_mask_statistics, last_mask_ms);
            ImGui::Checkbox("Show ambient eligibility overlay", &ambient_mask_enabled);
            ImGui::TextWrapped("Clouds trace the atmosphere's motion. Air and Cloud count as the same atmosphere for this mask; both still move normally. Full physics runs everywhere.");
            auto& s = ambient_mask_settings;
            ImGui::SliderFloat("Speed tolerance", &s.speed_tolerance, 0.01f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Pressure departure tolerance", &s.pressure_tolerance, 0.01f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Pressure acceleration tolerance", &s.acceleration_tolerance, 0.01f, 1000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Composition tolerance", &s.composition_tolerance, 0.00001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Required quiet time", &s.quiet_seconds, 0.0f, 2.0f);
            ImGui::SliderInt("Disturbance buffer (cells)", &s.halo_cells, 0, 8);
            ImGui::SliderFloat("Mask opacity", &mask_opacity, 0.0f, 1.0f);
            const auto& counts = ambient_mask_statistics;
            ImGui::Text("Candidates: %d / %d (%.1f%%)", counts.candidates, counts.cells,
                100.0 * counts.candidates / std::max(1, counts.cells));
            ImGui::Text("CPU mask: %.3f ms / frame", last_mask_ms);
            ImGui::Text("Above full tolerance (overlapping):");
            ImGui::Text("Speed %d | Pressure %d | Acceleration %d", counts.speed, counts.pressure, counts.acceleration);
            ImGui::TextWrapped("Green: reservoir-connected candidate. Yellow: disturbed. Blue: different material. Orange: buffer. Grey: waiting. Purple: quiet but disconnected.");
            ImGui::TextWrapped("Pressure is compared with the supplied ambient background. Each cell wakes at full tolerance and clears below half. Orange buffers these states; quiet time follows.");
            ImGui::EndCollapsingHeaderBox();
        }
    }
}

void FluidSim_Scene::observeAmbientMask(double dt)
{
    const auto started = std::chrono::steady_clock::now();
    [&]<class Grid>(const Grid& grid) {
        if constexpr (!Grid::requires_render_thread)
            ambient_mask.observe(grid, dt, ambient_mask_settings);
    }(fluid);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    mask_seconds += elapsed;
    last_mask_ms += elapsed * 1000;
    ambient_mask_statistics = ambient_mask.statistics;
    observed_ambient_mask_settings = ambient_mask_settings;
    ambient_mask_observed = true;
}

void FluidSim_Scene::initializePool(double amplitude)
{
    //constexpr double pi = 3.14159265358979323846;
    const size_t cell_count = size_t(fluid.width) * fluid.height;

    using FluidGridType = decltype(fluid);
    using Scalar = FluidGridType::Scalar;

    FluidGridType::State initial;
    initial.liquid_fraction.resize(cell_count);

    const double cell_volume = fluid.cell_size * fluid.cell_size;
    const double cloud_noise_scale = 6.0 / fluid.width;

    initial.liquid_mass.assign(fluid.liquid_mixture.components.size(),
        std::vector<Scalar>(cell_count, 0.0));

    initial.gas_mass.assign(fluid.gas_mixture.components.size(),
        std::vector<Scalar>(cell_count, 0.0));

    (void)amplitude;

    for (int y = 0; y < fluid.height; ++y)
    {
        for (int x = 0; x < fluid.width; ++x)
        {
            // the experiment starts with a sloped, rippled surface
            const double tx = double(x) / double(fluid.width);

            //const double surface = (fluid.height * 0.7) + (std::cos(tx * 20.0) * double(fluid.height) * 0.02);
            double surface;

            switch (scenario.type)
            {
                case ScenarioType::COSINE_WAVES:
                    surface = (grid_h * 0.7) + (std::cos(tx * 20.0) * double(grid_h) * 0.02);
                    break;

                case ScenarioType::CRASHING_WAVE:
                    surface = (double(grid_h) * 0.7) + tx * (double(grid_h) * 0.5);
                    break;
            }

            // measure how much of this cell lies below the liquid surface
            const int i = y * fluid.width + x;
            const double fraction = std::clamp(y + 1.0 - surface, 0.0, 1.0);

            initial.liquid_fraction[i] = Scalar(fraction);

            // Fill the left quarter's liquid with Red; use Water elsewhere.
            const double red_share = 0.0;// x < fluid.width / 4 ? 1.0 : 0.0;

            for (size_t k = 0; k < fluid.liquid_mixture.components.size(); ++k)
            {
                const auto& component = fluid.liquid_mixture.components[k];

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

            for (size_t k = 0; k < fluid.gas_mixture.components.size(); ++k)
            {
                const auto& component = fluid.gas_mixture.components[k];

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

    withFluidContext([&]() { fluid.upload(initial); });
}

void FluidSim_Scene::sceneStart()
{
    initializePool();
    surface.setFragmentSource(fluid_shader_src);
}

void FluidSim_Scene::restartSimulation()
{
    // Refill the current scenario; keep controls, camera and engine options intact.
    initializePool();
    simulation_failed = false;
    simulation_time = real_time = 0;
    physics_seconds = last_physics_ms = 0;
    mask_seconds = last_mask_ms = 0;
    ambient_mask.reset();
    ambient_mask_observed = false;
    ambient_mask_statistics = {};
#ifdef BITLOOP_DEV_MODE
    processed_frames = 0;
#endif
    if (ambient_mask_enabled) observeAmbientMask(0);
    requestRedraw(true);
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
        mask_display.destroy();
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
    if (!ambient_mask_enabled) ambient_mask_observed = false;
    else if (!ambient_mask_observed || observed_ambient_mask_settings != ambient_mask_settings)
        observeAmbientMask(0);
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
    //const double frame_time = 0.04;// std::min(dt, 0.05);
    const double frame_time = scenario.frame_time;
    const int steps_per_frame = 60;
    //const double play_speed = 0.8;

    //const int step_count = int(play_speed / frame_time);
    const int step_count = int(std::ceil(frame_time * double(steps_per_frame)));
    const double step = frame_time / step_count;

    ///const double previous_time = simulation_time;
    simulation_time += frame_time;
    real_time += dt;

    withFluidContext([&]()
    {
        // Spawn water by replacing gas; the ordinary fluid step handles the fall.
        ///constexpr double drops_per_second = 20.0;
        ///constexpr int drop_radius = 2;
        ///auto rain = fluid.beginEdit();
        ///for (int drop = int(previous_time * drops_per_second); drop < int(simulation_time * drops_per_second); ++drop)
        ///{
        ///    const int center_x = int(random(drop_radius + 1, fluid.width - drop_radius - 1));
        ///    for (int y = -drop_radius; y <= drop_radius; ++y)
        ///    for (int x = -drop_radius; x <= drop_radius; ++x)
        ///        if (x * x + y * y <= drop_radius * drop_radius)
        ///            rain.setCell(center_x + x, drop_radius + 2 + y, fluid::Phase::Liquid, water_element);
        ///}
        ///rain.commit();

        last_physics_ms = 0;
        last_mask_ms = 0;
        [&]<class Grid>(Grid& grid) {
            if constexpr (!Grid::requires_render_thread) grid.gas_damping_rate = atmosphere_damping;
        }(fluid);
        for (int substep = 0; substep < step_count; ++substep)
        {
            // use one float-compatible tolerance for backend comparisons
            const auto started = std::chrono::steady_clock::now();
            const auto result = fluid.step(step, 2000, 1e-3);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            physics_seconds += elapsed;
            last_physics_ms += elapsed * 1000;

            if (!result.converged)
            {
                blPrint() << "Projection failed; max divergence: "
                    << result.max_divergence;

                simulation_failed = true;
                return;
            }
            if (ambient_mask_enabled) observeAmbientMask(step);
        }
    });
#ifdef BITLOOP_DEV_MODE
    if (!simulation_failed)
    {
        if (getProject().isHeadless() && processed_frames >= scenario.capture_i0 && processed_frames < scenario.capture_i1)
            traceSnapshot("grid", "scenario capture interval");
        ++processed_frames;
    }
#endif
}

void FluidSim_Scene::renderFluidSurface() const
{
    surface.render(grid_w, grid_h, [&](const ShaderSurface& shader)
    {
        const auto colour = fluid_display.texture(fluid);
        shader.bindTexture2D("u_color", colour);
        const auto overlay = ambient_mask_enabled ? mask_display.texture(ambient_mask) : 0;
        shader.bindTexture2D("u_mask", overlay ? overlay : colour);
        shader.setUniform1f("u_mask_opacity", overlay ? mask_opacity : 0.0f);
    });
}

void FluidSim_Scene::viewportDraw(Viewport* ctx) const
{
    /// Draw the Scene on this viewport without modifying simulation state.
    ctx->transform(camera);
    renderFluidSurface();

    DQuad bounds{
        0.0, 0.0, world_w, world_h
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
void FluidSim_Scene::traceFrame(SimTraceFrame& frame) const
{
    frame.metric("scenario_index", scenario_index);
    frame.metric("simulation_time", simulation_time);
    frame.metric("frame_time", scenario.frame_time);
    frame.metric("simulation_failed", simulation_failed);
    frame.metric("simulated_cells", fluid.width * fluid.height);
    frame.metric("physics_seconds", physics_seconds);
    frame.metric("last_physics_ms", last_physics_ms);
    frame.metric("atmosphere_damping", atmosphere_damping);
    frame.metric("mask_opacity", ambient_mask_enabled ? mask_opacity : 0.0f);
    frame.metric("mask_seconds", mask_seconds);
    frame.metric("last_mask_ms", last_mask_ms);
    const auto& m = ambient_mask_statistics;
    frame.metric("ambient_cells", m.cells);
    frame.metric("ambient_candidates", m.candidates);
    frame.metric("ambient_material", m.material);
    frame.metric("ambient_disturbed", m.disturbed);
    frame.metric("ambient_buffer", m.buffer);
    frame.metric("ambient_waiting", m.waiting);
    frame.metric("ambient_isolated", m.isolated);
    frame.metric("ambient_speed_exceeded", m.speed);
    frame.metric("ambient_pressure_exceeded", m.pressure);
    frame.metric("ambient_acceleration_exceeded", m.acceleration);
}

bool FluidSim_Scene::captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const
{
    std::uint32_t texture;
    if (route == "ambient-mask" && ambient_mask_enabled)
    {
        image.resolution = IVec2(fluid.width, fluid.height);
        texture = mask_display.texture(ambient_mask);
    }
    else if (route == "grid")
    {
        renderFluidSurface();
        image.resolution = IVec2(grid_w, grid_h);
        texture = surface.texture();
    }
    else return false;
    CapturePreprocessParams params;
    params.src_resolution = params.dst_resolution = image.resolution;
    // texture row zero already represents the top grid row
    params.flip_y = false;
    return snapshot_preprocessor.preprocessToFrame(texture, params, image.pixels);
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
