#include "FluidSim.h"
#include "MouseAttraction.h"
#include "FluidCheckpointIO.h"
#include <cmath>
#include <cstdint>
#include <chrono>

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
    {
        auto scene = create<FluidSim_Scene>("scene-" + std::to_string(i));
        scene.setLaunchConfigParser([](FluidSim_Scene::Config& config, const JSON::json& values)
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
    }
}

/// ────────────────────────── scene ──────────────────────────

void FluidSim_Scene::UI::sidebar()
{
    ImGui::TextDisabled("Drop a saved FluidSim WebP onto the viewport to resume.");
    bl_view(snapshot_error);
    if (!snapshot_error.empty()) ImGui::TextWrapped("%s", snapshot_error.c_str());
    if (ImGui::Button("Restart Simulation"))
        bl_schedule([](FluidSim_Scene& scene) { scene.restartSimulation(); });

    bl_pull(projection_tolerance);
    const double before_input = projection_tolerance;
    ImGui::InputDouble("Pressure Quality", &projection_tolerance, 0., 0., "%.6g");
    if (ImGui::IsItemActivated())
        pressure_tolerance_before_edit = before_input;
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        // Commit completed edits only: clearing/typing the number must not change the live solver target.
        if (std::isfinite(projection_tolerance) && projection_tolerance > 0.)
        {
            bl_push(projection_tolerance);
        }
        else
            projection_tolerance = pressure_tolerance_before_edit;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Positive maximum cell divergence (per second).\nLarger values trade accuracy for speed.\nPress Enter or leave the field to apply.");

    bl_pull(surface_tension);
    bl_view(adaptive_mode);
    ImGui::BeginDisabled(adaptive_mode != fluid::AdaptiveMode::DISABLED);
    const double old_tension = surface_tension;
    ImGui::InputDouble("Surface Tension", &surface_tension, 0., 0., "%.6g");
    if (ImGui::IsItemActivated()) surface_tension_before_edit = old_tension;
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        if (std::isfinite(float(surface_tension)) && surface_tension >= 0.)
        {
            const double value = surface_tension;
            bl_schedule([value](FluidSim_Scene& scene) { scene.setSurfaceTension(value); });
        }
        else surface_tension = surface_tension_before_edit;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Liquid-gas surface tension in scene units; zero disables it.\nUniform CPU and WebGPU. Stronger tension may require extra physics steps.\nPress Enter or leave the field to apply.");
    ImGui::EndDisabled();

    bl_scoped(mouse_attraction);
    ImGui::BeginDisabled(adaptive_mode != fluid::AdaptiveMode::DISABLED);
    ImGui::Checkbox("Mouse attraction", &mouse_attraction);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Replace uniform gravity with attraction towards the pointer.\nMove inside the viewport to choose the target; leaving it keeps the last target.");
    ImGui::EndDisabled();

    if (ImGui::CollapsingHeaderBox("Frame statistics", true))
    {
        bl_scoped(profile_stages);
        bl_view(frame_statistics);
        const auto& stats = frame_statistics;
        const double render_ms = __target.last_render_ms;

        ImGui::Checkbox("Detailed stage timings", &profile_stages);
        if (ImGui::BeginTable("Frame timings", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
        {
            ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("ms / frame", ImGuiTableColumnFlags_WidthFixed, 95.0f);
            ImGui::TableHeadersRow();
            auto row = [](const char* label, double ms, bool available = true)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(label);
                ImGui::TableSetColumnIndex(1);
                if (available) ImGui::Text("%.3f", ms);
                else ImGui::TextUnformatted("--");
            };
            row("Total frame time", stats.update_ms + render_ms);
            row("Render (CPU submission)", render_ms);
            row("Fluid processing", stats.fluid_ms);
            row("    Transport", stats.stages.transport_ms, stats.detailed);
            if (!stats.adaptive)
                row("    Forces", stats.stages.forces_ms, stats.detailed);
            row(stats.adaptive ? "    Projection + forces" : "    Projection", stats.stages.pressure_ms, stats.detailed);
            if (stats.adaptive)
            {
                row("    Refinement selection", stats.stages.selection_ms, stats.detailed);
                row("    Regrid + restoration", stats.stages.regrid_ms, stats.detailed);
            }
            row("Display colour/readback", stats.colour_ms);
            row("Other frame work", std::max(0., stats.update_ms - stats.fluid_ms - stats.colour_ms));
            ImGui::EndTable();
        }
        ImGui::TextDisabled("Fluid stages sum %d physics substep(s).", stats.substeps);
        ImGui::TextDisabled("%d transport steps; %d pressure iterations.", stats.transport_steps, stats.pressure_iterations);
        ImGui::TextDisabled("GPU profiling adds completion waits.");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Disable detailed timings for normal throughput.\nTotal = scene update + latest render CPU time.\nUI/presentation and GPU draw completion are excluded.");
        ImGui::EndCollapsingHeaderBox();
    }

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
    if (resume_image.empty()) restartSimulation();
    else resumeFromImage(resume_image);
    surface.setFragmentSource(fluid_shader_src);
}

void FluidSim_Scene::restartSimulation()
{
    simulation_failed = false;
    simulation_time = real_time = 0.;
    frame_statistics = {};
    frame_statistics.projection_tolerance = projection_tolerance;
    initializePool();
    requestRedraw(true);
}

void FluidSim_Scene::setSurfaceTension(double coefficient)
{
    withFluidContext([&] { fluid.setSurfaceTension(coefficient); });
    surface_tension = coefficient;
}

void FluidSim_Scene::updateAccelerationField()
{
    if (mouse_attraction)
    {
        if (!mouse_field)
        {
            const fluid::FluidField::Layout layout{grid_w, grid_h, cell_size, fluid.getOrigin(), 2};
            if (fluid.getDevice() == fluid::FluidDevice::CPU)
                mouse_field.emplace(layout);
            else
            {
                const auto context = fluid.getWebGPUContext();
                mouse_field.emplace(layout, context);
                mouse_field_generator = std::make_unique<fluid::FluidFieldWebGPUGenerator>(context, mouse_attraction_shader);
            }
        }
        const std::array<float, 4> parameters[] = {
            {float(mouse_target.x), float(mouse_target.y),
                float(mouse_attraction_strength), float(mouse_attraction_radius)}
        };
        if (mouse_field->isWebGPU())
            mouse_field_generator->generate(*mouse_field, parameters);
        else
            fluid::generateFluidField(*mouse_field, [&](DVec2 position) { return mouseAttraction(position, parameters[0]); });
        fluid.setAccelerationField(*mouse_field);
        mouse_field_active = true;
    }
    else if (mouse_field_active)
    {
        fluid.clearAccelerationField();
        mouse_field_active = false;
    }
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
    withFluidContext([&]()
    {
        mouse_field_generator.reset();
        mouse_field.reset();
        fluid.destroy();
    });
    renderThreadQueue().post([this]()
    {
        fluid_display.destroy();
        surface.destroy();
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

    using Clock = std::chrono::steady_clock;
    const auto frame_started = Clock::now();
    frame_statistics = {};
    frame_statistics.detailed = profile_stages;
    frame_statistics.adaptive = fluid.getAdaptiveMode() != fluid::AdaptiveMode::DISABLED;
    frame_statistics.projection_tolerance = projection_tolerance;

    // Equal substeps avoid tiny leftover timesteps that spoil pressure reuse.
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
                FluidGrid::StepTimings timings;
                const auto started = Clock::now();
                // Generate once per scene frame; every complete physics step uses the stored field.
                if (substep == 0)
                    updateAccelerationField();
                // use one float-compatible tolerance for backend comparisons
                const auto result = fluid.step(step, projection_max_iterations, frame_statistics.projection_tolerance, profile_stages ? &timings : nullptr);
                frame_statistics.fluid_ms += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
                frame_statistics.stages.transport_ms += timings.transport_ms;
                frame_statistics.stages.forces_ms += timings.forces_ms;
                frame_statistics.stages.pressure_ms += timings.pressure_ms;
                frame_statistics.stages.selection_ms += timings.selection_ms;
                frame_statistics.stages.regrid_ms += timings.regrid_ms;
                frame_statistics.substeps += result.physical_steps;
                frame_statistics.transport_steps += result.transport_steps;
                frame_statistics.pressure_iterations += result.iterations;

                if (!result.converged)
                {
                    blPrint() << "Projection failed; max divergence: "
                        << result.max_divergence;

                    simulation_failed = true;
                    return;
                }
            }

            // Observe on the simulation/device owner; render only the derived image.
            const auto colour_started = Clock::now();
            fluid.writeColorPixels(display_pixels);
            frame_statistics.colour_ms = std::chrono::duration<double, std::milli>(Clock::now() - colour_started).count();
        });
    }
    catch (const std::runtime_error& error)
    {
        simulation_failed = true;
        blPrint() << error.what();
    }
    frame_statistics.update_ms = std::chrono::duration<double, std::milli>(Clock::now() - frame_started).count();
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
    const auto started = std::chrono::steady_clock::now();
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
    last_render_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

void FluidSim_Scene::onEvent(Event e)
{
    if (e.type() == EventType::DROP_FILE)
    {
        const auto target = e.ctx_hovered() ? e.ctx_hovered() : e.ctx_focused();
        const auto& mounted = mountedToViewports();
        if ((target && std::find(mounted.begin(), mounted.end(), target) != mounted.end()) ||
            (!target && getProject().scenes().size() == 1))
        {
            try
            {
                resumeFromImage(e.data().drop.data);
                snapshot_error.clear();
            }
            catch (const std::exception& error)
            {
                snapshot_error = error.what();
            }
        }
        return;
    }
    if (e.ownedBy(this))
        navigator.handleWorldNavigation(e, true);

    if (e.type() == EventType::MOUSE_MOTION)
    {
        // Hover motion has no drag owner; convert through the viewport under the cursor.
        auto* viewport = e.ctx_hovered();
        if (viewport && viewport->mountedScene() == this)
        {
            const auto& motion = e.data().motion;
            mouse_target = camera.transformFor(*viewport).toWorld<double>(
                viewport->toSurfaceX(motion.x), viewport->toSurfaceY(motion.y));
        }
    }
    else if (e.ownedBy(this) && (e.type() == EventType::FINGER_MOTION || e.isPressEvent()))
    {
        PointerEvent pointer(e);
        if (pointer.hoveredOver(this))
            mouse_target = pointer.worldPosition(camera);
    }
}

JSON::json FluidSim_Scene::captureSceneState() const
{
    return {
        {"version", 1}, {"simulation_time", simulation_time}, {"real_time", real_time},
        {"simulation_failed", simulation_failed}, {"projection_tolerance", projection_tolerance},
        {"profile_stages", profile_stages}, {"frame_time", frame_time},
        {"mouse_attraction", mouse_attraction},
        {"mouse_attraction_strength", mouse_attraction_strength},
        {"mouse_attraction_radius", mouse_attraction_radius},
        {"mouse_target", {mouse_target.x, mouse_target.y}},
        {"steps_per_frame", steps_per_frame}, {"projection_max_iterations", projection_max_iterations},
        {"device", int(fluid.getDevice())}, {"adaptive_mode", int(fluid.getAdaptiveMode())},
        {"scene_elements", {air_element, water_element, red_element, cloud_element}}
    };
}

bool FluidSim_Scene::captureSnapshot(SnapshotFrame& output) const
{
    fluid::FluidSnapshot snapshot;
    snapshot.width = fluid.getWidth();
    snapshot.height = fluid.getHeight();
    snapshot.cell_size = fluid.getCellSize();
    snapshot.scene = captureSceneState();
    bool saved = false;
    withFluidContext([&] { saved = fluid.saveCheckpoint(snapshot.checkpoint); });
    if (!saved) throw std::runtime_error("This fluid engine does not support resumable snapshots");
    output.resolution = {snapshot.width, snapshot.height};
    output.frame.data = fluid::fluidSnapshotPixels(display_pixels);
    output.frame.payload = fluid::encodeFluidSnapshot(snapshot);
    return true;
}

void FluidSim_Scene::restoreSceneState(const JSON::json& scene, const fluid::FluidCheckpoint& checkpoint,
    int width, int height, double spacing)
{
    if (scene.at("version") != 1 || scene.at("frame_time") != frame_time ||
        scene.at("steps_per_frame") != steps_per_frame || scene.at("projection_max_iterations") != projection_max_iterations)
        throw std::runtime_error("Snapshot scene timestep/solver policy does not match this build");
    const int device = scene.at("device").get<int>();
    const int mode = scene.at("adaptive_mode").get<int>();
    if ((device != int(fluid::FluidDevice::CPU) && device != int(fluid::FluidDevice::GPU)) ||
        (mode != 0 && mode != 2 && mode != 4 && mode != 8 && mode != 16 && mode != 32))
        throw std::runtime_error("Unknown saved fluid backend");
    const double saved_simulation_time = scene.at("simulation_time").get<double>();
    const double saved_real_time = scene.at("real_time").get<double>();
    const double saved_tolerance = scene.at("projection_tolerance").get<double>();
    if (!std::isfinite(saved_tolerance) || saved_tolerance <= 0)
        throw std::runtime_error("Invalid saved pressure quality");
    const bool saved_failure = scene.at("simulation_failed").get<bool>();
    const bool saved_profiling = scene.at("profile_stages").get<bool>();
    const auto ids = scene.at("scene_elements").get<std::array<ElementId, 4>>();
    const bool saved_mouse_attraction = scene.value("mouse_attraction", false);
    const double saved_strength = scene.value("mouse_attraction_strength", mouse_attraction_strength);
    const double saved_radius = scene.value("mouse_attraction_radius", mouse_attraction_radius);
    const auto saved_target = scene.value("mouse_target", std::array<double, 2>{width * spacing * 0.5, height * spacing * 0.5});
    if (!std::isfinite(saved_strength) || !std::isfinite(saved_radius) || saved_radius <= 0 ||
        !std::isfinite(saved_target[0]) || !std::isfinite(saved_target[1]))
        throw std::runtime_error("Invalid saved mouse attraction settings");

    // Build and validate a replacement before releasing the current simulation.
    FluidGrid restored(width, height, spacing);
    restored.setDevice(fluid::FluidDevice(device));
    restored.setAdaptiveMode(fluid::AdaptiveMode(mode));
    if (!restored.loadCheckpoint(checkpoint))
        throw std::runtime_error("The saved fluid backend does not support checkpoints");
    for (const auto id : ids)
        if (id >= restored.getElements().size()) throw std::runtime_error("Invalid saved scene element");
    std::vector<float> pixels;
    restored.writeColorPixels(pixels);
    withFluidContext([&]
    {
        mouse_field_generator.reset();
        mouse_field.reset();
        fluid.destroy();
        fluid = std::move(restored);
    });
    display_pixels = std::move(pixels);
    grid_w = width;
    grid_h = height;
    cell_size = spacing;
    world_w = width * spacing;
    world_h = height * spacing;
    fluid_device = fluid.getDevice();
    adaptive_mode = fluid.getAdaptiveMode();
    gravity_y = fluid.getGravityY();
    surface_tension = fluid.getSurfaceTension();
    mouse_attraction = saved_mouse_attraction;
    mouse_attraction_strength = saved_strength;
    mouse_attraction_radius = saved_radius;
    mouse_target = {saved_target[0], saved_target[1]};
    mouse_field_active = mouse_attraction;
    simulation_time = saved_simulation_time;
    real_time = saved_real_time;
    projection_tolerance = saved_tolerance;
    simulation_failed = saved_failure;
    profile_stages = saved_profiling;
    air_element = ids[0];
    water_element = ids[1];
    red_element = ids[2];
    cloud_element = ids[3];
    frame_statistics = {};
    frame_statistics.projection_tolerance = projection_tolerance;
    requestRedraw(true);
}

void FluidSim_Scene::resumeFromImage(const std::filesystem::path& image)
{
    const auto snapshot = fluid::readFluidSnapshot(image);
    restoreSceneState(snapshot.scene, snapshot.checkpoint, snapshot.width, snapshot.height, snapshot.cell_size);
    const auto origin = fluid.getOrigin();
    for (auto* viewport : mountedToViewports())
        camera.focusWorldRect(*viewport, origin.x, origin.y, origin.x + world_w, origin.y + world_h);
    snapshot_error.clear();
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
    frame.metric("simulation_time", simulation_time);
    frame.metric("real_time", real_time);
    frame.metric("update_ms", frame_statistics.update_ms);
    frame.metric("fluid_ms", frame_statistics.fluid_ms);
    frame.metric("colour_ms", frame_statistics.colour_ms);
    frame.metric("physics_substeps", frame_statistics.substeps);
    frame.metric("transport_substeps", frame_statistics.transport_steps);
    frame.metric("pressure_iterations", frame_statistics.pressure_iterations);
    frame.metric("surface_tension", fluid.getSurfaceTension());
    frame.metric("phase_timings_available", frame_statistics.detailed);
    frame.metric("projection_tolerance", frame_statistics.projection_tolerance);
    if (frame_statistics.detailed)
    {
        frame.metric("transport_ms", frame_statistics.stages.transport_ms);
        frame.metric("forces_ms", frame_statistics.stages.forces_ms);
        frame.metric("projection_ms", frame_statistics.stages.pressure_ms);
        frame.metric("selection_ms", frame_statistics.stages.selection_ms);
        frame.metric("regrid_ms", frame_statistics.stages.regrid_ms);
    }
    frame.metric("simulation_failed", simulation_failed);
}

bool FluidSim_Scene::captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const
{
    if (route != "grid" || display_pixels.empty())
        return false;

    SnapshotFrame snapshot;
    captureSnapshot(snapshot);
    image.resolution = snapshot.resolution;
    image.pixels = std::move(snapshot.frame.data);
    image.payload = std::move(snapshot.frame.payload);
    return true;
}

bool FluidSim_Scene::saveCheckpoint(SimCheckpointWriter& writer) const
{
    fluid::FluidCheckpoint checkpoint;
    bool saved = false;
    withFluidContext([&] { saved = fluid.saveCheckpoint(checkpoint); });
    if (!saved) return false;

    writer.writeJson("scene", captureSceneState());
    fluid::writeFluidCheckpoint(writer, checkpoint, checkpoint_brotli_quality);
    return true;
}

bool FluidSim_Scene::loadCheckpoint(const SimCheckpointReader& reader)
{
    restoreSceneState(reader.readJson("scene"), fluid::readFluidCheckpoint(reader),
        fluid.getWidth(), fluid.getHeight(), fluid.getCellSize());
    return true;
}
#endif

SIM_END;
