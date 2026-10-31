#include "MouseAttraction.h"
#include "FluidCheckpointIO.h"
#include <chrono>
#include <cmath>

SIM_BEG;

template<class Derived>
FluidScene<Derived>::FluidScene(const FluidSceneSettings& settings, const Config& config)
    : grid_w(settings.grid_w), grid_h(settings.grid_h), cell_size(settings.cell_size),
      fluid(grid_w, grid_h, cell_size),
      fluid_device(settings.fluid_device), adaptive_mode(settings.adaptive_mode),
      frame_time(settings.frame_time), steps_per_second(settings.steps_per_second),
      projection_max_iterations(settings.projection_max_iterations),
      projection_tolerance(settings.projection_tolerance), pressure_recovery_frames(settings.pressure_recovery_frames), gravity_y(settings.gravity_y),
      surface_tension(settings.surface_tension),
      checkpoint_brotli_quality(config.checkpoint_brotli_quality), resume_image(config.resume_image)
{
    adaptive_settings.prediction_horizon = adaptive_settings.check_interval + frame_time;
    fluid.setAdaptiveSettings(adaptive_settings);
    fluid.setDevice(fluid_device);
    fluid.setAdaptiveMode(adaptive_mode);
    fluid.setGravityY(gravity_y);
    fluid.setSurfaceTension(surface_tension);
    fluid.setPressureRecoveryTime(pressure_recovery_frames * frame_time);
}

template<class Derived>
void FluidScene<Derived>::sceneStart()
{
    if (resume_image.empty()) restartSimulation();
    else resumeFromImage(resume_image);
    surface.setFragmentSource(fluid_shader_src);
}

template<class Derived>
void FluidScene<Derived>::restartSimulation()
{
    if (grid_w != fluid.getWidth() || grid_h != fluid.getHeight() ||
        fluid_device != fluid.getDevice() || adaptive_mode != fluid.getAdaptiveMode())
    {
        const double proximity_scale = liquid_tint_depth /
            std::hypot(double(fluid.getWidth()), double(fluid.getHeight()));
        FluidGrid replacement(grid_w, grid_h, cell_size);
        replacement.setAdaptiveSettings(adaptive_settings);
        replacement.setDevice(fluid_device);
        replacement.setAdaptiveMode(adaptive_mode);
        replacement.configureMaterials(fluid.getSubstances(), fluid.getLiquidMixture(), fluid.getGasMixture());
        replacement.setOrigin(fluid.getOrigin());
        replacement.setGravityY(gravity_y);
        replacement.setSurfaceTension(surface_tension);
        replacement.setPressureRecoveryTime(pressure_recovery_frames * frame_time);
        withFluidContext([&]
        {
            mouse_field_generator.reset();
            mouse_field.reset();
            mouse_field_active = false;
            fluid.destroy();
            fluid = std::move(replacement);
        });
        liquid_tint_depth = float(proximity_scale * std::hypot(double(grid_w), double(grid_h)));
        world_w = grid_w * cell_size;
        world_h = grid_h * cell_size;
    }
    simulation_failed = false;
    simulation_time = real_time = 0.;
    frame_statistics = {};
    ++frame_statistics_generation;
    frame_statistics.projection_tolerance = projection_tolerance;
    try
    {
        withFluidContext([&]
        {
            fluid.upload(initialFluidState());
            fluid.writeColorPixels(display_pixels, true);
        });
    }
    catch (const std::runtime_error& error)
    {
        simulation_failed = true;
        blPrint() << error.what();
    }
    this->requestRedraw(true);
}

template<class Derived>
void FluidScene<Derived>::applyPreset(const FluidSceneSettings& settings)
{
    // Prepare all backend settings together, including accurate/relaxed mode,
    // before restarting. Applying them one UI control at a time can restart twice.
    auto next_adaptive_settings = adaptive_settings;
    next_adaptive_settings.prediction_horizon = next_adaptive_settings.check_interval + settings.frame_time;
    FluidGrid replacement(settings.grid_w, settings.grid_h, settings.cell_size);
    replacement.setAdaptiveSettings(next_adaptive_settings);
    replacement.setDevice(settings.fluid_device);
    replacement.setAdaptiveMode(settings.adaptive_mode);
    replacement.configureMaterials(fluid.getSubstances(), fluid.getLiquidMixture(), fluid.getGasMixture());
    replacement.setOrigin(fluid.getOrigin());
    replacement.setGravityY(settings.gravity_y);
    replacement.setSurfaceTension(settings.surface_tension);
    replacement.setPressureRecoveryTime(settings.pressure_recovery_frames * settings.frame_time);

    const bool resize = settings.grid_w != fluid.getWidth() || settings.grid_h != fluid.getHeight() ||
        settings.cell_size != fluid.getCellSize();
    const double proximity_scale = liquid_tint_depth /
        std::hypot(double(fluid.getWidth()), double(fluid.getHeight()));
    withFluidContext([&]
    {
        mouse_field_generator.reset();
        mouse_field.reset();
        mouse_field_active = false;
        fluid.destroy();
        fluid = std::move(replacement);
    });

    grid_w = settings.grid_w;
    grid_h = settings.grid_h;
    cell_size = settings.cell_size;
    world_w = grid_w * cell_size;
    world_h = grid_h * cell_size;
    fluid_device = settings.fluid_device;
    adaptive_mode = settings.adaptive_mode;
    adaptive_settings = next_adaptive_settings;
    frame_time = settings.frame_time;
    steps_per_second = settings.steps_per_second;
    projection_max_iterations = settings.projection_max_iterations;
    projection_tolerance = settings.projection_tolerance;
    pressure_recovery_frames = settings.pressure_recovery_frames;
    gravity_y = settings.gravity_y;
    surface_tension = settings.surface_tension;
    liquid_tint_depth = float(proximity_scale * std::hypot(double(grid_w), double(grid_h)));

    restartSimulation();
    snapshot_error.clear();
    if (resize)
    {
        const auto origin = fluid.getOrigin();
        for (auto* viewport : this->mountedToViewports())
            camera.focusWorldRect(*viewport, origin.x, origin.y, origin.x + world_w, origin.y + world_h);
        camera.uiSetCurrentAsDefault();
    }
}

template<class Derived>
void FluidScene<Derived>::setSurfaceTension(double coefficient)
{
    withFluidContext([&] { fluid.setSurfaceTension(coefficient); });
    surface_tension = coefficient;
}

template<class Derived>
void FluidScene<Derived>::setPressureRecoveryFrames(int frames)
{
    if (frames < 0 || frames > 10)
        throw std::invalid_argument("Pressure recovery must be between zero and ten frames");
    if (frames > 0 && (fluid_device != fluid::FluidDevice::GPU || adaptive_mode != fluid::AdaptiveMode::DISABLED))
        throw std::invalid_argument("Pressure recovery requires uniform WebGPU");
    const bool restart = (frames == 0) != (pressure_recovery_frames == 0);
    withFluidContext([&]
    {
        if (restart)
        {
            mouse_field_generator.reset();
            mouse_field.reset();
            mouse_field_active = false;
            fluid.destroy();
        }
        fluid.setPressureRecoveryTime(frames * frame_time);
    });
    pressure_recovery_frames = frames;
    if (restart) restartSimulation();
}

template<class Derived>
void FluidScene<Derived>::updateAccelerationField()
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

template<class Derived>
void FluidScene<Derived>::sceneMounted(Viewport* ctx)
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

template<class Derived>
void FluidScene<Derived>::sceneDestroy()
{
    withFluidContext([&]()
    {
        mouse_field_generator.reset();
        mouse_field.reset();
        fluid.destroy();
    });
    this->renderThreadQueue().post([this]()
    {
        fluid_display.destroy();
        surface.destroy();
    });
}

template<class Derived>
void FluidScene<Derived>::sceneProcess(double dt)
{
    this->requestRedraw(true);
    if (simulation_failed || dt <= 0.0)
        return;

    using Clock = std::chrono::steady_clock;
    const auto frame_started = Clock::now();
    frame_statistics = {};
    frame_statistics.detailed = profile_stages;
    frame_statistics.adaptive = fluid.getAdaptiveMode() != fluid::AdaptiveMode::DISABLED;
    frame_statistics.projection_tolerance = projection_tolerance;

    // Equal substeps avoid tiny leftover timesteps that spoil pressure reuse.
    int step_count = int(std::ceil(frame_time * double(steps_per_second)));
    if (fluid.getDevice() == fluid::FluidDevice::GPU &&
        fluid.getAdaptiveMode() == fluid::AdaptiveMode::DISABLED &&
        std::ceil(frame_time / fluid.capillaryStepLimit()) >= step_count)
    {
        // The engine's capillary steps already satisfy the scene's minimum rate.
        // Subdivide the whole frame there once (e.g. five steps instead of 2 x 3).
        step_count = 1;
    }
    const double step = frame_time / step_count;

    simulation_time += frame_time;
    real_time += dt;

    try
    {
        withFluidContext([&]()
        {
            beforeFluidFrame(frame_time);
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
                frame_statistics.max_volume_error = std::max(frame_statistics.max_volume_error, result.max_volume_error);
                frame_statistics.rms_volume_error = std::max(frame_statistics.rms_volume_error, result.rms_volume_error);

                if (!result.accepted())
                {
                    blPrint() << "Projection failed; max divergence: "
                        << result.max_divergence;

                    simulation_failed = true;
                    return;
                }
            }

            // Observe on the simulation/device owner; render only the derived image.
            const auto colour_started = Clock::now();
            fluid.writeColorPixels(display_pixels, true);
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

template<class Derived>
void FluidScene<Derived>::renderFluidSurface() const
{
    // Device initialization can fail before the first derived image is ready.
    if (display_pixels.empty()) return;
    surface.render(fluid.getWidth(), fluid.getHeight(), [&](const ShaderSurface& shader)
    {
        shader.bindTexture2D("u_color", fluid_display.texture(fluid.getWidth(), fluid.getHeight(), display_pixels));
        shader.bindTexture2D("u_proximity", fluid_display.proximityTexture(fluid.getWidth(), fluid.getHeight(), display_pixels, liquid_tint_depth));
        shader.setUniform1f("u_deep_brightness", liquid_deep_brightness);
    });
}

template<class Derived>
void FluidScene<Derived>::viewportDraw(Viewport* ctx) const
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
    drawScenario(ctx);
    ctx->drawWorldAxis(0.3, 0.0);

    ctx->worldHudMode();
    ctx->setFontSize(12.0);
    ctx->fillText<double>("Sim Time: " + std::to_string(simulation_time), {10.0, 10.0});
    ctx->fillText<double>("Real Time: " + std::to_string(real_time), {10.0, 30.0});
    last_render_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();

    auto& history = frame_time_history;
    if (history.generation != frame_statistics_generation)
    {
        history = {};
        history.generation = frame_statistics_generation;
    }
    // Paused redraws and extra viewports must not count the same simulation frame twice.
    if (frame_statistics.update_ms > 0. && history.last_simulation_time != simulation_time)
    {
        history.add(real_time, frame_statistics.update_ms + last_render_ms);
        history.last_simulation_time = simulation_time;
    }
}

template<class Derived>
void FluidScene<Derived>::onEvent(Event e)
{
    if (e.type() == EventType::DROP_FILE)
    {
        const auto target = e.ctx_hovered() ? e.ctx_hovered() : e.ctx_focused();
        const auto& mounted = this->mountedToViewports();
        if ((target && std::find(mounted.begin(), mounted.end(), target) != mounted.end()) ||
            (!target && this->getProject().scenes().size() == 1))
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

template<class Derived>
JSON::json FluidScene<Derived>::captureSceneState() const
{
    return {
        {"version", pressure_recovery_frames > 0 ? 4 : 3}, {"scenario", Derived::scenario_id}, {"simulation_time", simulation_time}, {"real_time", real_time},
        {"simulation_failed", simulation_failed}, {"projection_tolerance", projection_tolerance},
        {"pressure_recovery_frames", pressure_recovery_frames},
        {"profile_stages", profile_stages}, {"frame_time", frame_time},
        {"liquid_tint_depth", liquid_tint_depth}, {"liquid_deep_brightness", liquid_deep_brightness},
        {"mouse_attraction", mouse_attraction},
        {"mouse_attraction_strength", mouse_attraction_strength},
        {"mouse_attraction_radius", mouse_attraction_radius},
        {"mouse_target", {mouse_target.x, mouse_target.y}},
        {"steps_per_second", steps_per_second}, {"projection_max_iterations", projection_max_iterations},
        {"device", int(fluid.getDevice())}, {"adaptive_mode", int(fluid.getAdaptiveMode())},
        {"scenario_state", captureScenarioState()}
    };
}

template<class Derived>
void FluidScene<Derived>::captureFluidImage(CaptureFrame& output) const
{
    output.resolution = {fluid.getWidth(), fluid.getHeight()};
    const_cast<FluidScene*>(this)->renderThreadQueue().invokeBlocking([&]
    {
        GLStateGuard guard;
        const bool scissor = glIsEnabled(GL_SCISSOR_TEST);
        glDisable(GL_SCISSOR_TEST); // A viewport's clip must not crop a native capture.
        renderFluidSurface();
        GLuint framebuffer;
        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, surface.texture(), 0);
        output.frame.data.resize(size_t(output.resolution.x) * output.resolution.y * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, output.resolution.x, output.resolution.y, GL_RGBA, GL_UNSIGNED_BYTE, output.frame.data.data());
        glDeleteFramebuffers(1, &framebuffer);
        if (scissor) glEnable(GL_SCISSOR_TEST);
    });
}

template<class Derived>
bool FluidScene<Derived>::captureFrame(const CapturePreset& request, CaptureFrame& output) const
{
    output.frame.payload.clear();
    if (request.isVideo())
    {
        captureFluidImage(output);
        return true;
    }

    fluid::FluidSnapshot snapshot;
    snapshot.width = fluid.getWidth();
    snapshot.height = fluid.getHeight();
    snapshot.cell_size = fluid.getCellSize();
    snapshot.scene = captureSceneState();
    bool saved = false;
    withFluidContext([&] { saved = fluid.saveCheckpoint(snapshot.checkpoint); });
    if (!saved) throw std::runtime_error("This fluid engine does not support resumable snapshots");
    captureFluidImage(output);
    output.frame.payload = fluid::encodeFluidSnapshot(snapshot);
    return true;
}

template<class Derived>
void FluidScene<Derived>::restoreSceneState(const JSON::json& scene, const fluid::FluidCheckpoint& checkpoint,
    int width, int height, double spacing)
{
    if (scene.at("version") != 2 && scene.at("version") != 3 && scene.at("version") != 4)
        throw std::runtime_error("Unsupported fluid scene snapshot version");
    if (scene.at("scenario") != Derived::scenario_id)
        throw std::runtime_error("Snapshot belongs to a different fluid scenario");
    if (scene.at("steps_per_second") != steps_per_second || scene.at("projection_max_iterations") != projection_max_iterations)
        throw std::runtime_error("Snapshot timestep/solver policy does not match this scene");
    const double saved_frame_time = scene.at("frame_time").get<double>();
    const int saved_recovery = scene.at("version") == 4 ? scene.at("pressure_recovery_frames").get<int>() : 0;
    if (saved_recovery < 0 || saved_recovery > 10)
        throw std::runtime_error("Invalid saved pressure recovery horizon");
    if (!std::isfinite(saved_frame_time) || saved_frame_time <= 0.)
        throw std::runtime_error("Invalid saved frame time");
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
    const float saved_tint_depth = scene.value("liquid_tint_depth", liquid_tint_depth);
    const float saved_deep_brightness = scene.value("liquid_deep_brightness", liquid_deep_brightness);
    if (!std::isfinite(saved_tint_depth) || saved_tint_depth <= 0.f ||
        !std::isfinite(saved_deep_brightness) || saved_deep_brightness < 0.f || saved_deep_brightness > 1.f)
        throw std::runtime_error("Invalid saved liquid appearance settings");
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
    if (restored.getPressureRecoveryTime() != saved_recovery * saved_frame_time)
        throw std::runtime_error("Scene and engine pressure recovery settings disagree");
    validateScenarioState(scene.at("scenario_state"), restored);
    std::vector<float> pixels;
    restored.writeColorPixels(pixels, true);
    withFluidContext([&]
    {
        mouse_field_generator.reset();
        mouse_field.reset();
        fluid.destroy();
        fluid = std::move(restored);
    });
    display_pixels = std::move(pixels);
    frame_time = saved_frame_time;
    adaptive_settings.prediction_horizon = adaptive_settings.check_interval + frame_time;
    fluid.setAdaptiveSettings(adaptive_settings);
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
    pressure_recovery_frames = saved_recovery;
    simulation_failed = saved_failure;
    profile_stages = saved_profiling;
    liquid_tint_depth = saved_tint_depth;
    liquid_deep_brightness = saved_deep_brightness;
    restoreScenarioState(scene.at("scenario_state"));
    frame_statistics = {};
    ++frame_statistics_generation;
    frame_statistics.projection_tolerance = projection_tolerance;
    this->requestRedraw(true);
}

template<class Derived>
void FluidScene<Derived>::resumeFromImage(const std::filesystem::path& image)
{
    const auto snapshot = fluid::readFluidSnapshot(image);
    restoreSceneState(snapshot.scene, snapshot.checkpoint, snapshot.width, snapshot.height, snapshot.cell_size);
    const auto origin = fluid.getOrigin();
    for (auto* viewport : this->mountedToViewports())
        camera.focusWorldRect(*viewport, origin.x, origin.y, origin.x + world_w, origin.y + world_h);
    snapshot_error.clear();
}


#ifdef BITLOOP_DEV_MODE
template<class Derived>
void FluidScene<Derived>::traceFrame(SimTraceFrame& frame) const
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

template<class Derived>
bool FluidScene<Derived>::captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const
{
    if (route != "grid" || display_pixels.empty())
        return false;

    CaptureFrame snapshot;
    captureFrame(CapturePreset{}, snapshot);
    image.resolution = snapshot.resolution;
    image.pixels = std::move(snapshot.frame.data);
    image.payload = std::move(snapshot.frame.payload);
    return true;
}

template<class Derived>
bool FluidScene<Derived>::saveCheckpoint(SimCheckpointWriter& writer) const
{
    fluid::FluidCheckpoint checkpoint;
    bool saved = false;
    withFluidContext([&] { saved = fluid.saveCheckpoint(checkpoint); });
    if (!saved) return false;

    writer.writeJson("scene", captureSceneState());
    fluid::writeFluidCheckpoint(writer, checkpoint, checkpoint_brotli_quality);
    return true;
}

template<class Derived>
bool FluidScene<Derived>::loadCheckpoint(const SimCheckpointReader& reader)
{
    restoreSceneState(reader.readJson("scene"), fluid::readFluidCheckpoint(reader),
        fluid.getWidth(), fluid.getHeight(), fluid.getCellSize());
    return true;
}
#endif

SIM_END;
