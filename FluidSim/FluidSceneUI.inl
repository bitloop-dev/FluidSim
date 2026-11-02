// Shared buffered controls. Each concrete scene gets its own UI instance.
SIM_BEG;

template <class Derived> void FluidScene<Derived>::UI::sidebar()
{
    bl_pull(grid_w, grid_h, frame_time, gravity_y, pressure_recovery_frames, projection_tolerance, surface_tension);
    bl_pull(liquid_viscosities, gas_viscosities);
    bl_pull_temp(fluid_device, adaptive_mode);
    bl_view(cell_size, steps_per_second, projection_max_iterations);
    bl_view(elastic_material);

    const bool viscous =
        std::ranges::any_of(liquid_viscosities, [](double value)
        {
            return value != 0.0;
        }) ||
        std::ranges::any_of(gas_viscosities, [](double value)
        {
            return value != 0.0;
        });

    // Scenario presets.
    if constexpr (std::tuple_size_v<decltype(Derived::presets)> != 0)
    {
        bl_view(liquid_tint_depth, world_w, world_h);
        const float proximity_percent = float(100.0 * liquid_tint_depth * cell_size / std::hypot(world_w, world_h));
        const FluidSceneSettings current_settings{.grid_w = grid_w,
            .grid_h = grid_h,
            .cell_size = cell_size,
            .fluid_device = fluid_device,
            .adaptive_mode = adaptive_mode,
            .frame_time = frame_time,
            .steps_per_second = steps_per_second,
            .projection_max_iterations = projection_max_iterations,
            .projection_tolerance = projection_tolerance,
            .pressure_recovery_frames = pressure_recovery_frames,
            .gravity_y = gravity_y,
            .surface_tension = surface_tension,
            .liquid_viscosities = liquid_viscosities,
            .gas_viscosities = gas_viscosities,
            .gas_proximity_scale = proximity_percent};

        const char* preset_name = "Custom";
        for (const auto& preset : Derived::presets)
        {
            if (current_settings == preset.settings)
                preset_name = preset.name;
        }

        const FluidScenePreset* chosen_preset = nullptr;
        if (ImGui::BeginCombo("Preset", preset_name))
        {
            for (const auto& preset : Derived::presets)
            {
                const bool selected = current_settings == preset.settings;
                if (ImGui::Selectable(preset.name, selected))
                    chosen_preset = &preset;
                if (selected)
                    ImGui::SetItemDefaultFocus();
            }

            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("Load simulation settings and restart this scene.\n"
                              "Gas proximity scale is included; other appearance, mouse attraction and profiling "
                              "controls are preserved.\n"
                              "Manual changes show Custom; a different resolution refits the camera.");
        }

        if (chosen_preset)
        {
            bl_schedule([settings = chosen_preset->settings](Derived& scene)
            {
                scene.applyPreset(settings);
            });

            // Do not commit a deactivated old control after the preset has been queued.
            return;
        }
    }

    ImGui::TextDisabled("Drop a saved FluidSim WebP onto the viewport to resume.");
    bl_view(snapshot_error);
    if (!snapshot_error.empty())
        ImGui::TextWrapped("%s", snapshot_error.c_str());

    if (ImGui::Button("Restart Simulation"))
    {
        bl_schedule([width = std::clamp(grid_w, 32, 4096), height = std::clamp(grid_h, 32, 4096)](Derived& scene)
        {
            scene.grid_w = width;
            scene.grid_h = height;
            scene.restartSimulation();
        });
    }

    bl_scoped(liquid_tint_depth, liquid_deep_brightness);
    bl_view(world_w, world_h);

    // Use the active grid, since the resolution inputs below can contain unapplied edits.
    const double grid_diagonal = std::hypot(world_w, world_h) / cell_size;
    float proximity_percent = float(100.0 * liquid_tint_depth / grid_diagonal);
    if (ImGui::InputFloat("Gas proximity scale", &proximity_percent, 0.1f, 1.0f, "%.2f%%") &&
        std::isfinite(proximity_percent) && proximity_percent > 0.0f)
    {
        liquid_tint_depth = float(proximity_percent * 0.01 * grid_diagonal);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Percentage of the simulation grid diagonal. Preserved when changing resolution.");

    ImGui::SliderFloat("Minimum brightness", &liquid_deep_brightness, 0.1f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);

    int resolution[] = {grid_w, grid_h};
    if (ImGui::InputInt2("Resolution", resolution))
    {
        grid_w = resolution[0];
        grid_h = resolution[1];
    }

    // CPU / WebGPU combo box.
    int device = int(fluid_device);
    ImGui::BeginDisabled(pressure_recovery_frames != 0 || viscous);
    if (ImGui::Combo("Device", &device, "CPU\0WebGPU\0"))
    {
        bl_schedule([device](Derived& scene)
        {
            scene.fluid_device = fluid::FluidDevice(device);
            scene.restartSimulation();
        });
    }

    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        if (pressure_recovery_frames != 0 || viscous)
            ImGui::SetTooltip("Set Pressure recovery frames and material viscosities to 0 before changing the device.");
    }
    ImGui::EndDisabled();

    // Adaptive grid selection.
    bl_scoped(mouse_attraction);
    int mode = 0;
    for (int scale = int(adaptive_mode); scale > 1; scale /= 2)
        ++mode;

    ImGui::BeginDisabled(
        surface_tension != 0.0 || mouse_attraction || pressure_recovery_frames != 0 || viscous || elastic_material);
    if (ImGui::Combo("Adaptive Mode", &mode, "Disabled\0Max 2x2\0Max 4x4\0Max 8x8\0Max 16x16\0Max 32x32\0"))
    {
        bl_schedule([mode](Derived& scene)
        {
            scene.adaptive_mode = fluid::AdaptiveMode(mode == 0 ? 0 : 1 << mode);
            scene.restartSimulation();
        });
    }

    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        if (elastic_material)
            ImGui::SetTooltip("This elastic material requires a uniform grid.");
        else
            ImGui::SetTooltip("Restarts the simulation. Adaptive mode requires zero Surface Tension, zero Pressure "
                              "recovery frames, zero material viscosities and Mouse attraction off.");
    }

    ImGui::EndDisabled();

    ImGui::SliderDouble("Frame Time", &frame_time, 0.001, 0.2, "%.3f s", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        bl_schedule([value = frame_time](Derived& scene)
        {
            scene.frame_time = value;
            scene.adaptive_settings.prediction_horizon = scene.adaptive_settings.check_interval + value;
            scene.fluid.setAdaptiveSettings(scene.adaptive_settings);
            scene.fluid.setPressureRecoveryTime(scene.pressure_recovery_frames * value);
        });
    }

    ImGui::InputDouble("Gravity", &gravity_y, 0.0, 0.0, "%.6g");
    if (ImGui::IsItemDeactivatedAfterEdit() && std::isfinite(gravity_y))
    {
        bl_schedule([value = gravity_y](Derived& scene)
        {
            scene.withFluidContext([&]
            {
                scene.fluid.setGravityY(value);
            });

            scene.gravity_y = value;
        });
    }

    ImGui::BeginDisabled(
        fluid_device != fluid::FluidDevice::GPU || adaptive_mode != fluid::AdaptiveMode::DISABLED || elastic_material);
    ImGui::SliderInt("Pressure recovery frames", &pressure_recovery_frames, 0, 10, "%d", ImGuiSliderFlags_AlwaysClamp);

    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        bl_schedule([frames = pressure_recovery_frames](Derived& scene)
        {
            scene.setPressureRecoveryFrames(frames);
        });
    }

    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip(
            "0: original accurate solver.\n"
            "1-10: allow temporary compression and recover over roughly this many simulated frames.\n"
            "Higher values recover more slowly and do slightly less pressure work. Uniform WebGPU only.\n"
            "Crossing between 0 and a positive value restarts the simulation; other changes apply on release.");
    }

    ImGui::EndDisabled();
    ImGui::TextDisabled(pressure_recovery_frames == 0 ? "Accurate pressure" : "Relaxed pressure (experimental)");
    ImGui::TextDisabled("Switching to/from 0 restarts the scene.");

    ImGui::BeginDisabled(pressure_recovery_frames != 0);
    const double before_input = projection_tolerance;
    ImGui::InputDouble("Pressure Quality", &projection_tolerance, 0.0, 0.0, "%.6g");
    if (ImGui::IsItemActivated())
        pressure_tolerance_before_edit = before_input;
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        // Commit completed edits only: clearing/typing the number must not change the live solver target.
        if (std::isfinite(projection_tolerance) && projection_tolerance > 0.0)
        {
            bl_push(projection_tolerance);
        }
        else
            projection_tolerance = pressure_tolerance_before_edit;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Positive maximum cell divergence (per second).\nLarger values trade accuracy for "
                          "speed.\nPress Enter or leave the field to apply.");
    }

    ImGui::EndDisabled();

    ImGui::BeginDisabled(adaptive_mode != fluid::AdaptiveMode::DISABLED);
    const double old_tension = surface_tension;
    ImGui::InputDouble("Surface Tension", &surface_tension, 0.0, 0.0, "%.6g");
    if (ImGui::IsItemActivated())
        surface_tension_before_edit = old_tension;

    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        if (std::isfinite(float(surface_tension)) && surface_tension >= 0.0)
        {
            const double value = surface_tension;
            bl_schedule([value](Derived& scene)
            {
                scene.setSurfaceTension(value);
            });
        }
        else
            surface_tension = surface_tension_before_edit;
    }

    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip(
            "Liquid-gas surface tension in scene units; zero disables it.\nUniform CPU and WebGPU. Stronger tension "
            "may require extra physics steps.\nPress Enter or leave the field to apply.");
    }

    ImGui::EndDisabled();

    if (ImGui::CollapsingHeaderBox("Viscosity", true))
    {
        bl_view(liquid_viscosity_names, gas_viscosity_names);
        ImGui::BeginDisabled(fluid_device != fluid::FluidDevice::GPU || adaptive_mode != fluid::AdaptiveMode::DISABLED);

        auto controls = [&](fluid::Phase phase, const char* phase_name, std::vector<double>& values,
                            const std::vector<std::string>& names)
        {
            ImGui::PushID(phase_name);
            for (size_t i = 0; i < values.size(); ++i)
            {
                ImGui::PushID(int(i));
                const double before = values[i];
                const std::string label = names[i] + " (" + phase_name + ")";
                ImGui::InputDouble(label.c_str(), &values[i], 100.0, 1000.0, "%.6g");
                if (ImGui::IsItemActivated())
                    viscosity_before_edit = before;
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    if (std::isfinite(float(values[i])) && values[i] >= 0.0)
                    {
                        bl_schedule([phase, i, value = values[i]](Derived& scene)
                        {
                            scene.setComponentViscosity(phase, i, value);
                        });
                    }
                    else
                        values[i] = viscosity_before_edit;
                }
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                {
                    ImGui::SetTooltip(
                        "Dynamic viscosity in scene units; 0 is inviscid.\n"
                        "Mixed cells blend ingredient viscosities by volume.\n"
                        "Uniform WebGPU only. Applies on Enter or leaving the field, without restarting.");
                }

                ImGui::PopID();
            }

            ImGui::PopID();
        };

        controls(fluid::Phase::Liquid, "liquid", liquid_viscosities, liquid_viscosity_names);
        controls(fluid::Phase::Gas, "gas", gas_viscosities, gas_viscosity_names);

        ImGui::EndDisabled();
        ImGui::EndCollapsingHeaderBox();
    }

    ImGui::BeginDisabled(adaptive_mode != fluid::AdaptiveMode::DISABLED);
    ImGui::Checkbox("Mouse attraction", &mouse_attraction);
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Replace uniform gravity with attraction towards the pointer.\nMove inside the viewport to "
                          "choose the target; leaving it keeps the last target.");
    }

    ImGui::EndDisabled();

    if (ImGui::CollapsingHeaderBox("Frame statistics", true))
    {
        bl_scoped(profile_stages);
        bl_view(frame_statistics);
        const auto& stats = frame_statistics;
        if (pressure_recovery_frames != 0)
        {
            ImGui::Text("Volume error: max %.2f%%, RMS up to %.2f%%", 100.0 * stats.max_volume_error,
                100.0 * stats.rms_volume_error);
        }

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
                if (available)
                    ImGui::Text("%.3f", ms);
                else
                    ImGui::TextUnformatted("--");
            };

            row("Total frame time", stats.update_ms + render_ms);
            row("Render (CPU submission)", render_ms);
            row("Fluid processing", stats.fluid_ms);
            row("    Transport", stats.stages.transport_ms, stats.detailed);
            if (!stats.adaptive)
                row("    Viscosity", stats.stages.viscosity_ms, stats.detailed);
            row("    Elasticity", stats.stages.elasticity_ms, stats.detailed);
            if (!stats.adaptive)
                row("    Forces", stats.stages.forces_ms, stats.detailed);
            row(stats.adaptive ? "    Projection + forces" : "    Projection", stats.stages.pressure_ms,
                stats.detailed);
            if (stats.adaptive)
            {
                row("    Refinement selection", stats.stages.selection_ms, stats.detailed);
                row("    Regrid + restoration", stats.stages.regrid_ms, stats.detailed);
            }

            row("Display colour/readback", stats.colour_ms);
            row("Other frame work", std::max(0.0, stats.update_ms - stats.fluid_ms - stats.colour_ms));
            ImGui::EndTable();
        }

        ImGui::TextDisabled("Fluid stages sum %d physics substep(s).", stats.substeps);
        ImGui::TextDisabled(
            "%d transport steps; %d pressure iterations.", stats.transport_steps, stats.pressure_iterations);
        if (viscous)
            ImGui::TextDisabled("%d viscosity iteration slots (batches of 4).", stats.viscosity_iterations);

        if (stats.precision_limited_updates > 0)
        {
            ImGui::TextDisabled("Float precision limit reached; max divergence %.6g.", stats.max_divergence);
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("The remaining error is within local face-velocity rounding precision.\n"
                                  "Further pressure iterations cannot reliably reduce it at these speeds.");
            }
        }

        ImGui::TextDisabled("GPU profiling adds completion waits.");
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("Disable detailed timings for normal throughput.\nTotal = scene update + latest render "
                              "CPU time.\nUI/presentation and GPU draw completion are excluded.");
        }

        ImGui::EndCollapsingHeaderBox();
    }

    if (ImGui::CollapsingHeaderBox("View", true))
    {
        bl_scoped(camera);

        camera.populateUI();
        ImGui::EndCollapsingHeaderBox();
    }
}

SIM_END;
