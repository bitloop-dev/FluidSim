// Shared buffered controls. Each concrete scene gets its own UI instance.
SIM_BEG;

template<class Derived>
void FluidScene<Derived>::UI::sidebar()
{
    ImGui::TextDisabled("Drop a saved FluidSim WebP onto the viewport to resume.");
    bl_view(snapshot_error);
    if (!snapshot_error.empty()) ImGui::TextWrapped("%s", snapshot_error.c_str());
    bl_pull(grid_w, grid_h);
    if (ImGui::Button("Restart Simulation"))
        bl_schedule([width = std::clamp(grid_w, 32, 4096), height = std::clamp(grid_h, 32, 4096)](Derived& scene)
        {
            scene.grid_w = width;
            scene.grid_h = height;
            scene.restartSimulation();
        });

    bl_scoped(liquid_tint_depth, liquid_deep_brightness);
    ImGui::SliderFloat("Gas proximity range", &liquid_tint_depth, 0.5f, 32.f, "%.1f cells", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
    ImGui::SliderFloat("Minimum brightness", &liquid_deep_brightness, 0.1f, 1.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);

    int resolution[] = {grid_w, grid_h};
    if (ImGui::InputInt2("Resolution", resolution))
    {
        grid_w = resolution[0];
        grid_h = resolution[1];
    }
    bl_pull_temp(fluid_device, adaptive_mode);
    int device = int(fluid_device);
    if (ImGui::Combo("Device", &device, "CPU\0WebGPU\0"))
        bl_schedule([device](Derived& scene)
        {
            scene.fluid_device = fluid::FluidDevice(device);
            scene.restartSimulation();
        });

    bl_pull(surface_tension);
    bl_scoped(mouse_attraction);
    int mode = 0;
    for (int scale = int(adaptive_mode); scale > 1; scale /= 2)
        ++mode;
    ImGui::BeginDisabled(surface_tension != 0. || mouse_attraction);
    if (ImGui::Combo("Adaptive Mode", &mode, "Disabled\0Max 2x2\0Max 4x4\0Max 8x8\0Max 16x16\0Max 32x32\0"))
        bl_schedule([mode](Derived& scene)
        {
            scene.adaptive_mode = fluid::AdaptiveMode(mode == 0 ? 0 : 1 << mode);
            scene.restartSimulation();
        });
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Restarts the simulation. Adaptive mode requires zero Surface Tension and Mouse attraction off.");
    ImGui::EndDisabled();

    bl_pull(frame_time);
    ImGui::SliderDouble("Frame Time", &frame_time, 0.001, 0.2, "%.3f s", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemDeactivatedAfterEdit())
        bl_schedule([value = frame_time](Derived& scene)
        {
            scene.frame_time = value;
            scene.adaptive_settings.prediction_horizon = scene.adaptive_settings.check_interval + value;
            scene.fluid.setAdaptiveSettings(scene.adaptive_settings);
        });

    bl_pull(gravity_y);
    ImGui::InputDouble("Gravity", &gravity_y, 0., 0., "%.6g");
    if (ImGui::IsItemDeactivatedAfterEdit() && std::isfinite(gravity_y))
        bl_schedule([value = gravity_y](Derived& scene)
        {
            scene.withFluidContext([&] { scene.fluid.setGravityY(value); });
            scene.gravity_y = value;
        });

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

    ImGui::BeginDisabled(adaptive_mode != fluid::AdaptiveMode::DISABLED);
    const double old_tension = surface_tension;
    ImGui::InputDouble("Surface Tension", &surface_tension, 0., 0., "%.6g");
    if (ImGui::IsItemActivated()) surface_tension_before_edit = old_tension;
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        if (std::isfinite(float(surface_tension)) && surface_tension >= 0.)
        {
            const double value = surface_tension;
            bl_schedule([value](Derived& scene) { scene.setSurfaceTension(value); });
        }
        else surface_tension = surface_tension_before_edit;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Liquid-gas surface tension in scene units; zero disables it.\nUniform CPU and WebGPU. Stronger tension may require extra physics steps.\nPress Enter or leave the field to apply.");
    ImGui::EndDisabled();

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

SIM_END;
