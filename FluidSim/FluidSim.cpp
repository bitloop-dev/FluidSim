#include "FluidSim.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <numeric>

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
    for (int i = 0; i < viewport_count; i++) {
        auto scene=create<FluidSim_Scene>(i==0?"primary":"fluid-"+std::to_string(i));
#ifdef BITLOOP_DEV_MODE
        scene.setLaunchConfigParser([](FluidSim_Scene::Config& config,const JSON::json& values) {
            for (auto it=values.begin();it!=values.end();++it) {
                const auto& key=it.key(); const auto& value=it.value();
                if (key=="size") config.size=value.get<int>();
                else if (key=="scenario") config.scenario=value.get<int>();
                else if (key=="max_cell") config.max_cell=value.get<int>();
                else if (key=="engine") config.engine=value.get<int>();
                else if (key=="capture_stride") config.capture_stride=value.get<int>();
                else if (key=="webgpu") config.webgpu=value.get<bool>();
                else if (key=="adaptive") config.adaptive=value.get<bool>();
                else if (key=="coarse_control") config.coarse_control=value.get<bool>();
                else if (key=="clouds") config.clouds=value.get<bool>();
                else if (key=="profile_stages") config.profile_stages=value.get<bool>();
                else if (key=="damping") config.damping=value.get<double>();
                else if (key=="projection_tolerance") config.projection_tolerance=value.get<double>();
                else if (key=="frame_time") config.frame_time=value.get<double>();
                else if (key=="velocity_tolerance") config.velocity_tolerance=value.get<double>();
                else if (key=="pressure_tolerance") config.pressure_tolerance=value.get<double>();
                else if (key=="density_tolerance") config.density_tolerance=value.get<double>();
                else if (key=="ingredient_tolerance") config.ingredient_tolerance=value.get<double>();
                else throw std::invalid_argument("Unknown fluid scene option: "+key);
            }
            if (config.engine && (*config.engine<0 || *config.engine>4)) throw std::invalid_argument("engine must be 0..4");
            if (config.size && (*config.size<32 || *config.size>4096 || *config.size%32))
                throw std::invalid_argument("size must be a multiple of 32 between 32 and 4096");
            if (config.scenario && (*config.scenario<0 || *config.scenario>3)) throw std::invalid_argument("scenario must be 0..3");
            if (config.max_cell && (*config.max_cell<2 || *config.max_cell>32 || (*config.max_cell&(*config.max_cell-1))))
                throw std::invalid_argument("max_cell must be 2, 4, 8, 16 or 32");
            if (config.capture_stride && *config.capture_stride<1) throw std::invalid_argument("capture_stride must be positive");
            for (const auto& value:{config.frame_time,config.velocity_tolerance,config.pressure_tolerance,
                    config.density_tolerance,config.ingredient_tolerance,config.projection_tolerance})
                if (value && !(*value>0.)) throw std::invalid_argument("frame_time and tolerances must be positive");
            if (config.damping && *config.damping<0.) throw std::invalid_argument("damping cannot be negative");
        });
#endif
        layout << scene;
    }
}

/// ────────────────────────── scene ──────────────────────────

void FluidSim_Scene::UI::sidebar()
{
    if (ImGui::Button("Restart scene"))
        bl_schedule([](FluidSim_Scene& scene) { scene.restartSimulation(); });

    if (ImGui::CollapsingHeaderBox("Liquid appearance", true)) {
        bl_scoped(liquid_depth_shading, liquid_depth_scale, liquid_deep_brightness);
        ImGui::Checkbox("Depth shading", &liquid_depth_shading);
        ImGui::BeginDisabled(!liquid_depth_shading);
        ImGui::SliderFloat("Tint depth", &liquid_depth_scale, 2.0f, 128.0f, "%.1f cells", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Deep brightness", &liquid_deep_brightness, 0.03f, 0.7f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::EndDisabled();
        ImGui::EndCollapsingHeaderBox();
    }

    bl_pull_temp(engine_index);
    if (ImGui::Combo("Engine", &engine_index, "CPU uniform\0CPU adaptive\0GPU uniform (WebGPU)\0GPU adaptive (WebGPU)\0GPU uniform (WebGL)\0"))
        bl_schedule([selected=engine_index](FluidSim_Scene& scene) { scene.engine_index=selected; scene.restartSimulation(); });

    if (ImGui::CollapsingHeaderBox("Frame statistics", true))
    {
        bl_scoped(profile_stages);
        bl_view(last_frame_interval_ms, last_frame_ms, last_render_ms, last_physics_ms);
        bl_view(last_remap_ms, last_coarsening_ms, gpu_display_ms, adaptive_engine);
        bl_view(frame_physics_timings, frame_stage_timings_available, frame_substeps);

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
            row("Total frame time", last_frame_interval_ms);
            row("Measured frame work", last_frame_ms + last_render_ms);
            row("Scene update", last_frame_ms);
            row("Render (CPU submission)", last_render_ms);
            row("Fluid steps", last_physics_ms - last_remap_ms);
            row(adaptive_engine ? "    Transport + damping" : "    Transport",
                frame_physics_timings.transport_ms, frame_stage_timings_available);
            if (!adaptive_engine)
                row("    Forces + damping", frame_physics_timings.forces_ms, frame_stage_timings_available);
            row(adaptive_engine ? "    Projection + forces" : "    Projection",
                frame_physics_timings.pressure_ms, frame_stage_timings_available);
            row("Refinement selection", last_coarsening_ms);
            row("Regrid + restoration", last_remap_ms);
            row("WebGPU fields/readback", gpu_display_ms);
            row("Other update work", std::max(0., last_frame_ms - last_physics_ms - last_coarsening_ms - gpu_display_ms));
            ImGui::EndTable();
        }
        ImGui::TextWrapped("Latest completed frame; fluid stages sum all %d physics substeps. Child rows are included in their totals.", frame_substeps);
        ImGui::TextWrapped("Total frame time includes pacing. Measured work excludes UI/presentation; render time measures CPU calls, not completed GPU drawing.");
        if (engine_index >= GPU_UNIFORM)
            ImGui::TextWrapped("Detailed timings add GPU completion waits and can increase frame cost. Disable them for normal throughput.");
        ImGui::EndCollapsingHeaderBox();
    }

    if (ImGui::CollapsingHeaderBox("View", true))
    {
        bl_scoped(camera);

        camera.populateUI();
        ImGui::EndCollapsingHeaderBox();
    }
    if (ImGui::CollapsingHeaderBox("Atmosphere", true))
    {
        bl_scoped(atmosphere_damping);
        ImGui::SliderFloat("Gas damping (per second)", &atmosphere_damping, 0.0f, 2.0f, "%.2f");
        ImGui::TextWrapped("Zero disables damping. Applies to gas-only velocities in every scenario, with or without the overlay.");
        ImGui::EndCollapsingHeaderBox();
    }
    if (ImGui::CollapsingHeaderBox("Cell resolution", true))
    {
        bl_scoped(resolution_overlay, mask_opacity, refinement_interval);
        bl_pull_temp(adaptive_grid, uniform_coarse, coarse_scale);
        bl_view(amr_workers, adaptive_engine, simulated_cells, allocated_cells, simulation_blocks, fine_patches, level_count, grid_w, grid_h, failure_reason, last_display_ms, patch_changes, last_remap_ms);
        ImGui::BeginDisabled(!adaptive_engine);
        if (ImGui::Checkbox("Automatic refinement", &adaptive_grid))
            bl_schedule([enabled=adaptive_grid](FluidSim_Scene& scene) {
                scene.adaptive_grid=enabled;
                scene.restartSimulation();
            });
        int maximum=0;
        for (int scale=coarse_scale;scale>2;scale/=2) ++maximum;
        if (ImGui::Combo("Maximum cell size", &maximum, "2x2\0" "4x4\0" "8x8\0" "16x16\0" "32x32\0"))
            bl_schedule([scale=2<<maximum](FluidSim_Scene& scene) {
                scene.coarse_scale=scale;
                scene.restartSimulation();
            });
        ImGui::BeginDisabled(adaptive_grid);
        if (ImGui::Checkbox("Uniform coarse control", &uniform_coarse))
            bl_schedule([enabled=uniform_coarse](FluidSim_Scene& scene) {
                scene.uniform_coarse=enabled;
                scene.restartSimulation();
            });
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        ImGui::Checkbox("Show cell grid", &resolution_overlay);
        ImGui::SliderFloat("Grid opacity", &mask_opacity, 0.0f, 1.0f);
        const int cells = simulated_cells;
        ImGui::Text("Simulated cells: %d / %d (%.1f%%)", cells, grid_w * grid_h, 100.0 * cells / (grid_w * grid_h));
        ImGui::Text("Colour/overlay CPU: %.3f ms / draw", last_display_ms);
        ImGui::Text("Allocated level cells: %d; %d levels / %d FABs", allocated_cells, level_count, simulation_blocks);
        ImGui::Text("Refined patches: %d; AMR workers: %d", fine_patches, amr_workers);
        ImGui::Text("Layout changes: %d; this frame: %.3f ms", patch_changes, last_remap_ms);
        if (adaptive_engine) {
            ImGui::BeginDisabled(!adaptive_grid);
            ImGui::SliderDouble("Minimum check interval (seconds)", &refinement_interval, 0.08, 2.0, "%.2f", ImGuiSliderFlags_AlwaysClamp);
            ImGui::EndDisabled();
            ImGui::TextWrapped("Changing grid mode restarts this scene with the same settings. Adaptive uses up to the selected maximum, keeping liquid and its buffer fine. Turning automatic refinement off keeps the adaptive engine at fixed fine resolution; the coarse control uses the selected maximum everywhere. Outlines: green 2x2, amber 4x4, magenta 8x8, cyan 16x16, red 32x32. Fine 1x1 cells have no outline.");
        }
        else ImGui::TextWrapped("Uniform reference cells.");
        if (!failure_reason.empty()) ImGui::TextWrapped("Stopped: %s", failure_reason.c_str());
        ImGui::EndCollapsingHeaderBox();
    }
    if (ImGui::CollapsingHeaderBox("Resolution preview", true))
    {
        bl_scoped(show_coarsening_preview, coarsening_opacity, coarsening_tolerances, coarsening_stability, refinement_lookahead);
        bl_view(last_coarsening_ms, coarsening_counts, coarsening_statistics, adaptive_grid, adaptive_engine);
        ImGui::BeginDisabled(!adaptive_engine || !adaptive_grid);
        ImGui::Checkbox("Show refinement decisions", &show_coarsening_preview);
        ImGui::SliderFloat("Preview opacity", &coarsening_opacity, 0.0f, 1.0f);
        ImGui::SliderDouble("Velocity detail tolerance", &coarsening_tolerances.velocity, 0.001, 10.0, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderDouble("Pressure-gradient detail tolerance", &coarsening_tolerances.pressure_gradient, 0.001, 10.0, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderDouble("Relative density tolerance", &coarsening_tolerances.density, 0.0001, 0.5, "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderDouble("Ingredient fraction tolerance", &coarsening_tolerances.ingredient, 0.0001, 0.5, "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderDouble("Refinement look-ahead (seconds)", &refinement_lookahead, 0.0, 0.5, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderDouble("Quiet threshold / tolerance", &coarsening_stability.quiet_ratio, 0.1, 1.0, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderDouble("Quiet delay (simulated seconds)", &coarsening_stability.quiet_seconds, 0.0, 2.0, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        using P = fluid::FluidCoarseningPreview;
        if ((adaptive_engine && adaptive_grid) || (!adaptive_engine && show_coarsening_preview))
        {
            ImGui::Text("Estimate: %.3f ms / frame", last_coarsening_ms);
            ImGui::Text("Displayed groups: %d candidates; %d retain detail", coarsening_counts[P::Candidate], coarsening_counts[P::Detail]);
            ImGui::Text("%d protected; %d unassessed", coarsening_counts[P::LiquidBuffer], coarsening_counts[P::Unassessed]);
            ImGui::Text("Coarse: %d refine; %d approaching; %d no trigger", coarsening_counts[P::Refine], coarsening_counts[P::Approaching], coarsening_counts[P::Coarse]);
            ImGui::Text("Waiting to merge: %d; retained refinement: %d", coarsening_statistics.waiting, coarsening_statistics.held);
            ImGui::Text("Changed actions: %d raw / %d stabilised", coarsening_statistics.raw_changes, coarsening_statistics.decision_changes);
        }
        ImGui::TextWrapped("Fill: green = settled merge candidate; yellow = retain resolution / waiting; purple = liquid/buffer; grey = unassessed edge. Coarse: red = refine; orange = approaching detail. Requests stay until quiet; no fill = no detected trigger. Green/amber/magenta/cyan/red lines show actual 2x2/4x4/8x8/16x16/32x32 resolution. Fine 1x1 cells have no outline.");
        ImGui::TextWrapped("The estimates drive automatic atmosphere resolution on CPU and WebGPU, including waves. Liquid and its approaching region remain fine. Ingredient tolerance 0.02 means two percentage points locally, not a bound on accumulated transport error. Hiding this overlay does not disable automatic selection.");
        ImGui::TextWrapped("Merging and clearing refinement need sustained detail below the quiet threshold. New detail above the full tolerance, liquid and incoming features act immediately. Ratio 1 and delay 0 disable stabilisation. Changing thresholds or look-ahead restarts the quiet period. Layout changes reset only cells whose resolution changes.");
        ImGui::TextWrapped("Look-ahead sweeps known features using current velocities and gravity. Automatic selection uses at least the check interval plus one frame. It cannot recover discarded detail or predict future pressure changes or new sources.");
        ImGui::EndDisabled();
        ImGui::EndCollapsingHeaderBox();
    }
}

void FluidSim_Scene::updateCellResolution()
{
    last_coarsening_ms = 0;
    if (!adaptive_engine || !adaptive_grid)
    {
        coarsening_preview.reset();
        if (gpu_preview) gpu_preview->reset();
        coarsening_counts = {}; coarsening_statistics = {};
        return;
    }
    if (adaptive_engine && adaptive_grid && simulation_time < next_refinement_time) return;
    const auto started = std::chrono::steady_clock::now();
#ifdef FLUIDSIM_HAS_AMR
    if (adaptive_gpu) gpu_preview->update(*adaptive_gpu,coarsening_tolerances,coarsening_stability,simulation_time,
        std::max(refinement_lookahead,refinement_interval+scenario.frame_time),coarse_scale);
    else if (adaptive_cpu) coarsening_preview.update(*adaptive_cpu, coarsening_tolerances, coarsening_stability, simulation_time,
        adaptive_grid ? std::max(refinement_lookahead,refinement_interval+scenario.frame_time) : refinement_lookahead,coarse_scale);
    else
#endif
        return;
    last_coarsening_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count();
    coarsening_counts = adaptive_gpu ? gpu_preview->counts : coarsening_preview.counts;
    coarsening_statistics = adaptive_gpu ? gpu_preview->statistics : coarsening_preview.statistics;
#ifdef FLUIDSIM_HAS_AMR
    auto applyLayout=[&](auto& grid,auto& preview) {
        next_refinement_time=simulation_time+refinement_interval;
        const auto clustered=std::chrono::steady_clock::now();
        auto layout=preview.proposeLayout(grid);
        last_coarsening_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-clustered).count();
        const auto remap=std::chrono::steady_clock::now();
        if (grid.regrid(std::move(layout))) {
            const auto restored=grid.project(projection_tolerance);
            if (adaptive_gpu) adaptive_gpu->synchronize();
            if (!restored.converged) throw std::runtime_error("Pressure restoration after automatic refinement failed.");
            last_divergence=std::max(last_divergence,restored.max_divergence);
            updateCellCounts(); ++patch_changes;
            last_remap_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-remap).count();
            last_physics_ms+=last_remap_ms; physics_seconds+=last_remap_ms/1000.;
#ifdef BITLOOP_DEV_MODE
            frame_pressure_iterations+=restored.iterations;
#endif
        }
    };
    if (adaptive_grid) {
        if (adaptive_gpu) applyLayout(*adaptive_gpu,*gpu_preview);
        else if (adaptive_cpu) applyLayout(*adaptive_cpu,coarsening_preview);
    }
#endif
}

void FluidSim_Scene::updateCellCounts() {
#ifdef FLUIDSIM_HAS_AMR
    if (adaptive_gpu) {
        simulated_cells=int(adaptive_gpu->activeCellCount()); allocated_cells=int(adaptive_gpu->allocatedCellCount());
        level_count=int(adaptive_gpu->levels().size()); fine_patches=0; simulation_blocks=0;
        for (const auto& patches:adaptive_gpu->layout().refinedPatches) fine_patches+=int(patches.size());
        for (const auto& level:adaptive_gpu->levels()) simulation_blocks+=int(level->boxes.size());
        return;
    }
    if (adaptive_cpu) {
        simulated_cells=int(adaptive_cpu->activeCellCount()); allocated_cells=int(adaptive_cpu->allocatedCellCount());
        level_count=int(adaptive_cpu->levels().size()); fine_patches=0; simulation_blocks=0;
        for (const auto& patches:adaptive_cpu->layout().refinedPatches) fine_patches+=int(patches.size());
        for (const auto& level:adaptive_cpu->levels()) simulation_blocks+=level->conserved.size();
        return;
    }
#endif
    simulated_cells=allocated_cells=grid_w*grid_h; simulation_blocks=level_count=1; fine_patches=0;
}
std::uint32_t FluidSim_Scene::materialTexture() const {
#ifdef FLUIDSIM_HAS_AMR
    if (uniform_webgl) return uniform_webgl->colorTexture();
    if (adaptive_gpu || uniform_gpu) return fluid_display.texture(gpu_material,elements,displayMixture());
    if (adaptive_cpu) return fluid_display.texture(*adaptive_cpu);
#endif
    return fluid_display.texture(*uniform_cpu);
}
fluid::FluidMixture FluidSim_Scene::displayMixture() const {
#ifdef FLUIDSIM_HAS_AMR
    if (adaptive_gpu) return adaptive_gpu->materialMixture();
    if (adaptive_cpu) return adaptive_cpu->materialMixture();
#endif
    auto mixture=liquid_mixture;
    mixture.components.insert(mixture.components.end(),gas_mixture.components.begin(),gas_mixture.components.end());
    return mixture;
}
FluidGridDisplay::MaterialTextures FluidSim_Scene::materialTextures() const {
#ifdef FLUIDSIM_HAS_AMR
    if (uniform_webgl) return webgl_material;
    if (adaptive_gpu || uniform_gpu) return fluid_display.materialTextures(gpu_material);
    if (adaptive_cpu) return fluid_display.materialTextures(*adaptive_cpu);
#endif
    return fluid_display.materialTextures(*uniform_cpu);
}
std::uint32_t FluidSim_Scene::resolutionTexture() const {
#ifdef FLUIDSIM_HAS_AMR
    if (adaptive_gpu) return fluid_display.resolutionTexture(grid_w,grid_h,adaptive_gpu->layout());
    if (adaptive_cpu) return fluid_display.resolutionTexture(*adaptive_cpu);
#endif
    return fluid_display.resolutionTexture(grid_w,grid_h);
}

void FluidSim_Scene::prepareDisplay()
{
    gpu_display_ms=0;
#ifdef FLUIDSIM_HAS_AMR
    if (uniform_webgl) {
        renderThreadQueue().invokeBlocking([&] {
            webgl_material={uniform_webgl->fractionTexture(),uniform_webgl->liquidFractionTexture()};
        });
    }
    if (adaptive_gpu || uniform_gpu) {
        const auto started=std::chrono::steady_clock::now();
        // WebGPU belongs to this worker; GL consumes only derived display fields.
        gpu_material=adaptive_gpu ? adaptive_gpu->materialPixels() : uniform_gpu->materialPixels();
        if (adaptive_gpu && show_coarsening_preview && adaptive_grid && gpu_labels_revision!=gpu_preview->revision) {
            gpu_preview->downloadLabels(); gpu_labels_revision=gpu_preview->revision; ++gpu_preview_revision;
        }
        gpu_display_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
#endif
}

#ifdef BITLOOP_DEV_MODE
void FluidSim_Scene::inspectGPUState()
{
    if (adaptive_gpu) gpu_totals=adaptive_gpu->totals();
    else if (uniform_gpu) uniform_gpu_state=uniform_gpu->download();
    else if (uniform_webgl) renderThreadQueue().invokeBlocking([&] { uniform_gpu_state=uniform_webgl->download(); });
}
#endif

void FluidSim_Scene::initializePool(double amplitude)
{
    //constexpr double pi = 3.14159265358979323846;
    const size_t cell_count = size_t(grid_w) * grid_h;

    using Scalar = double;
    FluidGrid::State initial;
    initial.liquid_fraction.resize(cell_count);

    const double cell_volume = cell_size * cell_size;
    const double cloud_noise_scale = 10.0 / grid_w;

    initial.liquid_mass.assign(liquid_mixture.components.size(),
        std::vector<Scalar>(cell_count, 0.0));

    initial.gas_mass.assign(gas_mixture.components.size(),
        std::vector<Scalar>(cell_count, 0.0));

    (void)amplitude;

    for (int y = 0; y < grid_h; ++y)
    {
        for (int x = 0; x < grid_w; ++x)
        {
            // the experiment starts with a sloped, rippled surface
            const double tx = double(x) / double(grid_w);

            double surface;

            switch (scenario.type)
            {
                case ScenarioType::COSINE_WAVES:
                    surface = (grid_h * 0.7) + (std::cos(tx * 20.0) * double(grid_h) * 0.02);
                    break;

                case ScenarioType::CRASHING_WAVE:
                    surface = (double(grid_h) * 0.7) + tx * (double(grid_h) * 0.5);
                    break;
                case ScenarioType::LOCAL_COARSE_PATCH:
                case ScenarioType::MOVING_CLOUD_DETAIL:
                    surface = double(grid_h) + 1.0;
                    break;
            }

            // measure how much of this cell lies below the liquid surface
            const int i = y * grid_w + x;
            const double fraction = std::clamp(y + 1.0 - surface, 0.0, 1.0);

            initial.liquid_fraction[i] = Scalar(fraction);

            // Fill the left quarter's liquid with Red; use Water elsewhere.
            const double red_share = x < grid_w / 4 ? 1.0 : 0.0;

            for (size_t k = 0; k < liquid_mixture.components.size(); ++k)
            {
                const auto& component = liquid_mixture.components[k];

                double share = 0.0;

                if (component.element == water_element)
                    share = 1.0 - red_share;
                else if (component.element == red_element)
                    share = red_share;

                // Split the existing liquid volume between its ingredients.
                initial.liquid_mass[k][i] =
                    Scalar(component.density * fraction * cell_volume * share);
            }

            double cloud_share =  0.0;
            if (scenario.type==MOVING_CLOUD_DETAIL) {
                const double radius=.06*std::min(world_w,world_h);
                const double distance=std::hypot((x+.5)*cell_size-.3*world_w,(y+.5)*cell_size-.5*world_h);
                // A one-world-unit transition gives the moving cloud resolved
                // fine detail; the large quiet background does not dilute it.
                cloud_share=.5*(1.-std::tanh((distance-radius)/1.0));
            } else if (perlin_clouds) {
                // Layer three Perlin scales to give broad clouds some smaller detail.
                const double nx=(x+.5)*cloud_noise_scale, ny=(y+.5)*cloud_noise_scale;
                const double cloud_noise=(perlinNoise(nx,ny)+.5*perlinNoise(2.*nx,2.*ny)+.25*perlinNoise(4.*nx,4.*ny))/1.75;
                cloud_share=std::clamp(.15+2.5*cloud_noise,0.,1.);
            }

            for (size_t k = 0; k < gas_mixture.components.size(); ++k)
            {
                const auto& component = gas_mixture.components[k];

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

    if (gasScenario())
    {
        constexpr double pi = 3.14159265358979323846;
        // Moving-detail runs cross the same domain fraction at different sizes.
        const double speed=scenario.type==MOVING_CLOUD_DETAIL ? world_w/128. : 2.;
        // Curl of a vertex streamfunction: circulating gas, closed walls, no gravity.
        auto psi = [&](int x, int y) {
            if (x == 0 || y == 0 || x == grid_w || y == grid_h) return 0.0;
            return speed * world_w / pi * std::pow(std::sin(pi * x / grid_w) * std::sin(pi * y / grid_h), 2);
        };
        initial.u.resize((grid_w + 1) * grid_h);
        initial.v.resize(grid_w * (grid_h + 1));
        for (int y = 0; y < grid_h; ++y)
            for (int x = 0; x <= grid_w; ++x) initial.u[y * (grid_w + 1) + x] = (psi(x, y + 1) - psi(x, y)) / cell_size;
        for (int y = 0; y <= grid_h; ++y)
            for (int x = 0; x < grid_w; ++x) initial.v[y * grid_w + x] = -(psi(x + 1, y) - psi(x, y)) / cell_size;
    }
#ifdef BITLOOP_DEV_MODE
    initial_liquid_mass.clear(); initial_gas_mass.clear();
    for (const auto& mass : initial.liquid_mass) initial_liquid_mass.push_back(std::accumulate(mass.begin(), mass.end(), 0.0));
    for (const auto& mass : initial.gas_mass) initial_gas_mass.push_back(std::accumulate(mass.begin(), mass.end(), 0.0));
#endif
#ifdef FLUIDSIM_HAS_AMR
    if (adaptive_engine) {
        const fluid::GasLayout layout=!adaptive_grid && uniform_coarse ? fluid::GasLayout{coarse_scale,{}} : fluid::GasLayout{};
        if (engine_index==GPU_ADAPTIVE) {
            if (!adaptive_gpu) adaptive_gpu=std::make_unique<fluid::FluidGridAdaptiveGPU>(grid_w,grid_h,cell_size,elements,
                gasScenario()?fluid::FluidMixture{}:liquid_mixture,gas_mixture,layout);
            else adaptive_gpu->regrid(layout);
            if (!gpu_preview) gpu_preview=std::make_unique<fluid::FluidCoarseningGPU>();
            adaptive_gpu->upload(initial,projection_tolerance);
        } else {
            adaptive_cpu=std::make_unique<fluid::FluidGridAdaptive>(grid_w,grid_h,cell_size,elements,
                gasScenario()?fluid::FluidMixture{}:liquid_mixture,gas_mixture,layout);
            adaptive_cpu->upload(initial,projection_tolerance);
        }
    } else
#endif
    {
        if (uniform_cpu) uniform_cpu->upload(initial);
        else if (uniform_gpu) uniform_gpu->upload(fluid::convertFluidState<float>(initial));
        else renderThreadQueue().invokeBlocking([&] { uniform_webgl->upload(fluid::convertFluidState<float>(initial)); });
        if (gasScenario()) {
            fluid::FluidProjectionResult result;
            if (uniform_cpu) result=uniform_cpu->project(.016,2000,projection_tolerance);
            else if (uniform_gpu) result=uniform_gpu->project(.016,2000,projection_tolerance);
            else renderThreadQueue().invokeBlocking([&] { result=uniform_webgl->project(.016,2000,projection_tolerance); });
            if (!result.converged) throw std::runtime_error("Initial gas pressure projection failed.");
        }
    }
    updateCellCounts();
    updateCellResolution();
}

void FluidSim_Scene::sceneStart()
{
#ifdef FLUIDSIM_HAS_AMR
    adaptive_engine = engine_index==CPU_ADAPTIVE || engine_index==GPU_ADAPTIVE;
#endif
    restartSimulation();
    surface.setFragmentSource(fluid::fluid_shader_src);
}

void FluidSim_Scene::restartSimulation()
{
    adaptive_engine=engine_index==CPU_ADAPTIVE || engine_index==GPU_ADAPTIVE;
    // Refill the current scenario; keep controls, camera and engine options intact.
    const auto started=std::chrono::steady_clock::now();
    coarsening_preview.reset();
#ifdef FLUIDSIM_HAS_AMR
    // Retain device pipelines across same-backend restarts; only the physical state/history reset.
    if (engine_index!=GPU_ADAPTIVE || (adaptive_gpu && adaptive_gpu->commands().gpu.status()==bl::WebGpuDevice::Status::Failed)) {
        gpu_preview.reset(); adaptive_gpu.reset();
    } else if (gpu_preview) gpu_preview->reset();
    adaptive_cpu.reset(); amr_runtime.reset();
    amr_workers=0;
    if (engine_index==CPU_ADAPTIVE) { amr_runtime=std::make_unique<fluid::AmrGasRuntime>(); amr_workers=amr_runtime->workerCount(); }
#endif
    webgl_material={};
    gpu_material={}; gpu_labels_revision=~std::uint64_t(0);
    next_refinement_time=0;
    simulation_time = real_time = 0;
    simulation_failed = false;
    physics_seconds = last_physics_ms = 0;
    last_remap_ms = 0;
    processed_frames = patch_changes = 0;
    failure_reason.clear();
    last_display_ms = 0;
    gpu_display_ms=last_frame_ms=0;
    last_frame_interval_ms=last_render_ms=0;
    frame_substeps=0;
    frame_stage_timings_available=false;
    frame_physics_timings = {};
#ifdef BITLOOP_DEV_MODE
    frame_pressure_iterations = 0;
    gpu_totals={}; uniform_gpu_state={};
#endif
    try {
        uniform_cpu.reset();
        if (engine_index!=GPU_UNIFORM) uniform_gpu.reset();
        if (uniform_webgl) renderThreadQueue().invokeBlocking([&] { uniform_webgl.reset(); });
        if (engine_index==CPU_UNIFORM) {
            uniform_cpu=std::make_unique<FluidGrid>(grid_w,grid_h,cell_size);
            uniform_cpu->elements=elements; uniform_cpu->liquid_mixture=liquid_mixture; uniform_cpu->gas_mixture=gas_mixture;
            uniform_cpu->gravity_y=gasScenario()?0.:100.;
        } else if (engine_index==GPU_UNIFORM) {
            if (!uniform_gpu) uniform_gpu=std::make_unique<fluid::FluidGridGPU>(grid_w,grid_h,cell_size);
            uniform_gpu->elements=elements; uniform_gpu->liquid_mixture=liquid_mixture; uniform_gpu->gas_mixture=gas_mixture;
            uniform_gpu->gravity_y=gasScenario()?0.:100.;
        } else if (engine_index==GPU_WEBGL) {
            uniform_webgl=std::make_unique<fluid::FluidGridWebGL>(grid_w,grid_h,cell_size);
            uniform_webgl->elements=elements; uniform_webgl->liquid_mixture=liquid_mixture; uniform_webgl->gas_mixture=gas_mixture;
            uniform_webgl->gravity_y=gasScenario()?0.:100.;
        }
        initializePool();
        prepareDisplay();
    } catch (const std::runtime_error& error) {
        failure_reason=error.what(); simulation_failed=true; blPrint()<<failure_reason;
    }
    initialization_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    // Initial selection/restoration is already included in initialization_ms.
    physics_seconds=last_physics_ms=0;
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
#ifdef FLUIDSIM_HAS_AMR
    gpu_preview.reset();
    adaptive_gpu.reset();
    adaptive_cpu.reset();
    amr_runtime.reset();
#endif
    uniform_cpu.reset(); uniform_gpu.reset();
    renderThreadQueue().post([this]()
    {
        uniform_webgl.reset(); webgl_material={};
        fluid_display.destroy();
        liquid_depth_map.destroy();
        surface.destroy();
#ifdef BITLOOP_DEV_MODE
        snapshot_preprocessor.destroyGL();
#endif
    });
}

void FluidSim_Scene::viewportProcess(Viewport*, double) {}

void FluidSim_Scene::sceneProcess(double dt)
{
    requestRedraw(true);
    if (simulation_failed || dt <= 0.0)
        return;
    const auto frame_started=std::chrono::steady_clock::now();
    last_frame_interval_ms=frame_dt();

    // Equal substeps avoid tiny leftover timesteps that spoil pressure reuse.
    const double frame_time = scenario.frame_time;
    const int steps_per_frame = 16;

    const int step_count = int(std::ceil(frame_time * double(steps_per_frame)));
    const double step = frame_time / step_count;

    simulation_time += frame_time;
    real_time += dt;

    last_physics_ms = last_remap_ms = last_divergence = 0;
    frame_physics_timings = {};
    frame_substeps = 0;
    frame_stage_timings_available = false;
#ifdef BITLOOP_DEV_MODE
    frame_pressure_iterations = 0;
#endif
    if (uniform_cpu) uniform_cpu->gas_damping_rate=atmosphere_damping;
    if (uniform_gpu) uniform_gpu->gas_damping_rate=atmosphere_damping;
    if (uniform_webgl) uniform_webgl->gas_damping_rate=atmosphere_damping;
    for (int substep=0; substep<step_count; ++substep) {
        const auto started=std::chrono::steady_clock::now();
        fluid::FluidProjectionResult result;
        try {
#ifdef FLUIDSIM_HAS_AMR
            if (adaptive_gpu || adaptive_cpu) {
                const std::array<double,2> acceleration={0.,gasScenario()?0.:100.};
                const auto stepped=adaptive_gpu ? adaptive_gpu->step(step,projection_tolerance,double(atmosphere_damping),acceleration,profile_stages)
                    : adaptive_cpu->step(step,projection_tolerance,double(atmosphere_damping),acceleration);
                result=stepped.projection;
                if (profile_stages) {
                    frame_physics_timings.transport_ms+=stepped.transportMs;
                    frame_physics_timings.pressure_ms+=stepped.projectionMs;
                }
            } else
#endif
            {
                FluidGrid::StepTimings timings;
                auto* measured = profile_stages ? &timings : nullptr;
                if (uniform_cpu) result=uniform_cpu->step(step,2000,projection_tolerance,measured);
                else if (uniform_gpu) { result=uniform_gpu->step(step,2000,projection_tolerance,measured); if (!measured) uniform_gpu->synchronize(); }
                else renderThreadQueue().invokeBlocking([&] { result=uniform_webgl->step(step,2000,projection_tolerance,measured); if (!measured) glFinish(); });
                frame_physics_timings.transport_ms+=timings.transport_ms;
                frame_physics_timings.forces_ms+=timings.forces_ms;
                frame_physics_timings.pressure_ms+=timings.pressure_ms;
            }
            last_divergence=std::max(last_divergence,result.max_divergence);
#ifdef BITLOOP_DEV_MODE
            frame_pressure_iterations+=result.iterations;
#endif
            if (!result.converged) throw std::runtime_error("Pressure projection did not converge.");
        } catch (const std::runtime_error& error) {
            failure_reason=error.what(); simulation_failed=true; blPrint()<<failure_reason; return;
        }
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        physics_seconds+=elapsed; last_physics_ms+=elapsed*1000;
        ++frame_substeps;
    }
    if (!simulation_failed)
    {
        try {
            updateCellResolution();
            prepareDisplay();
        }
        catch (const std::runtime_error& error) {
            failure_reason=error.what(); simulation_failed=true; blPrint()<<failure_reason; return;
        }
        last_frame_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-frame_started).count();
        frame_stage_timings_available=profile_stages;
#ifdef BITLOOP_DEV_MODE
        // Health readback is a diagnostic, outside the reported simulation/display cost.
        if (getProject().isHeadless()) inspectGPUState();
        if (getProject().isHeadless() && processed_frames >= scenario.capture_i0 && processed_frames < scenario.capture_i1 &&
            ((processed_frames - scenario.capture_i0) % scenario.capture_stride == 0 ||
                (scenario.type == LOCAL_COARSE_PATCH && (last_remap_ms>0 || processed_frames + 1 == scenario.capture_i1))))
        {
            traceSnapshot("grid", "scenario capture interval");
            traceSnapshot("material", "material-only comparison");
        }
#endif
        ++processed_frames;
    }
}

void FluidSim_Scene::renderFluidSurface(bool overlays) const
{
    const auto started = std::chrono::steady_clock::now();
    // Subcell display samples make outlines visible in both resolutions when zoomed.
    surface.render(display_scale * grid_w, display_scale * grid_h, [&](const ShaderSurface& shader)
    {
        const auto material = materialTextures();
        const auto mixture=displayMixture();
        const int liquid_count=adaptive_gpu ? adaptive_gpu->liquidIngredientCount() :
            adaptive_cpu ? adaptive_cpu->liquidIngredientCount() : int(liquid_mixture.components.size());
        const bool preview_visible=overlays && show_coarsening_preview && (adaptive_gpu ? !gpu_preview->labels.empty() : !coarsening_preview.labels.empty());
        const auto preview=preview_visible ? (adaptive_gpu ? fluid_display.previewTexture(gpu_preview->width,gpu_preview->height,
            gpu_preview->labels,gpu_preview_revision,this) : fluid_display.texture(coarsening_preview)) : material.fractions;
        shader.bindTexture2D("u_fractions", material.fractions);
        shader.bindTexture2D("u_liquid", material.liquid);
        const bool depth_shading=liquid_depth_shading && liquid_count>0;
        shader.bindTexture2D("u_gas_seeds",depth_shading ? liquid_depth_map.texture(grid_w,grid_h,material.liquid) : material.liquid);
        shader.setUniform1f("u_depth_shading",depth_shading ? 1.f : 0.f);
        shader.setUniform1f("u_depth_scale",liquid_depth_scale);
        shader.setUniform1f("u_deep_brightness",liquid_deep_brightness);
        shader.bindTexture2D("u_palette", fluid_display.paletteTexture(elements,mixture));
        shader.setUniform2f("u_ingredient_counts",float(mixture.components.size()),float(liquid_count));
        shader.bindTexture2D("u_coarsening", preview);
        shader.setUniform1f("u_coarsening_opacity", preview_visible ? coarsening_opacity : 0.0f);
        shader.setUniform2f("u_grid_size", float(grid_w), float(grid_h));
        shader.bindTexture2D("u_resolution", resolutionTexture());
        shader.setUniform1f("u_mask_opacity", overlays && resolution_overlay ? mask_opacity : 0.0f);
    });
    last_display_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

void FluidSim_Scene::viewportDraw(Viewport* ctx) const
{
    const auto started = std::chrono::steady_clock::now();
    last_render_ms = 0;
    if (!uniform_cpu && !adaptive_cpu && gpu_material.values.empty() && !webgl_material.fractions) return;
    if (adaptive_gpu && gpu_material.values.empty()) return;
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
    last_render_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
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
    frame.metric("engine", engine_index);
    frame.metric("webgpu", engine_index==GPU_UNIFORM || engine_index==GPU_ADAPTIVE);
    frame.metric("perlin_clouds", perlin_clouds);
    frame.metric("simulation_time", simulation_time);
    frame.metric("frame_time", scenario.frame_time);
    frame.metric("simulation_failed", simulation_failed);
    if (simulation_failed) frame.metric("failure_reason", failure_reason);
    if (!uniform_cpu && !uniform_gpu && !uniform_webgl && !adaptive_cpu && !adaptive_gpu) return;
    frame.metric("adaptive_engine", adaptive_engine);
    frame.metric("amr_workers", amr_workers);
    frame.metric("simulated_cells", simulated_cells);
    frame.metric("allocated_level_cells", allocated_cells);
    frame.metric("amr_levels", level_count);
    frame.metric("fine_patches", fine_patches);
    frame.metric("max_divergence", last_divergence);
    frame.metric("simulation_blocks", simulation_blocks);
    frame.metric("coarse_scale", coarse_scale);
    frame.metric("display_ms", last_display_ms);
    frame.metric("render_cpu_ms", last_render_ms);
    frame.metric("gpu_display_readback_ms", gpu_display_ms);
    frame.metric("completed_frame_ms", last_frame_ms);
    frame.metric("initialization_ms", initialization_ms);
    frame.metric("physics_seconds", physics_seconds);
    frame.metric("last_physics_ms", last_physics_ms);
    frame.metric("remap_ms", last_remap_ms);
    frame.metric("patch_changes", patch_changes);
    frame.metric("coarsening_preview_ms", last_coarsening_ms);
    frame.metric("coarsening_preview_enabled", show_coarsening_preview);
    frame.metric("adaptive_grid", adaptive_engine && adaptive_grid);
    frame.metric("uniform_coarse", adaptive_engine && !adaptive_grid && uniform_coarse);
    frame.metric("coarsening_candidates", coarsening_counts[fluid::FluidCoarseningPreview::Candidate]);
    frame.metric("coarsening_detail", coarsening_counts[fluid::FluidCoarseningPreview::Detail]);
    frame.metric("coarsening_protected", coarsening_counts[fluid::FluidCoarseningPreview::LiquidBuffer]);
    frame.metric("coarsening_unassessed", coarsening_counts[fluid::FluidCoarseningPreview::Unassessed]);
    frame.metric("refinement_detail", coarsening_counts[fluid::FluidCoarseningPreview::Refine]);
    frame.metric("refinement_approaching", coarsening_counts[fluid::FluidCoarseningPreview::Approaching]);
    frame.metric("refinement_lookahead", refinement_lookahead);
    frame.metric("resolution_raw_changes", coarsening_statistics.raw_changes);
    frame.metric("resolution_decision_changes", coarsening_statistics.decision_changes);
    frame.metric("resolution_waiting", coarsening_statistics.waiting);
    frame.metric("resolution_held", coarsening_statistics.held);
    frame.metric("phase_timings_available", frame_stage_timings_available);
    frame.metric("physics_substeps", frame_substeps);
    if (frame_stage_timings_available) {
        frame.metric("transport_ms", frame_physics_timings.transport_ms);
        frame.metric("forces_ms", frame_physics_timings.forces_ms);
        frame.metric("pressure_ms", frame_physics_timings.pressure_ms);
    }
    frame.metric("pressure_iterations", frame_pressure_iterations);
    frame.metric("atmosphere_damping", atmosphere_damping);
    frame.metric("projection_tolerance", projection_tolerance);
    frame.metric("mask_opacity", resolution_overlay ? mask_opacity : 0.0f);
    // Inspect actual simulated cells; expanded display samples are not extra material.
    double volume_error = 0, mass_error = 0;
    double minimum_mass = 0;
    bool finite = true;
#ifdef FLUIDSIM_HAS_AMR
    if (adaptive_gpu || adaptive_cpu) {
        const auto totals=adaptive_gpu ? gpu_totals : adaptive_cpu->totals();
        const int liquid_count=adaptive_gpu ? adaptive_gpu->liquidIngredientCount() : adaptive_cpu->liquidIngredientCount();
        volume_error=totals.maxVolumeError; finite=totals.finite;
        // Native conserved ingredients are mass densities; make the trace unit explicit.
        frame.metric("minimum_material_density",totals.minMassDensity);
        frame.metric("max_speed",totals.maxSpeed);
        for (size_t k=0; k<totals.ingredientMass.size(); ++k) {
            const bool liquid=k<size_t(liquid_count);
            const size_t ingredient=liquid?k:k-liquid_count;
            const double initial=liquid?initial_liquid_mass[ingredient]:initial_gas_mass[ingredient];
            frame.metric(std::string(liquid?"liquid_ingredient_mass_":"gas_ingredient_mass_")+std::to_string(ingredient),totals.ingredientMass[k]);
            mass_error=std::max(mass_error,std::abs(totals.ingredientMass[k]-initial)/std::max(1.,initial));
        }
    } else
#endif
    {
        auto inspect=[&](const auto& g) {
        const double area=cell_size*cell_size;
        auto budget=[&](const auto& fields,const auto& exchange,const auto& initial) {
            for (size_t k=0; k<fields.size(); ++k) {
                const double total=std::accumulate(fields[k].begin(),fields[k].end(),0.)-(exchange.empty()?0.:exchange[k]);
                mass_error=std::max(mass_error,std::abs(total-initial[k])/std::max(1.,initial[k]));
                for (double m:fields[k]) { finite &= std::isfinite(m); minimum_mass=std::min(minimum_mass,m); }
            }
        };
        budget(g.liquid_mass,uniform_cpu?uniform_cpu->boundary_liquid_mass:std::vector<double>{},initial_liquid_mass);
        budget(g.gas_mass,uniform_cpu?uniform_cpu->boundary_gas_mass:std::vector<double>{},initial_gas_mass);
        for (size_t i=0; i<g.liquid_fraction.size(); ++i) {
            double liquid=0,gas=0;
            for (size_t k=0; k<g.liquid_mass.size(); ++k) liquid+=g.liquid_mass[k][i]/(liquid_mixture.components[k].density*area);
            for (size_t k=0; k<g.gas_mass.size(); ++k) gas+=g.gas_mass[k][i]/(gas_mixture.components[k].density*area);
            finite &= std::isfinite(liquid) && std::isfinite(gas);
            volume_error=std::max({volume_error,std::abs(liquid-g.liquid_fraction[i]),std::abs(gas-(1.-g.liquid_fraction[i]))});
        }
        for (double v:g.u) finite &= std::isfinite(v);
        for (double v:g.v) finite &= std::isfinite(v);
        };
        if (uniform_cpu) inspect(*uniform_cpu); else inspect(uniform_gpu_state);
    }
    frame.metric("mass_budget_relative_error", mass_error);
    frame.metric("phase_volume_error", volume_error);
    if (uniform_cpu) frame.metric("minimum_material_mass", minimum_mass);
    frame.metric("finite_state", finite);
}

bool FluidSim_Scene::captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const
{
    if ((!uniform_cpu && !uniform_gpu && !uniform_webgl && !adaptive_cpu && !adaptive_gpu) || (adaptive_gpu && gpu_material.values.empty())) return false;
    std::uint32_t texture;
    if (route == "grid" || route == "appearance")
    {
        renderFluidSurface(route == "grid");
        image.resolution = IVec2(display_scale * grid_w, display_scale * grid_h);
        texture = surface.texture();
    }
    else if (route == "material")
    {
        image.resolution = IVec2(grid_w, grid_h);
        texture = materialTexture();
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
