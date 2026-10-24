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
        ImGui::SliderFloat("Gas damping (per second)", &atmosphere_damping, 0.0f, 2.0f, "%.2f");
        ImGui::TextWrapped("Zero disables damping. Applies to gas-only velocities in every scenario, with or without the overlay.");
        ImGui::EndCollapsingHeaderBox();
    }
    if (ImGui::CollapsingHeaderBox("Cell resolution", true))
    {
        bl_scoped(resolution_overlay, mask_opacity, refinement_interval);
        bl_pull_temp(adaptive_grid, uniform_coarse, coarse_scale);
        bl_view(amr_workers, amr_native, simulated_cells, allocated_cells, simulation_blocks, fine_patches, level_count, grid_w, grid_h, failure_reason, last_display_ms, patch_changes, last_remap_ms);
        ImGui::BeginDisabled(!amr_native);
        if (ImGui::Checkbox("Adaptive grid", &adaptive_grid))
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
        if (amr_native) {
            ImGui::BeginDisabled(!adaptive_grid);
            ImGui::SliderDouble("Minimum check interval (seconds)", &refinement_interval, 0.08, 2.0, "%.2f", ImGuiSliderFlags_AlwaysClamp);
            ImGui::EndDisabled();
            ImGui::TextWrapped("Changing grid mode restarts this scene with the same settings. Adaptive uses up to the selected maximum, keeping liquid and its buffer fine. Off uses the same engine with uniform fine cells; the coarse control uses the selected maximum everywhere. Outlines: blue 1x1, green 2x2, amber 4x4, magenta 8x8, cyan 16x16, red 32x32.");
        }
        else ImGui::TextWrapped("Web builds use uniform CPU cells. Native wave scenes keep liquid fine and adapt the surrounding gas.");
        if (!failure_reason.empty()) ImGui::TextWrapped("Stopped: %s", failure_reason.c_str());
        ImGui::EndCollapsingHeaderBox();
    }
    if (ImGui::CollapsingHeaderBox("Resolution preview", true))
    {
        bl_scoped(show_coarsening_preview, coarsening_opacity, coarsening_tolerances, coarsening_stability, refinement_lookahead);
        bl_view(last_coarsening_ms, coarsening_counts, coarsening_statistics, adaptive_grid, amr_native);
        ImGui::BeginDisabled(amr_native && !adaptive_grid);
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
        if ((amr_native && adaptive_grid) || (!amr_native && show_coarsening_preview))
        {
            ImGui::Text("Estimate: %.3f ms / frame", last_coarsening_ms);
            ImGui::Text("Displayed groups: %d candidates; %d retain detail", coarsening_counts[P::Candidate], coarsening_counts[P::Detail]);
            ImGui::Text("%d protected; %d unassessed", coarsening_counts[P::LiquidBuffer], coarsening_counts[P::Unassessed]);
            ImGui::Text("Coarse: %d refine; %d approaching; %d no trigger", coarsening_counts[P::Refine], coarsening_counts[P::Approaching], coarsening_counts[P::Coarse]);
            ImGui::Text("Waiting to merge: %d; retained refinement: %d", coarsening_statistics.waiting, coarsening_statistics.held);
            ImGui::Text("Changed actions: %d raw / %d stabilised", coarsening_statistics.raw_changes, coarsening_statistics.decision_changes);
        }
        ImGui::TextWrapped("Fill: green = settled merge candidate; yellow = retain resolution / waiting; purple = liquid/buffer; grey = unassessed edge. Coarse: red = refine; orange = approaching detail. Requests stay until quiet; no fill = no detected trigger. Blue/green/amber/magenta/cyan/red lines show actual 1x1/2x2/4x4/8x8/16x16/32x32 resolution.");
        ImGui::TextWrapped("The estimates drive automatic atmosphere resolution in native scenes, including waves. Liquid and its approaching region remain fine. Ingredient tolerance 0.02 means two percentage points locally, not a bound on accumulated transport error. Hiding this overlay does not disable automatic selection.");
        ImGui::TextWrapped("Merging and clearing refinement need sustained detail below the quiet threshold. New detail above the full tolerance, liquid and incoming features act immediately. Ratio 1 and delay 0 disable stabilisation. Changing thresholds or look-ahead restarts the quiet period. Layout changes reset only cells whose resolution changes.");
        ImGui::TextWrapped("Look-ahead sweeps known features using current velocities and gravity. Automatic selection uses at least the check interval plus one frame. It cannot recover discarded detail or predict future pressure changes or new sources.");
        ImGui::EndDisabled();
        ImGui::EndCollapsingHeaderBox();
    }
}

void FluidSim_Scene::updateCellResolution()
{
    last_coarsening_ms = 0;
    if (amr_native ? !adaptive_grid : !show_coarsening_preview)
    {
        coarsening_preview.reset();
        coarsening_counts = {}; coarsening_statistics = {};
        return;
    }
    if (amr_native && adaptive_grid && simulation_time < next_refinement_time) return;
    const auto started = std::chrono::steady_clock::now();
#ifdef FLUIDSIM_HAS_AMR
    if (gas) coarsening_preview.update(*gas, coarsening_tolerances, coarsening_stability, simulation_time,
        adaptive_grid ? std::max(refinement_lookahead,refinement_interval+scenario.frame_time) : refinement_lookahead,coarse_scale);
    else
#endif
        coarsening_preview.update(*uniform, coarsening_tolerances, coarsening_stability, simulation_time, refinement_lookahead);
    last_coarsening_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count();
    coarsening_counts = coarsening_preview.counts;
    coarsening_statistics = coarsening_preview.statistics;
#ifdef FLUIDSIM_HAS_AMR
    if (gas && adaptive_grid) {
        next_refinement_time=simulation_time+refinement_interval;
        const auto clustered=std::chrono::steady_clock::now();
        auto layout=coarsening_preview.proposeLayout(*gas);
        last_coarsening_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-clustered).count();
        const auto remap=std::chrono::steady_clock::now();
        if (gas->regrid(std::move(layout))) {
            const auto restored=gas->project(1e-3);
            if (!restored.converged) throw std::runtime_error("Pressure restoration after automatic refinement failed.");
            last_divergence=std::max(last_divergence,restored.max_divergence);
            updateCellCounts(); ++patch_changes;
            last_remap_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-remap).count();
            last_physics_ms+=last_remap_ms; physics_seconds+=last_remap_ms/1000.;
#ifdef BITLOOP_DEV_MODE
            frame_pressure_iterations+=restored.iterations;
#endif
        }
    }
#endif
}

void FluidSim_Scene::updateCellCounts() {
#ifdef FLUIDSIM_HAS_AMR
    if (gas) {
        simulated_cells=int(gas->activeCellCount()); allocated_cells=int(gas->allocatedCellCount());
        level_count=int(gas->levels().size()); fine_patches=0; simulation_blocks=0;
        for (const auto& patches:gas->layout().refinedPatches) fine_patches+=int(patches.size());
        for (const auto& level:gas->levels()) simulation_blocks+=level->conserved.size();
        return;
    }
#endif
    simulated_cells=allocated_cells=grid_w*grid_h; simulation_blocks=level_count=1; fine_patches=0;
}
std::uint32_t FluidSim_Scene::materialTexture() const {
#ifdef FLUIDSIM_HAS_AMR
    if (gas) return fluid_display.texture(*gas);
#endif
    return fluid_display.texture(*uniform);
}
std::uint32_t FluidSim_Scene::resolutionTexture() const {
#ifdef FLUIDSIM_HAS_AMR
    if (gas) return fluid_display.resolutionTexture(*gas);
#endif
    return fluid_display.resolutionTexture(grid_w,grid_h);
}

void FluidSim_Scene::initializePool(double amplitude)
{
    //constexpr double pi = 3.14159265358979323846;
    const size_t cell_count = size_t(grid_w) * grid_h;

    using Scalar = double;
    FluidGrid::State initial;
    initial.liquid_fraction.resize(cell_count);

    const double cell_volume = cell_size * cell_size;
    //const double cloud_noise_scale = 1.0 / grid_w;

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

            //const double surface = (grid_h * 0.7) + (std::cos(tx * 20.0) * double(grid_h) * 0.02);
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
            const double red_share = 0.0;// x < grid_w / 4 ? 1.0 : 0.0;

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
            } else {
                // Layer three Perlin scales to give broad clouds some smaller detail.
                //const double nx=(x+.5)*cloud_noise_scale, ny=(y+.5)*cloud_noise_scale;
                //const double cloud_noise=(perlinNoise(nx,ny)+.5*perlinNoise(2.*nx,2.*ny)+.25*perlinNoise(4.*nx,4.*ny))/1.75;
                //cloud_share=std::clamp(.15+2.5*cloud_noise,0.,1.);
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
    if (amr_native) {
        const fluid::GasLayout layout=!adaptive_grid && uniform_coarse ? fluid::GasLayout{coarse_scale,{}} : fluid::GasLayout{};
        gas = std::make_unique<fluid::AmrGas>(grid_w,grid_h,cell_size,elements,
            gasScenario()?fluid::FluidMixture{}:liquid_mixture,gas_mixture,layout);
        gas->upload(initial,1e-3);
    } else
#endif
    {
        uniform->upload(initial);
        if (gasScenario() && !uniform->project(.016,2000,1e-3).converged)
            throw std::runtime_error("Initial gas pressure projection failed.");
    }
    updateCellCounts();
    updateCellResolution();
}

void FluidSim_Scene::sceneStart()
{
#ifdef FLUIDSIM_HAS_AMR
    amr_native = true;
    if (amr_native) { amr_runtime=std::make_unique<fluid::AmrGasRuntime>(); amr_workers=amr_runtime->workerCount(); }
    else
#endif
    {
        uniform=std::make_unique<FluidGrid>(grid_w,grid_h,cell_size);
        uniform->elements=elements; uniform->liquid_mixture=liquid_mixture; uniform->gas_mixture=gas_mixture;
        if (gasScenario()) uniform->gravity_y=0;
    }
    initializePool();
    surface.setFragmentSource(fluid_shader_src);
}

void FluidSim_Scene::restartSimulation()
{
    // Refill the current scenario; keep controls, camera and engine options intact.
    coarsening_preview.reset();
    next_refinement_time=0;
    simulation_time = real_time = 0;
    initializePool();
    simulation_failed = false;
    physics_seconds = last_physics_ms = 0;
    last_remap_ms = 0;
    processed_frames = patch_changes = 0;
    failure_reason.clear();
    last_display_ms = 0;
#ifdef BITLOOP_DEV_MODE
    frame_physics_timings = {};
    frame_pressure_iterations = 0;
#endif
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
    gas.reset();
    amr_runtime.reset();
#endif
    uniform.reset();
    renderThreadQueue().post([this]()
    {
        fluid_display.destroy();
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

    last_physics_ms = last_remap_ms = last_divergence = 0;
#ifdef BITLOOP_DEV_MODE
    frame_physics_timings = {}; frame_pressure_iterations = 0;
#endif
    if (uniform) uniform->gas_damping_rate=atmosphere_damping;
    for (int substep=0; substep<step_count; ++substep) {
        const auto started=std::chrono::steady_clock::now();
        fluid::FluidProjectionResult result;
        try {
#ifdef FLUIDSIM_HAS_AMR
            if (gas) {
                const auto stepped=gas->step(step,1e-3,double(atmosphere_damping),{0.,gasScenario()?0.:100.});
                result=stepped.projection;
#ifdef BITLOOP_DEV_MODE
                frame_physics_timings.transport_ms+=stepped.transportMs;
                frame_physics_timings.pressure_ms+=stepped.projectionMs;
#endif
            } else
#endif
            {
                FluidGrid::StepTimings timings;
                result=uniform->step(step,2000,1e-3,&timings);
#ifdef BITLOOP_DEV_MODE
                frame_physics_timings.transport_ms+=timings.transport_ms;
                frame_physics_timings.forces_ms+=timings.forces_ms;
                frame_physics_timings.pressure_ms+=timings.pressure_ms;
#endif
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
    }
    if (!simulation_failed)
    {
        try { updateCellResolution(); }
        catch (const std::runtime_error& error) {
            failure_reason=error.what(); simulation_failed=true; blPrint()<<failure_reason; return;
        }
#ifdef BITLOOP_DEV_MODE
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

void FluidSim_Scene::renderFluidSurface() const
{
    const auto started = std::chrono::steady_clock::now();
    // Subcell display samples make outlines visible in both resolutions when zoomed.
    surface.render(display_scale * grid_w, display_scale * grid_h, [&](const ShaderSurface& shader)
    {
        const auto colour = materialTexture();
        const bool preview_visible = show_coarsening_preview && !coarsening_preview.labels.empty();
        const auto preview = preview_visible ? fluid_display.texture(coarsening_preview) : colour;
        shader.bindTexture2D("u_color", colour);
        shader.bindTexture2D("u_coarsening", preview);
        shader.setUniform1f("u_coarsening_opacity", preview_visible ? coarsening_opacity : 0.0f);
        shader.setUniform2f("u_grid_size", float(grid_w), float(grid_h));
        shader.bindTexture2D("u_resolution", resolutionTexture());
        shader.setUniform1f("u_mask_opacity", resolution_overlay ? mask_opacity : 0.0f);
    });
    last_display_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
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
    frame.metric("amr_native", amr_native);
    frame.metric("amr_workers", amr_workers);
    frame.metric("simulated_cells", simulated_cells);
    frame.metric("allocated_level_cells", allocated_cells);
    frame.metric("amr_levels", level_count);
    frame.metric("fine_patches", fine_patches);
    frame.metric("max_divergence", last_divergence);
    frame.metric("simulation_blocks", simulation_blocks);
    frame.metric("coarse_scale", coarse_scale);
    frame.metric("display_ms", last_display_ms);
    frame.metric("physics_seconds", physics_seconds);
    frame.metric("last_physics_ms", last_physics_ms);
    frame.metric("remap_ms", last_remap_ms);
    frame.metric("patch_changes", patch_changes);
    frame.metric("coarsening_preview_ms", last_coarsening_ms);
    frame.metric("coarsening_preview_enabled", show_coarsening_preview);
    frame.metric("adaptive_grid", amr_native && adaptive_grid);
    frame.metric("uniform_coarse", amr_native && !adaptive_grid && uniform_coarse);
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
    {
        frame.metric("transport_ms", frame_physics_timings.transport_ms);
        frame.metric("forces_ms", frame_physics_timings.forces_ms);
        frame.metric("pressure_ms", frame_physics_timings.pressure_ms);
        frame.metric("pressure_iterations", frame_pressure_iterations);
    }
    frame.metric("atmosphere_damping", atmosphere_damping);
    frame.metric("mask_opacity", resolution_overlay ? mask_opacity : 0.0f);
    // Inspect actual simulated cells; expanded display samples are not extra material.
    double volume_error = 0, mass_error = 0;
    double minimum_mass = 0;
    bool finite = true;
#ifdef FLUIDSIM_HAS_AMR
    if (gas) {
        const auto totals=gas->totals();
        volume_error=totals.maxVolumeError; finite=totals.finite;
        // Native conserved ingredients are mass densities; make the trace unit explicit.
        frame.metric("minimum_material_density",totals.minMassDensity);
        frame.metric("max_speed",totals.maxSpeed);
        for (size_t k=0; k<totals.ingredientMass.size(); ++k) {
            const bool liquid=k<size_t(gas->liquidIngredientCount());
            const size_t ingredient=liquid?k:k-gas->liquidIngredientCount();
            const double initial=liquid?initial_liquid_mass[ingredient]:initial_gas_mass[ingredient];
            frame.metric(std::string(liquid?"liquid_ingredient_mass_":"gas_ingredient_mass_")+std::to_string(ingredient),totals.ingredientMass[k]);
            mass_error=std::max(mass_error,std::abs(totals.ingredientMass[k]-initial)/std::max(1.,initial));
        }
    } else
#endif
    {
        const auto& g=*uniform;
        const double area=cell_size*cell_size;
        auto budget=[&](const auto& fields,const auto& exchange,const auto& initial) {
            for (size_t k=0; k<fields.size(); ++k) {
                const double total=std::accumulate(fields[k].begin(),fields[k].end(),0.)-exchange[k];
                mass_error=std::max(mass_error,std::abs(total-initial[k])/std::max(1.,initial[k]));
                for (double m:fields[k]) { finite &= std::isfinite(m); minimum_mass=std::min(minimum_mass,m); }
            }
        };
        budget(g.liquid_mass,g.boundary_liquid_mass,initial_liquid_mass);
        budget(g.gas_mass,g.boundary_gas_mass,initial_gas_mass);
        for (size_t i=0; i<g.liquid_fraction.size(); ++i) {
            double liquid=0,gas=0;
            for (size_t k=0; k<g.liquid_mass.size(); ++k) liquid+=g.liquid_mass[k][i]/(liquid_mixture.components[k].density*area);
            for (size_t k=0; k<g.gas_mass.size(); ++k) gas+=g.gas_mass[k][i]/(gas_mixture.components[k].density*area);
            finite &= std::isfinite(liquid) && std::isfinite(gas);
            volume_error=std::max({volume_error,std::abs(liquid-g.liquid_fraction[i]),std::abs(gas-(1.-g.liquid_fraction[i]))});
        }
        for (double v:g.u) finite &= std::isfinite(v);
        for (double v:g.v) finite &= std::isfinite(v);
    }
    frame.metric("mass_budget_relative_error", mass_error);
    frame.metric("phase_volume_error", volume_error);
    if (uniform) frame.metric("minimum_material_mass", minimum_mass);
    frame.metric("finite_state", finite);
}

bool FluidSim_Scene::captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const
{
    std::uint32_t texture;
    if (route == "grid")
    {
        renderFluidSurface();
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
