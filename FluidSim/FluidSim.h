#pragma once
#include <bitloop.h>
#include "FluidGrid.h"
#include "FluidGridGPU.h"
#include "FluidGridWebGL.h"
#ifdef FLUIDSIM_HAS_AMR
#include "FluidGridAdaptive.h"
#include "FluidGridAdaptiveGPU.h"
#include "FluidCoarseningGPU.h"
#include "AmrGasRuntime.h"
#endif
#include <memory>
#include <optional>
#include "FluidGridDisplay.h"
#include "FluidShader.h"
#include "FluidLiquidDepth.h"
#include "FluidCoarseningPreview.h"
#include <utility>

SIM_BEG;

using namespace bl;
using fluid::FluidGrid;
using fluid::ElementId;
using fluid::FluidGridDisplay;

enum FluidEngine { CPU_UNIFORM, CPU_ADAPTIVE, GPU_UNIFORM, GPU_ADAPTIVE, GPU_WEBGL };

enum ScenarioType
{
    COSINE_WAVES,
    CRASHING_WAVE,
    LOCAL_COARSE_PATCH,
    MOVING_CLOUD_DETAIL
};

struct Scenario
{
    ScenarioType type;
    int capture_i0; // inclusive, zero-based post-step frame
    int capture_i1; // exclusive
    double frame_time;
    int capture_stride = 1;
};

struct FluidSim_Scene : public Scene<FluidSim_Scene>
{
    CameraInfo       camera;
    CameraNavigator  navigator;

    /// ─────── Variables ───────
    int grid_w = 1024;
    int grid_h = 720;

    std::vector<Scenario> scenarios = {
        {COSINE_WAVES, 0, 50, 0.04},
        {CRASHING_WAVE, 40, 120, 0.04},
        {LOCAL_COARSE_PATCH, 0, 121, 0.08, 25},
        {MOVING_CLOUD_DETAIL, 0, 401, 0.08, 100}
    };

    int scenario_index = CRASHING_WAVE;

    Scenario scenario = scenarios[scenario_index];
    bool gasScenario() const { return scenario.type==LOCAL_COARSE_PATCH || scenario.type==MOVING_CLOUD_DETAIL; }

    double cell_size = 1.0;

    bool simulation_failed = false;

    double simulation_time = 0.0;
    double real_time = 0.0;

    double world_w = double(grid_w) * cell_size;
    double world_h = double(grid_h) * cell_size;

    fluid::ElementCatalogue elements;
    fluid::FluidMixture liquid_mixture, gas_mixture;

    std::unique_ptr<FluidGrid>             uniform_cpu;
    std::unique_ptr<fluid::FluidGridGPU>   uniform_gpu;
    std::unique_ptr<fluid::FluidGridWebGL> uniform_webgl;
    FluidGridDisplay::MaterialTextures webgl_material{};
#ifdef FLUIDSIM_HAS_AMR
    std::unique_ptr<fluid::FluidGridAdaptive>    adaptive_cpu;
    std::unique_ptr<fluid::FluidGridAdaptiveGPU> adaptive_gpu;
    std::unique_ptr<fluid::FluidCoarseningGPU>   gpu_preview;

    std::unique_ptr<fluid::AmrGasRuntime> amr_runtime;
#endif

    int engine_index = GPU_ADAPTIVE;

    bool usesGPU() const { return engine_index >= GPU_UNIFORM; }
    bool perlin_clouds = true;
    fluid::FluidMaterialPixels gpu_material;
    std::uint64_t gpu_labels_revision = ~std::uint64_t(0);
    std::uint64_t gpu_preview_revision = 0;
    double gpu_display_ms = 0, last_frame_ms = 0, initialization_ms = 0;
    double last_frame_interval_ms = 0;
    mutable double last_render_ms = 0;
    bool profile_stages = true;
    bool frame_stage_timings_available = false;
    int frame_substeps = 0;
    FluidGrid::StepTimings frame_physics_timings;
    void prepareDisplay();
    bool adaptive_engine = true;
    int amr_workers = 0;
    int simulated_cells = 0, allocated_cells = 0, simulation_blocks = 0, fine_patches = 0, level_count = 1;
    int coarse_scale = 4; // maximum physical cell width in finest-cell units
    double last_divergence = 0;
    void updateCellCounts();
    std::uint32_t materialTexture() const;
    FluidGridDisplay::MaterialTextures materialTextures() const;
    fluid::FluidMixture displayMixture() const;
    std::uint32_t resolutionTexture() const;
    std::string failure_reason;
    double physics_seconds = 0, last_physics_ms = 0;
    double last_remap_ms = 0;
    int processed_frames = 0, patch_changes = 0;

    float atmosphere_damping = scenario.type==MOVING_CLOUD_DETAIL ? 0.0f : 0.0f; // s^-1; zero disables damping
    double projection_tolerance = 1e-3;
    fluid::FluidCoarseningPreview coarsening_preview;

    // The wave trial accepts some cloud smoothing; liquid still stays at 1x1.
    fluid::FluidCoarseningPreview::Tolerances coarsening_tolerances = scenario.type==CRASHING_WAVE
        //? fluid::FluidCoarseningPreview::Tolerances{0.4, 0.4, 0.04, 0.04}
        ? fluid::FluidCoarseningPreview::Tolerances{10.0, 10.0, 0.5, 0.5}
        : fluid::FluidCoarseningPreview::Tolerances{};

    fluid::FluidCoarseningPreview::Stability coarsening_stability;
    fluid::FluidCoarseningPreview::Statistics coarsening_statistics;

    bool show_coarsening_preview = true;
    bool adaptive_grid = true;
    bool uniform_coarse = false; // comparison control when automatic selection is off
    double refinement_interval = 0.32, next_refinement_time = 0;
    float coarsening_opacity = 0.35f;
    double refinement_lookahead = scenario.frame_time;
    double last_coarsening_ms = 0;
    std::array<int, fluid::FluidCoarseningPreview::LabelCount> coarsening_counts{};
    void updateCellResolution();


    ElementId air_element{};
    ElementId water_element{};
    ElementId red_element{};
    ElementId cloud_element{};

    ShaderSurface surface;
    fluid::FluidLiquidDepth liquid_depth_map;
    bool liquid_depth_shading = true;
    float liquid_depth_scale = 32.0f;
    float liquid_deep_brightness = 0.16f;
    FluidGridDisplay fluid_display;
    static constexpr int display_scale = 3;
    bool resolution_overlay = true;
    float mask_opacity = 0.2f;
    mutable double last_display_ms = 0;
    void initializePool(double amplitude = 0.04);
    void restartSimulation();
    void renderFluidSurface(bool overlays = true) const;

#ifdef BITLOOP_DEV_MODE
    int frame_pressure_iterations = 0;
    std::vector<double> initial_liquid_mass, initial_gas_mass;
    mutable CapturePreprocessor snapshot_preprocessor;
    fluid::GasTotals gpu_totals;
    fluid::FluidGridGPU::State uniform_gpu_state;
    void inspectGPUState();
#endif

    /// ─────── Methods ───────
    

    /// ─────── Launch config ───────
    struct Config {
        std::optional<int> size, scenario, max_cell, engine, capture_stride;
        std::optional<bool> webgpu, adaptive, coarse_control, clouds, profile_stages;
        std::optional<double> damping, frame_time, velocity_tolerance, pressure_tolerance, density_tolerance, ingredient_tolerance, projection_tolerance;
    };

    FluidSim_Scene(Config& info)
    {
        if (info.size) grid_w=grid_h=*info.size;
        if (info.scenario) scenario_index=*info.scenario;
        scenario=scenarios[scenario_index];
        if (info.frame_time) scenario.frame_time=*info.frame_time;
        if (info.capture_stride) scenario.capture_stride=*info.capture_stride;
        world_w=grid_w*cell_size; world_h=grid_h*cell_size;
        if (info.scenario) atmosphere_damping=scenario.type==MOVING_CLOUD_DETAIL?0.f:2.f;
        if (info.damping) atmosphere_damping=float(*info.damping);
        if (info.projection_tolerance) projection_tolerance=*info.projection_tolerance;
        refinement_lookahead=scenario.frame_time;
        if (info.scenario) coarsening_tolerances=scenario.type==CRASHING_WAVE ? fluid::FluidCoarseningPreview::Tolerances{10.,10.,.5,.5} : fluid::FluidCoarseningPreview::Tolerances{};
        if (info.velocity_tolerance) coarsening_tolerances.velocity=*info.velocity_tolerance;
        if (info.pressure_tolerance) coarsening_tolerances.pressure_gradient=*info.pressure_tolerance;
        if (info.density_tolerance) coarsening_tolerances.density=*info.density_tolerance;
        if (info.ingredient_tolerance) coarsening_tolerances.ingredient=*info.ingredient_tolerance;
        if (info.webgpu) engine_index=*info.webgpu ? GPU_ADAPTIVE : CPU_ADAPTIVE;
        engine_index=info.engine.value_or(engine_index);
        adaptive_grid=info.adaptive.value_or(adaptive_grid);
        uniform_coarse=info.coarse_control.value_or(uniform_coarse);
        perlin_clouds=info.clouds.value_or(perlin_clouds);
        profile_stages=info.profile_stages.value_or(profile_stages);
        coarse_scale=info.max_cell.value_or(coarse_scale);
        air_element = elements.add({
            "Air", {0.015f, 0.025f, 0.06f}
        });

        water_element = elements.add({
            "Water", {0.1f, 0.35f, 1.0f}
        });

        red_element = elements.add({
            "Red fluid", {0.95f, 0.25f, 0.15f}
        });

        cloud_element = elements.add({
            "Cloud", {0.0f, 1.0f, 1.0f}
        });

        liquid_mixture.components = {
            {water_element, 100.0},
            {red_element, 300.0}
        };

        const double atmosphere_density = 0.1;
        gas_mixture.components = {
            {air_element, atmosphere_density},
            {cloud_element, atmosphere_density}
        };
    }

    struct UI : BufferedInterfaceModel
    {
        using BufferedInterfaceModel::BufferedInterfaceModel;
        void sidebar() override;
        // void overlay() override; // ImGui viewport overlay

        /// ─────── Your UI-only variables ───────

    };

    /// ─────── Scene methods ───────
    void sceneStart() override;
    void sceneMounted(Viewport* viewport) override;
    void sceneUnmounted(Viewport* viewport) override;
    void sceneDestroy() override;
    void sceneProcess(double dt) override;

    /// ─────── Viewport methods ───────
    void viewportProcess(Viewport* ctx, double dt) override;
    void viewportDraw(Viewport* ctx) const override;

    /// ─────── Input handling ───────
    void onEvent(Event e) override; // receives all event types

    #ifdef BITLOOP_DEV_MODE
    void traceFrame(SimTraceFrame& frame) const override;
    bool captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const override;
    bool saveCheckpoint(SimCheckpointWriter& writer) const override;
    bool loadCheckpoint(const SimCheckpointReader& reader) override;
    #endif
};

struct FluidSim_Project : public Project<FluidSim_Project>
{
    static ProjectInfo info()
    {
        return ProjectInfo({ "New Projects", "FluidSim" });
    }

    struct UI : BufferedInterfaceModel
    {
        using BufferedInterfaceModel::BufferedInterfaceModel;
        void sidebar();
        // void overlay();
    };

    int viewport_count = 1;

    void projectPrepare(Layout& layout) override;
};

SIM_END;
