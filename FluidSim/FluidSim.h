#pragma once
#include <bitloop.h>
#include "FluidGrid.h"
#ifdef FLUIDSIM_HAS_AMR
#include "AmrGas.h"
#include "AmrGasRuntime.h"
#endif
#include <memory>
#include "FluidGridDisplay.h"
#include "FluidCoarseningPreview.h"
#include <utility>

SIM_BEG;

using namespace bl;
using fluid::FluidGrid;
using fluid::ElementId;
using fluid::FluidGridDisplay;

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
    int grid_w = 512;
    int grid_h = 512;

    std::vector<Scenario> scenarios = {
        {COSINE_WAVES, 0, 50, 0.04},
        {CRASHING_WAVE, 40, 120, 0.08},
        {LOCAL_COARSE_PATCH, 0, 121, 0.08, 25},
        {MOVING_CLOUD_DETAIL, 0, 401, 0.08, 100}
    };

    const int scenario_index = CRASHING_WAVE;

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
    std::unique_ptr<FluidGrid> uniform;
#ifdef FLUIDSIM_HAS_AMR
    std::unique_ptr<fluid::AmrGasRuntime> amr_runtime;
    std::unique_ptr<fluid::AmrGas> gas;
#endif
    bool amr_native = false;
    int amr_workers = 0;
    int simulated_cells = 0, allocated_cells = 0, simulation_blocks = 0, fine_patches = 0, level_count = 1;
    int coarse_scale = 32; // maximum physical cell width in finest-cell units
    double last_divergence = 0;
    void updateCellCounts();
    std::uint32_t materialTexture() const;
    std::uint32_t resolutionTexture() const;
    std::string failure_reason;
    double physics_seconds = 0, last_physics_ms = 0;
    double last_remap_ms = 0;
    int processed_frames = 0, patch_changes = 0;

    float atmosphere_damping = scenario.type==MOVING_CLOUD_DETAIL ? 0.0f : 2.0f; // s^-1; zero disables damping
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

    const char* fluid_shader_src = R"(
        #version 300 es
        precision highp float;

        uniform sampler2D u_color;
        uniform sampler2D u_coarsening;
        uniform float u_coarsening_opacity;
        uniform vec2 u_grid_size;
        uniform sampler2D u_resolution;
        uniform float u_mask_opacity;
        in vec2 v_uv;
        out vec4 outColor;
        void main()
        {
            outColor = texture(u_color, v_uv);
            vec2 cell = v_uv * u_grid_size;
            if (u_coarsening_opacity > 0.0)
            {
                int label = int(round(255.0 * texelFetch(u_coarsening, ivec2(cell) / 2, 0).r));
                vec3 fill = vec3(0.55); // unassessed block/physical edge
                if (label == 2) fill = vec3(0.65, 0.3, 0.85); // protected liquid/buffer
                if (label == 3) fill = vec3(1.0, 0.7, 0.05); // retain fine detail / wait for quiet
                if (label == 4) fill = vec3(0.15, 0.85, 0.3); // candidate merge
                if (label == 5) fill = vec3(1.0, 0.15, 0.15); // resolved coarse detail
                if (label == 6) fill = vec3(1.0, 0.45, 0.05); // approaching feature
                // No fill means no detected refinement trigger, not known lost detail.
                if (label != 0) outColor.rgb = mix(outColor.rgb, fill, u_coarsening_opacity);
            }
            float scale = round(255.0 * texelFetch(u_resolution, ivec2(cell) / 2, 0).r);
            vec3 tint = scale > 16.5 ? vec3(0.95, 0.25, 0.2) : scale > 8.5 ? vec3(0.1, 0.85, 0.9) : scale > 4.5 ? vec3(0.85, 0.3, 0.9) : scale > 2.5 ? vec3(0.95, 0.55, 0.15) : scale > 1.5 ? vec3(0.15, 0.8, 0.25) : vec3(0.15, 0.35, 0.95);
            vec2 q = cell / scale;
            vec2 edge = min(fract(q), 1.0 - fract(q));
            vec2 line = 1.0 - smoothstep(vec2(0.0), fwidth(q) * 1.25, edge);
            outColor.rgb = mix(outColor.rgb, tint, u_mask_opacity * max(line.x, line.y));
        }
    )";

    ShaderSurface surface;
    FluidGridDisplay fluid_display;
    static constexpr int display_scale = 3;
    bool resolution_overlay = true;
    float mask_opacity = 0.2f;
    mutable double last_display_ms = 0;
    void initializePool(double amplitude = 0.04);
    void restartSimulation();
    void renderFluidSurface() const;

#ifdef BITLOOP_DEV_MODE
    FluidGrid::StepTimings frame_physics_timings;
    int frame_pressure_iterations = 0;
    std::vector<double> initial_liquid_mass, initial_gas_mass;
    mutable CapturePreprocessor snapshot_preprocessor;
#endif

    /// ─────── Methods ───────
    

    /// ─────── Launch config ───────
    struct Config {};

    FluidSim_Scene(Config& info [[maybe_unused]])
    {
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
