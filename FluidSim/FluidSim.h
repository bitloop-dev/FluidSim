#pragma once
#include <bitloop.h>
#include "FluidGrid.h"
#include "FluidGridGPU.h"
#include "FluidGridDisplay.h"
#include "FluidAmbientMaskDisplay.h"
#include "FluidAmbientMask.h"
#include <utility>

SIM_BEG;

using namespace bl;
using fluid::FluidGrid;
using fluid::ElementId;
using fluid::FluidGridGPU;
using fluid::FluidGridDisplay;

enum ScenarioType
{
    COSINE_WAVES,
    CRASHING_WAVE
};

struct Scenario
{
    ScenarioType type;
    int capture_i0; // inclusive, zero-based post-step frame
    int capture_i1; // exclusive
    double frame_time;
};

struct FluidSim_Scene : public Scene<FluidSim_Scene>
{
    CameraInfo       camera;
    CameraNavigator  navigator;

    /// ─────── Variables ───────
    int grid_w = 256;
    int grid_h = 256;

    std::vector<Scenario> scenarios = {
        {COSINE_WAVES, 0, 50, 0.04},
        {CRASHING_WAVE, 40, 120, 0.08}
    };

    const int scenario_index = 1;
    Scenario scenario = scenarios[scenario_index];

    double cell_size = 1.0;

    bool simulation_failed = false;

    double simulation_time = 0.0;
    double real_time = 0.0;

    double world_w = double(grid_w) * cell_size;
    double world_h = double(grid_h) * cell_size;

    FluidGrid fluid;
    double physics_seconds = 0, last_physics_ms = 0;
    float atmosphere_damping = 0.2f; // s^-1, every scenario; zero disables damping

    // Reservoir gas shares, in mixture order; the initial cloud pattern varies their split.
    const std::vector<double> atmosphere_fractions = {0.5, 0.5}; // Air + Cloud, summing to one

    ElementId air_element{};
    ElementId water_element{};
    ElementId red_element{};
    ElementId cloud_element{};

    const char* fluid_shader_src = R"(
        #version 300 es
        precision highp float;

        uniform sampler2D u_color, u_mask;
        uniform float u_mask_opacity;
        in vec2 v_uv;
        out vec4 outColor;
        void main()
        {
            outColor = texture(u_color, v_uv);
            if (u_mask_opacity > 0.0)
                outColor.rgb = mix(outColor.rgb, texture(u_mask, v_uv).rgb, u_mask_opacity);
        }
    )";

    ShaderSurface surface;
    FluidGridDisplay fluid_display;
    fluid::FluidAmbientMaskDisplay mask_display;
    fluid::FluidAmbientMask ambient_mask;
    fluid::FluidAmbientMask::Settings ambient_mask_settings, observed_ambient_mask_settings;
    fluid::FluidAmbientMask::Statistics ambient_mask_statistics;
    bool ambient_mask_enabled = !decltype(fluid)::requires_render_thread, ambient_mask_observed = false;
    float mask_opacity = 0.35f;
    double mask_seconds = 0, last_mask_ms = 0;
    void observeAmbientMask(double dt);
    void initializePool(double amplitude = 0.04);
    void restartSimulation();
    void renderFluidSurface() const;

    // GPU work belongs to the render owner; CPU physics stays on the simulation worker
    template<typename Operation>
    void withFluidContext(Operation&& operation)
    {
        if constexpr (decltype(fluid)::requires_render_thread)
            renderThreadQueue().invokeBlocking(std::forward<Operation>(operation));
        else
            operation();
    }
#ifdef BITLOOP_DEV_MODE
    int processed_frames = 0;
    mutable CapturePreprocessor snapshot_preprocessor;
#endif

    /// ─────── Methods ───────
    

    /// ─────── Launch config ───────
    struct Config {};

    FluidSim_Scene(Config& info [[maybe_unused]])
        : fluid(grid_w, grid_h, cell_size)
    {
        air_element = fluid.elements.add({
            "Air", {0.015f, 0.025f, 0.06f}
        });

        water_element = fluid.elements.add({
            "Water", {0.1f, 0.35f, 1.0f}
        });

        red_element = fluid.elements.add({
            "Red fluid", {0.95f, 0.25f, 0.15f}
        });

        cloud_element = fluid.elements.add({
            "Cloud", {0.0f, 1.0f, 1.0f}
        });

        fluid.liquid_mixture.components = {
            {water_element, 100.0},
            {red_element, 300.0}
        };

        const double atmosphere_density = 0.1;
        fluid.gas_mixture.components = {
            {air_element, atmosphere_density},
            {cloud_element, atmosphere_density}
        };
        ambient_mask_settings.liquid_fractions = {0, 0};
        ambient_mask_settings.gas_fractions = atmosphere_fractions;
        // Air and Cloud have identical physics here; their colours trace atmospheric motion.
        ambient_mask_settings.gas_tracer_elements = {air_element, cloud_element};
        ambient_mask_settings.pressure_tolerance = 3.0f;
        // The scene supplies its background; the classifier knows neither gravity nor Air.
        ambient_mask_settings.pressure_gradient.y = atmosphere_density * fluid.gravity_y;
        // Restore the experiment's explicit ambient boundary and its anchored pressure gauge.
        [&]<class Grid>(Grid& grid) {
            if constexpr (!Grid::requires_render_thread)
                grid.top_ambient = FluidGrid::AmbientBoundary{0, atmosphere_fractions};
        }(fluid);
        ambient_mask_settings.reservoir_edges = fluid::FluidAmbientMask::Top;
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
