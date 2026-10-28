#pragma once
#include <bitloop.h>
#include "FluidGrid.h"
#include "FluidGridDisplay.h"
#include "FluidSnapshot.h"
#include <utility>

SIM_BEG;

using namespace bl;
using fluid::FluidGrid;
using fluid::ElementId;
using fluid::FluidGridDisplay;

struct FluidSim_Scene : public Scene<FluidSim_Scene>
{
    CameraInfo       camera;
    CameraNavigator  navigator;

    /// ─────── Variables ───────
    int grid_w = 1024;
    int grid_h = 720;

    double cell_size = 1.0;

    bool simulation_failed = false;

    double simulation_time = 0.0;
    double real_time = 0.0;

    double world_w = double(grid_w) * cell_size;
    double world_h = double(grid_h) * cell_size;

    FluidGrid fluid;
    fluid::FluidDevice fluid_device = fluid::FluidDevice::GPU;
    fluid::AdaptiveMode adaptive_mode = fluid::AdaptiveMode::DISABLED;
    fluid::FluidAdaptiveSettings adaptive_settings;

    static constexpr double frame_time = 0.02;
    static constexpr int steps_per_frame = 60;
    static constexpr int projection_max_iterations = 2000;
    double projection_tolerance = 1; // Scene policy, independent of engine API defaults.

    ElementId air_element{};
    ElementId water_element{};
    ElementId red_element{};
    ElementId cloud_element{};

    const char* fluid_shader_src = R"(
        #version 300 es
        precision highp float;

        uniform sampler2D u_color;
        
        in vec2 v_uv;
        out vec4 outColor;

        void main()
        {
            outColor = texture(u_color, v_uv);
        }
    )";

    ShaderSurface surface;
    FluidGridDisplay fluid_display;
    std::vector<float> display_pixels;
    bool profile_stages = true;
    int checkpoint_brotli_quality = -1; // Storage policy: -1=uncompressed, 0..11=Brotli.
    std::string resume_image;
    std::string snapshot_error;
    struct FrameStatistics
    {
        double update_ms = 0, fluid_ms = 0, colour_ms = 0;
        fluid::FluidStepTimings stages;
        int substeps = 0;
        bool detailed = false, adaptive = false;
        double projection_tolerance = 1e-2;
        bool operator==(const FrameStatistics&) const = default;
    };
    FrameStatistics frame_statistics;
    // Render and sidebar both run on the GUI owner; do not shadow-buffer this value.
    mutable double last_render_ms = 0;
    void initializePool(double amplitude = 0.04);
    void restartSimulation();
    void renderFluidSurface() const;
    JSON::json captureSceneState() const;
    void restoreSceneState(const JSON::json& state, const fluid::FluidCheckpoint& checkpoint,
        int width, int height, double spacing);
    bool captureSnapshot(SnapshotFrame& output) const override;
    void resumeFromImage(const std::filesystem::path& image);

    // Route operations to the engine's required owner (WebGPU stays on the simulation worker).
    template<typename Operation>
    void withFluidContext(Operation&& operation) const
    {
        if (fluid.requiresRenderThread())
        {
            // Bitloop's queue accessor is non-const; a read-only checkpoint still needs owner dispatch.
            const_cast<FluidSim_Scene*>(this)->renderThreadQueue().invokeBlocking(std::forward<Operation>(operation));
        }
        else
            operation();
    }

    /// ─────── Methods ───────
    

    /// ─────── Launch config ───────
    struct Config
    {
        int checkpoint_brotli_quality = -1;
        std::string resume_image;
    };

    FluidSim_Scene(Config& info)
        : fluid(grid_w, grid_h, cell_size)
    {
        checkpoint_brotli_quality = info.checkpoint_brotli_quality;
        resume_image = info.resume_image;
        // Scene-owned policy; selection, conservative regrid and restoration run in the backend.
        adaptive_settings.prediction_horizon = adaptive_settings.check_interval + frame_time;

        fluid.setAdaptiveSettings(adaptive_settings);
        fluid.setDevice(fluid_device);
        fluid.setAdaptiveMode(adaptive_mode);

        fluid::ElementCatalogue elements;
        fluid::FluidMixture liquid_mixture, gas_mixture;

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
        fluid.configureMaterials(std::move(elements), std::move(liquid_mixture), std::move(gas_mixture));
    }

    struct UI : BufferedInterfaceModel
    {
        using BufferedInterfaceModel::BufferedInterfaceModel;
        void sidebar() override;
        // void overlay() override; // ImGui viewport overlay

        /// ─────── Your UI-only variables ───────

        double pressure_tolerance_before_edit = 1e-2;

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
