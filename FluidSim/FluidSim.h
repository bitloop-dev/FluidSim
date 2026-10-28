#pragma once
#include <bitloop.h>
#include "FluidGrid.h"
#include "FluidGridDisplay.h"
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
    void initializePool(double amplitude = 0.04);
    void renderFluidSurface() const;

    // GPU work belongs to the render owner; CPU physics stays on the simulation worker
    template<typename Operation>
    void withFluidContext(Operation&& operation)
    {
        if (fluid.requiresRenderThread())
            renderThreadQueue().invokeBlocking(std::forward<Operation>(operation));
        else
            operation();
    }
#ifdef BITLOOP_DEV_MODE
    mutable CapturePreprocessor snapshot_preprocessor;
#endif

    /// ─────── Methods ───────
    

    /// ─────── Launch config ───────
    struct Config {};

    FluidSim_Scene(Config& info [[maybe_unused]])
        : fluid(grid_w, grid_h, cell_size)
    {
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
