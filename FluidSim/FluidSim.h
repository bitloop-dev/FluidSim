#pragma once
#include <bitloop.h>
#include "FluidGrid.h"

SIM_BEG;

using namespace bl;
using fluid::FluidGrid;

struct FluidSim_Scene : public Scene<FluidSim_Scene>
{
    CameraInfo       camera;
    CameraNavigator  navigator;

    /// ─────── Variables ───────
    int grid_w = 128;
    int grid_h = 128;

    double cell_size = 4.0;

    bool simulation_failed = false;

    double world_w = double(grid_w) * cell_size;
    double world_h = double(grid_h) * cell_size;

    FluidGrid fluid;

    const char* fluid_shader_src = R"(
        #version 300 es
        precision highp float;

        uniform sampler2D u_water;
        
        in vec2 v_uv;
        out vec4 outColor;

        void main()
        {
            float water_fraction = texture(u_water, v_uv).r;

            vec3 air_color = vec3(0.015, 0.025, 0.06);
            vec3 water_color = vec3(0.1, 0.75, 0.95);

            outColor = vec4(mix(air_color, water_color, water_fraction), 1.0);
        }
    )";

    ShaderSurface surface;
    mutable GLuint water_texture = 0;
    mutable std::vector<unsigned char> water_pixels;
    void initializePool(double amplitude = 0.04);
    void uploadWaterTexture() const;
    void renderFluidSurface() const;
#ifdef BITLOOP_DEV_MODE
    mutable CapturePreprocessor snapshot_preprocessor;
#endif

    /// ─────── Methods ───────
    

    /// ─────── Launch config ───────
    struct Config {};

    FluidSim_Scene(Config& info [[maybe_unused]])
        : fluid(grid_w, grid_h, cell_size)
    {
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
