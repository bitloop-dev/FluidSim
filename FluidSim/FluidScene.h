#pragma once
#include <bitloop.h>
#include "FluidGrid.h"
#include "FluidGridDisplay.h"
#include "FluidSnapshot.h"
#include <utility>
#include <optional>

SIM_BEG;

using namespace bl;
using fluid::FluidGrid;
using fluid::FluidGridDisplay;

// Initial values supplied by each scenario; live controls edit the scene instance.
struct FluidSceneSettings
{
    int grid_w;
    int grid_h;
    double cell_size = 1.0;
    fluid::FluidDevice fluid_device = fluid::FluidDevice::GPU;
    fluid::AdaptiveMode adaptive_mode = fluid::AdaptiveMode::DISABLED;
    double frame_time = 0.02;
    int steps_per_second = 60;
    int projection_max_iterations = 2000;
    double projection_tolerance = 0.1;
    double gravity_y = 0.0;
    double surface_tension = 0.0;
};

// Keep the concrete type visible to Bitloop's buffered UI.
template<class Derived>
struct FluidScene : bl::Scene<Derived>
{
    using BufferedInterfaceModel = typename bl::Scene<Derived>::BufferedInterfaceModel;

    CameraInfo camera;
    CameraNavigator navigator;

    int grid_w;
    int grid_h;
    double cell_size;
    double world_w = double(grid_w) * cell_size;
    double world_h = double(grid_h) * cell_size;

    FluidGrid fluid;
    fluid::FluidDevice fluid_device;
    fluid::AdaptiveMode adaptive_mode;
    fluid::FluidAdaptiveSettings adaptive_settings;

    double frame_time;
    int steps_per_second;
    int projection_max_iterations;
    double projection_tolerance;
    double gravity_y; // Positive Y points down.
    double surface_tension;

    bool simulation_failed = false;
    double simulation_time = 0.0;
    double real_time = 0.0;

    bool mouse_attraction = false;
    double mouse_attraction_strength = 500.0;
    double mouse_attraction_radius = 20.0;
    DVec2 mouse_target = {world_w * 0.5, world_h * 0.5};
    bool mouse_field_active = false;
    std::optional<fluid::FluidField> mouse_field;
    std::unique_ptr<fluid::FluidFieldWebGPUGenerator> mouse_field_generator;

    static constexpr const char* fluid_shader_src = R"(
        #version 300 es
        precision highp float;

        uniform sampler2D u_color;
        uniform sampler2D u_proximity;
        uniform float u_deep_brightness;

        in vec2 v_uv;
        out vec4 outColor;

        void main()
        {
            vec4 material = texture(u_color, v_uv); // Alpha carries liquid coverage.
            float proximity = sqrt(clamp(2.0 * texture(u_proximity, v_uv).r, 0.0, 1.0));
            vec3 transmission = pow(vec3(proximity), vec3(1.25, 0.9, 0.75));
            vec3 shallow = mix(material.rgb, vec3(0.48, 0.85, 0.95), 0.3);
            vec3 deep = material.rgb * u_deep_brightness;
            vec3 water = max(deep, mix(deep, shallow, transmission));
            outColor = vec4(mix(material.rgb, water, clamp(material.a, 0.0, 1.0)), 1.0);
        }
    )";

    ShaderSurface surface;
    FluidGridDisplay fluid_display;
    std::vector<float> display_pixels;
    float liquid_tint_depth = 6.0f; // Gas proximity range in cells.
    float liquid_deep_brightness = 0.6f; // Minimum liquid brightness.
    bool profile_stages = true;
    int checkpoint_brotli_quality = -1; // Storage policy: -1=uncompressed, 0..11=Brotli.
    std::string resume_image;
    std::string snapshot_error;

    struct FrameStatistics
    {
        double update_ms = 0, fluid_ms = 0, colour_ms = 0;
        fluid::FluidStepTimings stages;
        int substeps = 0;
        int transport_steps = 0, pressure_iterations = 0;
        bool detailed = false, adaptive = false;
        double projection_tolerance = 1e-2;
        bool operator==(const FrameStatistics&) const = default;
    };
    FrameStatistics frame_statistics;
    // Render and sidebar both run on the GUI owner; do not shadow-buffer this value.
    mutable double last_render_ms = 0;

    void restartSimulation();
    void setSurfaceTension(double coefficient);
    void updateAccelerationField();
    virtual void renderFluidSurface() const;
    JSON::json captureSceneState() const;
    void restoreSceneState(const JSON::json& state, const fluid::FluidCheckpoint& checkpoint,
        int width, int height, double spacing);
    virtual void captureFluidImage(CaptureFrame& output) const;
    bool captureFrame(const CapturePreset& request, CaptureFrame& output) const override;
    void resumeFromImage(const std::filesystem::path& image);

    // Route operations to the engine's required owner (WebGPU stays on the simulation worker).
    template<typename Operation>
    void withFluidContext(Operation&& operation) const
    {
        if (fluid.requiresRenderThread())
        {
            // Bitloop's queue accessor is non-const; a read-only checkpoint still needs owner dispatch.
            const_cast<FluidScene*>(this)->renderThreadQueue().invokeBlocking(std::forward<Operation>(operation));
        }
        else
            operation();
    }

    // Storage/launch options are separate from the scenario's initial settings.
    struct Config
    {
        int checkpoint_brotli_quality = -1;
        std::string resume_image;
    };

    FluidScene(const FluidSceneSettings& settings, const Config& config);

    // The base owns startup/restart and uploads the scenario's initial state.
    virtual FluidGrid::State initialFluidState() const = 0;
    virtual void beforeFluidFrame(double /*dt*/) {}
    virtual void drawScenario(Viewport* /*viewport*/) const {}

    // Validate external state before replacing the live grid. Restore only commits
    // the already validated values; it must not perform fallible resource work.
    virtual JSON::json captureScenarioState() const { return JSON::json::object(); }
    virtual void validateScenarioState(const JSON::json&, const FluidGrid&) const {}
    virtual void restoreScenarioState(const JSON::json&) {}

    struct UI : BufferedInterfaceModel
    {
        using BufferedInterfaceModel::BufferedInterfaceModel;
        void sidebar() override;

        // Bring buffered helpers into scope for the common templated UI.
        using BufferedInterfaceModel::__target;
        using BufferedInterfaceModel::_pull;
        using BufferedInterfaceModel::_temp_pull;
        using BufferedInterfaceModel::_commit;
        using BufferedInterfaceModel::_schedule;

        double pressure_tolerance_before_edit = 1e-2;
        double surface_tension_before_edit = 0.;
    };

    // Common lifecycle and rendering; scenarios customize the hooks above.
    void sceneStart() final;
    void sceneMounted(Viewport* viewport) override;
    void sceneDestroy() final;
    void sceneProcess(double dt) final;

    void viewportDraw(Viewport* ctx) const override;

    void onEvent(Event e) override;

    #ifdef BITLOOP_DEV_MODE
    void traceFrame(SimTraceFrame& frame) const override;
    bool captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const override;
    bool saveCheckpoint(SimCheckpointWriter& writer) const override;
    bool loadCheckpoint(const SimCheckpointReader& reader) override;
    #endif
};

SIM_END;

#include "FluidScene.inl"
#include "FluidSceneUI.inl"
