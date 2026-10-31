#include "scenarios/CrashingWave.h"
#include <array>
#include <iostream>

using namespace bl;
using FluidSim::CrashingWave;
using fluid::FluidGrid;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

bool sameState(const FluidGrid::State& a, const FluidGrid::State& b)
{
    return a.liquid_fraction == b.liquid_fraction && a.liquid_mass == b.liquid_mass &&
        a.gas_mass == b.gas_mass && a.u == b.u && a.v == b.v && a.pressure == b.pressure;
}

struct WaveProject : BasicProject
{
    std::array<SceneHandle<CrashingWave>, 4> waves;

    void projectPrepare(Layout& layout) override
    {
        for (int i = 0; i < 4; ++i)
        {
            CrashingWave::Config config;
            config.settings.grid_w = 32 + 4 * i;
            config.settings.grid_h = 24;
            config.settings.fluid_device = fluid::FluidDevice::CPU;
            config.settings.surface_tension = 0;
            config.settings.gravity_y = 10 + i;
            config.settings.frame_time = .01 + .005 * i;
            waves[i] = create<CrashingWave>("wave-" + std::to_string(i), config);
        }
        // Four views: one scene has two views, and another runs without a view.
        layout.add(waves[0], "first");
        layout.add(waves[0], "second");
        layout.add(waves[1], "third");
        layout.add(waves[2], "fourth");
    }
};

void checkIndependentWaves()
{
    ImDebugLog log;
    CaptureContext capture;
    WaveProject project;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();

    std::array<std::unique_ptr<fluid::FluidGrid>, 4> controls;
    for (int i = 0; i < 4; ++i)
    {
        auto& scene = *project.waves[i];
        scene.profile_stages = false;
        controls[i] = std::make_unique<fluid::FluidGrid>(scene.grid_w, scene.grid_h, scene.cell_size);
        controls[i]->configureMaterials(scene.fluid.getSubstances(),
            scene.fluid.getLiquidMixture(), scene.fluid.getGasMixture());
        controls[i]->setGravityY(scene.gravity_y);
        controls[i]->upload(scene.fluid.download());
    }

    for (int frame = 0; frame < 12; ++frame)
    {
        project.processHeadlessFrame(.02);
        for (int i = 0; i < 4; ++i)
        {
            const auto& scene = *project.waves[i];
            const int count = int(std::ceil(scene.frame_time * scene.steps_per_second));
            for (int step = 0; step < count; ++step)
                require(controls[i]->step(scene.frame_time / count,
                    scene.projection_max_iterations, scene.projection_tolerance).converged,
                    "Control projection failed");
            require(!scene.simulation_failed && sameState(scene.fluid.download(), controls[i]->download()),
                "Scene lifecycle differs from once-per-frame direct stepping");
            require(std::abs(scene.simulation_time - (frame + 1) * scene.frame_time) < 1e-12,
                "Scene clock depends on viewport count");
        }
    }

    const auto untouched = project.waves[1]->fluid.download();
    const double untouched_time = project.waves[1]->simulation_time;
    project.waves[0]->grid_w = 48;
    project.waves[0]->restartSimulation();
    require(project.waves[0]->simulation_time == 0 && project.waves[0]->fluid.getWidth() == 48,
        "Per-instance restart failed");
    require(sameState(project.waves[1]->fluid.download(), untouched) &&
        project.waves[1]->simulation_time == untouched_time, "Restart changed another scene");
    project.processHeadlessFrame(0.);
    require(sameState(project.waves[1]->fluid.download(), untouched), "Zero dt advanced a scene");

    auto& recovery = *project.waves[0];
    bool unsupported = false;
    try { recovery.setPressureRecoveryFrames(5); }
    catch (const std::invalid_argument&) { unsupported = true; }
    require(unsupported && recovery.pressure_recovery_frames == 0,
        "CPU silently accepted pressure recovery");
    recovery.fluid_device = fluid::FluidDevice::GPU;
    recovery.restartSimulation();
    const auto initial = recovery.fluid.download();
    project.processHeadlessFrame(.02);
    recovery.setPressureRecoveryFrames(5);
    const auto relaxed_initial = recovery.fluid.download();
    auto expected_initial = initial;
    // Relaxed geometry is derived from packed float masses, with one additional rounding step.
    for (size_t i = 0; i < initial.liquid_fraction.size(); ++i)
        require(std::abs(relaxed_initial.liquid_fraction[i] - initial.liquid_fraction[i]) < 2e-7,
            "Recovery restart changed initial geometry");
    expected_initial.liquid_fraction = relaxed_initial.liquid_fraction;
    require(recovery.simulation_time == 0. && !recovery.simulation_failed &&
        recovery.fluid.getPressureRecoveryTime() == 5 * recovery.frame_time &&
        sameState(expected_initial, relaxed_initial), "Entering recovery did not restart cleanly");
    for (int i = 0; i < 6; ++i) project.processHeadlessFrame(.02);
    require(!recovery.simulation_failed, "Recovery scene failed");
    const auto live = recovery.fluid.download();
    const double elapsed = recovery.simulation_time;
    recovery.setPressureRecoveryFrames(10);
    require(recovery.simulation_time == elapsed && sameState(live, recovery.fluid.download()) &&
        recovery.fluid.getPressureRecoveryTime() == 10 * recovery.frame_time,
        "Changing a positive recovery horizon reset the scene");
    const auto scene_state = recovery.captureSceneState();
    fluid::FluidCheckpoint saved;
    require(recovery.fluid.saveCheckpoint(saved), "Recovery scene checkpoint unsupported");
    recovery.setPressureRecoveryFrames(0);
    require(recovery.simulation_time == 0. && sameState(initial, recovery.fluid.download()) &&
        recovery.fluid.getPressureRecoveryTime() == 0., "Returning to accurate mode retained compressed state");
    recovery.restoreSceneState(scene_state, saved, recovery.grid_w, recovery.grid_h, recovery.cell_size);
    require(recovery.pressure_recovery_frames == 10 && recovery.simulation_time == elapsed &&
        sameState(live, recovery.fluid.download()), "Scene restore lost recovery mode or state");
    project.processHeadlessFrame(.02);
    require(!recovery.simulation_failed, "Restored recovery scene failed");
    std::cout << "PASS recovery mode transitions, time conversion, unsupported backend and scene restoration\n";
    project.stopHeadless();
    std::cout << "PASS four independent waves, zero/two viewport scheduling, restart isolation and direct-step parity" << std::endl;
}

// Test-only scenario exercises extension points without adding another app scenario.
struct ExtensionScene : FluidSim::FluidScene<ExtensionScene>
{
    using Base = FluidSim::FluidScene<ExtensionScene>;
    static constexpr const char* scenario_id = "test-extension";
    int marker = 1;
    int frames = 0;

    explicit ExtensionScene(const Config& config)
        : Base({.grid_w = 16, .grid_h = 12}, config)
    {
        fluid::SubstanceCatalogue substances;
        const auto liquid = substances.add({"Liquid", {0, 0, 1}});
        const auto gas = substances.add({"Gas", {0, 0, 0}});
        fluid.configureMaterials(std::move(substances), {{{liquid, 1.}}}, {{{gas, .1}}});
    }

    FluidGrid::State initialFluidState() const override
    {
        FluidGrid::State state;
        state.liquid_fraction.assign(grid_w * grid_h, 0.);
        state.liquid_mass = {std::vector<double>(grid_w * grid_h, 0.)};
        state.gas_mass = {std::vector<double>(grid_w * grid_h, .1)};
        return state;
    }

    void beforeFluidFrame(double) override { ++frames; }
    JSON::json captureScenarioState() const override { return {{"marker", marker}}; }
    void validateScenarioState(const JSON::json& state, const FluidGrid&) const override
    {
        (void)state.at("marker").get<int>();
    }
    void restoreScenarioState(const JSON::json& state) override { marker = state.at("marker").get<int>(); }
    void captureFluidImage(CaptureFrame& output) const override
    {
        output.resolution = {1, 1};
        output.frame.data = {12, 34, 56, 255};
    }

    struct UI : Base::UI
    {
        using Base::UI::UI;
        int sidebars = 0;

        void sidebar() override
        {
            Base::UI::sidebar();
            ++sidebars;
        }

        void edit()
        {
            bl_scoped(marker, projection_tolerance);
            marker = 7;
            projection_tolerance = .25;
        }

        void schedule()
        {
            bl_schedule([](ExtensionScene& scene) { scene.marker = 9; });
        }
    };
};

struct ExtensionProject : BasicProject
{
    SceneHandle<ExtensionScene> source;
    void projectPrepare(Layout&) override { source = create<ExtensionScene>(); }
};

void checkExtensions()
{
    ImDebugLog log;
    CaptureContext capture;
    ExtensionProject project;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();
    auto& scene = *project.source;
    auto& base = static_cast<SceneBase&>(scene);
    base._initGUI();
    auto* ui = dynamic_cast<ExtensionScene::UI*>(scene.ui);
    require(ui != nullptr, "Framework constructed the base UI instead of the concrete scenario UI");
    ui->edit();
    require(scene.marker == 1 && scene.projection_tolerance == .1, "UI bypassed shadow buffering");
    base.updateLiveBuffers();
    require(scene.marker == 7 && scene.projection_tolerance == .25, "Inherited/derived UI edits were not applied");
    ui->schedule();
    require(scene.marker == 7, "Scheduled action ran on the UI owner");
    project.processHeadlessFrame(.02);
    require(scene.marker == 9 && scene.frames == 1, "Concrete scheduled action/frame hook did not run");

    PlatformManager platform(nullptr); // Supplies UI scale without opening a window.
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().DisplaySize = {800, 600};
    ImGui::GetIO().Fonts->Build();
    ImGui::NewFrame();
    ImGui::Begin("Fluid scene test");
    ui->sidebar();
    ImGui::End();
    ImGui::Render();
    require(ui->sidebars == 1, "Extended common sidebar did not run");
    ImGui::DestroyContext();

    CaptureFrame image;
    require(scene.captureFrame(CapturePreset{}, image) && image.frame.data == bytebuf({12, 34, 56, 255}),
        "Shared capture did not use scenario image rendering");
    const auto snapshot = fluid::decodeFluidSnapshot(image.frame.payload);
    scene.marker = -1;
    scene.restoreSceneState(snapshot.scene, snapshot.checkpoint, snapshot.width, snapshot.height, snapshot.cell_size);
    require(scene.marker == 9, "Shared restore did not restore scenario state");
    base._destroyGUI();
    project.stopHeadless();
    std::cout << "PASS concrete buffered UI, shared sidebar, scheduled action, frame hook and custom capture/state\n";
}
}

int main()
{
    try
    {
        checkIndependentWaves();
        checkExtensions();
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
