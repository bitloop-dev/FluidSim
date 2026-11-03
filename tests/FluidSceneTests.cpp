#include "scenarios/CrashingWave.h"
#include "scenarios/ElasticTest.h"
#include <array>
#include <iostream>

using namespace bl;
using FluidSim::CrashingWave;
using fluid::FluidGrid;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

bool sameState(const FluidGrid::State& a, const FluidGrid::State& b)
{
    return a.liquid_fraction == b.liquid_fraction && a.liquid_mass == b.liquid_mass && a.gas_mass == b.gas_mass &&
           a.u == b.u && a.v == b.v && a.pressure == b.pressure && a.elastic_log_xx == b.elastic_log_xx &&
           a.elastic_log_xy == b.elastic_log_xy && a.elastic_log_yy == b.elastic_log_yy &&
           a.elastic_fraction == b.elastic_fraction;
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
            config.settings.frame_time = 0.01 + 0.005 * i;
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
        controls[i]->configureMaterials(
            scene.fluid.getSubstances(), scene.fluid.getLiquidMixture(), scene.fluid.getGasMixture());
        controls[i]->setGravityY(scene.gravity_y);
        controls[i]->upload(scene.fluid.download());
    }

    for (int frame = 0; frame < 12; ++frame)
    {
        project.processHeadlessFrame(0.02);
        for (int i = 0; i < 4; ++i)
        {
            const auto& scene = *project.waves[i];
            const int count = int(std::ceil(scene.frame_time * scene.steps_per_second));
            for (int step = 0; step < count; ++step)
            {
                require(
                    controls[i]
                        ->step(scene.frame_time / count, scene.projection_max_iterations, scene.projection_tolerance)
                        .converged,
                    "Control projection failed");
            }

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
    require(
        sameState(project.waves[1]->fluid.download(), untouched) && project.waves[1]->simulation_time == untouched_time,
        "Restart changed another scene");
    project.processHeadlessFrame(0.0);

    require(sameState(project.waves[1]->fluid.download(), untouched), "Zero dt advanced a scene");

    auto& recovery = *project.waves[0];
    bool unsupported = false;
    try
    {
        recovery.setPressureRecoveryFrames(5);
    }
    catch (const std::invalid_argument&)
    {
        unsupported = true;
    }

    require(unsupported && recovery.pressure_recovery_frames == 0, "CPU silently accepted pressure recovery");

    recovery.fluid_device = fluid::FluidDevice::GPU;
    recovery.restartSimulation();
    const auto initial = recovery.fluid.download();
    project.processHeadlessFrame(0.02);
    recovery.setPressureRecoveryFrames(5);
    const auto relaxed_initial = recovery.fluid.download();
    auto expected_initial = initial;

    // Relaxed geometry is derived from packed float masses, with one additional rounding step.
    for (size_t i = 0; i < initial.liquid_fraction.size(); ++i)
    {
        require(std::abs(relaxed_initial.liquid_fraction[i] - initial.liquid_fraction[i]) < 2e-7,
            "Recovery restart changed initial geometry");
    }

    expected_initial.liquid_fraction = relaxed_initial.liquid_fraction;

    require(recovery.simulation_time == 0.0 && !recovery.simulation_failed &&
                recovery.fluid.getPressureRecoveryTime() == 5 * recovery.frame_time &&
                sameState(expected_initial, relaxed_initial),
        "Entering recovery did not restart cleanly");

    for (int i = 0; i < 6; ++i)
        project.processHeadlessFrame(0.02);

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
    require(recovery.simulation_time == 0.0 && sameState(initial, recovery.fluid.download()) &&
                recovery.fluid.getPressureRecoveryTime() == 0.0,
        "Returning to accurate mode retained compressed state");

    recovery.restoreSceneState(scene_state, saved, recovery.grid_w, recovery.grid_h, recovery.cell_size);
    require(recovery.pressure_recovery_frames == 10 && recovery.simulation_time == elapsed &&
                sameState(live, recovery.fluid.download()),
        "Scene restore lost recovery mode or state");
    project.processHeadlessFrame(0.02);

    require(!recovery.simulation_failed, "Restored recovery scene failed");
    std::cout << "PASS recovery mode transitions, time conversion, unsupported backend and scene restoration\n";
    project.stopHeadless();
    std::cout << "PASS four independent waves, zero/two viewport scheduling, restart isolation and direct-step parity"
              << std::endl;
}

// Test-only scenario exercises extension points without adding another app scenario.
struct ExtensionScene : FluidSim::FluidScene<ExtensionScene>
{
    using Base = FluidSim::FluidScene<ExtensionScene>;
    static constexpr const char* scenario_id = "test-extension";
    int marker = 1;
    int frames = 0;

    explicit ExtensionScene(const Config& config) : Base({.grid_w = 16, .grid_h = 12}, config)
    {
        fluid::SubstanceCatalogue substances;
        const auto liquid = substances.add({"Liquid", {0, 0, 1}});
        const auto gas = substances.add({"Gas", {0, 0, 0}});
        fluid.configureMaterials(std::move(substances), {{{liquid, 1.0}}}, {{{gas, 0.1}}});
    }

    FluidGrid::State initialFluidState() const override
    {
        FluidGrid::State state;
        state.liquid_fraction.assign(grid_w * grid_h, 0.0);
        state.liquid_mass = {std::vector<double>(grid_w * grid_h, 0.0)};
        state.gas_mass = {std::vector<double>(grid_w * grid_h, 0.1)};

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
            projection_tolerance = 0.25;
        }

        void schedule()
        {
            bl_schedule([](ExtensionScene& scene)
            {
                scene.marker = 9;
            });
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
    require(scene.marker == 1 && scene.projection_tolerance == 0.1, "UI bypassed shadow buffering");

    base.updateLiveBuffers();
    require(scene.marker == 7 && scene.projection_tolerance == 0.25, "Inherited/derived UI edits were not applied");

    ui->schedule();
    require(scene.marker == 7, "Scheduled action ran on the UI owner");

    project.processHeadlessFrame(0.02);
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

namespace
{
void checkViscosityControls()
{
    ImDebugLog log;
    CaptureContext capture;
    WaveProject project;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();

    auto& scene = *project.waves[0];
    scene.fluid_device = fluid::FluidDevice::GPU;
    scene.restartSimulation();

    const auto initial = scene.fluid.download();
    scene.setComponentViscosity(fluid::Phase::Liquid, 1, 1000.0);

    require(sameState(initial, scene.fluid.download()) && scene.simulation_time == 0.0 &&
                scene.fluid.getLiquidMixture().components[1].dynamic_viscosity == 1000.0,
        "Viscosity edit reset or changed physical state");

    for (int i = 0; i < 4; ++i)
        project.processHeadlessFrame(0.02);

    require(!scene.simulation_failed && scene.frame_statistics.viscosity_iterations > 0,
        "Viscosity control did not reach complete physics steps");

    const auto settings = scene.captureSceneState();
    fluid::FluidCheckpoint checkpoint;
    scene.fluid.saveCheckpoint(checkpoint);
    scene.setComponentViscosity(fluid::Phase::Liquid, 1, 0.0);
    scene.restoreSceneState(settings, checkpoint, scene.grid_w, scene.grid_h, scene.cell_size);

    require(scene.liquid_viscosities[1] == 1000.0 && scene.liquid_viscosity_names[1] == "Glycerol",
        "Restored viscosity controls disagree with engine materials");

    scene.restartSimulation();
    require(scene.liquid_viscosities[1] == 1000.0, "Restart lost live viscosity");

    FluidSim::FluidSceneSettings preset{.grid_w = 32, .grid_h = 24, .gravity_y = 10.0};
    auto expanded = preset;
    expanded.liquid_viscosities = {0.0, 0.0};
    expanded.gas_viscosities = {0.0, 0.0};

    require(expanded == preset, "Expanded zero viscosities no longer identify the default preset");

    expanded.liquid_viscosities[1] = 100.0;
    require(!(expanded == preset), "Viscosity change did not identify custom settings");
    scene.applyPreset(expanded);

    require(scene.liquid_viscosities[1] == 100.0 && scene.simulation_time == 0.0,
        "Preset did not apply viscosity and restart");

    preset.fluid_device = fluid::FluidDevice::CPU;
    scene.applyPreset(preset);

    require(!scene.fluid.hasViscosity() && scene.fluid_device == fluid::FluidDevice::CPU,
        "Inviscid preset could not leave a viscous backend");

    project.stopHeadless();
    std::cout << "PASS viscosity live controls, restart, checkpoint and preset behavior\n";
}
}

namespace
{
struct ElasticProject : BasicProject
{
    SceneHandle<FluidSim::ElasticTest> source;

    void projectPrepare(Layout&) override
    {
        FluidSim::ElasticTest::Config config;
        config.settings.grid_w = 32;
        config.settings.grid_h = 32;
        source = create<FluidSim::ElasticTest>(config);
    }
};

void checkElasticScene()
{
    ImDebugLog log;
    CaptureContext capture;
    ElasticProject project;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();
    auto& scene = *project.source;
    require(scene.fluid.getDevice() == fluid::FluidDevice::GPU && scene.elastic_material,
        "ElasticTest did not select uniform WebGPU");
    const auto initial = scene.fluid.download();
    project.processHeadlessFrame(0.02);
    const auto live = scene.fluid.download();
    const double live_time = scene.simulation_time;
    scene.setElasticStiffness(12000.0);
    require(scene.elastic_stiffness == 12000.0 &&
                scene.fluid.getLiquidMixture().components[0].shear_modulus == 12000.0 &&
                scene.simulation_time == live_time && sameState(live, scene.fluid.download()),
        "Stiffness control restarted the scene or lost deformation");
    bool rejected = false;
    try
    {
        scene.setPressureRecoveryFrames(1);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected && sameState(live, scene.fluid.download()), "Rejected elastic recovery discarded live history");

    const auto scene_state = scene.captureSceneState();
    fluid::FluidCheckpoint checkpoint;
    require(scene.fluid.saveCheckpoint(checkpoint), "Elastic scene checkpoint unsupported");
    project.processHeadlessFrame(0.02);
    const auto continued = scene.fluid.download();
    scene.restoreSceneState(scene_state, checkpoint, scene.grid_w, scene.grid_h, scene.cell_size);
    require(scene.elastic_stiffness == 12000.0, "Snapshot did not restore stiffness control");
    project.processHeadlessFrame(0.02);
    require(!scene.simulation_failed && sameState(continued, scene.fluid.download()),
        "Elastic scene checkpoint changed continuation");
    scene.restartSimulation();
    require(sameState(initial, scene.fluid.download()), "Elastic scene restart lost reference configuration");
    require(scene.elastic_stiffness == 12000.0, "Restart lost edited stiffness");

    require(
        !scene.implicit_elasticity && !scene.fluid.getImplicitElasticity(), "Elastic scene default is not explicit");
    scene.setImplicitElasticity(true);
    require(sameState(initial, scene.fluid.download()), "Scene implicit toggle reset deformation");
    project.processHeadlessFrame(0.02);
    require(!scene.simulation_failed && scene.frame_statistics.elasticity_iterations > 0,
        "Scene implicit setting did not reach the coupled solver");

    const auto implicit_scene = scene.captureSceneState();
    fluid::FluidCheckpoint implicit_checkpoint;
    scene.fluid.saveCheckpoint(implicit_checkpoint);
    require(implicit_scene.at("version") == 5 && implicit_scene.at("implicit_elasticity") == true,
        "Scene JSON lost implicit mode");
    project.processHeadlessFrame(0.02);
    const auto implicit_continued = scene.fluid.download();
    scene.setImplicitElasticity(false);
    scene.restoreSceneState(implicit_scene, implicit_checkpoint, scene.grid_w, scene.grid_h, scene.cell_size);
    require(scene.implicit_elasticity && scene.fluid.getImplicitElasticity(),
        "Restored implicit control disagrees with engine");
    project.processHeadlessFrame(0.02);
    require(!scene.simulation_failed && sameState(implicit_continued, scene.fluid.download()),
        "Implicit scene checkpoint changed continuation");
    scene.restartSimulation();
    require(scene.implicit_elasticity && sameState(initial, scene.fluid.download()),
        "Restart lost implicit mode or initial deformation");

    auto preset = FluidSim::ElasticTest::presets.front().settings;
    preset.grid_w = scene.grid_w;
    preset.grid_h = scene.grid_h;
    auto implicit_preset = preset;
    implicit_preset.implicit_elasticity = true;
    require(!(implicit_preset == preset), "Implicit toggle did not mark preset Custom");
    auto stiffness_preset = preset;
    stiffness_preset.elastic_stiffness = 12000.0;
    require(!(stiffness_preset == preset), "Stiffness slider did not mark preset Custom");
    scene.applyPreset(preset);
    require(scene.elastic_stiffness == 50000.0, "Preset did not restore stiffness");
    require(!scene.implicit_elasticity && !scene.fluid.getImplicitElasticity() &&
                sameState(initial, scene.fluid.download()),
        "Explicit preset failed to restore default mode");
    scene.applyPreset(implicit_preset);
    require(scene.implicit_elasticity && scene.fluid.getImplicitElasticity(), "Preset lost implicit mode");
    scene.restoreSceneState(scene_state, checkpoint, scene.grid_w, scene.grid_h, scene.cell_size);
    require(!scene.implicit_elasticity && !scene.fluid.getImplicitElasticity(),
        "Old explicit scene inherited implicit mode");

    project.stopHeadless();
    std::cout << "PASS GPU ElasticTest modes, recovery rejection, checkpoint continuation, presets and restart\n";
}

void checkElasticPoolScene()
{
    using FluidSim::ElasticTest;

    ImDebugLog log;
    CaptureContext capture;
    ElasticProject project;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();
    auto& scene = *project.source;
    const auto old_scene = scene.captureSceneState();
    fluid::FluidCheckpoint old_checkpoint;
    require(scene.fluid.saveCheckpoint(old_checkpoint), "Original elastic checkpoint unsupported");
    const auto old_initial = scene.fluid.download();

    const auto& pool = ElasticTest::presets.back();
    require(pool.settings == ElasticTest::presets[1].settings &&
                pool.material_setup != ElasticTest::presets[1].material_setup,
        "Pool and High Gravity presets no longer distinguish material setup from settings");
    scene.applyPreset(pool);
    require(!scene.simulation_failed && scene.grid_w == 720 && scene.grid_h == 480 && scene.elastic_with_liquid &&
                scene.material_setup == pool.material_setup,
        "Pool preset did not configure the requested grid and materials");

    const auto initial = scene.fluid.download();
    const auto& liquid = scene.fluid.getLiquidMixture().components;
    require(liquid.size() == 2 && liquid[0].shear_modulus == 50000.0 && liquid[1].shear_modulus == 0.0 &&
                liquid[0].dynamic_viscosity == 1000.0 && liquid[1].dynamic_viscosity == 0.0 &&
                liquid[0].substance != liquid[1].substance && initial.elastic_fraction.size() == 720 * 480,
        "Pool is not distinct zero-stiffness water with independent elastic occupancy");

    double water_volume = 0.0;
    double elastic_volume = 0.0;
    for (int y = 0; y < 480; ++y)
    {
        for (int x = 0; x < 720; ++x)
        {
            const size_t i = size_t(y) * 720 + x;
            const double water = initial.liquid_mass[1][i] / liquid[1].density;
            require(water == (y >= 360 ? 1.0 : 0.0), "Pool does not occupy exactly the bottom 120 rows");
            require(std::abs(initial.liquid_fraction[i] - initial.elastic_fraction[i] - water) < 1e-6,
                "Pool condensed occupancy does not partition into water and elastic material");
            require(std::abs(initial.liquid_mass[0][i] / liquid[0].density - initial.elastic_fraction[i]) < 1e-6,
                "Block mass does not match elastic occupancy");
            if (water > 0.0)
            {
                require(initial.elastic_fraction[i] == 0.0 && initial.elastic_log_xx[i] == 0.0 &&
                            initial.elastic_log_xy[i] == 0.0 && initial.elastic_log_yy[i] == 0.0 &&
                            initial.gas_mass[0][i] == 0.0,
                    "Initial pool contains elastic history or atmosphere");
            }

            water_volume += water;
            elastic_volume += initial.elastic_fraction[i];
        }
    }
    require(
        water_volume == 86400.0 && std::abs(elastic_volume - 144.0 * 144.0) < 0.001, "Pool or block volume changed");
    const size_t water_pixel = (size_t(400) * 720 + 360) * 4;
    const size_t block_pixel = (size_t(240) * 720 + 360) * 4;
    require(scene.display_pixels[water_pixel + 2] > scene.display_pixels[water_pixel + 1] &&
                scene.display_pixels[block_pixel + 1] > scene.display_pixels[block_pixel + 2],
        "Water and elastic block do not render with distinct material colours");

    // The smaller geometry keeps continuation/restart checks cheap; behavior horizons live in the engine gates.
    auto small_pool = pool;
    small_pool.settings.grid_w = 48;
    small_pool.settings.grid_h = 32;
    scene.applyPreset(small_pool);
    const auto small_initial = scene.fluid.download();
    project.processHeadlessFrame(0.02);
    require(!scene.simulation_failed, "Small pool scene failed its first frame");
    const auto live = scene.fluid.download();
    scene.setElasticStiffness(12000.0);
    require(
        sameState(live, scene.fluid.download()) && scene.fluid.getLiquidMixture().components[1].shear_modulus == 0.0,
        "Pool stiffness edit changed state or stiffened water");

    fluid::FluidSnapshot saved;
    saved.width = scene.grid_w;
    saved.height = scene.grid_h;
    saved.cell_size = scene.cell_size;
    saved.scene = scene.captureSceneState();
    require(scene.fluid.saveCheckpoint(saved.checkpoint) && saved.checkpoint.metadata.at("version") == 12,
        "Pool did not save the independent-occupancy checkpoint");
    const auto snapshot = fluid::decodeFluidSnapshot(fluid::encodeFluidSnapshot(saved));
    project.processHeadlessFrame(0.02);
    const auto continued = scene.fluid.download();
    scene.restoreSceneState(snapshot.scene, snapshot.checkpoint, snapshot.width, snapshot.height, snapshot.cell_size);
    require(
        scene.elastic_with_liquid && scene.material_setup == pool.material_setup && scene.elastic_stiffness == 12000.0,
        "Restored pool lost material setup or live controls");
    project.processHeadlessFrame(0.02);
    require(
        !scene.simulation_failed && sameState(continued, scene.fluid.download()), "Pool snapshot changed continuation");

    bool rejected = false;
    auto broken = snapshot.scene;
    broken["scenario_state"]["water_substance"] = broken["scenario_state"]["substances"][0];
    try
    {
        scene.restoreSceneState(broken, snapshot.checkpoint, snapshot.width, snapshot.height, snapshot.cell_size);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    require(rejected && sameState(continued, scene.fluid.download()), "Invalid pool snapshot changed the active state");

    scene.restartSimulation();
    require(sameState(small_initial, scene.fluid.download()), "Restored pool did not restart its initial geometry");
    rejected = false;
    try
    {
        scene.setSurfaceTension(1.0);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected && scene.surface_tension == 0.0 && sameState(small_initial, scene.fluid.download()),
        "Unsupported pool surface tension changed active state");

    auto cpu = small_pool;
    cpu.settings.fluid_device = fluid::FluidDevice::CPU;
    cpu.settings.liquid_viscosities = {0.0, 0.0};
    rejected = false;
    try
    {
        scene.applyPreset(cpu);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(
        rejected && scene.fluid_device == fluid::FluidDevice::GPU && sameState(small_initial, scene.fluid.download()),
        "Unsupported pool CPU preset changed active state");

    // Ingredient order is not an elasticity API contract.
    auto reordered = scene.fluid.getLiquidMixture();
    std::swap(reordered.components[0], reordered.components[1]);
    scene.fluid.destroy();
    scene.fluid.configureMaterials(scene.fluid.getSubstances(), reordered, scene.fluid.getGasMixture());
    std::swap(scene.liquid_viscosities[0], scene.liquid_viscosities[1]);
    scene.restartSimulation();
    require(!scene.simulation_failed && scene.elastic_stiffness == 12000.0,
        "Stiffness display assumed the first ingredient is elastic");
    scene.setElasticStiffness(15000.0);
    require(scene.fluid.getLiquidMixture().components[0].shear_modulus == 0.0 &&
                scene.fluid.getLiquidMixture().components[1].shear_modulus == 15000.0,
        "Reordered scene stiffness edit changed the water ingredient");

    // Old scene JSON has no water field and retains its original single-material path.
    scene.restoreSceneState(old_scene, old_checkpoint, 32, 32, 1.0);
    scene.restartSimulation();
    require(!scene.elastic_with_liquid && scene.material_setup == ElasticTest::blockMaterials &&
                sameState(old_initial, scene.fluid.download()),
        "Old elastic snapshot inherited the pool on restart");
    scene.applyPreset(small_pool);
    for (size_t index = 0; index < 2; ++index)
    {
        auto block = ElasticTest::presets[index];
        block.settings.grid_w = 32;
        block.settings.grid_h = 32;
        scene.applyPreset(block);
        require(!scene.elastic_with_liquid && scene.fluid.getLiquidMixture().components.size() == 1 &&
                    scene.fluid.download().elastic_fraction.empty(),
            "Original block preset retained pool materials or selected the new transport path");
    }

    project.stopHeadless();
    std::cout
        << "PASS 120-row pool geometry, colours, live stiffness, snapshot continuation, old-file restart and preset transitions\n";
}
}

int main()
{
    try
    {
        checkIndependentWaves();
        checkExtensions();
        checkViscosityControls();
        checkElasticScene();
        checkElasticPoolScene();
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
