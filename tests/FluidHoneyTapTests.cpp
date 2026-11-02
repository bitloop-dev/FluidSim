#include "scenarios/HoneyTap.h"
#include "scenarios/CrashingWave.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

using namespace bl;
using FluidSim::HoneyTap;
namespace fs = std::filesystem;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void checkAtmosphere()
{
    for (const IVec2 size : {IVec2{64, 48}, IVec2{67, 101}, IVec2{33, 19}})
    {
        HoneyTap::Config config;
        config.settings.grid_w = size.x;
        config.settings.grid_h = size.y;
        config.settings.cell_size = 0.5;
        HoneyTap tap(config);
        FluidSim::CrashingWave::Config wave_config;
        wave_config.settings = config.settings;
        FluidSim::CrashingWave wave(wave_config);
        const auto a = tap.initialFluidState();
        const auto b = wave.initialFluidState();

        require(a.u.empty() && a.v.empty(), "HoneyTap must start at rest");
        require(a.liquid_mass.size() == 1 && a.gas_mass.size() == 2, "Unexpected HoneyTap ingredients");

        for (size_t k = 0; k < 2; ++k)
        {
            const auto& gas = tap.fluid.getGasMixture().components[k];
            const auto& reference = wave.fluid.getGasMixture().components[k];

            require(gas.density == reference.density && tap.fluid.getSubstances()[gas.substance].color ==
                                                            wave.fluid.getSubstances()[reference.substance].color,
                "HoneyTap atmosphere properties differ from CrashingWave");
        }

        for (size_t i = 0; i < a.liquid_fraction.size(); ++i)
        {
            require(a.liquid_fraction[i] == 0.0 && a.liquid_mass[0][i] == 0.0, "Initial grid is not all atmosphere");
            require(std::abs((a.gas_mass[0][i] + a.gas_mass[1][i]) / 0.025 - 1.0) < 1e-12,
                "Atmosphere does not fill each cell");
            if (b.liquid_fraction[i] < 1.0)
            {
                for (size_t k = 0; k < 2; ++k)
                {
                    require(std::abs(a.gas_mass[k][i] - b.gas_mass[k][i] / (1.0 - b.liquid_fraction[i])) < 1e-13,
                        "HoneyTap gas pattern differs from CrashingWave");
                }
            }
        }
    }

    std::cout << "PASS initial atmosphere, rest state and matching gas properties/pattern\n";
}

int nozzleWidth(const HoneyTap& scene)
{
    return std::clamp(int(scene.fluid.getWidth() * scene.tap_width_percent / 100.0), 1, scene.fluid.getWidth() - 2);
}

double honeyMass(const fluid::FluidGrid::State& state)
{
    return std::accumulate(state.liquid_mass[0].begin(), state.liquid_mass[0].end(), 0.0);
}

double restVolume(const HoneyTap& scene, const fluid::FluidGrid::State& state)
{
    double volume = honeyMass(state) / scene.fluid.getLiquidMixture().components[0].density;
    for (size_t k = 0; k < state.gas_mass.size(); ++k)
    {
        volume += std::accumulate(state.gas_mass[k].begin(), state.gas_mass[k].end(), 0.0) /
                  scene.fluid.getGasMixture().components[k].density;
    }

    return volume;
}

double ventHoneyFraction(const HoneyTap& scene, const fluid::FluidGrid::State& state)
{
    double maximum = 0.0;
    for (const auto& vent : scene.fluid.getBoundaryFlows().outflows)
    {
        for (int x = vent.opening.first; x < vent.opening.first + vent.opening.count; ++x)
            maximum = std::max(maximum, state.liquid_fraction[x]);
    }

    return maximum;
}

double ventHoneyMass(const HoneyTap& scene, const fluid::FluidGrid::State& state)
{
    double total = 0.0;
    for (const auto& vent : scene.fluid.getBoundaryFlows().outflows)
    {
        for (int x = vent.opening.first; x < vent.opening.first + vent.opening.count; ++x)
            total += state.liquid_mass[0][x];
    }

    return total;
}

void checkBoundaryLayout(const HoneyTap& scene)
{
    const auto& flows = scene.fluid.getBoundaryFlows();
    if (scene.tap_speed == 0.0)
    {
        require(flows.empty(), "Zero speed did not close the inlet and vents");
        return;
    }

    const int width = scene.fluid.getWidth();
    const int count = nozzleWidth(scene);
    const int first = (width - count) / 2;

    require(flows.inflows.size() == 1 && flows.outflows.size() == 2, "Expected one inlet and two vents");

    const auto& inlet = flows.inflows.front();

    require(inlet.opening == fluid::BoundaryOpening{fluid::BoundarySide::Top, first, count} &&
                inlet.phase == fluid::Phase::Liquid && inlet.substance == scene.honey_substance &&
                inlet.speed == scene.tap_speed,
        "Honey inlet side, interval, material or speed is incorrect");

    const auto& left = flows.outflows[0];
    const auto& right = flows.outflows[1];

    require(left.opening == fluid::BoundaryOpening{fluid::BoundarySide::Top, 0, first} &&
                right.opening == fluid::BoundaryOpening{fluid::BoundarySide::Top, first + count, width - first - count},
        "Ceiling vent intervals are incorrect");
    require(left.opening.count >= 1 && right.opening.count >= 1, "Inlet left no vent face on one side");
    require(left.speed == right.speed && std::abs(left.speed * (width - count) - inlet.speed * count) < 1e-10,
        "Ceiling vent volume rate does not balance the honey inlet");
}

void checkBoundaryVelocities(const HoneyTap& scene, const fluid::FluidGrid::State& state)
{
    const int width = scene.fluid.getWidth();
    const int height = scene.fluid.getHeight();
    std::vector<double> ceiling(width, 0.0);

    for (const auto& inlet : scene.fluid.getBoundaryFlows().inflows)
    {
        for (int x = inlet.opening.first; x < inlet.opening.first + inlet.opening.count; ++x)
            ceiling[x] = float(inlet.speed);
    }
    for (const auto& vent : scene.fluid.getBoundaryFlows().outflows)
    {
        for (int x = vent.opening.first; x < vent.opening.first + vent.opening.count; ++x)
            ceiling[x] = -float(vent.speed);
    }

    for (int x = 0; x < width; ++x)
    {
        require(state.v[x] == ceiling[x] && state.v[height * width + x] == 0.0,
            "Incorrect ceiling or floor normal velocity");
    }
    for (int y = 0; y < height; ++y)
        require(state.u[y * (width + 1)] == 0.0 && state.u[y * (width + 1) + width] == 0.0, "Open side wall");
}

void checkNarrowGrids()
{
    for (const int width : {1, 2, 3, 4, 5, 67})
    {
        HoneyTap::Config config;
        config.settings.grid_w = width;
        config.settings.grid_h = 12;
        HoneyTap scene(config);
        if (width < 3)
        {
            bool rejected = false;
            try
            {
                scene.initialFluidState();
            }
            catch (const std::runtime_error& error)
            {
                rejected = std::string(error.what()).find("three columns") != std::string::npos;
            }

            require(rejected, "Unsupported HoneyTap width was not rejected clearly");
            continue;
        }

        for (const double percent : {0.1, 100.0 / 24.0, 100.0})
        {
            scene.tap_width_percent = percent;
            scene.beforeFluidFrame(scene.frame_time);
            checkBoundaryLayout(scene);
        }
    }

    std::cout << "PASS narrow/odd grids and ceiling vent reservation\n";

    // Removing viscosity can enable the shared backend selector; the scene's
    // boundary model must still reject unsupported engines before processing.
    HoneyTap::Config cpu_config;
    cpu_config.settings.fluid_device = fluid::FluidDevice::CPU;
    cpu_config.settings.liquid_viscosities = {0.0};
    HoneyTap cpu_scene(cpu_config);
    bool rejected = false;
    try
    {
        cpu_scene.initialFluidState();
    }
    catch (const std::runtime_error& error)
    {
        rejected = std::string(error.what()).find("uniform WebGPU") != std::string::npos;
    }

    require(rejected, "Unsupported HoneyTap backend was not rejected clearly");
}

struct TapProject : BasicProject
{
    HoneyTap::Config config;
    SceneHandle<HoneyTap> tap;

    void projectPrepare(Layout&) override { tap = create<HoneyTap>("tap", config); }
};

void checkSourceAndRestore()
{
    ImDebugLog log;
    CaptureContext capture;
    TapProject project;
    project.config.settings.grid_w = 64;
    project.config.settings.grid_h = 48;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();
    auto& scene = *project.tap;
    scene.profile_stages = false;
    const auto initial = scene.fluid.download();
    const double density = scene.fluid.getLiquidMixture().components[0].density;

    require(
        scene.fluid.getLiquidMixture().components[0].dynamic_viscosity == project.config.settings.liquid_viscosities[0],
        "Configured honey viscosity missing");

    scene.beforeFluidFrame(scene.frame_time);
    checkBoundaryLayout(scene);
    const auto configured = scene.fluid.download();

    require(configured.liquid_fraction == initial.liquid_fraction && configured.liquid_mass == initial.liquid_mass &&
                configured.gas_mass == initial.gas_mass && configured.pressure == initial.pressure,
        "Configuring the inlet replaced interior material or pressure");
    checkBoundaryVelocities(scene, configured);

    fluid::FluidCheckpoint pending, unchanged;
    scene.fluid.saveCheckpoint(pending);
    scene.beforeFluidFrame(scene.frame_time);
    scene.fluid.saveCheckpoint(unchanged);

    require(pending.data == unchanged.data, "Unchanged boundary configuration changed the checkpoint");

    const double input_rate = density * nozzleWidth(scene) * scene.cell_size * scene.tap_speed;
    bool honey_at_vent = false;
    int analytic_frames = 0;
    double previous_honey = 0.0;

    for (int frame = 1; frame <= 20; ++frame)
    {
        // Observe the first transport before another substep can move honey sideways to a vent.
        scene.frame_time = frame == 1 ? 0.001 : 0.02;
        project.processHeadlessFrame(0.02);

        require(!scene.simulation_failed && scene.frame_statistics.viscosity_iterations > 0,
            "Viscous inlet failed to advance");

        const auto state = scene.fluid.download();
        checkBoundaryVelocities(scene, state);
        honey_at_vent = honey_at_vent || ventHoneyFraction(scene, state) > 0.0 || ventHoneyMass(scene, state) > 0.0;
        const double honey = honeyMass(state);
        const double supplied = input_rate * scene.simulation_time;
        if (!honey_at_vent)
        {
            ++analytic_frames;

            require(std::abs(honey - supplied) < 1e-4 * supplied,
                "Honey injection disagrees with inlet area times normal speed while vents contain only gas");
        }

        require(honey > previous_honey && honey <= supplied * 1.0001,
            "Continuous injection did not increase honey within the analytic input bound");
        previous_honey = honey;

        require(std::abs(restVolume(scene, state) - restVolume(scene, initial)) < 1e-4 * restVolume(scene, initial),
            "Balanced inflow/outflow changed total rest volume");
        if (frame == 1 || frame == 20)
        {
            std::cout << "startup frame=" << frame << " honey=" << honey << " input=" << supplied
                      << " vent_honey=" << ventHoneyFraction(scene, state) << std::endl;
        }
    }

    require(analytic_frames > 0, "Analytic injection gate did not observe a gas-only vent interval");

    scene.tap_width_percent = 12.5;
    scene.tap_speed = 80.0;
    std::cout << "Checking live width/speed change\n";
    project.processHeadlessFrame(0.02);

    require(!scene.simulation_failed, "Live inlet controls failed");
    checkBoundaryLayout(scene);
    checkBoundaryVelocities(scene, scene.fluid.download());

    const auto settings = scene.captureSceneState();
    fluid::FluidCheckpoint saved, uninterrupted, resumed;
    scene.fluid.saveCheckpoint(saved);

    for (int frame = 0; frame < 10; ++frame)
        project.processHeadlessFrame(0.02);

    require(!scene.simulation_failed, "Continuous live inlet failed");
    scene.fluid.saveCheckpoint(uninterrupted);

    std::cout << "Checking checkpoint continuation\n";
    scene.tap_width_percent = 25.0;
    scene.tap_speed = 0.0;
    scene.restoreSceneState(settings, saved, scene.grid_w, scene.grid_h, scene.cell_size);

    require(scene.tap_width_percent == 12.5 && scene.tap_speed == 80.0, "Snapshot lost tap controls");
    checkBoundaryLayout(scene);

    for (int frame = 0; frame < 10; ++frame)
        project.processHeadlessFrame(0.02);

    require(!scene.simulation_failed, "Resumed inlet failed");
    scene.fluid.saveCheckpoint(resumed);

    require(resumed.data == uninterrupted.data, "Honey inlet checkpoint continuation changed");
    std::cout << "Checking closure, restart and resized preset\n";

    const double before_close = honeyMass(scene.fluid.download());
    scene.tap_speed = 0.0;
    project.processHeadlessFrame(0.02);

    require(!scene.simulation_failed, "Closing the source failed");
    checkBoundaryLayout(scene);
    const auto closed = scene.fluid.download();
    checkBoundaryVelocities(scene, closed);

    require(std::abs(honeyMass(closed) - before_close) < 1e-5 * before_close, "Zero speed still supplied honey");

    scene.tap_speed = 80.0;
    scene.restartSimulation();

    require(scene.fluid.download().gas_mass == initial.gas_mass && scene.simulation_time == 0.0,
        "Restart did not restore atmosphere");
    require(scene.tap_width_percent == 12.5 && scene.tap_speed == 80.0, "Restart reset tap controls");
    project.processHeadlessFrame(0.02);

    require(!scene.simulation_failed && honeyMass(scene.fluid.download()) > 0.0, "Inlet did not resume after restart");
    checkBoundaryLayout(scene);

    auto preset = HoneyTap::presets.front().settings;
    preset.grid_w = 67;
    preset.grid_h = 49;
    scene.applyPreset(preset);

    require(scene.tap_width_percent == 12.5 && scene.tap_speed == 80.0, "Physics preset reset independent tap controls");
    project.processHeadlessFrame(0.02);

    require(
        !scene.simulation_failed && honeyMass(scene.fluid.download()) > 0.0, "New grid lost its boundary configuration");
    checkBoundaryLayout(scene);
    checkBoundaryVelocities(scene, scene.fluid.download());

    // Earlier scene saves have neither tap controls nor an engine boundary configuration.
    std::cout << "Checking old source state and sidebar\n";
    auto old_settings = scene.captureSceneState();
    old_settings["scenario_state"].erase("tap_width_percent");
    old_settings["scenario_state"].erase("tap_speed");
    scene.fluid.setBoundaryFlows({});
    scene.fluid.saveCheckpoint(saved);
    scene.restoreSceneState(old_settings, saved, scene.grid_w, scene.grid_h, scene.cell_size);

    require(scene.tap_width_percent == 100.0 / 24.0 && scene.tap_speed == 20.0,
        "Old snapshot did not use historical tap defaults");
    project.processHeadlessFrame(0.02);

    require(!scene.simulation_failed, "Old closed source state did not adopt boundary flow");
    checkBoundaryLayout(scene);

    auto& base = static_cast<SceneBase&>(scene);
    base._initGUI();

    require(dynamic_cast<HoneyTap::UI*>(scene.ui) != nullptr, "HoneyTap controls UI was not constructed");

    PlatformManager platform(nullptr);
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().DisplaySize = {800, 600};
    ImGui::GetIO().Fonts->Build();
    ImGui::NewFrame();
    ImGui::Begin("HoneyTap controls");
    scene.ui->sidebar();
    ImGui::End();
    ImGui::Render();
    ImGui::DestroyContext();
    base._destroyGUI();
    project.stopHeadless();
    std::cout
        << "PASS continuous boundary injection, balanced vents, live/zero controls, exact resume, restart, presets and UI\n";
}

// Optional default-size behavioral gate; output directory alone runs 32 simulated seconds.
void evolve(const fs::path& output, int frames, double speed = 20.0, double width_percent = 100.0 / 24.0)
{
    fs::create_directories(output);

    require(SDL_Init(SDL_INIT_VIDEO), "SDL initialization failed");
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

    auto* window = SDL_CreateWindow("HoneyTap checks", 32, 32, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    require(window != nullptr, "GL window failed");

    auto context = SDL_GL_CreateContext(window);
    require(context != nullptr && gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress), "GL context failed");

    ImDebugLog log;
    CaptureContext capture;
    TapProject project;
    project.config.tap_speed = speed;
    project.config.tap_width_percent = width_percent;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();
    auto& scene = *project.tap;
    scene.profile_stages = false;

    const int w = scene.grid_w;
    const int h = scene.grid_h;
    const double area = scene.cell_size * scene.cell_size;
    const double density = scene.fluid.getLiquidMixture().components[0].density;
    const double honey_cell_mass = density * area;
    const auto& gases = scene.fluid.getGasMixture().components;
    const double initial_volume = restVolume(scene, scene.fluid.download());
    const double input_rate = density * nozzleWidth(scene) * scene.cell_size * speed;
    std::ofstream csv(output / "health.csv");

    csv << "frame,time,rest_volume_error,phase_error,divergence,honey_mass,honey_input,honey_input_minus_inventory,"
           "analytic_honey_error,vent_honey_fraction,vent_honey_mass,ever_honey_at_vent,bottom_liquid,honey_lowest_row,max_speed,"
           "pressure_iterations,viscosity_iterations,physical_steps,hydrogen_mass,helium_mass\n"
        << std::setprecision(17);
    double final_bottom = 0.0;
    int budget_failures = 0;
    bool honey_at_vent = false;

    for (int frame = 1; frame <= frames; ++frame)
    {
        project.processHeadlessFrame(scene.frame_time);
        if (scene.simulation_failed)
        {
            std::cerr << "HoneyTap stopped at frame " << frame << ", sim time " << scene.simulation_time << '\n';
            std::lock_guard lock(log.log_mutex);
            for (const auto& line : log.logLines)
                std::cerr << line.getFullText() << '\n';
        }

        require(!scene.simulation_failed, "HoneyTap solver failed");

        const auto state = scene.fluid.download();
        const double honey = honeyMass(state);
        const double input = input_rate * scene.simulation_time;
        const double vent_honey = ventHoneyFraction(scene, state);
        const double vent_mass = ventHoneyMass(scene, state);

        honey_at_vent = honey_at_vent || vent_honey > 0.0 || vent_mass > 0.0;
        const double volume_error = std::abs(restVolume(scene, state) - initial_volume) / initial_volume;

        // Once honey reaches a vent, inventory includes real outflow. The difference from
        // supplied mass is reported, not mistaken for closed-box numerical mass loss.
        const double analytic_error =
            honey_at_vent ? std::numeric_limits<double>::quiet_NaN() : std::abs(honey - input) / std::max(1.0, input);
        double phase_error = 0.0;
        double divergence = 0.0;
        double bottom = 0.0;
        double maximum_speed = 0.0;
        int lowest_row = -1;

        for (const auto* field : {&state.liquid_mass[0], &state.gas_mass[0], &state.gas_mass[1]})
        {
            for (const double value : *field)
                require(std::isfinite(value) && value >= -1e-5, "Invalid component mass");
        }
        for (const auto* field : {&state.u, &state.v, &state.pressure})
        {
            for (const double value : *field)
                require(std::isfinite(value), "Nonfinite velocity/pressure");
        }

        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const int i = y * w + x;
                const double f = state.liquid_fraction[i];

                require(std::isfinite(f) && f >= -1e-6 && f <= 1.000001, "Invalid fraction");
                phase_error = std::max({phase_error, std::abs(state.liquid_mass[0][i] / honey_cell_mass - f),
                    std::abs(
                        (state.gas_mass[0][i] / gases[0].density + state.gas_mass[1][i] / gases[1].density) / area -
                        (1.0 - f))});
                divergence = std::max(divergence, std::abs(state.u[y * (w + 1) + x + 1] - state.u[y * (w + 1) + x] +
                                                           state.v[(y + 1) * w + x] - state.v[y * w + x]) /
                                                      scene.cell_size);
                maximum_speed = std::max(maximum_speed, std::hypot(state.u[y * (w + 1) + x], state.v[y * w + x]));
                if (f > 0.01)
                    lowest_row = y;
                if (y == h - 1)
                    bottom += f;
            }
        }

        checkBoundaryVelocities(scene, state);

        // Continue after a quality miss to distinguish drift from a transient; fail at the end.
        if (volume_error > 1e-4 || analytic_error > 1e-4 || honey > input + 1e-4 * std::max(1.0, input) ||
            phase_error > 0.002 || divergence > scene.projection_tolerance + 1e-4)
        {
            if (budget_failures++ == 0)
            {
                std::cout << "First quality budget miss: frame=" << frame << " rest_volume=" << volume_error
                          << " analytic_honey=" << analytic_error << " phase=" << phase_error
                          << " divergence=" << divergence << std::endl;
            }
        }

        csv << frame << ',' << scene.simulation_time << ',' << volume_error << ',' << phase_error << ',' << divergence
            << ',' << honey << ',' << input << ',' << input - honey << ',' << analytic_error << ',' << vent_honey << ','
            << vent_mass << ',' << honey_at_vent << ',' << bottom << ',' << lowest_row << ',' << maximum_speed << ','
            << scene.frame_statistics.pressure_iterations << ',' << scene.frame_statistics.viscosity_iterations << ','
            << scene.frame_statistics.substeps << ','
            << std::accumulate(state.gas_mass[0].begin(), state.gas_mass[0].end(), 0.0) << ','
            << std::accumulate(state.gas_mass[1].begin(), state.gas_mass[1].end(), 0.0) << '\n';

        // Startup, the old 4-second stall, the old 10.48-second failure, and developed deposition.
        if (frame == 1 || frame == 50 || frame == 200 || frame == 524 || frame == 800 || frame == 1200 ||
            frame == frames)
        {
            CaptureFrame image;
            scene.captureFluidImage(image);
            bytebuf webp;

            require(webp_snapshot_encode_rgba(
                        image.frame.data.data(), image.resolution.x, image.resolution.y, 100, 6, true, 100, 100, webp),
                "Image encoding failed");

            std::ofstream file(output / ("frame-" + std::to_string(frame) + ".webp"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(webp.data()), std::streamsize(webp.size()));
            std::cout << "frame=" << frame << " time=" << scene.simulation_time << " honey=" << honey
                      << " bottom=" << bottom << " lowest_row=" << lowest_row << " vent_honey=" << vent_honey
                      << std::endl;
            csv.flush();
        }

        final_bottom = bottom;
    }

    require(final_bottom > 1.0, "Run did not reach floor accumulation; extend the behavior horizon");

    project.stopHeadless();
    SDL_GL_DestroyContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    require(budget_failures == 0,
        "HoneyTap boundary-flow conservation/pressure budget failed (complete health trace retained)");
    std::cout << "PASS HoneyTap continuous injection, floor accumulation and per-frame health over " << frames
              << " frames\n";
}
}

int main(int argc, char** argv)
{
    try
    {
        std::cout << std::unitbuf;
        if (argc == 2 || argc == 3 || argc == 5)
        {
            evolve(argv[1], argc == 2 ? 1600 : std::stoi(argv[2]), argc == 5 ? std::stod(argv[3]) : 20.0,
                argc == 5 ? std::stod(argv[4]) : 100.0 / 24.0);
        }
        else
        {
            checkAtmosphere();
            checkNarrowGrids();
            checkSourceAndRestore();
        }

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
