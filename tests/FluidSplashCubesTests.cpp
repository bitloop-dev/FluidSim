#include "scenarios/SplashCubes.h"
#include "scenarios/CrashingWave.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>

using namespace bl;
using FluidSim::SplashCubes;
namespace fs = std::filesystem;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void checkInitialState()
{
    for (const IVec2 size : {IVec2{80, 60}, IVec2{83, 57}, IVec2{40, 80}})
    {
        SplashCubes::Config config;
        config.settings.grid_w = size.x;
        config.settings.grid_h = size.y;
        config.settings.cell_size = .5;
        config.settings.fluid_device = fluid::FluidDevice::CPU;
        SplashCubes scene(config);
        const auto state = scene.initialFluidState();
        const auto& substances = scene.fluid.getSubstances();
        require(substances.size() == 4 && substances.elementCount() == 4, "Unexpected chemical catalogue size");
        const auto checkComposition = [&](fluid::SubstanceId id, const char* name, double molar_mass,
            std::initializer_list<std::pair<const char*, unsigned>> expected)
        {
            const auto& substance = substances[id];
            require(substance.name == name && substance.composition.size() == expected.size(), "Incorrect substance identity");
            size_t i = 0;
            for (const auto& [symbol, count] : expected)
            {
                const auto& part = substance.composition[i++];
                require(substances.getElement(part.element).symbol == symbol && part.count == count,
                    "Incorrect elemental composition");
            }
            require(std::abs(substances.molarMass(id) - molar_mass) < 1e-10, "Incorrect molar mass");
        };
        checkComposition(scene.hydrogen_substance, "Hydrogen", 2.016, {{"H", 2}});
        checkComposition(scene.helium_substance, "Helium", 4.0026, {{"He", 1}});
        checkComposition(scene.water_substance, "Water", 18.015, {{"H", 2}, {"O", 1}});
        checkComposition(scene.glycerol_substance, "Glycerol", 92.094, {{"C", 3}, {"H", 8}, {"O", 3}});
        require(substances[scene.hydrogen_substance].color == std::array<float, 3>{0, 0, 0} &&
            substances[scene.helium_substance].color == std::array<float, 3>{1, 1, 1}, "Atmosphere is not black/white");

        FluidSim::CrashingWave::Config wave_config;
        wave_config.settings = config.settings;
        FluidSim::CrashingWave wave(wave_config);
        const auto wave_state = wave.initialFluidState();
        require(substances[scene.water_substance].color == wave.fluid.getSubstances()[wave.water_substance].color &&
            substances[scene.glycerol_substance].color == wave.fluid.getSubstances()[wave.glycerol_substance].color,
            "Existing liquid colours changed");
        require(state.u.empty() && state.v.empty(), "Cubes should start at rest");

        const double side = .3 * std::min(size.x, size.y);
        const double expected_volume = side * side * .25;
        const auto& liquid = scene.fluid.getLiquidMixture().components;
        const auto& gas_components = scene.fluid.getGasMixture().components;
        require(std::abs(liquid[0].density - 9.9820675) < 1e-12 &&
            std::abs(liquid[1].density - 12.613) < 1e-12 &&
            std::abs(gas_components[0].density - .1) < 1e-12 &&
            std::abs(gas_components[1].density - .1) < 1e-12,
            "Scenario liquid reference or scaled atmosphere densities are incorrect");
        for (size_t k = 0; k < 2; ++k)
        {
            const auto& component = scene.fluid.getLiquidMixture().components[k];
            const double mass = std::accumulate(state.liquid_mass[k].begin(), state.liquid_mass[k].end(), 0.);
            require(std::abs(mass / component.density - expected_volume) < 1e-9, "Incorrect square volume");
        }
        for (int y = 0; y < size.y; ++y)
        for (int x = 0; x < size.x; ++x)
        {
            const int i = y * size.x + x;
            const double blue = state.liquid_mass[0][i] / (liquid[0].density * .25);
            const double red = state.liquid_mass[1][i] / (liquid[1].density * .25);
            require(std::abs(blue + red - state.liquid_fraction[i]) < 1e-12 && blue * red == 0.,
                "Initial liquids overlap or disagree with coverage");
            if (red > 0.) require(x + 1 < size.x * .5, "Red square is not on the left");
            if (blue > 0.) require(x > size.x * .5, "Blue square is not on the right");
            const double top = .8 * (size.y - side);
            if (blue + red > 0.) require(y + 1 > top && y < top + side, "Squares changed their vertical placement");
            const double gas = state.gas_mass[0][i] / gas_components[0].density +
                state.gas_mass[1][i] / gas_components[1].density;
            require(std::abs(gas / .25 + blue + red - 1.) < 1e-12, "Gas does not fill remaining volume");
            const double wave_gas = wave_state.gas_mass[0][i] / wave.fluid.getGasMixture().components[0].density +
                wave_state.gas_mass[1][i] / wave.fluid.getGasMixture().components[1].density;
            if (gas > 0. && wave_gas > 0.)
                require(std::abs(state.gas_mass[1][i] / gas_components[1].density / gas -
                    wave_state.gas_mass[1][i] / wave.fluid.getGasMixture().components[1].density / wave_gas) < 1e-12,
                    "Perlin atmosphere differs from CrashingWave");
        }
    }
    std::cout << "PASS chemical identities/compositions/densities, square volumes/placement, rest velocities, colours and Perlin atmosphere\n";
}

template<class Scenario>
struct MaterialProject : BasicProject
{
    SceneHandle<Scenario> source;
    void projectPrepare(Layout&) override { source = create<Scenario>(); }
};

template<class Scenario>
void evolve(const fs::path& output, int frames)
{
    fs::create_directories(output);
    require(SDL_Init(SDL_INIT_VIDEO), "SDL initialization failed");
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    auto* window = SDL_CreateWindow("SplashCubes checks", 32, 32, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    require(window != nullptr, "GL window failed");
    auto context = SDL_GL_CreateContext(window);
    require(context != nullptr && gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress), "GL context failed");

    ImDebugLog log;
    CaptureContext capture;
    MaterialProject<Scenario> project;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();
    auto& scene = *project.source;
    scene.profile_stages = false;
    const auto save = [&](int frame)
    {
        CaptureFrame image;
        scene.captureFluidImage(image);
        bytebuf webp;
        require(webp_snapshot_encode_rgba(image.frame.data.data(), image.resolution.x, image.resolution.y,
            100, 6, true, 100, 100, webp), "Image encoding failed");
        std::ofstream file(output / ("frame-" + std::to_string(frame) + ".webp"), std::ios::binary);
        file.write(reinterpret_cast<const char*>(webp.data()), std::streamsize(webp.size()));
    };
    save(0);

    const auto initial = scene.fluid.download();
    std::vector<double> initial_mass;
    for (const auto* phase : {&initial.liquid_mass, &initial.gas_mass})
        for (const auto& component : *phase)
            initial_mass.push_back(std::accumulate(component.begin(), component.end(), 0.));
    std::ofstream csv(output / "health.csv");
    csv << "frame,time,mass_error,phase_error,mixed_volume,liquid_centroid_y,bottom_liquid,max_speed\n" << std::setprecision(17);
    bool reached_floor = false;
    double mixed = 0.;
    for (int frame = 1; frame <= frames; ++frame)
    {
        project.processHeadlessFrame(.02);
        require(!scene.simulation_failed, "SplashCubes projection failed");
        const auto state = scene.fluid.download();
        double mass_error = 0., phase_error = 0., volume = 0., centre_y = 0., bottom = 0., max_speed = 0.;
        mixed = 0.;
        size_t component_index = 0;
        for (const auto* phase : {&state.liquid_mass, &state.gas_mass})
        for (const auto& component : *phase)
        {
            for (const double mass : component)
                require(std::isfinite(mass) && mass >= -1e-5, "Invalid component mass");
            const double total = std::accumulate(component.begin(), component.end(), 0.);
            mass_error = std::max(mass_error, std::abs(total / initial_mass[component_index++] - 1.));
        }
        for (const auto* field : {&state.u, &state.v, &state.pressure})
            for (const double value : *field)
                require(std::isfinite(value), "Nonfinite velocity/pressure");
        for (int y = 0; y < scene.grid_h; ++y)
        for (int x = 0; x < scene.grid_w; ++x)
        {
            const int i = y * scene.grid_w + x;
            const double f = state.liquid_fraction[i];
            require(std::isfinite(f) && f >= -1e-6 && f <= 1.000001, "Invalid liquid fraction");
            const double area = scene.cell_size * scene.cell_size;
            const double blue = state.liquid_mass[0][i] / (scene.fluid.getLiquidMixture().components[0].density * area);
            const double red = state.liquid_mass[1][i] / (scene.fluid.getLiquidMixture().components[1].density * area);
            const double gas = (state.gas_mass[0][i] / scene.fluid.getGasMixture().components[0].density +
                state.gas_mass[1][i] / scene.fluid.getGasMixture().components[1].density) / area;
            phase_error = std::max({phase_error, std::abs(blue + red - f), std::abs(gas - (1. - f))});
            mixed += std::min(blue, red);
            volume += f;
            centre_y += f * (y + .5);
            if (y == scene.grid_h - 1) bottom += f;
            const double u = .5 * (state.u[y * (scene.grid_w + 1) + x] + state.u[y * (scene.grid_w + 1) + x + 1]);
            const double v = .5 * (state.v[y * scene.grid_w + x] + state.v[(y + 1) * scene.grid_w + x]);
            max_speed = std::max(max_speed, std::hypot(u, v));
        }
        reached_floor |= bottom > 1.;
        require(mass_error < 1e-4 && phase_error < .002, "Conservation/phase consistency exceeded existing float budgets");
        csv << frame << ',' << scene.simulation_time << ',' << mass_error << ',' << phase_error << ','
            << mixed << ',' << centre_y / volume << ',' << bottom << ',' << max_speed << '\n';
        if (frame == 50 || frame == 75 || frame == 100 || frame == 200 || frame == frames)
        {
            save(frame);
            std::cout << "frame=" << frame << " mixed_volume=" << mixed << " bottom_liquid=" << bottom << std::endl;
        }
    }
    require(reached_floor && mixed > 1., "Run did not reach impact and mixing");
    project.stopHeadless();
    SDL_GL_DestroyContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << "PASS " << Scenario::scenario_id << " impact, mixing and per-frame health over " << frames << " frames\n";
}
}

int main(int argc, char** argv)
{
    try
    {
        checkInitialState();
        if (argc == 3) evolve<SplashCubes>(argv[1], std::stoi(argv[2]));
        else if (argc == 4 && std::string(argv[3]) == "wave")
            evolve<FluidSim::CrashingWave>(argv[1], std::stoi(argv[2]));
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
