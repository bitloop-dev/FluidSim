// Evolving benchmark of the actual scene, including its initializer and frame scheduling.
#include "scenarios/CrashingWave.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace bl;
namespace fs = std::filesystem;
struct WaveProject : BasicProject
{
    SceneHandle<FluidSim::CrashingWave> source;
    void projectPrepare(Layout&) override { source = create<FluidSim::CrashingWave>(); }
};

void compareUniformBackends(const std::string& prefix)
{
    fs::create_directories(fs::path(prefix).parent_path());
    ImDebugLog cpu_log, gpu_log;
    CaptureContext cpu_capture, gpu_capture;
    WaveProject cpu_project, gpu_project;
    cpu_project.configureHeadless(0, &cpu_log, {800, 600}, &cpu_capture);
    gpu_project.configureHeadless(0, &gpu_log, {800, 600}, &gpu_capture);
    cpu_project.startHeadless();
    gpu_project.startHeadless();
    auto& cpu = *cpu_project.source.get();
    auto& gpu = *gpu_project.source.get();
    const auto prepare = [](auto& scene, fluid::FluidDevice device)
    {
        fluid::FluidGrid replacement(96, 72, 1.);
        replacement.setDevice(device);
        replacement.configureMaterials(scene.fluid.getElements(), scene.fluid.getLiquidMixture(), scene.fluid.getGasMixture());
        replacement.setGravityY(scene.gravity_y);
        // Keep this enabled even when the interactive scene currently defaults to zero.
        scene.surface_tension = 20000.;
        replacement.setSurfaceTension(scene.surface_tension);
        scene.fluid.destroy();
        scene.fluid = std::move(replacement);
        scene.fluid_device = device;
        scene.grid_w = 96;
        scene.grid_h = 72;
        scene.cell_size = 1.;
        scene.world_w = 96.;
        scene.world_h = 72.;
        scene.profile_stages = false;
        scene.mouse_attraction = false;
        scene.restartSimulation();
    };
    prepare(cpu, fluid::FluidDevice::CPU);
    prepare(gpu, fluid::FluidDevice::GPU);
    std::cout << "paired wave: 96x72 dx=1 sigma=" << cpu.fluid.getSurfaceTension()
        << " gravity=" << cpu.gravity_y << " tolerance=" << cpu.projection_tolerance << std::endl;
    const auto initial = cpu.fluid.download();
    const double initial_volume = std::accumulate(initial.liquid_fraction.begin(), initial.liquid_fraction.end(), 0.);
    std::vector<double> initial_mass;
    for (const auto* phase : {&initial.liquid_mass, &initial.gas_mass})
        for (const auto& component : *phase) initial_mass.push_back(std::accumulate(component.begin(), component.end(), 0.));
    double worst_mass = 0., worst_volume = 0., worst_phase = 0., worst_fraction_mean = 0., worst_velocity = 0.;
    std::ofstream csv(prefix + ".csv");
    csv << "frame,fraction_mean_difference,fraction_max_difference,velocity_max_difference,cpu_volume,gpu_volume,cpu_ms,gpu_ms,physical_steps\n" << std::setprecision(17);
    const auto observe = [&](const auto& scene, const auto& state)
    {
        size_t component_index = 0;
        for (const auto* phase : {&state.liquid_mass, &state.gas_mass})
            for (const auto& component : *phase)
            {
                const double total = std::accumulate(component.begin(), component.end(), 0.);
                const double before = initial_mass[component_index++];
                worst_mass = std::max(worst_mass, std::abs(total / before - 1.));
            }
        for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
        {
            const double f = state.liquid_fraction[i];
            if (!std::isfinite(f) || f < -1e-6 || f > 1.000001) throw std::runtime_error("Invalid paired-wave fraction");
            double liquid = 0., gas = 0.;
            for (size_t k = 0; k < state.liquid_mass.size(); ++k)
                liquid += state.liquid_mass[k][i] / scene.fluid.getLiquidMixture().components[k].density;
            for (size_t k = 0; k < state.gas_mass.size(); ++k)
                gas += state.gas_mass[k][i] / scene.fluid.getGasMixture().components[k].density;
            if (!std::isfinite(liquid + gas)) throw std::runtime_error("Nonfinite paired-wave mass");
            worst_phase = std::max({worst_phase, std::abs(liquid - f), std::abs(gas - (1. - f))});
        }
        for (const auto* field : {&state.u, &state.v, &state.pressure})
            for (const double value : *field)
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite paired-wave field");
        for (int y = 0; y < 72; ++y)
            if (state.u[y * 97] != 0 || state.u[y * 97 + 96] != 0) throw std::runtime_error("Open paired-wave X wall");
        for (int x = 0; x < 96; ++x)
            if (state.v[x] != 0 || state.v[72 * 96 + x] != 0) throw std::runtime_error("Open paired-wave Y wall");
        const double volume = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.);
        worst_volume = std::max(worst_volume, std::abs(volume / initial_volume - 1.));
        return volume;
    };
    const auto image = [&](const auto& scene, const char* device, int frame)
    {
        std::ofstream output(prefix + "-" + device + "-" + std::to_string(frame) + ".ppm", std::ios::binary);
        output << "P6\n96 72\n255\n";
        for (size_t i = 0; i < scene.display_pixels.size(); i += 4)
            for (int c = 0; c < 3; ++c)
            {
                const char value = char(std::clamp(scene.display_pixels[i + c], 0.f, 1.f) * 255.f + .5f);
                output.write(&value, 1);
            }
    };
    for (int frame = 1; frame <= 200; ++frame)
    {
        cpu.sceneProcess(.02);
        gpu.sceneProcess(.02);
        if (cpu.simulation_failed || gpu.simulation_failed) throw std::runtime_error("Paired wave failed at frame " + std::to_string(frame));
        if (cpu.frame_statistics.substeps != gpu.frame_statistics.substeps)
            throw std::runtime_error("CPU/GPU physical subdivision differs");
        const auto a = cpu.fluid.download(), b = gpu.fluid.download();
        const double cpu_volume = observe(cpu, a), gpu_volume = observe(gpu, b);
        double fraction_sum = 0., fraction_max = 0., velocity_max = 0.;
        for (size_t i = 0; i < a.liquid_fraction.size(); ++i)
        {
            const double difference = std::abs(a.liquid_fraction[i] - b.liquid_fraction[i]);
            fraction_sum += difference;
            fraction_max = std::max(fraction_max, difference);
        }
        for (size_t i = 0; i < a.u.size(); ++i) velocity_max = std::max(velocity_max, std::abs(a.u[i] - b.u[i]));
        for (size_t i = 0; i < a.v.size(); ++i) velocity_max = std::max(velocity_max, std::abs(a.v[i] - b.v[i]));
        const double mean = fraction_sum / a.liquid_fraction.size();
        worst_fraction_mean = std::max(worst_fraction_mean, mean);
        worst_velocity = std::max(worst_velocity, velocity_max);
        csv << frame << ',' << mean << ',' << fraction_max << ',' << velocity_max << ',' << cpu_volume << ',' << gpu_volume
            << ',' << cpu.frame_statistics.fluid_ms << ',' << gpu.frame_statistics.fluid_ms
            << ',' << cpu.frame_statistics.substeps << '\n';
        if (frame == 1 || frame == 50 || frame == 100 || frame == 200)
        {
            image(cpu, "cpu", frame);
            image(gpu, "gpu", frame);
            std::cout << "wave frame=" << frame << " mean_fraction_difference=" << mean << " max_velocity_difference=" << velocity_max << std::endl;
        }
    }
    std::cout << "wave maxima mass_drift=" << worst_mass << " volume_drift=" << worst_volume << " phase_volume_error=" << worst_phase
        << " mean_fraction_difference=" << worst_fraction_mean << " velocity_difference=" << worst_velocity << std::endl;
    if (worst_mass > 1e-4 || worst_volume > 1e-4 || worst_phase > .002)
        throw std::runtime_error("Paired wave exceeded conservation/phase-consistency budget");
    cpu_project.stopHeadless();
    gpu_project.stopHeadless();
    std::cout << "PASS paired uniform wave: 200 frames /4 seconds" << std::endl;
}

int main(int argc, char** argv)
{
    try
    {
        if (argc == 3 && std::string(argv[1]) == "--pair")
        {
            compareUniformBackends(argv[2]);
            return 0;
        }
        if (argc != 5 && argc != 6)
            throw std::runtime_error("Usage: FluidSurfaceTensionWave sigma tolerance frames output-prefix [fixed|moving]");
        const double sigma = std::stod(argv[1]), tolerance = std::stod(argv[2]);
        const int frames = std::stoi(argv[3]);
        const std::string prefix = argv[4];
        const std::string attraction = argc == 6 ? argv[5] : "";
        if (!attraction.empty() && attraction != "fixed" && attraction != "moving")
            throw std::runtime_error("Attraction mode must be fixed or moving");
        fs::create_directories(fs::path(prefix).parent_path());
        ImDebugLog log;
        CaptureContext capture;
        WaveProject project;
        project.configureHeadless(0, &log, {800, 600}, &capture);
        project.startHeadless();
        auto& scene = *project.source.get();
        scene.projection_tolerance = tolerance;
        scene.profile_stages = true;
        scene.setSurfaceTension(sigma);
        scene.mouse_attraction = !attraction.empty();
        scene.mouse_target = {scene.world_w * .5, scene.world_h * .55};
        const auto initial = scene.fluid.download();
        std::vector<double> masses;
        for (const auto* phase : {&initial.liquid_mass, &initial.gas_mass})
            for (const auto& component : *phase) masses.push_back(std::accumulate(component.begin(), component.end(), 0.0));
        const double initial_volume = std::accumulate(initial.liquid_fraction.begin(), initial.liquid_fraction.end(), 0.0);
        std::ofstream csv(prefix + ".csv"), health(prefix + "-health.csv");
        csv << "frame,update_ms,fluid_ms,transport_ms,forces_ms,projection_ms,colour_ms,physical_steps,transport_steps,pressure_iterations\n" << std::setprecision(17);
        health << "frame,mass_error,phase_volume_error,liquid_volume_drift,fraction_error,max_speed,energy,interface_length\n" << std::setprecision(17);
        for (int frame = 0; frame < frames; ++frame)
        {
            if (attraction == "moving")
            {
                const double angle = frame * .02;
                scene.mouse_target = {
                    scene.world_w * (.5 + .25 * std::sin(angle)),
                    scene.world_h * (.55 + .2 * std::cos(angle))
                };
            }
            project.processHeadlessFrame(.02);
            if (scene.simulation_failed) throw std::runtime_error("Wave projection failed at frame " + std::to_string(frame));
            const auto& t = scene.frame_statistics;
            if (t.substeps == 0) throw std::runtime_error("Wave fixture did not advance the actual scene");
            csv << frame << ',' << t.update_ms << ',' << t.fluid_ms << ',' << t.stages.transport_ms << ',' << t.stages.forces_ms
                << ',' << t.stages.pressure_ms << ',' << t.colour_ms << ',' << t.substeps << ',' << t.transport_steps << ',' << t.pressure_iterations << '\n';
            if (frame % 20 == 0 || frame + 1 == frames)
            {
                const auto state = scene.fluid.download();
                double mass_error = 0, volume_error = 0, fraction_error = 0, max_speed = 0, energy = 0, length = 0;
                size_t component_index = 0;
                for (const auto* phase : {&state.liquid_mass, &state.gas_mass})
                    for (const auto& component : *phase)
                    {
                        const double before = masses[component_index++];
                        mass_error = std::max(mass_error, std::abs(std::accumulate(component.begin(), component.end(), 0.0) - before) / std::max(before, 1.0));
                    }
                const int w = scene.grid_w, h = scene.grid_h;
                const double dx = scene.cell_size, area = dx * dx;
                for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
                {
                    const size_t i = size_t(y) * w + x;
                    const double f = state.liquid_fraction[i];
                    fraction_error = std::max({fraction_error, -f, f - 1.0});
                    double lv = 0, gv = 0, mass = 0;
                    for (size_t k = 0; k < state.liquid_mass.size(); ++k) { mass += state.liquid_mass[k][i]; lv += state.liquid_mass[k][i] / scene.fluid.getLiquidMixture().components[k].density / area; }
                    for (size_t k = 0; k < state.gas_mass.size(); ++k) { mass += state.gas_mass[k][i]; gv += state.gas_mass[k][i] / scene.fluid.getGasMixture().components[k].density / area; }
                    volume_error = std::max({volume_error, std::abs(lv - f), std::abs(gv - (1 - f))});
                    const double u = .5 * (state.u[y*(w+1)+x] + state.u[y*(w+1)+x+1]);
                    const double v = .5 * (state.v[y*w+x] + state.v[(y+1)*w+x]);
                    max_speed = std::max(max_speed, std::hypot(u,v));
                    double potential = scene.fluid.getGravityY() * (h-y-.5)*dx;
                    if (scene.mouse_attraction)
                    {
                        const auto origin = scene.fluid.getOrigin();
                        const double distance = std::hypot(origin.x + (x+.5)*dx - scene.mouse_target.x,
                            origin.y + (y+.5)*dx - scene.mouse_target.y);
                        const double radius = scene.mouse_attraction_radius;
                        potential = scene.mouse_attraction_strength * (distance < radius
                            ? .5 * distance * distance / radius : distance - .5 * radius);
                    }
                    energy += mass * (.5*(u*u+v*v) + potential);
                    // Gradient-based interface-length diagnostic; same estimator in every control.
                    const double gx = .5*(state.liquid_fraction[y*w+std::min(x+1,w-1)]-state.liquid_fraction[y*w+std::max(x-1,0)]);
                    const double gy = .5*(state.liquid_fraction[std::min(y+1,h-1)*w+x]-state.liquid_fraction[std::max(y-1,0)*w+x]);
                    length += std::hypot(gx,gy)*dx;
                    if (!std::isfinite(f+lv+gv+u+v+mass)) throw std::runtime_error("Nonfinite wave state");
                }
                const double volume_drift = (std::accumulate(state.liquid_fraction.begin(),state.liquid_fraction.end(),0.0)-initial_volume)/initial_volume;
                health << frame << ',' << mass_error << ',' << volume_error << ',' << volume_drift << ',' << fraction_error << ',' << max_speed << ',' << energy+sigma*length << ',' << length << '\n';
            }
            if (frame == 0 || (frame+1)%200 == 0 || frame+1 == frames)
            {
                std::ofstream image(prefix + "-" + std::to_string(frame+1) + ".ppm", std::ios::binary);
                image << "P6\n" << scene.grid_w << ' ' << scene.grid_h << "\n255\n";
                for (size_t i = 0; i < scene.display_pixels.size(); i += 4)
                    for (int c=0;c<3;++c) { const char value=char(std::clamp(scene.display_pixels[i+c],0.f,1.f)*255.f+.5f); image.write(&value,1); }
                std::cout << "frame=" << frame+1 << " fluid_ms=" << t.fluid_ms << std::endl;
            }
        }
        fluid::FluidCheckpoint checkpoint;
        if (scene.fluid.saveCheckpoint(checkpoint))
        {
            std::ofstream data(prefix+".bin", std::ios::binary);
            data.write(reinterpret_cast<const char*>(checkpoint.data.data()), std::streamsize(checkpoint.data.size()));
            std::ofstream(prefix+".json") << checkpoint.metadata.dump(2);
        }
        std::cout << "PASS wave " << frames << " frames sigma=" << sigma << " tolerance=" << tolerance << std::endl;
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << std::endl; return 1; }
}
