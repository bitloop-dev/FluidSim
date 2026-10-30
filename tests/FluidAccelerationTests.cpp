#include "FluidCheckpointTestHelpers.h"
#include "UniformWebGPU/FluidGridGPU.h"
#include "FluidGrid.h"
#include "FluidTestSetup.h"
#include "MouseAttraction.h"
#include "FluidFieldInternal.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

using namespace fluid;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class Grid>
void initialize(Grid& grid)
{
    typename Grid::State state;
    state.liquid_fraction.resize(size_t(grid.width) * grid.height);
    for (int y = 0; y < grid.height; ++y)
    {
        for (int x = 0; x < grid.width; ++x)
        {
            state.liquid_fraction[y * grid.width + x] = std::clamp(float(y + 1) - 8.5f - 0.25f * x, 0.f, 1.f);
        }
    }
    test::uploadTestMaterials(grid, state);
}

void requireSame(const FluidGridGPU& left, const FluidGridGPU& right)
{
    FluidCheckpoint a, b;
    left.saveCheckpoint(a);
    right.saveCheckpoint(b);
    require(a.data == b.data, "Acceleration paths changed native state or pressure history");
}

std::vector<float> storedField(const FluidGridGPU& grid)
{
    FluidCheckpoint checkpoint;
    grid.saveCheckpoint(checkpoint);
    require(checkpoint.metadata.at("acceleration_field"), "Acceleration field was not saved");
    std::vector<float> values(size_t(grid.width) * grid.height * 2);
    const size_t bytes = values.size() * sizeof(float);
    std::memcpy(values.data(), checkpoint.data.data() + checkpoint.data.size() - bytes, bytes);
    return values;
}

void uniformAndClear()
{
    FluidGridGPU original(32, 24, 1), supplied(32, 24, 1);
    original.gravity_y = supplied.gravity_y = 128;
    initialize(original);
    initialize(supplied);
    std::vector<float> values(32 * 24 * 2);
    for (size_t i = 1; i < values.size(); i += 2) values[i] = 128;
    FluidField cpu({32, 24, 1, {}, 2});
    cpu.write(values);
    supplied.setAccelerationField(cpu.upload(supplied.getWebGPUContext()));

    for (int i = 0; i < 40; ++i)
    {
        require(original.step(1. / 128, 2000, .001).converged, "Uniform gravity failed");
        require(supplied.step(1. / 128, 2000, .001).converged, "Uniform field failed");
    }
    supplied.clearAccelerationField();
    requireSame(original, supplied);
    for (int i = 0; i < 40; ++i)
    {
        require(original.step(1. / 128, 2000, .001).converged, "Original continuation failed");
        require(supplied.step(1. / 128, 2000, .001).converged, "Cleared continuation failed");
        requireSame(original, supplied);
    }
    std::cout << "PASS uniform field replaces gravity exactly; clear restores the original path\n";
}

constexpr std::string_view affine_shader = R"wgsl(
fn fieldValue(position: vec2f) -> vec4f {
    let origin = fieldParameter(0u).xy;
    let offset = position - origin;
    return vec4f(-0.25 * offset.y, 0.5 * offset.x, 0, 0);
}
)wgsl";

void generationSubstepsAndResume()
{
    // Generate before either engine exists, then share the chosen device explicitly.
    FluidWebGPUContext context;
    FluidField acceleration({32, 24, .5, {3, -2}, 2}, context);
    FluidFieldWebGPUGenerator generator(context, affine_shader);
    const std::array<float, 4> parameters[] = {{3, -2, 0, 0}};
    generator.generate(acceleration, parameters);
    FluidGridGPU generated(32, 24, .5), uploaded(32, 24, .5);
    generated.setWebGPUContext(context);
    generated.origin = uploaded.origin = {3, -2};
    generated.setSurfaceTension(40000);
    uploaded.setSurfaceTension(40000);
    initialize(generated);
    initialize(uploaded);
    generated.setAccelerationField(acceleration);
    std::vector<float> values(32 * 24 * 2);
    for (int y = 0; y < 24; ++y)
    {
        for (int x = 0; x < 32; ++x)
        {
            const size_t index = 2 * size_t(y * 32 + x);
            values[index] = -.25f * (y + .5f) * .5f;
            values[index + 1] = .5f * (x + .5f) * .5f;
        }
    }
    require(storedField(generated) == values, "Generated world positions/XY channels differ from CPU values");
    FluidField cpu(acceleration.layout());
    cpu.write(values);
    auto uploaded_field = cpu.upload(uploaded.getWebGPUContext());
    uploaded.setAccelerationField(uploaded_field);

    const double dt = .0125;
    const int count = int(std::ceil(dt / generated.capillaryStepLimit()));
    FluidStepTimings timings;
    const auto result = generated.step(dt, 2000, .001, &timings);
    require(result.converged && result.physical_steps == count && count > 1, "Full acceleration substeps failed");
    int iterations = 0;
    for (int i = 0; i < count; ++i)
    {
        const auto part = uploaded.step(dt / count, 2000, .001);
        require(part.converged, "Manual acceleration substep failed");
        iterations += part.iterations;
    }
    require(iterations == result.iterations && timings.forces_ms > 0, "Field timing/iteration accounting failed");
    requireSame(generated, uploaded);

    // Both engine bindings must see later writes, without another set call.
    for (auto& value : values) value *= .5f;
    acceleration.write(values);
    uploaded_field.write(values);
    require(storedField(generated) == values && storedField(uploaded) == values,
        "Binding copied the field instead of retaining its shared storage");
    require(generated.step(.001, 2000, .001).converged && uploaded.step(.001, 2000, .001).converged,
        "Updated bound field failed to advance");
    requireSame(generated, uploaded);

    FluidCheckpoint saved;
    generated.saveCheckpoint(saved);
    FluidGridGPU resumed(32, 24, .5);
    resumed.loadCheckpoint(saved);
    for (int i = 0; i < 30; ++i)
    {
        require(generated.step(.001, 2000, .001).converged, "Original field evolution failed");
        require(resumed.step(.001, 2000, .001).converged, "Restored field evolution failed");
        requireSame(generated, resumed);
    }

    generated.clearAccelerationField();
    generated.saveCheckpoint(saved);
    makeLegacyCheckpoint(saved, 2);
    resumed.loadCheckpoint(saved);
    requireSame(generated, resumed);
    FluidCheckpoint legacy;
    resumed.saveCheckpoint(legacy);
    require(!legacy.metadata.at("acceleration_field").get<bool>(), "Legacy load retained a stale field");
    std::cout << "PASS shader/CPU equality, capillary substeps, timed parity, exact field resume and legacy clear\n";
}

void mouseParametersAndValidation()
{
    FluidGridGPU grid(17, 13, 1);
    initialize(grid);
    const auto context = grid.getWebGPUContext();
    FluidField field({17, 13, 1, {}, 2}, context);
    FluidFieldWebGPUGenerator generator(context, mouse_attraction_shader);
    std::array<float, 4> parameters[] = {{8.5f, 6.5f, 100.f, 2.f}};
    generator.generate(field, parameters);
    grid.setAccelerationField(field);
    const auto first = storedField(grid);
    const size_t centre = 2 * (6 * 17 + 8);
    require(first[centre] == 0 && first[centre + 1] == 0, "Mouse-centred acceleration is not zero");
    require(first[centre - 2] > 0 && first[centre + 2] < 0, "Attraction points away from the target");
    parameters[0][0] += 2;
    generator.generate(field, parameters);
    require(storedField(grid)[centre] == 100, "Cached shader ignored updated parameters");

    FluidCheckpoint before, after;
    grid.saveCheckpoint(before);
    bool rejected = false;
    try { field.write(std::span<const float>(first.data(), 1)); }
    catch (const std::invalid_argument&) { rejected = true; }
    grid.saveCheckpoint(after);
    require(rejected && before.data == after.data, "Invalid upload changed the active field");

    FluidGrid unsupported(8, 8, 1);
    unsupported.setAdaptiveMode(AdaptiveMode::MAX_2);
    rejected = false;
    try { unsupported.setAccelerationField(field); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Unsupported backend silently accepted a field");
    std::cout << "PASS mouse direction/centre, cached parameter update and input/backend rejection\n";
}

void independentFieldsAndBinding()
{
    FluidWebGPUContext context;
    constexpr std::string_view shader = R"wgsl(
fn fieldValue(position: vec2f) -> vec4f {
    return vec4f(position, position.x + position.y, 7);
}
)wgsl";
    FluidFieldWebGPUGenerator generator(context, shader);
    for (const int channels : {1, 2, 3, 4})
    {
        FluidField cpu({17, 13, .5, {3, -2}, channels});
        generateFluidField(cpu, [](bl::DVec2 position)
        {
            return std::array<float, 4>{float(position.x), float(position.y), float(position.x + position.y), 7};
        });
        auto alias = cpu;
        require(alias.values().data() == cpu.values().data(), "CPU handles do not share storage");
        auto gpu = cpu.upload(context);
        require(gpu.read() == cpu.read(), "Upload changed the field values");
        auto gpu_alias = gpu;
        require(detail::FluidFieldAccess::gpuField(gpu).buffer.Get() == detail::FluidFieldAccess::gpuField(gpu_alias).buffer.Get(),
            "GPU handle copy allocated another buffer");
        gpu.write(std::vector<float>(cpu.values().size(), -777.f));
        generator.generate(gpu);
        const auto downloaded = gpu.download();
        require(cpu.read() == downloaded.read(), "Independent CPU/GPU generation or transfer changed samples");
        alias.values()[0] = -99;
        require(cpu.values()[0] == -99 && downloaded.values()[0] != -99 && gpu.read()[0] != -99,
            "Transfers failed to make independent copies");
    }

    FluidGridGPU grid(17, 13, 1);
    grid.setWebGPUContext(context);
    initialize(grid);
    FluidField field({17, 13, 1, {}, 2}, context);
    generator.generate(field);
    grid.setAccelerationField(field);
    FluidCheckpoint before, after;
    grid.saveCheckpoint(before);
    const auto rejectBinding = [&](const FluidField& invalid)
    {
        bool rejected = false;
        try { grid.setAccelerationField(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        grid.saveCheckpoint(after);
        require(rejected && before.data == after.data, "Invalid binding changed the active field");
    };
    rejectBinding(FluidField(field.layout())); // No implicit upload.
    rejectBinding(FluidField({17, 13, 1, {}, 1}, context));
    rejectBinding(FluidField({16, 13, 1, {}, 2}, context));
    rejectBinding(FluidField({17, 13, .5, {}, 2}, context));
    rejectBinding(FluidField({17, 13, 1, {1, 0}, 2}, context));
    FluidWebGPUContext other_context;
    FluidField foreign(field.layout(), other_context);
    rejectBinding(foreign);
    bool rejected = false;
    try { generator.generate(foreign); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Generator accepted a foreign device");
    rejected = false;
    try { grid.setWebGPUContext(other_context); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected && grid.getWebGPUContext() == context, "Live grid allowed its device to change");

    const auto original = field.read();
    const std::vector<float> changed(original.size(), -3.f);
    field.write(changed);
    grid.loadCheckpoint(before);
    require(field.read() == changed && storedField(grid) == original,
        "Checkpoint restore mutated the external field or failed to restore its own values");

    // The engine's lifetime does not own the producer or the field.
    grid.destroy();
    generator.generate(field);
    require(!field.read().empty(), "Destroying the engine invalidated an independent field");
    grid.setWebGPUContext(context);
    initialize(grid);
    grid.setAccelerationField(field);
    grid.upload({});
    require(storedField(grid) == field.read(), "Ordinary grid upload lost the binding");
    std::cout << "PASS independent scalar/vector generation, explicit copies, lifetime and binding validation\n";
}

bool sameState(const FluidGridCPU::State& a, const FluidGridCPU::State& b)
{
    return a.liquid_fraction == b.liquid_fraction && a.u == b.u && a.v == b.v &&
        a.pressure == b.pressure && a.liquid_mass == b.liquid_mass && a.gas_mass == b.gas_mass;
}

void cpuBindingAndClear()
{
    FluidGridCPU gravity(32, 24, 1), supplied(32, 24, 1);
    initialize(gravity);
    initialize(supplied);
    gravity.gravity_y = supplied.gravity_y = 128;
    FluidField field({32, 24, 1, {}, 2});
    generateFluidField(field, [](bl::DVec2) { return std::array<float, 4>{0, 128, 0, 0}; });
    supplied.setAccelerationField(field);
    bool rejected = false;
    try { supplied.setAccelerationField(FluidField({32, 24, 1, {1, 0}, 2})); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "CPU accepted a mismatched field origin");
    const auto advance = [&]
    {
        require(gravity.step(1. / 128, 2000, .001).converged && supplied.step(1. / 128, 2000, .001).converged,
            "CPU uniform-field projection failed");
        require(sameState(gravity.download(), supplied.download()), "CPU field differs from uniform gravity");
    };
    for (int i = 0; i < 5; ++i) advance();
    gravity.gravity_y = 64;
    generateFluidField(field, [](bl::DVec2) { return std::array<float, 4>{0, 64, 0, 0}; });
    for (int i = 0; i < 5; ++i) advance(); // Live updates, no rebind.
    supplied.clearAccelerationField();
    gravity.gravity_y = 128;
    for (int i = 0; i < 5; ++i) advance();

    supplied.setAccelerationField(field);
    supplied.upload(gravity.download()); // Ordinary upload retains the binding.
    gravity.upload(gravity.download());
    gravity.gravity_y = 64;
    advance();
    supplied.destroy();
    gravity.gravity_y = 128;
    advance();

    FluidGrid api(32, 24, 1);
    api.configureMaterials(gravity.substances, gravity.liquid_mixture, gravity.gas_mixture);
    api.upload(gravity.download());
    gravity.upload(gravity.download());
    gravity.gravity_y = 64;
    api.setAccelerationField(field.download()); // Temporary handle; the engine retains the storage.
    require(api.step(.001, 2000, .001).converged && gravity.step(.001, 2000, .001).converged,
        "Common CPU field binding failed");
    require(sameState(api.download(), gravity.download()), "Common CPU binding lost a temporary field");

    std::cout << "PASS CPU gravity equivalence, bound updates, clear, upload, destroy and shared API lifetime\n";
}

void uniformCpuGpuParity()
{
    FluidGridCPU cpu(33, 25, .5);
    FluidGridGPU gpu(33, 25, .5);
    cpu.origin = gpu.origin = {3, -2};
    initialize(cpu);
    initialize(gpu);
    FluidField field({33, 25, .5, cpu.origin, 2});
    auto device_field = field.upload(gpu.getWebGPUContext());
    cpu.setAccelerationField(field);
    gpu.setAccelerationField(device_field);
    bool rejected = false;
    try { cpu.setAccelerationField(device_field); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "CPU silently consumed a GPU field without an explicit download");
    double fraction_error = 0., velocity_error = 0., mass_density_error = 0., pressure_error = 0.;
    double conservation_error = 0., volume_drift = 0., divergence = 0., pressure_scale = 0.;
    const auto initial = cpu.download();
    const double initial_volume = std::accumulate(initial.liquid_fraction.begin(), initial.liquid_fraction.end(), 0.);
    const auto compare = [](const auto& a, const auto& b, double& worst, double scale = 1.)
    {
        for (size_t i = 0; i < a.size(); ++i)
        {
            require(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite CPU/GPU field");
            worst = std::max(worst, std::abs(double(a[i]) - double(b[i])) / scale);
        }
    };
    // Short requested parity check: 40 steps / 0.4 seconds, with a moving field.
    for (int step = 0; step < 40; ++step)
    {
        const std::array<float, 4> settings = {float(10. + 2. * std::sin(step * .1)), 3.f, 64.f, 2.f};
        generateFluidField(field, [&](bl::DVec2 position) { return mouseAttraction(position, settings); });
        device_field.write(field.values()); // Both engines consume identical float samples.
        const auto c = cpu.step(.01, 2000, .001);
        const auto g = gpu.step(.01, 2000, .001);
        require(c.converged && g.converged, "CPU/GPU acceleration parity projection failed");
        divergence = std::max({divergence, c.max_divergence, g.max_divergence});
        const auto a = cpu.download();
        const auto b = gpu.download();
        compare(a.liquid_fraction, b.liquid_fraction, fraction_error);
        compare(a.u, b.u, velocity_error);
        compare(a.v, b.v, velocity_error);
        compare(a.pressure, b.pressure, pressure_error);
        for (const double value : a.pressure) pressure_scale = std::max(pressure_scale, std::abs(value));
        compare(a.liquid_mass[0], b.liquid_mass[0], mass_density_error, 100. * .25);
        compare(a.gas_mass[0], b.gas_mass[0], mass_density_error, .1 * .25);
        const auto health = [&](const auto& state)
        {
            for (const auto f : state.liquid_fraction)
                require(f >= -1e-6 && f <= 1. + 1e-6, "CPU/GPU field caused an out-of-range fraction");
            const double volume = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.);
            volume_drift = std::max(volume_drift, std::abs(volume / initial_volume - 1.));
            for (int phase = 0; phase < 2; ++phase)
            {
                const auto& before = phase == 0 ? initial.liquid_mass[0] : initial.gas_mass[0];
                const auto& after = phase == 0 ? state.liquid_mass[0] : state.gas_mass[0];
                const double old_mass = std::accumulate(before.begin(), before.end(), 0.);
                const double new_mass = std::accumulate(after.begin(), after.end(), 0.);
                conservation_error = std::max(conservation_error, std::abs(new_mass / old_mass - 1.));
            }
            for (int y = 0; y < cpu.height; ++y)
                require(state.u[cpu.uIndex(0, y)] == 0 && state.u[cpu.uIndex(cpu.width, y)] == 0, "Open X wall");
            for (int x = 0; x < cpu.width; ++x)
                require(state.v[cpu.vIndex(x, 0)] == 0 && state.v[cpu.vIndex(x, cpu.height)] == 0, "Open Y wall");
        };
        health(a);
        health(b);
    }
    std::cout << "CPU/GPU 40-step maxima: fraction=" << fraction_error << " velocity=" << velocity_error
        << " component-volume=" << mass_density_error << " pressure=" << pressure_error
        << " mass-drift=" << conservation_error << " liquid-volume-drift=" << volume_drift
        << " divergence=" << divergence << " peak-pressure=" << pressure_scale << '\n';
    require(fraction_error < 5e-4 && velocity_error < 5e-3 && mass_density_error < 5e-4 && pressure_error < .5,
        "Uniform CPU/GPU acceleration trajectories diverged");
    require(conservation_error < 1e-5 && volume_drift < 1e-5, "Short CPU/GPU field evolution lost mass or volume");

    FluidFieldWebGPUGenerator generator(gpu.getWebGPUContext(), mouse_attraction_shader);
    const std::array<float, 4> settings[] = {{10.f, 3.f, 64.f, 2.f}};
    generateFluidField(field, [&](bl::DVec2 position) { return mouseAttraction(position, settings[0]); });
    generator.generate(device_field, settings);
    double generator_error = 0.;
    compare(field.values(), device_field.read(), generator_error);
    require(generator_error < 2e-5, "CPU/WGSL mouse producers disagree");
    std::cout << "PASS uniform CPU/GPU moving-field parity; mouse generator max difference=" << generator_error << '\n';
}
}

int main(int, char**)
{
    try
    {
        uniformAndClear();
        generationSubstepsAndResume();
        mouseParametersAndValidation();
        independentFieldsAndBinding();
        cpuBindingAndClear();
        uniformCpuGpuParity();
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
