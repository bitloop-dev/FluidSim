#include "FluidGrid.h"
#include "FluidFieldInternal.h"
#include "UniformWebGPU/ElasticLiquidGPU.h"
#include "UniformWebGPU/FluidGridGPU.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace
{
using fluid::FluidGridGPU;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void configure(FluidGridGPU& grid, const fluid::FluidWebGPUContext& context, int solid = 0)
{
    grid.setWebGPUContext(context);
    const auto elastic = grid.substances.add({"Elastic", {.2f, .8f, .6f}});
    const auto water = grid.substances.add({"Water", {.1f, .4f, 1.f}});
    const auto dye = grid.substances.add({"Dye", {.6f, .1f, .8f}});
    const auto gas = grid.substances.add({"Gas", {0, 0, 0}});
    grid.liquid_mixture = {{{elastic, 10, 2, 1000}, {water, 3}, {dye, 6}}};
    std::swap(grid.liquid_mixture.components[0], grid.liquid_mixture.components[solid]);
    grid.gas_mixture = {{{gas, .01}}};
    grid.gravity_y = 0;
}

FluidGridGPU::State initial(int n, int solid, bool triple)
{
    FluidGridGPU::State state;
    state.liquid_fraction.resize(n * n);
    state.elastic_fraction.resize(n * n);
    state.liquid_mass.assign(3, std::vector<float>(n * n));
    state.gas_mass.assign(1, std::vector<float>(n * n));
    state.elastic_log_xx.resize(n * n);
    state.elastic_log_xy.resize(n * n);
    state.elastic_log_yy.resize(n * n);
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            const int i = y * n + x;
            float s = x >= n / 3 && x < n * 2 / 3 && y >= n / 4 && y < n / 2 ? 1.f : 0.f;
            float l = y >= n * 3 / 4 ? 1.f : 0.f;
            if (triple)
            {
                // Exercise all three split winners, including a small intermediate liquid phase.
                const float bands[3][2] = {{.49f, .02f}, {.334f, .333f}, {.02f, .49f}};
                s = bands[(x / 3 + y / 3) % 3][0];
                l = bands[(x / 3 + y / 3) % 3][1];
            }
            state.liquid_fraction[i] = s + l;
            state.elastic_fraction[i] = s;
            state.liquid_mass[0][i] = 10 * s;
            state.liquid_mass[1][i] = 3 * .7f * l;
            state.liquid_mass[2][i] = 6 * .3f * l;
            std::swap(state.liquid_mass[0][i], state.liquid_mass[solid][i]);
            state.gas_mass[0][i] = .01f * (1 - s - l);
            state.elastic_log_xx[i] = .02f * std::sin(float(x));
            state.elastic_log_yy[i] = -state.elastic_log_xx[i];
        }
    }
    state.u.resize((n + 1) * n);
    state.v.resize(n * (n + 1));
    const double pi = std::acos(-1.0);
    auto stream = [&](int x, int y)
    {
        return 2 * std::sin(pi * x / n) * std::sin(pi * y / n);
    };
    for (int y = 0; y < n; ++y)
    {
        for (int x = 1; x < n; ++x)
            state.u[y * (n + 1) + x] = float(stream(x, y + 1) - stream(x, y));
    }
    for (int y = 1; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
            state.v[y * n + x] = float(stream(x, y) - stream(x + 1, y));
    }
    return state;
}

double total(const std::vector<float>& values)
{
    return std::accumulate(values.begin(), values.end(), 0.0);
}

void inspect(const FluidGridGPU::State& state, const FluidGridGPU::State& before, int solid)
{
    for (size_t k = 0; k < state.liquid_mass.size(); ++k)
        require(std::abs(total(state.liquid_mass[k]) / total(before.liquid_mass[k]) - 1) < 1e-4,
            "Liquid ingredient mass drift");
    require(std::abs(total(state.gas_mass[0]) / total(before.gas_mass[0]) - 1) < 1e-4, "Gas mass drift");
    const double density[3] = {10, 3, 6};
    for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
    {
        const float s = state.elastic_fraction[i];
        const float c = state.liquid_fraction[i];
        require(std::isfinite(s) && s >= -2e-5f && s <= c + 2e-5f && c <= 1.00002f, "Invalid phase partition");
        require(std::abs(state.liquid_mass[solid][i] / 10 - s) < .002, "Elastic mass/volume mismatch");
        double l = 0;
        for (int k = 0; k < 3; ++k)
        {
            const int original = k == solid ? 0 : (k == 0 ? solid : k);
            if (k != solid)
                l += state.liquid_mass[k][i] / density[original];
            require(
                std::isfinite(state.liquid_mass[k][i]) && state.liquid_mass[k][i] > -2e-4, "Invalid ingredient mass");
        }
        require(std::abs(l - (c - s)) < .002, "Water mass/volume mismatch");
        require(std::abs(state.gas_mass[0][i] / .01 - (1 - c)) < .002, "Gas mass/volume mismatch");
        require(std::isfinite(state.elastic_log_xx[i]) && std::isfinite(state.elastic_log_xy[i]) &&
                    std::isfinite(state.elastic_log_yy[i]),
            "Invalid elastic history");
        if (s <= 0)
            require(state.elastic_log_xx[i] == 0 && state.elastic_log_xy[i] == 0 && state.elastic_log_yy[i] == 0,
                "History escaped elastic material");
    }
    for (const auto* field : {&state.u, &state.v, &state.pressure})
    {
        for (float value : *field)
            require(std::isfinite(value), "Nonfinite mechanical state");
    }
}

void sweptPartition(const fluid::FluidWebGPUContext& context)
{
    using namespace fluid::webgpu;
    auto& compute = fluid::detail::FluidFieldAccess::commands(context);
    constexpr int n = 16;
    ElasticLiquidGPU material(compute, n, n);
    auto condensed = compute.field(n, n);
    auto outer = compute.field(n, n, 4);
    auto density = compute.field(n, n, 4);
    auto flow = compute.field(n + 1, n);
    auto flux = compute.field(n + 1, n, 4);
    std::vector<float> c(n * n);
    std::vector<float> solid(n * n);
    std::vector<float> planes(4 * n * n);
    std::vector<float> inner(4 * n * n);
    std::vector<float> rho(4 * n * n);
    const float occupancies[] = {0, 1e-7f, .02f, .33333334f, .49f, .51f, .9999999f, 1, 1.0000001f};
    for (int y = 0; y < n; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            const int i = y * n + x;
            c[i] = occupancies[x % std::size(occupancies)];
            solid[i] = c[i] * occupancies[y % std::size(occupancies)];
            planes[4 * i] = 1;
            planes[4 * i + 2] = c[i];
            inner[4 * i + 1] = 1;
            inner[4 * i + 2] = occupancies[y % std::size(occupancies)];
            rho[4 * i] = 10 * solid[i] + 3 * (c[i] - solid[i]);
            rho[4 * i + 1] = .01f * (1 - c[i]);
            rho[4 * i + 2] = 10 * solid[i];
            rho[4 * i + 3] = 3 * (c[i] - solid[i]);
        }
    }
    compute.write(condensed, c);
    compute.write(material.fraction, solid);
    compute.write(outer, planes);
    compute.write(material.interfaces, inner);
    compute.write(density, rho);
    for (int direction = 0; direction < 2; ++direction)
    {
        flow = compute.field(n + (direction == 0), n + (direction == 1));
        flux = compute.field(flow.width, flow.height, 4);
        std::vector<float> velocity(flow.width * flow.height);
        for (int y = 0; y < flow.height; ++y)
        {
            for (int x = 0; x < flow.width; ++x)
            {
                const int coordinate = direction == 0 ? x : y;
                velocity[y * flow.width + x] =
                    coordinate == 0 || coordinate == n ? 0 : (coordinate % 2 == 0 ? .16f : -.16f);
            }
        }
        compute.write(flow, velocity);
        Parameters parameters;
        parameters.grid = {n, n, direction, 0};
        parameters.value = {1, 1, 0, 0};
        compute.run(material.transfers, {&flux},
            {{"condensed", &condensed}, {"solid", &material.fraction}, {"outer", &outer},
                {"inner", &material.interfaces}, {"flow", &flow}, {"density", &density}},
            parameters);
        const auto result = compute.read(flux);
        for (int y = 0; y < flow.height; ++y)
        {
            for (int x = 0; x < flow.width; ++x)
            {
                const int face = y * flow.width + x;
                const float q = velocity[face];
                if (q == 0)
                    continue;
                const int dx = x - (direction == 0 && q > 0);
                const int dy = y - (direction == 1 && q > 0);
                const int i = dy * n + dx;
                const double strip = std::abs(q);
                const double height = std::min(1.0, double(inner[4 * i + 2]));
                const double width = std::min(1.0, double(c[i]));
                double total = width * strip;
                double elastic = width * (q > 0 ? std::max(0.0, height - (1 - strip)) : std::min(height, strip));
                if (direction == 0)
                {
                    total = q > 0 ? std::max(0.0, width - (1 - strip)) : std::min(width, strip);
                    elastic = total * height;
                }
                const double sign = q > 0 ? 1 : -1;
                require(std::abs(result[4 * face] - sign * total) < 2e-7, "Wrong condensed swept volume");
                require(std::abs(result[4 * face + 3] - sign * elastic) < 2e-7, "Wrong elastic swept volume");
                const double mass = 10 * elastic + 3 * (total - elastic) + .01 * (strip - total);
                require(std::abs(result[4 * face + 1] - sign * mass) < 3e-6, "Inconsistent swept mass");
            }
        }
    }
    std::cout << "PASS analytic swept partitions, signed fluxes, absent phases and thin slivers\n";
}

void absentLiquid(const fluid::FluidWebGPUContext& context)
{
    FluidGridGPU mixed(24, 24, 1);
    FluidGridGPU original(24, 24, 1);
    configure(mixed, context);
    configure(original, context);
    original.liquid_mixture.components.resize(1);
    auto state = initial(24, 0, false);
    for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
    {
        state.liquid_fraction[i] = state.elastic_fraction[i];
        state.liquid_mass[1][i] = 0;
        state.liquid_mass[2][i] = 0;
        state.gas_mass[0][i] = .01f * (1 - state.liquid_fraction[i]);
    }
    // The same small displacement uses one transport pair on both CFL paths.
    mixed.upload(state);
    state.elastic_fraction.clear();
    state.liquid_mass.resize(1);
    original.upload(state);
    mixed.advect(.001);
    original.advect(.001);
    const auto a = mixed.download();
    const auto b = original.download();
    for (size_t i = 0; i < a.liquid_fraction.size(); ++i)
    {
        require(std::abs(a.elastic_fraction[i] - b.liquid_fraction[i]) < 2e-7, "Zero-water geometry changed");
        require(std::abs(a.liquid_mass[0][i] - b.liquid_mass[0][i]) < 3e-6, "Zero-water mass changed");
        require(std::abs(a.elastic_log_xx[i] - b.elastic_log_xx[i]) < 2e-6, "Zero-water history changed");
    }
    for (size_t i = 0; i < a.u.size(); ++i)
        require(std::abs(a.u[i] - b.u[i]) < 2e-6, "Zero-water momentum changed");
    std::cout << "PASS absent-water reduction against the existing transport path\n";
}

void transport(const fluid::FluidWebGPUContext& context)
{
    for (int solid : {0, 2})
    {
        FluidGridGPU grid(24, 24, 1);
        configure(grid, context, solid);
        const auto before = initial(24, solid, true);
        grid.upload(before);
        for (int i = 0; i < 80; ++i)
        {
            require(grid.project(.1, 2000, 1e-6).accepted(), "Transport projection rejected");
            grid.advect(.1);
            inspect(grid.download(), before, solid);
        }
    }
    std::cout << "PASS three-phase transport, split winners, ingredient order and history confinement\n";
}

void nestedReconstruction(const fluid::FluidWebGPUContext& context)
{
    using namespace fluid::webgpu;
    auto& compute = fluid::detail::FluidFieldAccess::commands(context);
    constexpr int n = 16;
    ElasticLiquidGPU material(compute, n, n);
    auto condensed = compute.field(n, n);
    auto outer = compute.field(n, n, 4);
    auto u = compute.field(n + 1, n);
    auto v = compute.field(n, n + 1);
    std::vector<float> c(n * n);
    std::vector<float> s(n * n);
    std::vector<float> planes(4 * n * n);
    for (int i = 0; i < n * n; ++i)
    {
        s[i] = .2f + .12f * std::sin(float(i) * .17f);
        c[i] = s[i] + (i % 5 == 0 ? 1e-7f : .4f);
    }
    compute.write(condensed, c);
    compute.write(material.fraction, s);
    compute.write(u, std::vector<float>((n + 1) * n));
    compute.write(v, std::vector<float>(n * (n + 1)));
    Kernel plain;
    compute.build(plain, "test selected occupancy reconstruction", 1, {"liquid", "u", "v"},
        R"wgsl(
            fn execute()
            {
                write0(cell(), reconstructInterface(cell()));
            }
        )wgsl",
        std::string(geometry) + std::string(interface_reconstruction));
    Parameters parameters;
    parameters.grid = {n, n, 0, 0};
    parameters.value = {1, .1f, 0, 0};
    compute.run(plain, {&material.interfaces}, {{"liquid", &material.fraction}, {"u", &u}, {"v", &v}}, parameters);
    const auto normals = compute.read(material.interfaces);
    using Point = std::array<double, 2>;
    using Polygon = std::vector<Point>;
    auto clip = [](const Polygon& input, double nx, double ny, double offset)
    {
        Polygon result;
        if (input.empty())
            return result;
        Point previous = input.back();
        double before = nx * previous[0] + ny * previous[1] - offset;
        for (const auto& current : input)
        {
            const double after = nx * current[0] + ny * current[1] - offset;
            if ((before <= 0) != (after <= 0))
            {
                const double t = before / (before - after);
                result.push_back(
                    {previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1])});
            }
            if (after <= 0)
                result.push_back(current);
            previous = current;
            before = after;
        }
        return result;
    };
    auto area = [](const Polygon& polygon)
    {
        double sum = 0;
        for (size_t i = 0; i < polygon.size(); ++i)
        {
            const auto& a = polygon[i];
            const auto& b = polygon[(i + 1) % polygon.size()];
            sum += a[0] * b[1] - a[1] * b[0];
        }
        return std::abs(sum) / 2;
    };
    const Polygon square{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    for (int i = 0; i < n * n; ++i)
    {
        // Nearly parallel and opposing cuts stress the nested offset solve.
        const double angle = i % 2 == 0 ? 1e-6 : 2.3;
        const double nx = normals[4 * i] * std::cos(angle) - normals[4 * i + 1] * std::sin(angle);
        const double ny = normals[4 * i] * std::sin(angle) + normals[4 * i + 1] * std::cos(angle);
        planes[4 * i] = float(nx);
        planes[4 * i + 1] = float(ny);
        double lo = -2;
        double hi = 2;
        for (int iteration = 0; iteration < 60; ++iteration)
        {
            const double middle = .5 * (lo + hi);
            if (area(clip(square, planes[4 * i], planes[4 * i + 1], middle)) < c[i])
                lo = middle;
            else
                hi = middle;
        }
        planes[4 * i + 2] = float(.5 * (lo + hi));
    }
    compute.write(outer, planes);
    compute.run(material.reconstruct, {&material.interfaces},
        {{"liquid", &material.fraction}, {"condensed", &condensed}, {"outer", &outer}, {"u", &u}, {"v", &v}},
        parameters);
    const auto nested = compute.read(material.interfaces);
    for (int i = 0; i < n * n; ++i)
    {
        const auto polygon = clip(square, planes[4 * i], planes[4 * i + 1], planes[4 * i + 2]);
        const double volume = area(clip(polygon, nested[4 * i], nested[4 * i + 1], nested[4 * i + 2]));
        require(std::abs(volume - s[i]) < 2e-6, "Nested reconstruction does not preserve elastic area");
    }
    std::cout << "PASS nested reconstruction against double geometry, near-parallel cuts and thin liquid layers\n";
}

void uniformVelocity(const fluid::FluidWebGPUContext& context)
{
    constexpr int n = 64;
    FluidGridGPU grid(n, n, 1);
    configure(grid, context);
    auto state = initial(n, 0, true);
    std::fill(state.u.begin(), state.u.end(), .5f);
    std::fill(state.v.begin(), state.v.end(), -.25f);
    for (int i = 0; i < n; ++i)
    {
        state.u[i * (n + 1)] = 0;
        state.u[i * (n + 1) + n] = 0;
        state.v[i] = 0;
        state.v[n * n + i] = 0;
    }
    grid.upload(state);
    for (int step = 0; step < 6; ++step)
        grid.advect(.1);
    const auto result = grid.download();
    // The central patch is outside the finite transport stencil's wall domain of influence.
    for (int y = 24; y < 40; ++y)
    {
        for (int x = 24; x < 40; ++x)
        {
            require(std::abs(result.u[y * (n + 1) + x] - .5f) < 2e-6f && std::abs(result.v[y * n + x] + .25f) < 2e-6f,
                "Material transport changed uniform velocity");
        }
    }
    std::cout << "PASS uniform velocity through unequal-density three-phase transport\n";
}

void layoutTransitions(const fluid::FluidWebGPUContext& context)
{
    FluidGridGPU grid(16, 16, 1);
    configure(grid, context);
    bool rejected = false;
    try
    {
        grid.setSurfaceTension(1);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected && grid.getSurfaceTension() == 0, "Mixed surface tension accepted before upload");
    grid.upload(initial(16, 0, true));
    fluid::FluidField acceleration({16, 16, 1, {}, 2}, context);
    acceleration.write(std::vector<float>(16 * 16 * 2, 0));
    grid.setAccelerationField(acceleration);
    fluid::FluidCheckpoint mixed;
    grid.saveCheckpoint(mixed);
    auto bare = initial(16, 0, false);
    bare.liquid_fraction = bare.elastic_fraction;
    bare.elastic_fraction.clear();
    bare.liquid_mass.resize(1);
    for (size_t i = 0; i < bare.liquid_fraction.size(); ++i)
        bare.gas_mass[0][i] = .01f * (1 - bare.liquid_fraction[i]);
    for (int iteration = 0; iteration < 4; ++iteration)
    {
        grid.liquid_mixture.components.resize(1);
        grid.upload(bare);
        fluid::FluidCheckpoint old;
        grid.saveCheckpoint(old);
        require(old.metadata["acceleration_field"] == true, "Layout change discarded external acceleration");
        grid.clearAccelerationField();
        grid.origin = {3, 4};
        grid.saveCheckpoint(old);
        require(grid.loadCheckpoint(mixed), "Mixed layout load failed");
        require(grid.loadCheckpoint(old), "Old layout load with different origin failed");
        fluid::FluidCheckpoint restored;
        grid.saveCheckpoint(restored);
        require(restored.metadata["acceleration_field"] == false && grid.origin.x == 3,
            "Old checkpoint retained acceleration or lost origin");
        require(grid.loadCheckpoint(mixed), "Repeated mixed layout load failed");
        require(grid.step(.001, 2000, 1e-5).accepted(), "Layout transition continuation failed");
    }
    std::cout << "PASS repeated storage-layout changes, origins and acceleration restoration\n";
}

void componentGroups(const fluid::FluidWebGPUContext& context)
{
    FluidGridGPU grid(16, 16, 1);
    configure(grid, context);
    auto state = initial(16, 0, true);
    // Nineteen liquid ingredients cross both four-lane packing and four-group density batches.
    for (int k = 0; k < 16; ++k)
    {
        const auto substance = grid.substances.add({"Additional liquid", {.3f, .4f, .5f}});
        grid.liquid_mixture.components.push_back({substance, 6});
        state.liquid_mass.push_back(state.liquid_mass[2]);
    }
    for (size_t k = 2; k < state.liquid_mass.size(); ++k)
    {
        for (auto& mass : state.liquid_mass[k])
            mass /= 17;
    }
    grid.upload(state);
    for (int step = 0; step < 20; ++step)
    {
        require(grid.project(.1, 2000, 1e-6).accepted(), "Grouped transport projection rejected");
        grid.advect(.1);
    }
    const auto result = grid.download();
    for (size_t k = 0; k < result.liquid_mass.size(); ++k)
        require(std::abs(total(result.liquid_mass[k]) / total(state.liquid_mass[k]) - 1) < 1e-5,
            "Packed group lost component mass");
    for (size_t i = 0; i < result.liquid_fraction.size(); ++i)
    {
        double ordinary = result.liquid_mass[1][i] / 3;
        for (size_t k = 2; k < result.liquid_mass.size(); ++k)
            ordinary += result.liquid_mass[k][i] / 6;
        require(std::abs(ordinary - (result.liquid_fraction[i] - result.elastic_fraction[i])) < 2e-5,
            "Packed group lost phase consistency");
    }
    auto invalid = result;
    invalid.elastic_fraction[0] = 2;
    bool rejected = false;
    try
    {
        grid.upload(invalid);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected && grid.download().elastic_fraction == result.elastic_fraction,
        "Invalid occupancy changed live state");
    std::cout << "PASS multiple ingredient groups and external occupancy validation\n";
}

void coupled(const fluid::FluidWebGPUContext& context)
{
    for (bool implicit : {false, true})
    {
        FluidGridGPU grid(24, 24, 1);
        configure(grid, context);
        grid.gravity_y = 10;
        const auto before = initial(24, 0, false);
        grid.upload(before);
        grid.setImplicitElasticity(implicit);
        for (int i = 0; i < 60; ++i)
        {
            require(grid.step(.02, 2000, 1e-5).accepted(), "Coupled step rejected");
            inspect(grid.download(), before, 0);
        }
        fluid::FluidCheckpoint saved;
        require(grid.saveCheckpoint(saved) && saved.metadata["version"] == 12, "Missing mixed checkpoint");
        FluidGridGPU restored(24, 24, 1);
        restored.setWebGPUContext(context);
        require(restored.loadCheckpoint(saved), "Checkpoint load failed");
        require(restored.download().elastic_fraction == grid.download().elastic_fraction, "Occupancy restore changed");
        for (int i = 0; i < 3; ++i)
        {
            require(grid.step(.02, 2000, 1e-5).accepted() && restored.step(.02, 2000, 1e-5).accepted(),
                "Checkpoint continuation rejected");
            const auto a = grid.download();
            const auto b = restored.download();
            require(a.elastic_fraction == b.elastic_fraction && a.liquid_mass == b.liquid_mass && a.u == b.u,
                "Checkpoint continuation changed");
        }
    }
    std::cout << "PASS explicit/implicit pool coupling and native checkpoint continuation\n";
}
}

int main()
{
    try
    {
        fluid::FluidWebGPUContext context;
        sweptPartition(context);
        nestedReconstruction(context);
        uniformVelocity(context);
        layoutTransitions(context);
        absentLiquid(context);
        transport(context);
        componentGroups(context);
        coupled(context);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL elastic liquid: " << error.what() << '\n';
        return 1;
    }
}
