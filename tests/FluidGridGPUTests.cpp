#include "FluidGrid.h"
#include "FluidTestSetup.h"
#include "FluidGridEditTests.h"
#include "FluidGridGPU.h"
#include "FluidGridGPUCoarseTests.h"
#include "FluidAmbientMaskDisplayTests.h"
#include "FluidGridDisplay.h"
#include "PressureMultigridGPU.h"
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace
{
using fluid::FluidGrid;
using fluid::FluidGridGPU;
using fluid::test::initializeTestMaterials;
using fluid::test::uploadTestMaterials;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Context
{
    SDL_Window* window = nullptr;
    SDL_GLContext context = nullptr;
    Context()
    {
        if (!SDL_Init(SDL_INIT_VIDEO)) throw std::runtime_error(SDL_GetError());
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
        window = SDL_CreateWindow("FluidGridGPU tests", 32, 32, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
        if (!window) throw std::runtime_error(SDL_GetError());
        context = SDL_GL_CreateContext(window);
        if (!context) throw std::runtime_error(SDL_GetError());
        SDL_GL_MakeCurrent(window, context);
        require(gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress) != 0, "GL entry points unavailable");
        SDL_GL_SetSwapInterval(0);
        std::cout << "GPU: " << glGetString(GL_RENDERER) << "\nGL: " << glGetString(GL_VERSION) << '\n';
    }
    ~Context()
    {
        if (context) SDL_GL_DestroyContext(context);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

FluidGridGPU::State pool(int width, int height)
{
    FluidGridGPU::State state;
    state.liquid_fraction.resize(size_t(width) * height);
    for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
    {
        const double surface = x + height * (0.5 + 0.04 * std::cos(3.14159265358979323846 * (x * 3 + 0.5) / width));
        state.liquid_fraction[y * width + x] = float(std::clamp(y + 1.0 - surface, 0.0, 1.0));
    }
    return state;
}

void pressureBalancesGravity(int cpu_cells = 1024)
{
    for (const auto& [w, h] : { std::pair{15, 11}, std::pair{1, 13}, std::pair{13, 1} })
    {
        FluidGridGPU grid(w, h, 4);
        grid.cpu_pressure_cells = cpu_cells;
        FluidGridGPU::State state;
        state.liquid_fraction.assign(w * h, 1);
        uploadTestMaterials(grid, state);
        grid.applyGravity(1.0 / 120);
        auto result = grid.project(1.0 / 120, 2000, 1e-4);
        std::cout << "hydrostatic " << w << 'x' << h << " iterations=" << result.iterations << " divergence=" << result.max_divergence << '\n';
        require(result.converged, "GPU hydrostatic projection failed");
        state = grid.download();
        for (float v : state.v) require(std::abs(v) < 0.005, "GPU hydrostatic velocity failed to settle");
    }
}

void projectionMatchesCPU(int cpu_cells = 1024)
{
    for (const auto& [w, h] : { std::pair{17, 13}, std::pair{128, 128} })
    {
        FluidGrid cpu(w, h, 4);
        auto state = pool(w, h);
        cpu.liquid_fraction.assign(state.liquid_fraction.begin(), state.liquid_fraction.end());
        initializeTestMaterials(cpu);
        cpu.applyGravity(1.0 / 120);
        const auto reference = cpu.project(1.0 / 120, 2000, 1e-8);
        require(reference.converged, "CPU projection reference failed");
        for (auto preconditioner : {FluidGridGPU::PressurePreconditioner::Diagonal, FluidGridGPU::PressurePreconditioner::Multigrid})
        {
            FluidGridGPU gpu(w, h, 4);
            gpu.cpu_pressure_cells = cpu_cells;
            gpu.pressure_preconditioner = preconditioner;
            uploadTestMaterials(gpu, state);
            gpu.applyGravity(1.0 / 120);
            const auto result = gpu.project(1.0 / 120, 2000, 1e-4);
            const auto output = gpu.download();
            double error = 0;
            for (size_t i = 0; i < output.u.size(); ++i) error = std::max(error, std::abs(output.u[i] - cpu.u[i]));
            for (size_t i = 0; i < output.v.size(); ++i) error = std::max(error, std::abs(output.v[i] - cpu.v[i]));
            std::cout << "mixed projection " << w << 'x' << h << " iterations=" << result.iterations
                      << " divergence=" << result.max_divergence << " max_velocity_error=" << error << '\n';
            require(result.converged, "mixed-density GPU projection failed");
            require(error < 0.005, "GPU pressure correction differs from the CPU reference");
        }
    }
}

void zeroTimeAndFailedProjectionAreHandled()
{
    FluidGridGPU grid(8, 8, 1);
    uploadTestMaterials(grid, pool(8, 8));
    const auto before = grid.download();
    require(!grid.step(0).converged, "zero-time step unexpectedly advanced");
    const auto after = grid.download();
    require(before.liquid_fraction == after.liquid_fraction && before.u == after.u && before.v == after.v, "zero-time step changed state");
    grid.applyGravity(0.01);
    require(!grid.project(0.01, 0).converged, "zero-iteration projection unexpectedly converged");
}

void multigridPreservesPCGContract(int cpu_cells = 0)
{
    using namespace fluid::gpu_detail;
    ComputeState guard;
    PassRunner runner;
    for (auto [w, h] : {std::pair{1, 1}, {1, 129}, {131, 1}, {2, 67}, {13, 9}, {65, 37}, {128, 128}, {512, 512}, {513, 257}})
    {
        PressureMultigridGPU multigrid(runner, w, h);
        multigrid.useCPUCoarsePressure(cpu_cells);
        Texture coefficients(w, h, 4), rhs(w, h), stats(1, 1, 4);
        std::vector<float> weights(size_t(w) * h * 4), a(w * h), b(w * h), combined(w * h);
        auto density = [](int x, int y) { return (x / 3 + y / 4) % 2 == 0 ? 0.1f : 100.0f; };
        for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            float* face = weights.data() + (y * w + x) * 4;
            const float center = density(x, y);
            if (x > 0) face[0] = 2.0f / (center + density(x - 1, y));
            if (x + 1 < w) face[1] = 2.0f / (center + density(x + 1, y));
            if (y > 0) face[2] = 2.0f / (center + density(x, y - 1));
            if (y + 1 < h) face[3] = 2.0f / (center + density(x, y + 1));
            const int i = y * w + x;
            a[i] = float(std::sin(i * 0.31) + std::cos(i * 0.073));
            b[i] = float(std::cos(i * 0.17) - std::sin(i * 0.019));
        }
        glBindTexture(GL_TEXTURE_2D, coefficients.id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, weights.data());
        multigrid.rebuild(coefficients);
        auto center = [](auto& values)
        {
            const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            for (auto& value : values) value = typename std::decay_t<decltype(values)>::value_type(value - mean);
        };
        center(a); center(b);
        for (size_t i = 0; i < a.size(); ++i) combined[i] = 1.3f * a[i] - 0.7f * b[i];
        auto apply = [&](const std::vector<float>& input)
        {
            glBindTexture(GL_TEXTURE_2D, rhs.id);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, input.data());
            const float reduction[] = {float(std::accumulate(input.begin(), input.end(), 0.0)), 0, 0, 0};
            glBindTexture(GL_TEXTURE_2D, stats.id);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, reduction);
            const auto pixels = runner.read(multigrid.apply(rhs, stats));
            std::vector<double> result(input.size());
            for (size_t i = 0; i < result.size(); ++i) result[i] = pixels[4 * i];
            center(result);
            return result;
        };
        const auto ma = apply(a), mb = apply(b), mc = apply(combined), repeat = apply(a);
        require(ma == repeat, "GPU multigrid retained an old correction");
        double error = 0, scale = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            error = std::max(error, std::abs(mc[i] - (1.3 * ma[i] - 0.7 * mb[i])));
            scale = std::max(scale, std::abs(mc[i]));
        }
        require(error < 5e-4 * std::max(1.0, scale), "GPU multigrid is not linear within float accuracy");
        const double a_mb = std::inner_product(a.begin(), a.end(), mb.begin(), 0.0);
        const double b_ma = std::inner_product(b.begin(), b.end(), ma.begin(), 0.0);
        const double energy_a = std::inner_product(a.begin(), a.end(), ma.begin(), 0.0);
        const double energy_b = std::inner_product(b.begin(), b.end(), mb.begin(), 0.0);
        require(std::abs(a_mb - b_ma) < 5e-4 * std::max(1.0, std::sqrt(energy_a * energy_b)), "GPU multigrid is not symmetric within float accuracy");
        require(w * h == 1 ? energy_a == 0 : energy_a > 0 && energy_b > 0, "GPU multigrid is not positive");
        std::fill(a.begin(), a.end(), 7.0f);
        const auto constant = apply(a);
        double constant_error = 0.0;
        for (double value : constant) constant_error = std::max(constant_error, std::abs(value));
        std::cout << "multigrid " << w << 'x' << h << " cpu_cells=" << cpu_cells << " linearity=" << error / std::max(1.0, scale)
                  << " constant_error=" << constant_error << " scale=" << scale << '\n';
        require(constant_error <= 5e-6 * std::max(1.0, scale), "GPU multigrid did not remove its nullspace within float accuracy");
    }
}

void largeMixedPressureConvergesWithinDefaultBudget()
{
    FluidGridGPU grid(512, 512, 4);
    auto state = pool(512, 512);
    uploadTestMaterials(grid, state);
    grid.applyGravity(1.0 / 120);
    const auto result = grid.project(1.0 / 120);
    require(result.converged, "512x512 GPU multigrid exhausted the default pressure budget");
    uploadTestMaterials(grid, state);
    grid.applyGravity(1.0 / 120);
    const auto limited = grid.project(1.0 / 120, 1, 1e-6);
    require(!limited.converged && limited.iterations == 1, "GPU pressure exceeded or misreported a one-iteration budget");
}

void cachedVelocityStatsRemainCurrent()
{
    FluidGridGPU cached(48, 32, 4), refreshed(48, 32, 4);
    const auto initial = pool(48, 32);
    uploadTestMaterials(cached, initial);
    uploadTestMaterials(refreshed, initial);
    for (int tick = 0; tick < 20; ++tick)
    {
        // A zero force leaves velocities unchanged while forcing the uncached CFL path.
        refreshed.applyGravity(0.0);
        const double dt = tick < 10 ? 1.0 / 120.0 : 0.25;
        require(cached.step(dt, 2000, 1e-3).converged && refreshed.step(dt, 2000, 1e-3).converged, "CFL cache comparison failed to converge");
        const auto a = cached.download(), b = refreshed.download();
        require(a.liquid_fraction == b.liquid_fraction && a.u == b.u && a.v == b.v && a.pressure == b.pressure,
                "cached velocity statistics changed transport or pressure");
    }
    const auto restart = cached.download();
    uploadTestMaterials(cached, restart);
    uploadTestMaterials(refreshed, restart);
    refreshed.applyGravity(0.0);
    cached.advect(0.1);
    refreshed.advect(0.1);
    const auto a = cached.download(), b = refreshed.download();
    require(a.liquid_fraction == b.liquid_fraction && a.u == b.u && a.v == b.v, "upload retained stale velocity statistics");
}

template<typename Grid>
void commonInterfaceAndDisplay(int cpu_cells = 1024)
{
    Grid grid(9, 7, 2.5);
    if constexpr (std::is_same_v<Grid, FluidGridGPU>) grid.cpu_pressure_cells = cpu_cells;
    typename Grid::State initial;
    initial.liquid_fraction.resize(9 * 7);
    initial.liquid_fraction[2 * 9 + 3] = typename Grid::Scalar(0.375);
    uploadTestMaterials(grid, initial);
    const auto before = grid.download();
    require(before.liquid_fraction == initial.liquid_fraction && before.u.size() == 10 * 7 && before.v.size() == 9 * 8 && before.pressure.size() == 9 * 7,
            "common state transfer has incorrect field sizes or contents");

    fluid::FluidGridDisplay display;
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    auto checkTexture = [&](float expected)
    {
        const auto texture = display.texture(grid);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        require(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "common display texture is incomplete");
        float pixel[4];
        glReadPixels(3, 2, 1, 1, GL_RGBA, GL_FLOAT, pixel);
        require(pixel[0] == 0.0f && pixel[1] == 0.0f && pixel[2] == expected && pixel[3] == 1.0f, "display returned stale or incorrect component colours");
    };
    checkTexture(0.375f);
    require(before.liquid_fraction == grid.download().liquid_fraction, "display mutated the material state");
    auto edit = grid.beginEdit();
    edit.setCell(3, 2, fluid::Phase::Liquid, grid.liquid_mixture.components[0].element);
    edit.commit();
    checkTexture(1.0f);
    edit.setCell(3, 2, fluid::Phase::Gas, grid.gas_mixture.components[0].element);
    edit.commit();
    checkTexture(0.0f);
    require(grid.step(1.0 / 120, 2000, 1e-3).converged, "common physics call failed");
    grid.destroy();
    grid.upload({});
    checkTexture(0.0f);
    const auto reset = grid.download();
    for (const auto* field : { &reset.liquid_fraction, &reset.u, &reset.v, &reset.pressure })
        for (auto value : *field) require(value == 0, "empty upload failed to reset a field");
    for (const auto* phase : { &reset.liquid_mass, &reset.gas_mass })
        for (const auto& field : *phase) for (auto value : field) require(value == 0, "empty upload failed to reset component masses");
    grid.liquid_mixture.components.clear();
    grid.gas_mixture.components.clear();
    grid.upload({});
    checkTexture(0.0f);
    display.destroy();
    grid.destroy();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &framebuffer);
}

void graphicsStateIsPreserved(int w = 8, int h = 8, int cpu_cells = 1024)
{
    GLuint texture, framebuffer, buffers[2];
    glGenTextures(1, &texture);
    glGenFramebuffers(1, &framebuffer);
    glGenBuffers(2, buffers);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, buffers[0]);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffers[1]);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 19);
    glPixelStorei(GL_PACK_ROW_LENGTH, 23);
    glViewport(7, 9, 11, 13);
    glEnable(GL_BLEND);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 0, 0);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    {
        FluidGridGPU gpu(w, h, 1);
        gpu.cpu_pressure_cells = cpu_cells;
        uploadTestMaterials(gpu, pool(w, h));
        auto edit = gpu.beginEdit();
        edit.setCell(0, h - 1, fluid::Phase::Liquid, gpu.liquid_mixture.components[0].element);
        edit.commit();
        require(gpu.step(1.0 / 120, 2000, 1e-3).converged, "graphics state changed the numerical passes");
        const auto state = gpu.download();
        require(state.liquid_fraction[(h - 1) * w] > 0.9f, "pixel buffers or color mask corrupted GPU state");
        GLint value, viewport[4];
        GLboolean mask[4];
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &value);
        require(GLuint(value) == framebuffer, "draw framebuffer was not restored");
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &value);
        require(GLuint(value) == framebuffer, "read framebuffer was not restored");
        glGetIntegerv(GL_ACTIVE_TEXTURE, &value);
        require(value == GL_TEXTURE3, "active texture was not restored");
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &value);
        require(GLuint(value) == texture, "texture binding was not restored");
        glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &value);
        require(GLuint(value) == buffers[0], "pack buffer was not restored");
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &value);
        require(GLuint(value) == buffers[1], "unpack buffer was not restored");
        glGetIntegerv(GL_UNPACK_ROW_LENGTH, &value);
        require(value == 19, "unpack layout was not restored");
        glGetIntegerv(GL_PACK_ROW_LENGTH, &value);
        require(value == 23, "pack layout was not restored");
        glGetIntegerv(GL_VIEWPORT, viewport);
        require(viewport[0] == 7 && viewport[1] == 9 && viewport[2] == 11 && viewport[3] == 13, "viewport was not restored");
        glGetBooleanv(GL_COLOR_WRITEMASK, mask);
        require(!mask[0] && !mask[1] && !mask[2] && !mask[3], "color mask was not restored");
        require(glIsEnabled(GL_BLEND) && glIsEnabled(GL_SCISSOR_TEST), "drawing capabilities were not restored");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glDeleteFramebuffers(1, &framebuffer);
    glDeleteTextures(1, &texture);
    glDeleteBuffers(2, buffers);
}

void transportMatchesCPU()
{
    for (const auto velocity : { bl::DVec2{1, 0.75}, bl::DVec2{-1, 0.75}, bl::DVec2{1, -0.75}, bl::DVec2{-1, -0.75} })
    {
        FluidGrid cpu(48, 48, 1);
        FluidGridGPU gpu(48, 48, 1);
        FluidGridGPU::State state;
        state.liquid_fraction.resize(48 * 48);
        state.u.assign(49 * 48, float(velocity.x));
        state.v.assign(48 * 49, float(velocity.y));
        for (int y = 18; y < 30; ++y)
        for (int x = 18; x < 30; ++x)
            state.liquid_fraction[y * 48 + x] = float(std::clamp(6.0 - std::hypot(x - 23.5, y - 23.5), 0.0, 1.0));
        cpu.liquid_fraction.assign(state.liquid_fraction.begin(), state.liquid_fraction.end());
        cpu.u.assign(state.u.begin(), state.u.end());
        cpu.v.assign(state.v.begin(), state.v.end());
        initializeTestMaterials(cpu);
        const double initial = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.0);
        uploadTestMaterials(gpu, state);
        cpu.advect(1);
        gpu.advect(1);
        state = gpu.download();
        double fraction_error = 0, velocity_error = 0;
        for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
            fraction_error = std::max(fraction_error, std::abs(state.liquid_fraction[i] - cpu.liquid_fraction[i]));
        for (int y = 8; y < 40; ++y)
        for (int x = 8; x < 40; ++x)
        {
            velocity_error = std::max(velocity_error, std::abs(state.u[y * 49 + x] - velocity.x));
            velocity_error = std::max(velocity_error, std::abs(state.v[y * 48 + x] - velocity.y));
        }
        const double drift = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.0) - initial;
        std::cout << "diagonal " << velocity.x << ',' << velocity.y << " fraction_error=" << fraction_error
                  << " velocity_error=" << velocity_error << " volume_drift=" << drift << '\n';
        require(fraction_error < 2e-6, "GPU geometric transport differs from CPU");
        require(velocity_error < 2e-4, "GPU density changes altered uniform motion");
        require(std::abs(drift) < initial * 1e-6, "GPU uniform transport lost liquid");
    }
}

void transportPreservesPureMaterial()
{
    constexpr int size = 19;
    FluidGridGPU grid(size, size, 4);
    FluidGridGPU::State state;
    state.liquid_fraction.assign(size * size, 1.0f);
    state.u.resize((size + 1) * size);
    state.v.resize(size * (size + 1));
    // Differences of a streamfunction give a closed, circulating flow without a pressure solve.
    auto stream = [](int x, int y) {
        return x == 0 || y == 0 || x == size || y == size ? 0.0f : float(30 * std::sin(x * 3.141592653589793 / size) * std::sin(y * 3.141592653589793 / size));
    };
    for (int y = 0; y < size; ++y)
    for (int x = 0; x <= size; ++x) state.u[grid.uIndex(x, y)] = stream(x, y + 1) - stream(x, y);
    for (int y = 0; y <= size; ++y)
    for (int x = 0; x < size; ++x) state.v[grid.vIndex(x, y)] = stream(x, y) - stream(x + 1, y);
    uploadTestMaterials(grid, state);
    grid.advect(0.137);
    for (float value : grid.download().liquid_fraction)
        require(value == 1.0f, "GPU transport accumulated roundoff in a pure material");
}

void transportKeepsStraightInterfacesSharp()
{
    FluidGridGPU gpu(40, 8, 1);
    FluidGridGPU::State state;
    state.liquid_fraction.resize(40 * 8);
    state.u.assign(41 * 8, 1);
    for (int y = 0; y < 8; ++y)
    for (int x = 8; x < 16; ++x) state.liquid_fraction[y * 40 + x] = 1;
    uploadTestMaterials(gpu, state);
    gpu.advect(8);
    state = gpu.download();
    for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 40; ++x)
        require(std::abs(state.liquid_fraction[y * 40 + x] - (x >= 16 && x < 24 ? 1.0 : 0.0)) < 1e-6,
                "GPU whole-cell translation blurred a straight interface");
}

void transportKeepsShallowInterfacesStraight()
{
    double largest_error = 0.0;
    for (bool vertical : {false, true})
    for (bool reverse_phase : {false, true})
    for (double slope : {0.0, -0.1, -0.05, 0.05, 0.1})
    for (double speed : {-1.0, 1.0})
    {
        FluidGridGPU grid(96, 96, 1.0);
        FluidGridGPU::State state;
        state.liquid_fraction.resize(96 * 96);
        auto& flow = vertical ? state.v : state.u;
        flow.assign(96 * 97, float(speed));

        auto exactFraction = [&](double along, double depth)
        {
            // Integrate the clipped liquid height across a cell under a known straight line.
            auto integral = [](double h) { return h <= 0.0 ? 0.0 : h >= 1.0 ? h - 0.5 : 0.5 * h * h; };
            const double a = depth + 1.0 - (48.25 + slope * (along - 48.0));
            const double b = a - slope;
            const double fraction = slope == 0.0 ? std::clamp(a, 0.0, 1.0) : (integral(a) - integral(b)) / slope;
            return reverse_phase ? 1.0 - fraction : fraction;
        };
        for (int y = 0; y < grid.height; ++y)
        for (int x = 0; x < grid.width; ++x)
            state.liquid_fraction[y * grid.width + x] = float(vertical ? exactFraction(y, x) : exactFraction(x, y));
        uploadTestMaterials(grid, state);

        for (int step = 0; step < 20; ++step)
            grid.advect(0.25);
        state = grid.download();

        // Inspect the interior, where uniform translation is unaffected by the closed walls.
        for (int y = 24; y < 72; ++y)
        for (int x = 24; x < 72; ++x)
        {
            const double expected = vertical ? exactFraction(y - 5.0 * speed, x) : exactFraction(x - 5.0 * speed, y);
            largest_error = std::max(largest_error, std::abs(state.liquid_fraction[y * grid.width + x] - expected));
        }
    }
    std::cout << "GPU shallow-interface fraction error=" << largest_error << '\n';
    require(largest_error < 1e-5, "GPU uniform transport corrugated a shallow straight interface");
}

void nonuniformTransportMatchesCPU()
{
    const int w = 32, h = 24;
    auto state = pool(w, h);
    state.u.resize((w + 1) * h);
    state.v.resize(w * (h + 1));
    // differences of one stream function give a closed, divergence-free swirling flow
    auto stream = [&](int x, int y) { return 80.0 * std::sin(3.141592653589793 * x / w) * std::sin(3.141592653589793 * y / h); };
    for (int y = 0; y < h; ++y)
    for (int x = 0; x <= w; ++x) state.u[y * (w + 1) + x] = float(stream(x, y + 1) - stream(x, y));
    for (int y = 0; y <= h; ++y)
    for (int x = 0; x < w; ++x) state.v[y * w + x] = float(stream(x, y) - stream(x + 1, y));
    FluidGrid cpu(w, h, 1);
    cpu.liquid_fraction.assign(state.liquid_fraction.begin(), state.liquid_fraction.end());
    cpu.u.assign(state.u.begin(), state.u.end());
    cpu.v.assign(state.v.begin(), state.v.end());
    initializeTestMaterials(cpu);
    FluidGridGPU gpu(w, h, 1);
    uploadTestMaterials(gpu, state);
    cpu.advect(0.1);
    gpu.advect(0.1);
    state = gpu.download();
    double fraction_error = 0, velocity_error = 0;
    for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
        fraction_error = std::max(fraction_error, std::abs(state.liquid_fraction[i] - cpu.liquid_fraction[i]));
    for (size_t i = 0; i < state.u.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.u[i] - cpu.u[i]));
    for (size_t i = 0; i < state.v.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.v[i] - cpu.v[i]));
    std::cout << "nonuniform transport fraction_error=" << fraction_error << " velocity_error=" << velocity_error << '\n';
    require(fraction_error < 2e-6 && velocity_error < 0.005, "GPU nonuniform momentum transport differs from CPU");
}

void fullStepsMatchCPU()
{
    FluidGrid cpu(32, 24, 4);
    FluidGridGPU gpu(32, 24, 4);
    auto state = pool(32, 24);
    cpu.liquid_fraction.assign(state.liquid_fraction.begin(), state.liquid_fraction.end());
    initializeTestMaterials(cpu);
    uploadTestMaterials(gpu, state);
    double fraction_error = 0, velocity_error = 0;
    for (int tick = 0; tick < 20; ++tick)
    {
        require(cpu.step(1.0 / 120, 2000, 1e-8).converged, "CPU step reference failed");
        require(gpu.step(1.0 / 120, 2000, 1e-3).converged, "GPU comparison step failed");
        state = gpu.download();
        for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
            fraction_error = std::max(fraction_error, std::abs(state.liquid_fraction[i] - cpu.liquid_fraction[i]));
        for (size_t i = 0; i < state.u.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.u[i] - cpu.u[i]));
        for (size_t i = 0; i < state.v.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.v[i] - cpu.v[i]));
    }
    std::cout << "20 full steps fraction_error=" << fraction_error << " velocity_error=" << velocity_error << '\n';
    require(fraction_error < 2e-4 && velocity_error < 0.01, "GPU full steps differ from CPU reference");
}

void tinyFractionsRemainFinite()
{
    for (float amount : { 1e-8f, 1e-20f, 1e-35f })
    {
        FluidGridGPU gpu(8, 8, 1);
        FluidGridGPU::State state;
        state.liquid_fraction.resize(64);
        state.liquid_fraction[3 * 8 + 3] = amount;
        state.liquid_fraction[3 * 8 + 4] = amount * 0.5f;
        state.liquid_fraction[4 * 8 + 3] = amount * 0.25f;
        state.u.assign(9 * 8, 1);
        state.v.assign(8 * 9, 0.75f);
        uploadTestMaterials(gpu, state);
        gpu.advect(0.25);
        state = gpu.download();
        for (const auto* field : { &state.liquid_fraction, &state.u, &state.v })
            for (float value : *field) require(std::isfinite(value), "tiny liquid fractions produced non-finite transport");
    }
    // this near-empty remnant reproduced a float 0/0 during the sloshing scene
    FluidGridGPU gpu(8, 8, 4);
    FluidGridGPU::State state;
    state.liquid_fraction.resize(64);
    state.liquid_fraction[3 * 8 + 3] = 1.654361225e-24f;
    state.liquid_fraction[3 * 8 + 4] = 1.051768366e-22f;
    state.liquid_fraction[4 * 8 + 3] = 6.335743373e-20f;
    state.u.assign(9 * 8, -28.16068459f);
    state.v.assign(8 * 9, 0.01431973279f);
    uploadTestMaterials(gpu, state);
    gpu.advect(1.0 / 120);
    state = gpu.download();
    for (const auto* field : { &state.liquid_fraction, &state.u, &state.v })
        for (float value : *field) require(std::isfinite(value), "near-empty sloshing cell produced non-finite transport");
}

// Vary ingredient proportions without adding extra liquid/gas interfaces.
template<class Grid>
typename Grid::State componentPool(Grid& grid, int liquid_count, int gas_count, bool varied_density = true)
{
    using Scalar = typename Grid::Scalar;
    const auto water = grid.elements.add({"Water", {0.1f, 0.75f, 0.95f}});
    const auto red = grid.elements.add({"Red", {0.95f, 0.25f, 0.15f}});
    const auto air = grid.elements.add({"Air", {0.015f, 0.025f, 0.06f}});
    const auto cloud = grid.elements.add({"Cloud", {0, 1, 1}});
    for (int k = 0; k < liquid_count; ++k)
        grid.liquid_mixture.components.push_back({k % 2 ? red : water, varied_density && k == 1 ? 200.0 : 100.0});
    for (int k = 0; k < gas_count; ++k)
        grid.gas_mixture.components.push_back({k == 0 ? air : k == 1 ? cloud : water, varied_density && k > 1 ? 0.2 : 0.1});
    typename Grid::State state;
    auto fractions = pool(grid.width, grid.height).liquid_fraction;
    if (liquid_count == 0) std::fill(fractions.begin(), fractions.end(), 0.0f);
    if (gas_count == 0) std::fill(fractions.begin(), fractions.end(), 1.0f);
    state.liquid_fraction.assign(fractions.begin(), fractions.end());
    const double volume = grid.cell_size * grid.cell_size;
    auto seed = [&](const auto& mixture, auto& fields, bool liquid)
    {
        const size_t count = mixture.components.size();
        fields.assign(count, std::vector<Scalar>(fractions.size()));
        for (size_t i = 0; i < fractions.size(); ++i)
        {
            std::vector<double> weights(count);
            double sum = 0;
            for (size_t k = 0; k < count; ++k)
            {
                // Include an identically empty ingredient when the fixture has room.
                weights[k] = varied_density && count > 2 && k + 1 == count ? 0.0 :
                    1.0 + 0.8 * std::sin(0.17 * (i % grid.width) + 0.11 * (i / grid.width) + k * 1.7);
                sum += weights[k];
            }
            for (size_t k = 0; k < count; ++k)
                fields[k][i] = Scalar(mixture.components[k].density * volume *
                    (liquid ? fractions[i] : 1.0 - fractions[i]) * weights[k] / sum);
        }
    };
    seed(grid.liquid_mixture, state.liquid_mass, true);
    seed(grid.gas_mixture, state.gas_mass, false);
    return state;
}

struct ComponentErrors { double volume = 0, drift = 0, negative = 0; };
ComponentErrors checkComponents(const FluidGridGPU& grid, const FluidGridGPU::State& state,
    const FluidGridGPU::State& initial, double volume_tolerance = 5e-6)
{
    ComponentErrors error;
    auto check = [&](const auto& mixture, const auto& fields, const auto& original, bool liquid)
    {
        std::vector<double> fractions(state.liquid_fraction.size());
        for (size_t k = 0; k < fields.size(); ++k)
        {
            const double scale = mixture.components[k].density * grid.cell_size * grid.cell_size;
            const double before = std::accumulate(original[k].begin(), original[k].end(), 0.0);
            const double after = std::accumulate(fields[k].begin(), fields[k].end(), 0.0);
            if (before == 0) require(after == 0, "an empty component gained mass");
            else error.drift = std::max(error.drift, std::abs(after - before) / before);
            for (size_t i = 0; i < fractions.size(); ++i)
            {
                require(std::isfinite(fields[k][i]), "non-finite component mass");
                const double fraction = fields[k][i] / scale;
                fractions[i] += fraction;
                error.negative = std::max(error.negative, -fraction);
            }
        }
        for (size_t i = 0; i < fractions.size(); ++i)
            error.volume = std::max(error.volume, std::abs(fractions[i] - (liquid ? state.liquid_fraction[i] : 1.0 - state.liquid_fraction[i])));
    };
    check(grid.liquid_mixture, state.liquid_mass, initial.liquid_mass, true);
    check(grid.gas_mixture, state.gas_mass, initial.gas_mass, false);
    // Check local volume accuracy separately from conservation of each ingredient's total mass.
    require(error.volume < volume_tolerance && error.drift < 1e-4 && error.negative < 2e-5, "component mass or phase volume invariant failed");
    return error;
}

void componentsMatchCPU()
{
    for (auto [liquids, gases] : {std::pair{1, 0}, std::pair{0, 1}, std::pair{2, 2}, std::pair{3, 4}, std::pair{9, 9}})
    {
        FluidGridGPU gpu(31, 23, 2.5);
        FluidGrid cpu(31, 23, 2.5);
        auto initial = componentPool(gpu, liquids, gases);
        auto reference = componentPool(cpu, liquids, gases);
        // Start from exactly the same representable masses, so the comparison isolates arithmetic.
        for (size_t k = 0; k < initial.liquid_mass.size(); ++k)
            reference.liquid_mass[k].assign(initial.liquid_mass[k].begin(), initial.liquid_mass[k].end());
        for (size_t k = 0; k < initial.gas_mass.size(); ++k)
            reference.gas_mass[k].assign(initial.gas_mass[k].begin(), initial.gas_mass[k].end());
        gpu.upload(initial);
        cpu.upload(reference);
        const auto roundtrip = gpu.download();
        require(roundtrip.liquid_mass == initial.liquid_mass && roundtrip.gas_mass == initial.gas_mass, "packed component upload/download changed mass");
        double mass_error = 0, velocity_error = 0;
        for (int tick = 0; tick < 20; ++tick)
        {
            require(cpu.step(1.0 / 120, 2000, 1e-8).converged, "CPU component reference failed");
            require(gpu.step(1.0 / 120, 2000, 1e-3).converged, "GPU component projection failed");
            const auto state = gpu.download();
            checkComponents(gpu, state, initial);
            auto compare = [&](const auto& a, const auto& b, const auto& mixture) {
                for (size_t k = 0; k < a.size(); ++k)
                    for (size_t i = 0; i < a[k].size(); ++i)
                        mass_error = std::max(mass_error, std::abs(a[k][i] - b[k][i]) / (mixture.components[k].density * 6.25));
            };
            compare(state.liquid_mass, cpu.liquid_mass, cpu.liquid_mixture);
            compare(state.gas_mass, cpu.gas_mass, cpu.gas_mixture);
            for (size_t i = 0; i < state.u.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.u[i] - cpu.u[i]));
            for (size_t i = 0; i < state.v.size(); ++i) velocity_error = std::max(velocity_error, std::abs(state.v[i] - cpu.v[i]));
        }
        std::cout << "components=" << liquids + gases << " mass_fraction_error=" << mass_error << " velocity_error=" << velocity_error << '\n';
        require(mass_error < 2e-4 && velocity_error < 0.01, "GPU components differ from CPU reference");
        fluid::FluidGridDisplay cpu_display, gpu_display;
        const auto gpu_texture = gpu_display.texture(gpu);
        const auto cpu_texture = cpu_display.texture(cpu);
        fluid::gpu_detail::ComputeState guard;
        GLuint framebuffer;
        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        std::vector<float> a(31 * 23 * 4), b(a.size());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gpu_texture, 0);
        glReadPixels(0, 0, 31, 23, GL_RGBA, GL_FLOAT, a.data());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cpu_texture, 0);
        glReadPixels(0, 0, 31, 23, GL_RGBA, GL_FLOAT, b.data());
        for (size_t i = 0; i < a.size(); ++i) require(std::abs(a[i] - b[i]) < 5e-4, "GPU catalogue colors differ from CPU");
        glDeleteFramebuffers(1, &framebuffer);
        cpu_display.destroy();
    }
}

void vanishingSliversCannotExportExcessMass()
{
    // This develops tiny liquid remnants that previously produced negative density at tick 120.
    FluidGridGPU grid(512, 512, 4);
    const auto initial = componentPool(grid, 2, 2, false);
    grid.upload(initial);
    for (int tick = 0; tick < 140; ++tick)
        require(grid.step(1.0 / 120, 2000, 1e-3).converged, "vanishing donor sliver destabilized component transport");
    checkComponents(grid, grid.download(), initial);
}

void componentBenchmark(int steps, int width, int height, int count)
{
    FluidGridGPU grid(width, height, 4);
    grid.upload(componentPool(grid, count / 2, count - count / 2, false));
    glFinish();
    const auto started = std::chrono::steady_clock::now();
    long long iterations = 0;
    for (int tick = 0; tick < steps; ++tick)
    {
        const auto result = grid.step(1.0 / 120, 2000, 1e-3);
        if (!result.converged)
            std::cout << "benchmark failure tick=" << tick + 1 << " iterations=" << result.iterations << " divergence=" << result.max_divergence << '\n';
        require(result.converged, "component benchmark failed to converge");
        iterations += result.iterations;
    }
    glFinish();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "component_benchmark grid=" << width << 'x' << height << " components=" << count << " steps=" << steps
              << " seconds=" << seconds << " fps=" << steps / (6 * seconds) << " mean_iterations=" << double(iterations) / steps << '\n';
}

void benchmark(int steps)
{
    const auto initial = pool(128, 128);
    auto time = [&](auto& grid, const char* name, double tolerance)
    {
        const auto start = std::chrono::steady_clock::now();
        long long iterations = 0;
        for (int tick = 0; tick < steps; ++tick)
        {
            auto result = grid.step(1.0 / 120, 2000, tolerance);
            require(result.converged, "benchmark projection failed");
            iterations += result.iterations;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << name << " steps=" << steps << " tolerance=" << tolerance << " seconds=" << seconds
                  << " frames_per_second=" << (steps / 6.0) / seconds << " mean_iterations=" << double(iterations) / steps << '\n';
    };
    for (double tolerance : { 1e-8, 1e-3 })
    {
        FluidGrid cpu(128, 128, 4);
        cpu.liquid_fraction.assign(initial.liquid_fraction.begin(), initial.liquid_fraction.end());
        initializeTestMaterials(cpu);
        time(cpu, "CPU", tolerance);
    }
    FluidGridGPU gpu(128, 128, 4);
    uploadTestMaterials(gpu, initial);
    time(gpu, "GPU", 1e-3);
}

void editBenchmark()
{
    FluidGridGPU grid(1024, 1024, 1);
    grid.upload(componentPool(grid, 2, 2, false));
    const auto water = grid.liquid_mixture.components[0].element;
    auto edit = grid.beginEdit();
    edit.reserve(3600);
    auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    for (int drops : {1, 10, 100})
    {
        std::vector<double> staging, commits;
        for (int repeat = 0; repeat < 25; ++repeat)
        {
            glFinish();
            const auto start = std::chrono::steady_clock::now();
            for (int drop = 0; drop < drops; ++drop)
            for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 6; ++x)
                edit.setCell(4 + (drop % 10) * 100 + x, 4 + (drop / 10) * 100 + y, fluid::Phase::Liquid, water);
            const auto queued = std::chrono::steady_clock::now();
            edit.commit();
            glFinish();
            const auto finished = std::chrono::steady_clock::now();
            if (repeat >= 5)
            {
                staging.push_back(std::chrono::duration<double, std::micro>(queued - start).count());
                commits.push_back(std::chrono::duration<double, std::micro>(finished - queued).count());
            }
        }
        std::cout << "edit drops=" << drops << " cells=" << drops * 36 << " texture_bytes=" << drops * 36 * 20
                  << " staging_us=" << median(staging) << " commit_gpu_complete_us=" << median(commits) << '\n';
    }
    std::vector<double> full_transfers;
    for (int repeat = 0; repeat < 7; ++repeat)
    {
        glFinish();
        const auto start = std::chrono::steady_clock::now();
        const auto state = grid.download();
        grid.upload(state);
        glFinish();
        if (repeat >= 2)
            full_transfers.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::cout << "full_state_round_trip_ms=" << median(full_transfers) << '\n';
}

void closedPool(int steps, int size, double tolerance, int count)
{
    FluidGridGPU grid(size, size, 4);
    auto state = componentPool(grid, count / 2, count - count / 2);
    grid.upload(state);
    const auto initial = state;
    ComponentErrors component_errors;
    const double initial_volume = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.0);
    // use the MAC faces' dual-cell masses for kinetic energy, and the box bottom as zero potential
    auto energy = [&](const FluidGridGPU::State& sample)
    {
        auto density = [&](int x, int y) {
            double mass = 0; for (const auto* phase : { &sample.liquid_mass, &sample.gas_mass }) for (const auto& component : *phase) mass += component[y * size + x];
            return mass / (grid.cell_size * grid.cell_size);
        };
        double total = 0;
        for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
        {
            total += density(x, y) * grid.gravity_y * grid.cell_size * (size - y - 0.5);
            if (x > 0)
            {
                double u = sample.u[y * (size + 1) + x];
                total += 0.25 * (density(x - 1, y) + density(x, y)) * u * u;
            }
            if (y > 0)
            {
                double v = sample.v[y * size + x];
                total += 0.25 * (density(x, y - 1) + density(x, y)) * v * v;
            }
        }
        return total * grid.cell_size * grid.cell_size;
    };
    const double initial_energy = energy(grid.download());
    double max_energy_ratio = 1;
    double largest_drift = 0, largest_divergence = 0, min_fraction = 1, max_fraction = 0, max_speed = 0;
    long long iterations = 0;
    double solve_seconds = 0;
    const auto started = std::chrono::steady_clock::now();
    for (int tick = 1; tick <= steps; ++tick)
    {
        const auto solve_start = std::chrono::steady_clock::now();
        const auto result = grid.step(1.0 / 120, 2000, tolerance);
        solve_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
        iterations += result.iterations;
        largest_divergence = std::max(largest_divergence, result.max_divergence);
        if (!result.converged)
        {
            std::cout << "pool failure tick=" << tick << " iterations=" << result.iterations << " divergence=" << result.max_divergence << '\n';
            const auto failed = grid.download();
            for (const auto& [name, field] : { std::pair{"liquid", &failed.liquid_fraction}, std::pair{"u", &failed.u},
                                             std::pair{"v", &failed.v}, std::pair{"pressure", &failed.pressure} })
                for (size_t i = 0; i < field->size(); ++i)
                    if (!std::isfinite((*field)[i])) { std::cout << "first non-finite " << name << " index=" << i << '\n'; break; }
        }
        require(result.converged, "GPU pool projection failed");
        state = grid.download();
        // Many CFL splits accumulate float mismatch between mass and independent phase geometry.
        // Long-run quality budget: 0.2% of one cell, with the same stricter global mass check.
        const auto errors = checkComponents(grid, state, initial, 2e-3);
        component_errors.volume = std::max(component_errors.volume, errors.volume);
        component_errors.drift = std::max(component_errors.drift, errors.drift);
        component_errors.negative = std::max(component_errors.negative, errors.negative);
        const double volume = std::accumulate(state.liquid_fraction.begin(), state.liquid_fraction.end(), 0.0);
        largest_drift = std::max(largest_drift, std::abs(volume - initial_volume) / initial_volume);
        for (float f : state.liquid_fraction)
        {
            min_fraction = std::min(min_fraction, double(f));
            max_fraction = std::max(max_fraction, double(f));
            require(std::isfinite(f) && f >= -2e-5 && f <= 1.0 + 2e-5, "GPU fraction left [0,1]");
        }
        for (const auto* field : { &state.u, &state.v, &state.pressure })
            for (float value : *field) require(std::isfinite(value), "GPU pool has non-finite state");
        for (const auto* field : { &state.u, &state.v })
            for (float value : *field) max_speed = std::max(max_speed, double(std::abs(value)));
        for (int y = 0; y < size; ++y)
            require(state.u[y * (size + 1)] == 0 && state.u[y * (size + 1) + size] == 0, "GPU side wall leaked");
        for (int x = 0; x < size; ++x)
            require(state.v[x] == 0 && state.v[size * size + x] == 0, "GPU top or bottom wall leaked");
        require(largest_drift < 1e-4, "GPU pool lost more than 0.01 percent of its liquid");
        const double energy_ratio = energy(state) / initial_energy;
        max_energy_ratio = std::max(max_energy_ratio, energy_ratio);
        require(max_energy_ratio < 1.02, "GPU pool energy grew more than two percent above its initial value");
        if (tick <= 3 || tick % 120 == 0 || tick == steps)
        {
            double center_x = 0, center_y = 0;
            for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
            {
                center_x += state.liquid_fraction[y * size + x] * (x + 0.5) * grid.cell_size;
                center_y += state.liquid_fraction[y * size + x] * (y + 0.5) * grid.cell_size;
            }
            std::cout << "pool tick=" << tick << " time=" << tick / 120.0 << " max_divergence=" << largest_divergence
                      << " fraction=[" << min_fraction << ',' << max_fraction << "] max_relative_drift=" << largest_drift
                      << " max_speed=" << max_speed << " mean_iterations=" << double(iterations) / tick
                      << " centroid=" << center_x / volume << ',' << center_y / volume
                      << " energy_ratio=" << energy_ratio << " max_energy_ratio=" << max_energy_ratio
                      << " component_volume_error=" << component_errors.volume << " component_drift=" << component_errors.drift << " negative_mass_fraction=" << component_errors.negative
                      << " physics_seconds=" << solve_seconds << " elapsed=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n';
        }
    }
}
}

int main(int argc, char** argv)
{
    try
    {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;
        Context context;
        if (argc == 2 && std::string(argv[1]) == "--mask-display-only")
        {
            fluid::test::ambientMaskDisplayRegressions();
            return 0;
        }
        if (argc >= 2 && (std::string(argv[1]) == "--coarse-benchmark" || std::string(argv[1]) == "--coarse-compare"))
        {
            fluid::test::coarsePressureBenchmark(argc > 2 ? std::stoi(argv[2]) : 512, argc > 3 ? std::stoi(argv[3]) : 0,
                                                 std::string(argv[1]) == "--coarse-compare");
            return 0;
        }
        if (argc >= 3 && std::string(argv[1]) == "--coarse-pool-steps")
        {
            fluid::test::coarsePressurePool(std::stoi(argv[2]), argc > 3 ? std::stoi(argv[3]) : 128);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--edit-benchmark")
        {
            editBenchmark();
            require(glGetError() == GL_NO_ERROR, "GL error during edit benchmark");
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--component-benchmark")
        {
            componentBenchmark(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--benchmark-steps")
        {
            benchmark(std::stoi(argv[2]));
            return 0;
        }
        if (argc >= 3 && std::string(argv[1]) == "--pool-steps")
        {
            const int steps = std::stoi(argv[2]);
            const double tolerance = argc > 3 ? std::stod(argv[3]) : 1e-3;
            const int size = argc > 4 ? std::stoi(argv[4]) : 128;
            const int count = argc > 5 ? std::stoi(argv[5]) : 2;
            closedPool(steps, size, tolerance, count);
            require(glGetError() == GL_NO_ERROR, "GL error during GPU pool");
            return 0;
        }
        pressureBalancesGravity();
        pressureBalancesGravity(0);
        multigridPreservesPCGContract();
        multigridPreservesPCGContract(64);
        multigridPreservesPCGContract(1024);
        largeMixedPressureConvergesWithinDefaultBudget();
        cachedVelocityStatsRemainCurrent();
        fluid::test::materialEditingIsSparseAndReusable<FluidGridGPU>();
        fluid::test::bulkMaterialEditsPreserveEveryCell<FluidGridGPU>();
        fluid::test::repeatedMaterialEditsPreservePhaseVolumes<FluidGridGPU>();
        commonInterfaceAndDisplay<FluidGrid>();
        commonInterfaceAndDisplay<FluidGridGPU>();
        commonInterfaceAndDisplay<FluidGridGPU>(0);
        projectionMatchesCPU();
        projectionMatchesCPU(0);
        tinyFractionsRemainFinite();
        transportMatchesCPU();
        transportPreservesPureMaterial();
        transportKeepsStraightInterfacesSharp();
        transportKeepsShallowInterfacesStraight();
        nonuniformTransportMatchesCPU();
        fullStepsMatchCPU();
        componentsMatchCPU();
        vanishingSliversCannotExportExcessMass();
        zeroTimeAndFailedProjectionAreHandled();
        graphicsStateIsPreserved();
        graphicsStateIsPreserved(8, 8, 0);
        graphicsStateIsPreserved(65, 37);
        fluid::test::coarsePressureRegressions();
        fluid::test::ambientMaskDisplayRegressions();
        require(glGetError() == GL_NO_ERROR, "GL error during GPU regressions");
        std::cout << "PASS GPU regressions\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
