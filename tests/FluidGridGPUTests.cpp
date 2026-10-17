#include "FluidGrid.h"
#include "FluidGridGPU.h"
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
    state.water_fraction.resize(size_t(width) * height);
    for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
    {
        const double surface = x + height * (0.5 + 0.04 * std::cos(3.14159265358979323846 * (x * 3 + 0.5) / width));
        state.water_fraction[y * width + x] = float(std::clamp(y + 1.0 - surface, 0.0, 1.0));
    }
    return state;
}

void pressureBalancesGravity()
{
    for (const auto& [w, h] : { std::pair{15, 11}, std::pair{1, 13}, std::pair{13, 1} })
    {
        FluidGridGPU grid(w, h, 4);
        FluidGridGPU::State state;
        state.water_fraction.assign(w * h, 1);
        grid.upload(state);
        grid.applyGravity(1.0 / 120);
        auto result = grid.project(1.0 / 120, 2000, 1e-4);
        std::cout << "hydrostatic " << w << 'x' << h << " iterations=" << result.iterations << " divergence=" << result.max_divergence << '\n';
        require(result.converged, "GPU hydrostatic projection failed");
        state = grid.download();
        for (float v : state.v) require(std::abs(v) < 0.005, "GPU hydrostatic velocity failed to settle");
    }
}

void projectionMatchesCPU()
{
    for (const auto& [w, h] : { std::pair{17, 13}, std::pair{128, 128} })
    {
        FluidGrid cpu(w, h, 4);
        auto state = pool(w, h);
        cpu.water_fraction.assign(state.water_fraction.begin(), state.water_fraction.end());
        cpu.applyGravity(1.0 / 120);
        const auto reference = cpu.project(1.0 / 120, 2000, 1e-8);
        require(reference.converged, "CPU projection reference failed");
        for (auto preconditioner : {FluidGridGPU::PressurePreconditioner::Diagonal, FluidGridGPU::PressurePreconditioner::Multigrid})
        {
            FluidGridGPU gpu(w, h, 4);
            gpu.pressure_preconditioner = preconditioner;
            gpu.upload(state);
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
    grid.upload(pool(8, 8));
    const auto before = grid.download();
    require(!grid.step(0).converged, "zero-time step unexpectedly advanced");
    const auto after = grid.download();
    require(before.water_fraction == after.water_fraction && before.u == after.u && before.v == after.v, "zero-time step changed state");
    grid.applyGravity(0.01);
    require(!grid.project(0.01, 0).converged, "zero-iteration projection unexpectedly converged");
}

void multigridPreservesPCGContract()
{
    using namespace fluid::gpu_detail;
    ComputeState guard;
    PassRunner runner;
    for (auto [w, h] : {std::pair{1, 1}, {1, 129}, {131, 1}, {2, 67}, {13, 9}, {65, 37}, {128, 128}, {512, 512}, {513, 257}})
    {
        PressureMultigridGPU multigrid(runner, w, h);
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
        std::cout << "multigrid " << w << 'x' << h << " linearity=" << error / std::max(1.0, scale)
                  << " constant_error=" << constant_error << " scale=" << scale << '\n';
        require(constant_error <= 5e-6 * std::max(1.0, scale), "GPU multigrid did not remove its nullspace within float accuracy");
    }
}

void largeMixedPressureConvergesWithinDefaultBudget()
{
    FluidGridGPU grid(512, 512, 4);
    auto state = pool(512, 512);
    grid.upload(state);
    grid.applyGravity(1.0 / 120);
    const auto result = grid.project(1.0 / 120);
    require(result.converged, "512x512 GPU multigrid exhausted the default pressure budget");
    grid.upload(state);
    grid.applyGravity(1.0 / 120);
    const auto limited = grid.project(1.0 / 120, 1, 1e-6);
    require(!limited.converged && limited.iterations == 1, "GPU pressure exceeded or misreported a one-iteration budget");
}

void cachedVelocityStatsRemainCurrent()
{
    FluidGridGPU cached(48, 32, 4), refreshed(48, 32, 4);
    const auto initial = pool(48, 32);
    cached.upload(initial);
    refreshed.upload(initial);
    for (int tick = 0; tick < 20; ++tick)
    {
        // A zero force leaves velocities unchanged while forcing the uncached CFL path.
        refreshed.applyGravity(0.0);
        const double dt = tick < 10 ? 1.0 / 120.0 : 0.25;
        require(cached.step(dt, 2000, 1e-3).converged && refreshed.step(dt, 2000, 1e-3).converged, "CFL cache comparison failed to converge");
        const auto a = cached.download(), b = refreshed.download();
        require(a.water_fraction == b.water_fraction && a.u == b.u && a.v == b.v && a.pressure == b.pressure,
                "cached velocity statistics changed transport or pressure");
    }
    const auto restart = cached.download();
    cached.upload(restart);
    refreshed.upload(restart);
    refreshed.applyGravity(0.0);
    cached.advect(0.1);
    refreshed.advect(0.1);
    const auto a = cached.download(), b = refreshed.download();
    require(a.water_fraction == b.water_fraction && a.u == b.u && a.v == b.v, "upload retained stale velocity statistics");
}

template<typename Grid>
void commonInterfaceAndDisplay()
{
    Grid grid(9, 7, 2.5);
    typename Grid::State initial;
    initial.water_fraction.resize(9 * 7);
    initial.water_fraction[2 * 9 + 3] = typename Grid::Scalar(0.375);
    grid.upload(initial);
    const auto before = grid.download();
    require(before.water_fraction == initial.water_fraction && before.u.size() == 10 * 7 && before.v.size() == 9 * 8 && before.pressure.size() == 9 * 7,
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
        require(pixel[0] == expected, "common display returned stale or incorrect water fractions");
    };
    checkTexture(0.375f);
    require(before.water_fraction == grid.download().water_fraction, "display mutated the material state");
    require(grid.step(1.0 / 120, 2000, 1e-3).converged, "common physics call failed");
    grid.destroy();
    grid.upload({});
    checkTexture(0.0f);
    const auto reset = grid.download();
    for (const auto* field : { &reset.water_fraction, &reset.u, &reset.v, &reset.pressure })
        for (auto value : *field) require(value == 0, "empty upload failed to reset a field");
    display.destroy();
    grid.destroy();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &framebuffer);
}

void graphicsStateIsPreserved()
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
        FluidGridGPU gpu(8, 8, 1);
        gpu.upload(pool(8, 8));
        require(gpu.step(1.0 / 120, 2000, 1e-3).converged, "graphics state changed the numerical passes");
        const auto state = gpu.download();
        require(state.water_fraction[7 * 8] > 0.9f, "pixel buffers or color mask corrupted GPU state");
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
        state.water_fraction.resize(48 * 48);
        state.u.assign(49 * 48, float(velocity.x));
        state.v.assign(48 * 49, float(velocity.y));
        for (int y = 18; y < 30; ++y)
        for (int x = 18; x < 30; ++x)
            state.water_fraction[y * 48 + x] = float(std::clamp(6.0 - std::hypot(x - 23.5, y - 23.5), 0.0, 1.0));
        cpu.water_fraction.assign(state.water_fraction.begin(), state.water_fraction.end());
        cpu.u.assign(state.u.begin(), state.u.end());
        cpu.v.assign(state.v.begin(), state.v.end());
        const double initial = std::accumulate(state.water_fraction.begin(), state.water_fraction.end(), 0.0);
        gpu.upload(state);
        cpu.advect(1);
        gpu.advect(1);
        state = gpu.download();
        double fraction_error = 0, velocity_error = 0;
        for (size_t i = 0; i < state.water_fraction.size(); ++i)
            fraction_error = std::max(fraction_error, std::abs(state.water_fraction[i] - cpu.water_fraction[i]));
        for (int y = 8; y < 40; ++y)
        for (int x = 8; x < 40; ++x)
        {
            velocity_error = std::max(velocity_error, std::abs(state.u[y * 49 + x] - velocity.x));
            velocity_error = std::max(velocity_error, std::abs(state.v[y * 48 + x] - velocity.y));
        }
        const double drift = std::accumulate(state.water_fraction.begin(), state.water_fraction.end(), 0.0) - initial;
        std::cout << "diagonal " << velocity.x << ',' << velocity.y << " fraction_error=" << fraction_error
                  << " velocity_error=" << velocity_error << " volume_drift=" << drift << '\n';
        require(fraction_error < 2e-6, "GPU geometric transport differs from CPU");
        require(velocity_error < 2e-4, "GPU density changes altered uniform motion");
        require(std::abs(drift) < initial * 1e-6, "GPU uniform transport lost water");
    }
}

void transportPreservesPureMaterial()
{
    constexpr int size = 19;
    FluidGridGPU grid(size, size, 4);
    FluidGridGPU::State state;
    state.water_fraction.assign(size * size, 1.0f);
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
    grid.upload(state);
    grid.advect(0.137);
    for (float value : grid.download().water_fraction)
        require(value == 1.0f, "GPU transport accumulated roundoff in a pure material");
}

void transportKeepsStraightInterfacesSharp()
{
    FluidGridGPU gpu(40, 8, 1);
    FluidGridGPU::State state;
    state.water_fraction.resize(40 * 8);
    state.u.assign(41 * 8, 1);
    for (int y = 0; y < 8; ++y)
    for (int x = 8; x < 16; ++x) state.water_fraction[y * 40 + x] = 1;
    gpu.upload(state);
    gpu.advect(8);
    state = gpu.download();
    for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 40; ++x)
        require(std::abs(state.water_fraction[y * 40 + x] - (x >= 16 && x < 24 ? 1.0 : 0.0)) < 1e-6,
                "GPU whole-cell translation blurred a straight interface");
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
    cpu.water_fraction.assign(state.water_fraction.begin(), state.water_fraction.end());
    cpu.u.assign(state.u.begin(), state.u.end());
    cpu.v.assign(state.v.begin(), state.v.end());
    FluidGridGPU gpu(w, h, 1);
    gpu.upload(state);
    cpu.advect(0.1);
    gpu.advect(0.1);
    state = gpu.download();
    double fraction_error = 0, velocity_error = 0;
    for (size_t i = 0; i < state.water_fraction.size(); ++i)
        fraction_error = std::max(fraction_error, std::abs(state.water_fraction[i] - cpu.water_fraction[i]));
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
    cpu.water_fraction.assign(state.water_fraction.begin(), state.water_fraction.end());
    gpu.upload(state);
    double fraction_error = 0, velocity_error = 0;
    for (int tick = 0; tick < 20; ++tick)
    {
        require(cpu.step(1.0 / 120, 2000, 1e-8).converged, "CPU step reference failed");
        require(gpu.step(1.0 / 120, 2000, 1e-3).converged, "GPU comparison step failed");
        state = gpu.download();
        for (size_t i = 0; i < state.water_fraction.size(); ++i)
            fraction_error = std::max(fraction_error, std::abs(state.water_fraction[i] - cpu.water_fraction[i]));
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
        state.water_fraction.resize(64);
        state.water_fraction[3 * 8 + 3] = amount;
        state.water_fraction[3 * 8 + 4] = amount * 0.5f;
        state.water_fraction[4 * 8 + 3] = amount * 0.25f;
        state.u.assign(9 * 8, 1);
        state.v.assign(8 * 9, 0.75f);
        gpu.upload(state);
        gpu.advect(0.25);
        state = gpu.download();
        for (const auto* field : { &state.water_fraction, &state.u, &state.v })
            for (float value : *field) require(std::isfinite(value), "tiny water fractions produced non-finite transport");
    }
    // this near-empty remnant reproduced a float 0/0 during the sloshing scene
    FluidGridGPU gpu(8, 8, 4);
    FluidGridGPU::State state;
    state.water_fraction.resize(64);
    state.water_fraction[3 * 8 + 3] = 1.654361225e-24f;
    state.water_fraction[3 * 8 + 4] = 1.051768366e-22f;
    state.water_fraction[4 * 8 + 3] = 6.335743373e-20f;
    state.u.assign(9 * 8, -28.16068459f);
    state.v.assign(8 * 9, 0.01431973279f);
    gpu.upload(state);
    gpu.advect(1.0 / 120);
    state = gpu.download();
    for (const auto* field : { &state.water_fraction, &state.u, &state.v })
        for (float value : *field) require(std::isfinite(value), "near-empty sloshing cell produced non-finite transport");
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
        cpu.water_fraction.assign(initial.water_fraction.begin(), initial.water_fraction.end());
        time(cpu, "CPU", tolerance);
    }
    FluidGridGPU gpu(128, 128, 4);
    gpu.upload(initial);
    time(gpu, "GPU", 1e-3);
}

void closedPool(int steps, int size, double tolerance)
{
    FluidGridGPU grid(size, size, 4);
    auto state = pool(size, size);
    grid.upload(state);
    const double initial_volume = std::accumulate(state.water_fraction.begin(), state.water_fraction.end(), 0.0);
    // use the MAC faces' dual-cell masses for kinetic energy, and the box bottom as zero potential
    auto energy = [&](const FluidGridGPU::State& sample)
    {
        auto density = [&](int x, int y) {
            double fraction = std::clamp(double(sample.water_fraction[y * size + x]), 0.0, 1.0);
            return grid.air_density + (grid.water_density - grid.air_density) * fraction;
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
            for (const auto& [name, field] : { std::pair{"water", &failed.water_fraction}, std::pair{"u", &failed.u},
                                             std::pair{"v", &failed.v}, std::pair{"pressure", &failed.pressure} })
                for (size_t i = 0; i < field->size(); ++i)
                    if (!std::isfinite((*field)[i])) { std::cout << "first non-finite " << name << " index=" << i << '\n'; break; }
        }
        require(result.converged, "GPU pool projection failed");
        state = grid.download();
        const double volume = std::accumulate(state.water_fraction.begin(), state.water_fraction.end(), 0.0);
        largest_drift = std::max(largest_drift, std::abs(volume - initial_volume) / initial_volume);
        for (float f : state.water_fraction)
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
        require(largest_drift < 1e-4, "GPU pool lost more than 0.01 percent of its water");
        const double energy_ratio = energy(state) / initial_energy;
        max_energy_ratio = std::max(max_energy_ratio, energy_ratio);
        require(max_energy_ratio < 1.02, "GPU pool energy grew more than two percent above its initial value");
        if (tick <= 3 || tick % 120 == 0 || tick == steps)
        {
            double center_x = 0, center_y = 0;
            for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
            {
                center_x += state.water_fraction[y * size + x] * (x + 0.5) * grid.cell_size;
                center_y += state.water_fraction[y * size + x] * (y + 0.5) * grid.cell_size;
            }
            std::cout << "pool tick=" << tick << " time=" << tick / 120.0 << " max_divergence=" << largest_divergence
                      << " fraction=[" << min_fraction << ',' << max_fraction << "] max_relative_drift=" << largest_drift
                      << " max_speed=" << max_speed << " mean_iterations=" << double(iterations) / tick
                      << " centroid=" << center_x / volume << ',' << center_y / volume
                      << " energy_ratio=" << energy_ratio << " max_energy_ratio=" << max_energy_ratio
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
            closedPool(steps, size, tolerance);
            require(glGetError() == GL_NO_ERROR, "GL error during GPU pool");
            return 0;
        }
        pressureBalancesGravity();
        multigridPreservesPCGContract();
        largeMixedPressureConvergesWithinDefaultBudget();
        cachedVelocityStatsRemainCurrent();
        commonInterfaceAndDisplay<FluidGrid>();
        commonInterfaceAndDisplay<FluidGridGPU>();
        projectionMatchesCPU();
        tinyFractionsRemainFinite();
        transportMatchesCPU();
        transportPreservesPureMaterial();
        transportKeepsStraightInterfacesSharp();
        nonuniformTransportMatchesCPU();
        fullStepsMatchCPU();
        zeroTimeAndFailedProjectionAreHandled();
        graphicsStateIsPreserved();
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
