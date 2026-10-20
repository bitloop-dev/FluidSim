#include "FluidGridHybrid.h"
#include "FluidGridDisplay.h"
#include "FluidGridEditTests.h"
#include <bitloop/core/threads.h>
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

namespace
{
using fluid::FluidGridGPU;
using fluid::FluidGridHybrid;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Context
{
    SDL_Window* window;
    SDL_GLContext context;
    Context()
    {
        if (!SDL_Init(SDL_INIT_VIDEO)) throw std::runtime_error(SDL_GetError());
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
        window = SDL_CreateWindow("FluidGridHybrid tests", 32, 32, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
        if (!window) throw std::runtime_error(SDL_GetError());
        context = SDL_GL_CreateContext(window);
        if (!context) throw std::runtime_error(SDL_GetError());
        require(gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress) != 0, "GL entry points unavailable");
        SDL_GL_SetSwapInterval(0);
        std::cout << "GPU: " << glGetString(GL_RENDERER) << '\n';
    }
    ~Context() { SDL_GL_DestroyContext(context); SDL_DestroyWindow(window); SDL_Quit(); }
};

template<class Grid>
typename Grid::State pool(Grid& grid)
{
    const auto water = grid.elements.add({"Water", {0, 0, 1}});
    const auto red = grid.elements.add({"Red", {1, 0, 0}});
    const auto air = grid.elements.add({"Air", {0, 0, 0}});
    const auto cloud = grid.elements.add({"Cloud", {0, 1, 1}});
    grid.liquid_mixture.components = {{water, 100}, {red, 100}};
    grid.gas_mixture.components = {{air, 0.1}, {cloud, 0.1}};
    typename Grid::State state;
    const size_t count = size_t(grid.width) * grid.height;
    state.liquid_fraction.resize(count);
    state.liquid_mass.assign(2, std::vector<float>(count));
    state.gas_mass.assign(2, std::vector<float>(count));
    const double volume = grid.cell_size * grid.cell_size;
    for (int y = 0; y < grid.height; ++y)
    for (int x = 0; x < grid.width; ++x)
    {
        const size_t i = size_t(y) * grid.width + x;
        const float f = float(std::clamp(y + 1.0 - grid.height * (0.5 + 0.12 * std::cos(3.141592653589793 * x / grid.width)), 0.0, 1.0));
        state.liquid_fraction[i] = f;
        const double blend = 0.5 + 0.3 * std::sin(0.05 * x + 0.07 * y);
        state.liquid_mass[0][i] = float(100 * volume * f * blend);
        state.liquid_mass[1][i] = float(100 * volume * f * (1 - blend));
        state.gas_mass[0][i] = float(0.1 * volume * (1 - f) * blend);
        state.gas_mass[1][i] = float(0.1 * volume * (1 - f) * (1 - blend));
    }
    return state;
}

struct Errors { double phase = 0, drift = 0, negative = 0; };

void checkState(const FluidGridHybrid& grid, const FluidGridGPU::State& state, const FluidGridGPU::State& initial, Errors& errors)
{
    const double volume = grid.cell_size * grid.cell_size;
    for (const auto* field : {&state.u, &state.v, &state.pressure})
        for (float value : *field) require(std::isfinite(value), "hybrid non-finite state");
    for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
    {
        const double f = state.liquid_fraction[i];
        require(std::isfinite(f) && f >= -2e-5 && f <= 1.00002, "hybrid invalid fraction");
        require(std::abs((state.liquid_mass[0][i] + state.liquid_mass[1][i]) / (100 * volume) - f) < 0.002,
                "hybrid liquid mass/geometry mismatch");
        require(std::abs((state.gas_mass[0][i] + state.gas_mass[1][i]) / (0.1 * volume) - (1 - f)) < 0.002,
                "hybrid gas mass/geometry mismatch");
        errors.phase = std::max({errors.phase,
            std::abs((state.liquid_mass[0][i] + state.liquid_mass[1][i]) / (100 * volume) - f),
            std::abs((state.gas_mass[0][i] + state.gas_mass[1][i]) / (0.1 * volume) - (1 - f))});
    }
    for (auto fields : {std::pair{&state.liquid_mass, &initial.liquid_mass}, std::pair{&state.gas_mass, &initial.gas_mass}})
    for (size_t k = 0; k < fields.first->size(); ++k)
    {
        const auto& values = (*fields.first)[k];
        const double before = std::accumulate((*fields.second)[k].begin(), (*fields.second)[k].end(), 0.0);
        const double after = std::accumulate(values.begin(), values.end(), 0.0);
        errors.drift = std::max(errors.drift, std::abs(after - before) / before);
        require(std::abs(after - before) < 1e-4 * before, "hybrid ingredient mass drift");
        const double cell_mass = fields.first == &state.liquid_mass ? 100 * volume : 0.1 * volume;
        for (float value : values)
        {
            require(std::isfinite(value) && value >= -2e-5 * cell_mass, "hybrid invalid mass");
            errors.negative = std::max(errors.negative, -double(value) / cell_mass);
        }
    }
    for (int y = 0; y < grid.height; ++y)
        require(state.u[grid.uIndex(0, y)] == 0 && state.u[grid.uIndex(grid.width, y)] == 0, "hybrid side wall leaked");
    for (int x = 0; x < grid.width; ++x)
        require(state.v[grid.vIndex(x, 0)] == 0 && state.v[grid.vIndex(x, grid.height)] == 0, "hybrid top/bottom wall leaked");
}

void projectionAndAPI()
{
    for (const auto [w, h] : {std::pair{31, 19}, std::pair{1, 17}, std::pair{17, 1}, std::pair{257, 129}})
    for (int cutoff : {16, 64, 256})
    {
        FluidGridHybrid hybrid(w, h, 4);
        hybrid.cpu_pressure_cells = cutoff;
        auto state = pool(hybrid);
        state.u.assign(size_t(w + 1) * h, 0.0f);
        state.v.assign(size_t(w) * (h + 1), 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 1; x < w; ++x) state.u[y * (w + 1) + x] = float(std::sin(0.1 * x + y));
        for (int y = 1; y < h; ++y)
            for (int x = 0; x < w; ++x) state.v[y * w + x] = float(std::cos(x + 0.1 * y));
        hybrid.upload(state);
        const auto unchanged = hybrid.download();
        require(!hybrid.step(0).converged && fluid::test::sameEditState(hybrid.download(), unchanged), "hybrid zero-time step changed state");
        require(!hybrid.project(0.01, 0, 1e-3).converged, "hybrid zero iteration cap ignored");
        require(hybrid.project(0.01, 1, 1e-6).iterations <= 1, "hybrid exceeded its pressure iteration cap");
        hybrid.upload(state);
        auto result = hybrid.project(0.01, 2000, 1e-3);
        if (!result.converged) std::cout << "failure grid=" << w << 'x' << h << " cutoff=" << cutoff << " divergence=" << result.max_divergence << '\n';
        require(result.converged && result.max_divergence <= 1e-3, "hybrid projection failed");
        require(hybrid.maxDivergence() <= 1e-3, "hybrid actual divergence failed");
        const auto projected = hybrid.download();
        require(projected.liquid_mass == state.liquid_mass && projected.gas_mass == state.gas_mass, "hybrid pressure changed masses");
        // Check the downloaded MAC field independently of GPU reduction/stopping state.
        for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            const double div = (double(projected.u[hybrid.uIndex(x + 1, y)]) - projected.u[hybrid.uIndex(x, y)] +
                                projected.v[hybrid.vIndex(x, y + 1)] - projected.v[hybrid.vIndex(x, y)]) / hybrid.cell_size;
            require(std::abs(div) < 1.001e-3, "hybrid downloaded velocity is divergent");
        }
    }
    FluidGridHybrid hybrid(32, 24, 4);
    FluidGridGPU gpu(32, 24, 4);
    hybrid.cpu_pressure_cells = 0;
    auto state = pool(hybrid);
    pool(gpu);
    hybrid.upload(state);
    gpu.upload(state);
    for (int tick = 0; tick < 8; ++tick)
    {
        require(hybrid.step(0.004, 2000).converged && gpu.step(0.004, 2000).converged, "disabled hybrid step failed");
        const auto a = hybrid.download(), b = gpu.download();
        require(fluid::test::sameEditState(a, b), "disabled hybrid differs from original GPU path");
    }
    hybrid.cpu_pressure_cells = 64;
    hybrid.upload(state);
    fluid::FluidGridDisplay display;
    require(display.texture(hybrid) != 0, "hybrid display unavailable");
    display.destroy();
    hybrid.destroy();
    hybrid.upload(state);
    require(hybrid.step(0.004, 2000).converged, "hybrid reupload after destroy failed");
}

void behavioral(int steps, int size = 128)
{
    FluidGridHybrid grid(size, size, 4);
    const auto initial = pool(grid);
    grid.upload(initial);
    int largest_iterations = 0;
    double max_divergence = 0;
    Errors errors;
    for (int tick = 0; tick < steps; ++tick)
    {
        const auto result = grid.step(1.0 / 120, 2000, 1e-3);
        require(result.converged, "hybrid pool projection failed");
        largest_iterations = std::max(largest_iterations, result.iterations);
        max_divergence = std::max(max_divergence, result.max_divergence);
        checkState(grid, grid.download(), initial, errors);
    }
    std::cout << "behavior size=" << size << " steps=" << steps << " simulated_seconds=" << steps / 120.0
              << " max_iterations=" << largest_iterations << " max_divergence=" << max_divergence
              << " phase_error=" << errors.phase << " mass_drift=" << errors.drift << " negative_volume=" << errors.negative << '\n';
}

void benchmark(int size, int history, bool paired)
{
    FluidGridGPU reference(size, size, 4);
    auto initial = pool(reference);
    reference.upload(initial);
    for (int tick = 0; tick < history; ++tick) require(reference.step(1.0 / 120, 2000).converged, "benchmark history failed");
    if (history) initial = reference.download();
    for (int repeat = 0; repeat < (paired ? 5 : 3); ++repeat)
    {
      std::vector<int> cutoffs = paired ? std::vector<int>{0, 1024} : std::vector<int>{0, 16, 64, 256, 1024, 4096};
      if (repeat % 2) std::reverse(cutoffs.begin(), cutoffs.end());
      for (int cutoff : cutoffs)
      {
        FluidGridHybrid grid(size, size, 4);
        pool(grid);
        grid.cpu_pressure_cells = cutoff;
        grid.upload(initial);
        for (int tick = 0; tick < 5; ++tick) require(grid.step(0.004, 2000).converged, "benchmark warmup failed");
        glFinish();
        const auto start = std::chrono::steady_clock::now();
        int iterations = 0;
        const int samples = paired ? 48 : 24;
        for (int tick = 0; tick < samples; ++tick)
        {
            const auto result = grid.step(0.004, 2000);
            require(result.converged, "benchmark projection failed");
            iterations += result.iterations;
        }
        glFinish();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / samples;
        std::cout << "benchmark size=" << size << " history=" << history << " repeat=" << repeat << " cpu_cells=" << cutoff
                  << " step_ms=" << ms << " iterations=" << double(iterations) / samples << std::endl;
      }
    }
}
}

int main(int argc, char** argv)
{
    try
    {
        bl::Thread::setMaxThreads(8);
        Context context;
        if (argc >= 2 && (std::string(argv[1]) == "--benchmark" || std::string(argv[1]) == "--compare"))
        {
            benchmark(argc > 2 ? std::stoi(argv[2]) : 512, argc > 3 ? std::stoi(argv[3]) : 0, std::string(argv[1]) == "--compare");
            return 0;
        }
        if (argc >= 3 && std::string(argv[1]) == "--pool-steps")
        {
            behavioral(std::stoi(argv[2]), argc > 3 ? std::stoi(argv[3]) : 128);
            return 0;
        }
        projectionAndAPI();
        fluid::test::materialEditingIsSparseAndReusable<FluidGridHybrid>();
        fluid::test::bulkMaterialEditsPreserveEveryCell<FluidGridHybrid>();
        fluid::test::repeatedMaterialEditsPreservePhaseVolumes<FluidGridHybrid>();
        behavioral(24);
        require(glGetError() == GL_NO_ERROR, "hybrid left a GL error");
        std::cout << "FluidGridHybrid tests passed\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
