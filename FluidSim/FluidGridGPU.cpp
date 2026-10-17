#include "FluidGridGPU.h"
#include "FluidGridGPUInternal.h"
#include "PressureMultigridGPU.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#ifdef __EMSCRIPTEN__
#include <GLES3/gl3.h>
#include <emscripten/html5.h>
#else
#include <glad/glad.h>
#endif

namespace fluid
{
using namespace gpu_detail;


struct FluidGridGPU::Impl
{
    const int width, height;
    float cell_size = 1, air_density = 0.1f, water_density = 100, gravity_y = 100, tolerance = 0;
    PassRunner runner;
    Texture u, v, next_u, next_v, water_fraction, next_water_fraction, density;
    Texture momentum_u, momentum_v, next_momentum_u, next_momentum_v;
    Texture interfaces, split_weights, transfers_u, transfers_v;
    Texture pressure, next_pressure, previous_pressure, pressure_base, pressure_rhs, pressure_residual, next_residual;
    Texture pressure_correction, pressure_direction, next_direction, pressure_product;
    Texture pressure_coefficients;
    Texture residual_stats, correction_stats, denominator, solver_state, next_solver_state;
    std::vector<Texture> reduction_levels;
    PressureMultigridGPU multigrid;
    bool use_multigrid = true;
    int previous_solve_iterations = 7;
    float cached_outflow = 0;
    bool velocity_stats_valid = false;

    Program fill, density_pass, pressure_coefficient, divergence, pressure_operator;
    Program predict_pressure, add_pressure, subtract_mean, residual, reduce_pass, precondition, coefficients, direction, update_pressure;
    Program correct_velocity, gravity, velocity_stats;
    Program reconstruct, freeze_weights, transfers, transport_momentum, transport_water, momentum;

    Impl(int w, int h);
    void configure(const FluidGridGPU& grid)
    {
        cell_size = float(grid.cell_size);
        air_density = float(grid.air_density);
        water_density = float(grid.water_density);
        gravity_y = float(grid.gravity_y);
        use_multigrid = grid.pressure_preconditioner == FluidGridGPU::PressurePreconditioner::Multigrid;
    }
    using Input = std::pair<const char*, const Texture*>;
    void run(Program& program, std::initializer_list<const Texture*> outputs,
             std::initializer_list<Input> inputs = {}, float dt = 0, int mode = 0)
    {
        runner.bind(program, outputs, inputs);
        glUniform2i(program.location("grid_size"), width, height);
        program.scalar("cell_size", cell_size);
        program.scalar("air_density", air_density);
        program.scalar("water_density", water_density);
        program.scalar("time_step", dt);
        program.scalar("tolerance", tolerance);
        program.integer("mode", mode);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    void reduce(const Texture& input, Texture& result, int mode = 0, const Texture* other = nullptr, size_t first_level = 0)
    {
        const Texture* source = &input;
        for (size_t i = first_level; i < reduction_levels.size(); ++i)
        {
            const Texture& level = reduction_levels[i];
            run(reduce_pass, { &level }, { {"source", source}, {"other", other ? other : source} }, 0, mode);
            source = &level;
            mode = mode == 4 ? 4 : 3;
        }
        run(reduce_pass, { &result }, { {"source", source}, {"other", other ? other : source} }, 0, mode);
    }
    std::array<float, 4> readScalar(const Texture& texture) const
    {
        std::array<float, 4> values;
        runner.target({ &texture });
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, values.data());
        return values;
    }
    void removeMean(Texture& field, Texture& scratch)
    {
        reduce(field, residual_stats);
        run(subtract_mean, { &scratch }, { {"source", &field}, {"stats", &residual_stats} });
        field.swap(scratch);
    }
    void rebuildDensity() { run(density_pass, { &density }, { {"water", &water_fraction} }); }
    void rebuildPressureCoefficients();
    void applyPressureOperator(const Texture& input);
    void rebuildResidual();
    void applyPreconditioner(bool restart);
    int solvePressure(int max_iterations, float residual_tolerance, bool refinement);
    FluidGridGPU::ProjectionResult project(double dt, int max_iterations, double divergence_tolerance);
    double maxDivergence();
    void applyGravity(double dt);
    void buildTransportPrograms();
    void rebuildInterfaces();
    void rebuildTransfersX(double dt);
    void rebuildTransfersY(double dt);
    void rebuildMomentum();
    void advectMomentumX(double dt);
    void advectMomentumY(double dt);
    void recoverVelocity();
    void advect(double dt);
};

FluidGridGPU::Impl::Impl(int w, int h) :
    width(w), height(h),
    u(w + 1, h), v(w, h + 1), next_u(w + 1, h), next_v(w, h + 1),
    water_fraction(w, h), next_water_fraction(w, h), density(w, h),
    momentum_u(w + 1, h), momentum_v(w, h + 1), next_momentum_u(w + 1, h), next_momentum_v(w, h + 1),
    interfaces(w, h, 4), split_weights(w, h), transfers_u(w + 1, h, 2), transfers_v(w, h + 1, 2),
    pressure(w, h), next_pressure(w, h), previous_pressure(w, h), pressure_base(w, h), pressure_rhs(w, h),
    pressure_residual(w, h), next_residual(w, h), pressure_correction(w, h),
    pressure_direction(w, h), next_direction(w, h), pressure_product(w, h),
    pressure_coefficients(w, h, 4),
    residual_stats(1, 1, 4), correction_stats(1, 1, 4), denominator(1, 1, 4),
    solver_state(1, 1, 4), next_solver_state(1, 1, 4), multigrid(runner, w, h)
{
    for (int rw = (w + 3) / 4, rh = (h + 3) / 4; rw > 1 || rh > 1; rw = (rw + 3) / 4, rh = (rh + 3) / 4)
        reduction_levels.emplace_back(rw, rh, 4);
    // float textures must be renderable, not merely sampleable
    for (const Texture* texture : { &u, &transfers_u, &interfaces })
    {
        runner.target({ texture });
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("FluidGridGPU requires renderable R32F, RG32F and RGBA32F textures");
    }
    auto build = [&](Program& p, const char* name, const char* body) { runner.build(p, name, common, body); };
    build(fill, "fill", R"(
        void main() { output0 = vec4(time_step); }
    )");
    build(density_pass, "rebuildDensity", R"(
        uniform sampler2D water;
        void main() { output0 = vec4(densityOf(at(water, cell()))); }
    )");
    build(pressure_coefficient, "pressure face weights", R"(
        uniform sampler2D water;
        void main()
        {
            ivec2 p = cell();
            float center = densityOf(at(water, p));
            vec4 neighbors = vec4(
                densityOf(at(water, max(p - ivec2(1, 0), ivec2(0)))), densityOf(at(water, min(p + ivec2(1, 0), grid_size - 1))),
                densityOf(at(water, max(p - ivec2(0, 1), ivec2(0)))), densityOf(at(water, min(p + ivec2(0, 1), grid_size - 1))));
            // RGBA stores left/right/up/down weights, with zero pressure flux through walls.
            vec4 weights = 2.0 / (vec4(center) + neighbors);
            if (p.x == 0) weights.x = 0.0;
            if (p.x + 1 == grid_size.x) weights.y = 0.0;
            if (p.y == 0) weights.z = 0.0;
            if (p.y + 1 == grid_size.y) weights.w = 0.0;
            output0 = weights;
        }
    )");
    build(divergence, "divergence", R"(
        uniform sampler2D u, v;
        void main()
        {
            ivec2 p = cell();
            float value = (at(u, p + ivec2(1, 0)) - at(u, p) + at(v, p + ivec2(0, 1)) - at(v, p)) / cell_size;
            output0 = vec4(mode == 0 ? value : -value * cell_size * cell_size / time_step);
        }
    )");
    build(velocity_stats, "divergence and CFL reduction", R"(
        uniform sampler2D u, v;
        void main()
        {
            ivec2 base = cell() * 4;
            vec4 result = vec4(0);
            // Reuse the same face samples for actual divergence and the next transport CFL limit.
            for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
            {
                ivec2 p = base + ivec2(x, y);
                if (any(greaterThanEqual(p, grid_size))) continue;
                float left = at(u, p), right = at(u, p + ivec2(1, 0));
                float up = at(v, p), down = at(v, p + ivec2(0, 1));
                float divergence = (right - left + down - up) / cell_size;
                float outflow = max(right, 0.0) + max(-left, 0.0) + max(down, 0.0) + max(-up, 0.0);
                if (!finite(divergence) || !finite(outflow)) { result.a = 1.0; continue; }
                result.r += divergence;
                result.g = max(result.g, abs(divergence));
                result.b = max(result.b, outflow);
            }
            output0 = result;
        }
    )");
    build(pressure_operator, "applyPressureOperator", R"(
        uniform sampler2D source, coefficients;
        void main()
        {
            ivec2 p = cell();
            float center = at(source, p);
            vec4 a = texelFetch(coefficients, p, 0);
            float value =
                (a.x * (center - at(source, max(p - ivec2(1, 0), ivec2(0)))) + a.y * (center - at(source, min(p + ivec2(1, 0), grid_size - 1)))) +
                (a.z * (center - at(source, max(p - ivec2(0, 1), ivec2(0)))) + a.w * (center - at(source, min(p + ivec2(0, 1), grid_size - 1))));
            output0 = vec4(value);
        }
    )");
    build(predict_pressure, "predict pressure", R"(
        uniform sampler2D latest, previous;
        void main() { output0 = vec4(2.0 * at(latest, cell()) - at(previous, cell())); }
    )");
    build(add_pressure, "accumulate pressure", R"(
        uniform sampler2D base, correction;
        void main() { output0 = vec4(at(base, cell()) + at(correction, cell())); }
    )");
    build(subtract_mean, "removeMean", R"(
        uniform sampler2D source, stats;
        void main() { output0 = vec4(at(source, cell()) - at(stats, ivec2(0)) / float(grid_size.x * grid_size.y)); }
    )");
    build(residual, "true residual", R"(
        uniform sampler2D rhs, product;
        void main() { output0 = vec4(at(rhs, cell()) - at(product, cell())); }
    )");
    build(reduce_pass, "reduction", R"(
        uniform sampler2D source, other;
        void main()
        {
            ivec2 base = cell() * 4, size = textureSize(source, 0);
            vec4 result = vec4(0);
            // exclude missing texels so odd dimensions contribute each cell exactly once
            for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
            {
                ivec2 p = base + ivec2(x, y);
                if (any(greaterThanEqual(p, size))) continue;
                vec4 value = texelFetch(source, p, 0);
                if (mode == 0) value = vec4(value.r, abs(value.r), 0, finite(value.r) ? 0 : 1);
                if (mode == 1) value = vec4(value.r * at(other, p), 0, 0, 0);
                if (mode == 2) value = vec4(value.r, 0, value.r * at(other, p), 0);
                if (!finite(value.r) || !finite(value.g) || !finite(value.b)) value = vec4(0, 0, 0, 1);
                result.r += value.r;
                result.b = mode == 4 ? max(result.b, value.b) : result.b + value.b;
                result.g = max(result.g, value.g);
                result.a = max(result.a, value.a);
            }
            output0 = result;
        }
    )");
    build(precondition, "applyPreconditioner", R"(
        uniform sampler2D residual, stats, coefficients;
        void main()
        {
            float mean = at(stats, ivec2(0)) / float(grid_size.x * grid_size.y);
            vec4 a = texelFetch(coefficients, cell(), 0);
            float diagonal = (a.x + a.y) + (a.z + a.w);
            output0 = vec4(diagonal > 0.0 ? (at(residual, cell()) - mean) / diagonal : 0.0);
        }
    )");
    build(coefficients, "PCG coefficients", R"(
        uniform sampler2D residual_stats, correction_stats, previous, denominator;
        void main()
        {
            vec4 r = texelFetch(residual_stats, ivec2(0), 0);
            vec4 z = texelFetch(correction_stats, ivec2(0), 0);
            vec4 old = texelFetch(previous, ivec2(0), 0);
            float mean = z.r / float(grid_size.x * grid_size.y);
            // removing the correction mean changes its dot product by mean times the residual sum
            float dot = z.b - mean * r.r;
            float failed = max(r.a, z.a);
            if (!finite(dot) || (dot <= 0.0 && r.g > tolerance)) failed = 1.0;
            if (mode == 0)
            {
                vec4 denominator_value = texelFetch(denominator, ivec2(0), 0);
                failed = max(failed, old.r < 0.0 ? 1.0 : 0.0);
                if (old.b > tolerance && (denominator_value.r <= 0.0 || denominator_value.a != 0.0)) failed = 1.0;
            }
            // Keep the active iteration count on-device; negative rho marks solver breakdown.
            float iterations = mode == 1 ? 0.0 : old.a + (old.b > tolerance && old.r > 0.0 ? 1.0 : 0.0);
            output0 = vec4(failed != 0.0 ? -1.0 : dot, mean, r.g, iterations);
        }
    )");
    build(direction, "pressure direction", R"(
        uniform sampler2D correction, old_direction, state, previous;
        void main()
        {
            vec4 s = texelFetch(state, ivec2(0), 0), old = texelFetch(previous, ivec2(0), 0);
            if (s.r < 0.0) { output0 = vec4(at(old_direction, cell())); return; }
            float beta = mode == 1 || old.r <= 0.0 ? 0.0 : s.r / old.r;
            output0 = vec4(at(correction, cell()) - s.g + beta * at(old_direction, cell()));
        }
    )");
    build(update_pressure, "updatePressureAndResidual", R"(
        OUT1 vec4 output1;
        uniform sampler2D pressure, residual, direction, product, state, denominator;
        void main()
        {
            vec4 s = texelFetch(state, ivec2(0), 0), d = texelFetch(denominator, ivec2(0), 0);
            if (s.b <= tolerance || s.r < 0.0 || d.r <= 0.0 || d.a != 0.0)
            {
                output0 = vec4(at(pressure, cell()));
                output1 = vec4(at(residual, cell()));
                return;
            }
            float alpha = s.r / d.r;
            output0 = vec4(at(pressure, cell()) + alpha * at(direction, cell()));
            output1 = vec4(at(residual, cell()) - alpha * at(product, cell()));
        }
    )");
    build(correct_velocity, "pressure gradient", R"(
        uniform sampler2D velocity, pressure, coefficients;
        void main()
        {
            ivec2 p = cell(), axis = mode == 0 ? ivec2(1, 0) : ivec2(0, 1);
            int coordinate = mode == 0 ? p.x : p.y, extent = mode == 0 ? grid_size.x : grid_size.y;
            if (coordinate == 0 || coordinate == extent) { output0 = vec4(0); return; }
            float gradient = (at(pressure, p) - at(pressure, p - axis)) / cell_size;
            vec4 a = texelFetch(coefficients, p, 0);
            output0 = vec4(at(velocity, p) - gradient * time_step * (mode == 0 ? a.x : a.z));
        }
    )");
    build(gravity, "applyGravity", R"(
        uniform sampler2D velocity;
        void main() { output0 = vec4(cell().y == 0 || cell().y == grid_size.y ? 0.0 : at(velocity, cell()) + time_step); }
    )");
    buildTransportPrograms();
}

void FluidGridGPU::Impl::rebuildPressureCoefficients()
{
    run(pressure_coefficient, { &pressure_coefficients }, { {"water", &water_fraction} });
    if (use_multigrid) multigrid.rebuild(pressure_coefficients);
}

void FluidGridGPU::Impl::applyPressureOperator(const Texture& input)
{
    run(pressure_operator, { &pressure_product }, { {"source", &input}, {"coefficients", &pressure_coefficients} });
}

void FluidGridGPU::Impl::rebuildResidual()
{
    applyPressureOperator(pressure);
    run(residual, { &pressure_residual }, { {"rhs", &pressure_rhs}, {"product", &pressure_product} });
    reduce(pressure_residual, residual_stats);
}

void FluidGridGPU::Impl::applyPreconditioner(bool restart)
{
    const Texture* correction = &pressure_correction;
    if (use_multigrid)
        correction = &multigrid.apply(pressure_residual, residual_stats);
    else
        run(precondition, { &pressure_correction }, {
            {"residual", &pressure_residual}, {"stats", &residual_stats}, {"coefficients", &pressure_coefficients} });
    reduce(*correction, correction_stats, 2, &pressure_residual);
    run(coefficients, { &next_solver_state }, {
        {"residual_stats", &residual_stats}, {"correction_stats", &correction_stats},
        {"previous", &solver_state}, {"denominator", &denominator} }, 0, restart ? 1 : 0);
    run(direction, { &next_direction }, {
        {"correction", correction}, {"old_direction", &pressure_direction},
        {"state", &next_solver_state}, {"previous", &solver_state} }, 0, restart ? 1 : 0);
    pressure_direction.swap(next_direction);
    solver_state.swap(next_solver_state);
}

int FluidGridGPU::Impl::solvePressure(int max_iterations, float residual_tolerance, bool refinement)
{
    tolerance = residual_tolerance;
    // solve a correction from zero so small updates are not added to a large warm-start pressure
    run(fill, { &pressure });
    rebuildResidual();
    auto stats = readScalar(residual_stats);
    if (stats[1] <= tolerance || max_iterations == 0 || stats[3] != 0) return 0;

    run(fill, { &solver_state });
    run(fill, { &pressure_direction });
    run(fill, { &denominator });
    applyPreconditioner(true);
    int submitted = 0;
    int batch = use_multigrid ? (refinement ? 2 : std::clamp(previous_solve_iterations + 1, 2, 64)) : 8;
    int extra_batch = 4;
    std::array<float, 4> state = {};
    while (submitted < max_iterations)
    {
        // Predict the check point from the previous solve; frozen updates cannot overshoot convergence.
        const int end = std::min(submitted + batch, max_iterations);
        for (; submitted < end; ++submitted)
        {
            applyPressureOperator(pressure_direction);
            reduce(pressure_direction, denominator, 1, &pressure_product);
            run(update_pressure, { &next_pressure, &next_residual }, {
                {"pressure", &pressure}, {"residual", &pressure_residual}, {"direction", &pressure_direction},
                {"product", &pressure_product}, {"state", &solver_state}, {"denominator", &denominator} });
            pressure.swap(next_pressure);
            pressure_residual.swap(next_residual);
            reduce(pressure_residual, residual_stats);
            applyPreconditioner(false);
        }
        state = readScalar(solver_state);
        // project verifies the true remaining error on velocity before accepting this correction
        if (state[0] < 0.0 || state[2] <= tolerance) break;
        batch = use_multigrid ? extra_batch : 8;
        extra_batch = std::min(extra_batch * 2, 128);
    }
    removeMean(pressure, next_pressure);
    return int(state[3] + 0.5f);
}

FluidGridGPU::ProjectionResult FluidGridGPU::Impl::project(double dt, int max_iterations, double divergence_tolerance)
{
    velocity_stats_valid = false;
    rebuildPressureCoefficients();

    run(predict_pressure, { &pressure_base }, { {"latest", &pressure}, {"previous", &previous_pressure} });
    previous_pressure.swap(pressure);
    removeMean(pressure_base, next_pressure);
    // apply the warm start first, then solve the remaining divergence as a smaller pressure correction
    run(correct_velocity, { &next_u }, { {"velocity", &u}, {"pressure", &pressure_base}, {"coefficients", &pressure_coefficients} }, float(dt));
    run(correct_velocity, { &next_v }, { {"velocity", &v}, {"pressure", &pressure_base}, {"coefficients", &pressure_coefficients} }, float(dt), 1);
    u.swap(next_u);
    v.swap(next_v);
    ProjectionResult result;
    for (;;)
    {
        run(divergence, { &pressure_rhs }, { {"u", &u}, {"v", &v} }, float(dt), 1);
        removeMean(pressure_rhs, pressure_product);
        // leave room for float rounding when applying the pressure gradient to velocity
        const int iterations = solvePressure(max_iterations - result.iterations, float(0.25 * divergence_tolerance * cell_size * cell_size / dt), result.iterations != 0);
        if (result.iterations == 0) previous_solve_iterations = iterations;
        result.iterations += iterations;
        run(correct_velocity, { &next_u }, { {"velocity", &u}, {"pressure", &pressure}, {"coefficients", &pressure_coefficients} }, float(dt));
        run(correct_velocity, { &next_v }, { {"velocity", &v}, {"pressure", &pressure}, {"coefficients", &pressure_coefficients} }, float(dt), 1);
        u.swap(next_u);
        v.swap(next_v);
        run(add_pressure, { &next_pressure }, { {"base", &pressure_base}, {"correction", &pressure} });
        pressure_base.swap(next_pressure);

        // measure actual divergence; refine from zero if float recurrence drift left too much error
        result.max_divergence = maxDivergence();
        result.converged = result.max_divergence <= divergence_tolerance;
        if (result.converged || iterations == 0 || result.iterations >= max_iterations) break;
    }
    // retain total pressure for next step's prediction, after applying the corrections separately
    pressure.swap(pressure_base);
    removeMean(pressure, next_pressure);
    return result;
}

double FluidGridGPU::Impl::maxDivergence()
{
    if (reduction_levels.empty())
        run(velocity_stats, { &residual_stats }, { {"u", &u}, {"v", &v} });
    else
    {
        run(velocity_stats, { &reduction_levels.front() }, { {"u", &u}, {"v", &v} });
        reduce(reduction_levels.front(), residual_stats, 4, nullptr, 1);
    }
    const auto result = readScalar(residual_stats);
    cached_outflow = result[2];
    velocity_stats_valid = result[3] == 0;
    return velocity_stats_valid ? result[1] : std::numeric_limits<double>::infinity();
}

void FluidGridGPU::Impl::applyGravity(double dt)
{
    velocity_stats_valid = false;
    run(gravity, { &next_v }, { {"velocity", &v} }, float(gravity_y * dt));
    v.swap(next_v);
}

FluidGridGPU::FluidGridGPU(int width, int height, double cell_size) : width(width), height(height), cell_size(cell_size) {}
FluidGridGPU::~FluidGridGPU() = default;
void FluidGridGPU::destroy() { gpu.reset(); }
std::uint32_t FluidGridGPU::waterTexture() const { return gpu->water_fraction.id; }

void FluidGridGPU::upload(const State& state)
{
#ifdef __EMSCRIPTEN__
    if (!emscripten_webgl_get_current_context())
        throw std::runtime_error("FluidGridGPU requires a current graphics context");
    if (!emscripten_webgl_enable_extension(emscripten_webgl_get_current_context(), "EXT_color_buffer_float"))
        throw std::runtime_error("FluidGridGPU requires EXT_color_buffer_float");
#else
    // numeric-only headless runs do not create a GL context; --snapshots enables one
    if (!glad_glGetString || !glGetString(GL_VERSION))
        throw std::runtime_error("FluidGridGPU requires a current graphics context; use --snapshots when running headlessly");
#endif
    ComputeState guard;
    if (!gpu) gpu = std::make_unique<Impl>(width, height);
    gpu->configure(*this);
    auto upload = [&](Texture& field, const std::vector<float>& values)
    {
        if (values.empty()) { gpu->run(gpu->fill, { &field }); return; }
        // host-provided arrays must match the destination to avoid reading past their storage
        if (values.size() != size_t(field.width) * field.height)
            throw std::invalid_argument("FluidGridGPU upload dimensions do not match the grid");
        glBindTexture(GL_TEXTURE_2D, field.id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, field.width, field.height, GL_RED, GL_FLOAT, values.data());
    };
    upload(gpu->water_fraction, state.water_fraction);
    upload(gpu->u, state.u);
    upload(gpu->v, state.v);
    upload(gpu->pressure, state.pressure);
    upload(gpu->previous_pressure, state.pressure);
    gpu->previous_solve_iterations = 7;
    gpu->cached_outflow = 0;
    gpu->velocity_stats_valid = state.u.empty() && state.v.empty();
}

FluidGridGPU::State FluidGridGPU::download() const
{
    ComputeState guard;
    auto read = [&](const Texture& field)
    {
#ifndef __EMSCRIPTEN__
        std::vector<float> values(size_t(field.width) * field.height);
        gpu->runner.target({ &field });
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, field.width, field.height, GL_RED, GL_FLOAT, values.data());
#else
        // WebGL2 float attachments use the portable RGBA readback path.
        const auto rgba = gpu->runner.read(field);
        std::vector<float> values(rgba.size() / 4);
        for (size_t i = 0; i < values.size(); ++i) values[i] = rgba[4 * i];
#endif
        return values;
    };
    return { read(gpu->water_fraction), read(gpu->u), read(gpu->v), read(gpu->pressure) };
}

FluidGridGPU::ProjectionResult FluidGridGPU::step(double dt, int max_iterations, double divergence_tolerance)
{
    if (dt <= 0.0) return {};
    ComputeState guard;
    gpu->configure(*this);
    gpu->advect(dt);
    gpu->applyGravity(dt);
    return gpu->project(dt, max_iterations, divergence_tolerance);
}

void FluidGridGPU::advect(double dt)
{
    if (dt <= 0.0) return;
    ComputeState guard;
    gpu->configure(*this);
    gpu->advect(dt);
}

void FluidGridGPU::applyGravity(double dt)
{
    ComputeState guard;
    gpu->configure(*this);
    gpu->applyGravity(dt);
}

FluidGridGPU::ProjectionResult FluidGridGPU::project(double dt, int max_iterations, double divergence_tolerance)
{
    if (dt <= 0.0) return {};
    ComputeState guard;
    gpu->configure(*this);
    return gpu->project(dt, max_iterations, divergence_tolerance);
}

double FluidGridGPU::maxDivergence() const
{
    ComputeState guard;
    gpu->configure(*this);
    return gpu->maxDivergence();
}

void FluidGridGPU::Impl::buildTransportPrograms()
{
    const std::string geometry = common + R"(
        float waterArea(vec2 normal, float offset, vec2 rect_min, vec2 rect_max)
        {
            vec2 size = rect_max - rect_min;
            float area = size.x * size.y;
            // measure the line from the rectangle's furthest water-side corner
            float min_projection =
                normal.x * (normal.x >= 0.0 ? rect_min.x : rect_max.x) +
                normal.y * (normal.y >= 0.0 ? rect_min.y : rect_max.y);
            float cut = offset - min_projection;
            vec2 span = abs(normal) * size;
            float small_span = min(span.x, span.y), large_span = max(span.x, span.y);
            if (cut <= 0.0) return 0.0;
            if (cut >= small_span + large_span) return area;
            if (small_span == 0.0) return area * cut / large_span;
            // divide before multiplying so tiny triangles cannot underflow into 0/0
            if (cut < small_span) return area * 0.5 * (cut / small_span) * (cut / large_span);
            if (cut < large_span) return area * (cut - 0.5 * small_span) / large_span;
            float dry_cut = small_span + large_span - cut;
            return area * (1.0 - 0.5 * (dry_cut / small_span) * (dry_cut / large_span));
        }
        float interfaceOffset(vec2 normal, float fraction)
        {
            float min_projection = min(normal.x, 0.0) + min(normal.y, 0.0);
            vec2 span = abs(normal);
            float small_span = min(span.x, span.y), large_span = max(span.x, span.y);
            float triangle_fraction = 0.5 * small_span / large_span;
            float cut;
            if (fraction < triangle_fraction)
                cut = sqrt(2.0 * small_span * large_span * fraction);
            else if (fraction <= 1.0 - triangle_fraction)
                cut = large_span * fraction + 0.5 * small_span;
            else
                cut = small_span + large_span - sqrt(2.0 * small_span * large_span * (1.0 - fraction));
            return min_projection + cut;
        }
        vec2 interfaceNormal(sampler2D water, ivec2 p)
        {
            // point from water towards air using the same four neighbouring fractions
            vec2 normal = vec2(
                at(water, max(p - ivec2(1, 0), ivec2(0))) - at(water, min(p + ivec2(1, 0), grid_size - 1)),
                at(water, max(p - ivec2(0, 1), ivec2(0))) - at(water, min(p + ivec2(0, 1), grid_size - 1)));
            // scaling the normal leaves the interface unchanged and keeps float geometry well scaled
            float scale = max(abs(normal.x), abs(normal.y));
            return scale == 0.0 ? vec2(1, 0) : normal / scale;
        }
    )";
    runner.build(reconstruct, "rebuildInterfaces", geometry, R"(
        uniform sampler2D water;
        void main()
        {
            float fraction = at(water, cell());
            if (fraction <= 0.0 || fraction >= 1.0) { output0 = vec4(0); return; }
            vec2 normal = interfaceNormal(water, cell());
            output0 = vec4(normal, interfaceOffset(normal, fraction), 0);
        }
    )");
    runner.build(transfers, "rebuildTransfersX/Y", geometry, R"(
        uniform sampler2D water, interfaces, flow;
        void main()
        {
            ivec2 p = cell(), axis = mode == 0 ? ivec2(1, 0) : ivec2(0, 1);
            int coordinate = mode == 0 ? p.x : p.y, extent = mode == 0 ? grid_size.x : grid_size.y;
            if (coordinate == 0 || coordinate == extent) { output0 = vec4(0); return; }
            float fluid = at(flow, p) * time_step / cell_size;
            ivec2 donor = fluid > 0.0 ? p - axis : p;
            float fraction = at(water, donor), water_amount = 0.0;
            if (fraction >= 1.0) water_amount = abs(fluid);
            else if (fraction > 0.0 && fluid != 0.0)
            {
                // intersect the reconstructed water with the outgoing edge strip
                vec2 rect_min = vec2(0), rect_max = vec2(1);
                if (fluid > 0.0) rect_min = vec2(axis) * (1.0 - abs(fluid));
                else rect_max -= vec2(axis) * (1.0 - abs(fluid));
                vec3 geometry = texelFetch(interfaces, donor, 0).rgb;
                water_amount = waterArea(geometry.xy, geometry.z, rect_min, rect_max);
            }
            water_amount *= sign(fluid);
            float mass = water_density * water_amount + air_density * (fluid - water_amount);
            output0 = vec4(water_amount, mass, 0, 0);
        }
    )");
    runner.build(freeze_weights, "split weights", common, R"(
        uniform sampler2D water;
        void main() { output0 = vec4(at(water, cell()) > 0.5 ? 1.0 : 0.0); }
    )");
    runner.build(momentum, "momentum and velocity", common, R"(
        OUT1 vec4 output1;
        uniform sampler2D density, source;
        void main()
        {
            bool horizontal = mode == 0 || mode == 2;
            ivec2 p = cell(), axis = horizontal ? ivec2(1, 0) : ivec2(0, 1);
            output1 = vec4(at(source, p));
            int coordinate = horizontal ? p.x : p.y, extent = horizontal ? grid_size.x : grid_size.y;
            if (coordinate == 0 || coordinate == extent) { output0 = vec4(0); return; }
            // a face's momentum region contains half of each adjacent cell
            float face_density = 0.5 * (at(density, p - axis) + at(density, p));
            float value = at(source, p);
            output0 = vec4(mode < 2 ? face_density * value : value / face_density);
        }
    )");
    runner.build(transport_momentum, "advectMomentumX/Y", common, R"(
        uniform sampler2D momentum, donor_velocity, original_velocity, flow, weights, transfers;
        float mass(ivec2 p) { return texelFetch(transfers, p, 0).g; }
        float massCorrection(ivec2 p, ivec2 axis)
        {
            float weight = at(weights, p) != 0.0 ? water_density : air_density;
            return weight * (time_step / cell_size) * (at(flow, p + axis) - at(flow, p));
        }
        float momentumTransfer(float amount, float before, float after)
        {
            return amount * (amount >= 0.0 ? before : after);
        }
        void main()
        {
            // mode selects X/u, X/v, Y/u or Y/v; both components travel in either direction
            bool horizontal_component = mode == 0 || mode == 2;
            bool horizontal_transport = mode < 2;
            ivec2 p = cell();
            ivec2 component_axis = horizontal_component ? ivec2(1, 0) : ivec2(0, 1);
            ivec2 transport_axis = horizontal_transport ? ivec2(1, 0) : ivec2(0, 1);
            int normal_coordinate = horizontal_component ? p.x : p.y;
            int normal_extent = horizontal_component ? grid_size.x : grid_size.y;
            if (normal_coordinate == 0 || normal_coordinate == normal_extent) { output0 = vec4(0); return; }

            // average the two adjacent cells' mass transfers onto the face region's boundaries
            float before_mass = 0.5 * (mass(p - component_axis) + mass(p));
            float after_mass = 0.5 * (mass(p - component_axis + transport_axis) + mass(p + transport_axis));
            int coordinate = horizontal_transport ? p.x : p.y;
            ivec2 size = textureSize(donor_velocity, 0);
            int extent = horizontal_transport ? size.x : size.y;
            float before = coordinate > 0
                ? momentumTransfer(before_mass, at(donor_velocity, p - transport_axis), at(donor_velocity, p)) : 0.0;
            float after = coordinate + 1 < extent
                ? momentumTransfer(after_mass, at(donor_velocity, p), at(donor_velocity, p + transport_axis)) : 0.0;
            float correction = 0.5 * (massCorrection(p - component_axis, transport_axis) + massCorrection(p, transport_axis));
            output0 = vec4(at(momentum, p) + before - after + at(original_velocity, p) * correction);
        }
    )");
    runner.build(transport_water, "transport water", common, R"(
        uniform sampler2D water, transfers, flow, weights;
        void main()
        {
            ivec2 p = cell(), axis = mode == 0 ? ivec2(1, 0) : ivec2(0, 1);
            float weight = at(weights, p);
            // Subtract the split correction at each face, so pure water's equal fluxes cancel exactly.
            float before = at(transfers, p) - weight * (at(flow, p) * time_step / cell_size);
            float after = at(transfers, p + axis) - weight * (at(flow, p + axis) * time_step / cell_size);
            output0 = vec4(at(water, p) + (before - after));
        }
    )");

}

void FluidGridGPU::Impl::rebuildInterfaces()
{
    run(reconstruct, { &interfaces }, { {"water", &water_fraction} });
}

void FluidGridGPU::Impl::rebuildTransfersX(double dt)
{
    run(transfers, { &transfers_u }, { {"water", &water_fraction}, {"interfaces", &interfaces}, {"flow", &u} }, float(dt));
}

void FluidGridGPU::Impl::rebuildTransfersY(double dt)
{
    run(transfers, { &transfers_v }, { {"water", &water_fraction}, {"interfaces", &interfaces}, {"flow", &v} }, float(dt), 1);
}

void FluidGridGPU::Impl::rebuildMomentum()
{
    rebuildDensity();
    // Initialize donor velocities alongside momentum, without multiplying and dividing them by density.
    run(momentum, { &momentum_u, &next_u }, { {"density", &density}, {"source", &u} });
    run(momentum, { &momentum_v, &next_v }, { {"density", &density}, {"source", &v} }, 0, 1);
}

void FluidGridGPU::Impl::recoverVelocity()
{
    rebuildDensity();
    run(momentum, { &next_u }, { {"density", &density}, {"source", &momentum_u} }, 0, 2);
    run(momentum, { &next_v }, { {"density", &density}, {"source", &momentum_v} }, 0, 3);
}

void FluidGridGPU::Impl::advectMomentumX(double dt)
{
    run(transport_momentum, { &next_momentum_u }, {
        {"momentum", &momentum_u}, {"donor_velocity", &next_u}, {"original_velocity", &u},
        {"flow", &u}, {"weights", &split_weights}, {"transfers", &transfers_u} }, float(dt), 0);
    run(transport_momentum, { &next_momentum_v }, {
        {"momentum", &momentum_v}, {"donor_velocity", &next_v}, {"original_velocity", &v},
        {"flow", &u}, {"weights", &split_weights}, {"transfers", &transfers_u} }, float(dt), 1);
    momentum_u.swap(next_momentum_u);
    momentum_v.swap(next_momentum_v);
}

void FluidGridGPU::Impl::advectMomentumY(double dt)
{
    run(transport_momentum, { &next_momentum_u }, {
        {"momentum", &momentum_u}, {"donor_velocity", &next_u}, {"original_velocity", &u},
        {"flow", &v}, {"weights", &split_weights}, {"transfers", &transfers_v} }, float(dt), 2);
    run(transport_momentum, { &next_momentum_v }, {
        {"momentum", &momentum_v}, {"donor_velocity", &next_v}, {"original_velocity", &v},
        {"flow", &v}, {"weights", &split_weights}, {"transfers", &transfers_v} }, float(dt), 3);
    momentum_u.swap(next_momentum_u);
    momentum_v.swap(next_momentum_v);
}

void FluidGridGPU::Impl::advect(double dt)
{
    rebuildMomentum();
    if (!velocity_stats_valid && !std::isfinite(maxDivergence()))
        throw std::runtime_error("FluidGridGPU encountered non-finite transport velocity");
    const double max_outflow = cached_outflow;
    const double max_step = max_outflow > 0.0 ? 0.5 * cell_size / max_outflow : dt;

    // u/v stay fixed throughout transport; next_u/v contain the evolving donor velocities
    for (double remaining = dt; remaining > 0.0;)
    {
        const double step = std::min(remaining, max_step);
        run(freeze_weights, { &split_weights }, { {"water", &water_fraction} });
        rebuildInterfaces();
        rebuildTransfersX(step);
        advectMomentumX(step);
        run(transport_water, { &next_water_fraction }, {
            {"water", &water_fraction}, {"transfers", &transfers_u}, {"flow", &u}, {"weights", &split_weights} }, float(step));
        water_fraction.swap(next_water_fraction);
        recoverVelocity();

        rebuildInterfaces();
        rebuildTransfersY(step);
        advectMomentumY(step);
        run(transport_water, { &next_water_fraction }, {
            {"water", &water_fraction}, {"transfers", &transfers_v}, {"flow", &v}, {"weights", &split_weights} }, float(step), 1);
        water_fraction.swap(next_water_fraction);
        recoverVelocity();
        remaining -= step;
    }
    u.swap(next_u);
    v.swap(next_v);
    velocity_stats_valid = false;
}

}
