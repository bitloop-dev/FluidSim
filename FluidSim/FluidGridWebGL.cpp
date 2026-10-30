#include "FluidGridWebGL.h"
#include "FluidGridWebGLInternal.h"
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


struct FluidGridWebGL::Impl
{
    const int width, height;
    float cell_size = 1, gravity_y = 100, tolerance = 0;
    PassRunner runner;
    Texture u, v, next_u, next_v, liquid_fraction, next_liquid_fraction;
    // RG holds liquid mass / cell volume and gas mass / cell volume separately.
    Texture density, next_density;
    Texture color, next_color;
    // Each RGBA group holds four ingredients, liquid first and then gas.
    struct ComponentGroup
    {
        Texture mass, next_mass, correction;
        std::array<float, 4> liquid_mask = {}, gas_mask = {};
        ComponentGroup(int w, int h) : mass(w, h, 4), next_mass(w, h, 4), correction(w, h, 4) {}
    };
    std::vector<ComponentGroup> components;
    std::vector<float> edit_upload;
    size_t liquid_count = 0, gas_count = 0;
    bool density_valid = false;
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

    Program fill, density_pass, color_pass, pressure_coefficient, divergence, pressure_operator;
    Program transport_components_x, transport_components_y, transport_single_x, transport_single_y;
    Program predict_pressure, add_pressure, subtract_mean, residual, reduce_pass, precondition, coefficients, direction, update_pressure;
    Program correct_velocity, gravity, velocity_stats;
    Program reconstruct, freeze_weights, transfers, transport_momentum, transport_liquid, momentum;

    Impl(int w, int h);
    void configure(const FluidGridWebGL& grid)
    {
        if (cell_size != float(grid.cell_size)) density_valid = false;
        cell_size = float(grid.cell_size);
        gravity_y = float(grid.gravity_y);
        use_multigrid = grid.pressure_preconditioner == FluidGridWebGL::PressurePreconditioner::Multigrid;
    }
    using Input = std::pair<const char*, const Texture*>;
    void run(Program& program, std::initializer_list<const Texture*> outputs,
             std::initializer_list<Input> inputs = {}, float dt = 0, int mode = 0)
    {
        runner.bind(program, outputs, inputs);
        glUniform2i(program.location("grid_size"), width, height);
        program.scalar("cell_size", cell_size);
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
    void uploadComponents(const FluidGridWebGL& grid, const State& state);
    void rebuildDensity();
    void rebuildColor(const FluidGridWebGL& grid);
    void sumGroups(Program& program, Texture& result, Texture& scratch, size_t first);
    void advectComponents(double dt, int direction);
    void rebuildPressureCoefficients();
    void applyPressureOperator(const Texture& input);
    void rebuildResidual();
    void applyPreconditioner(bool restart);
    int solvePressure(int max_iterations, float residual_tolerance, bool refinement);
    FluidGridWebGL::ProjectionResult project(double dt, int max_iterations, double divergence_tolerance);
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

FluidGridWebGL::Impl::Impl(int w, int h) :
    width(w), height(h),
    u(w + 1, h), v(w, h + 1), next_u(w + 1, h), next_v(w, h + 1),
    liquid_fraction(w, h), next_liquid_fraction(w, h), density(w, h, 2), next_density(w, h, 2),
    color(w, h, 4), next_color(w, h, 4),
    momentum_u(w + 1, h), momentum_v(w, h + 1), next_momentum_u(w + 1, h), next_momentum_v(w, h + 1),
    interfaces(w, h, 4), split_weights(w, h, 2), transfers_u(w + 1, h, 2), transfers_v(w, h + 1, 2),
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
    for (const Texture* texture : { &u, &density, &interfaces })
    {
        runner.target({ texture });
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("FluidGridWebGL requires renderable R32F, RG32F and RGBA32F textures");
    }
    auto build = [&](Program& p, const char* name, const char* body) { runner.build(p, name, common, body); };
    build(fill, "fill", R"(
        void main() { output0 = mode == 1 ? vec4(0, 0, 0, 1) : vec4(time_step); }
    )");
    build(density_pass, "rebuildDensity", R"(
        uniform sampler2D masses0, masses1, masses2, masses3, previous;
        uniform vec4 liquid_masks[4], gas_masks[4];
        uniform int group_count;
        vec2 phases(sampler2D field, int group)
        {
            vec4 mass = texelFetch(field, cell(), 0) / (cell_size * cell_size);
            return vec2(dot(mass, liquid_masks[group]), dot(mass, gas_masks[group]));
        }
        void main()
        {
            vec2 sum = phases(masses0, 0);
            if (group_count > 1) sum += phases(masses1, 1);
            if (group_count > 2) sum += phases(masses2, 2);
            if (group_count > 3) sum += phases(masses3, 3);
            if (mode != 0) sum += texelFetch(previous, cell(), 0).rg;
            output0 = vec4(sum, 0, 0);
        }
    )");
    build(color_pass, "catalogue colors", R"(
        uniform sampler2D masses0, masses1, masses2, masses3, previous;
        uniform vec4 red[4], green[4], blue[4];
        uniform int group_count;
        vec3 color(sampler2D field, int group)
        {
            vec4 mass = texelFetch(field, cell(), 0);
            return vec3(dot(mass, red[group]), dot(mass, green[group]), dot(mass, blue[group]));
        }
        void main()
        {
            vec3 sum = color(masses0, 0);
            if (group_count > 1) sum += color(masses1, 1);
            if (group_count > 2) sum += color(masses2, 2);
            if (group_count > 3) sum += color(masses3, 3);
            if (mode != 0) sum += texelFetch(previous, cell(), 0).rgb;
            output0 = vec4(sum, 1);
        }
    )");
    build(pressure_coefficient, "pressure face weights", R"(
        uniform sampler2D density;
        void main()
        {
            ivec2 p = cell();
            float center = densityAt(density, p);
            vec4 neighbors = vec4(
                densityAt(density, max(p - ivec2(1, 0), ivec2(0))), densityAt(density, min(p + ivec2(1, 0), grid_size - 1)),
                densityAt(density, max(p - ivec2(0, 1), ivec2(0))), densityAt(density, min(p + ivec2(0, 1), grid_size - 1)));
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

void FluidGridWebGL::Impl::rebuildPressureCoefficients()
{
    rebuildDensity();
    run(pressure_coefficient, { &pressure_coefficients }, { {"density", &density} });
    if (use_multigrid) multigrid.rebuild(pressure_coefficients);
}

void FluidGridWebGL::Impl::applyPressureOperator(const Texture& input)
{
    run(pressure_operator, { &pressure_product }, { {"source", &input}, {"coefficients", &pressure_coefficients} });
}

void FluidGridWebGL::Impl::rebuildResidual()
{
    applyPressureOperator(pressure);
    run(residual, { &pressure_residual }, { {"rhs", &pressure_rhs}, {"product", &pressure_product} });
    reduce(pressure_residual, residual_stats);
}

void FluidGridWebGL::Impl::applyPreconditioner(bool restart)
{
    const Texture* correction = &pressure_correction;
    if (use_multigrid)
        correction = &multigrid.apply(pressure_residual, residual_stats, tolerance);
    else
        run(precondition, { &pressure_correction }, {
            {"residual", &pressure_residual}, {"stats", &residual_stats}, {"coefficients", &pressure_coefficients} });
    if (use_multigrid && multigrid.cpuConverged()) return;
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

int FluidGridWebGL::Impl::solvePressure(int max_iterations, float residual_tolerance, bool refinement)
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
    const bool cpu_coarse = use_multigrid && multigrid.usesCPUCoarsePressure();
    // Check short GPU-only solves directly; CPU coarse cycles carry convergence with their coarse data.
    const bool check_each_iteration = use_multigrid && !cpu_coarse && previous_solve_iterations <= 8;
    if (check_each_iteration) batch = 1;
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
            if (check_each_iteration)
            {
                const auto current = readScalar(residual_stats);
                if (current[3] == 0 && current[1] <= tolerance)
                {
                    // A converged correction needs no further V-cycle or search direction.
                    removeMean(pressure, next_pressure);
                    return submitted + 1;
                }
            }
            applyPreconditioner(false);
            if (cpu_coarse && multigrid.cpuConverged())
            {
                removeMean(pressure, next_pressure);
                return submitted + 1;
            }
        }
        state = readScalar(solver_state);
        // project verifies the true remaining error on velocity before accepting this correction
        if (state[0] < 0.0 || state[2] <= tolerance) break;
        batch = check_each_iteration ? 1 : use_multigrid ? extra_batch : 8;
        extra_batch = std::min(extra_batch * 2, 128);
    }
    removeMean(pressure, next_pressure);
    return int(state[3] + 0.5f);
}

FluidGridWebGL::ProjectionResult FluidGridWebGL::Impl::project(double dt, int max_iterations, double divergence_tolerance)
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

double FluidGridWebGL::Impl::maxDivergence()
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

void FluidGridWebGL::Impl::applyGravity(double dt)
{
    velocity_stats_valid = false;
    run(gravity, { &next_v }, { {"velocity", &v} }, float(gravity_y * dt));
    v.swap(next_v);
}

FluidGridWebGL::FluidGridWebGL(int width, int height, double cell_size) : width(width), height(height), cell_size(cell_size) {}
FluidGridWebGL::~FluidGridWebGL() = default;
void FluidGridWebGL::destroy() { gpu.reset(); }
std::uint32_t FluidGridWebGL::liquidFractionTexture() const { return gpu->liquid_fraction.id; }
std::uint32_t FluidGridWebGL::colorTexture() const
{
    ComputeState guard;
    gpu->configure(*this);
    gpu->rebuildColor(*this);
    return gpu->color.id;
}

void FluidGridWebGL::Impl::uploadComponents(const FluidGridWebGL& grid, const State& state)
{
    liquid_count = grid.liquid_mixture.components.size();
    gas_count = grid.gas_mixture.components.size();
    const size_t cell_count = size_t(width) * height;
    auto validate = [&](const auto& fields, size_t count)
    {
        if (!fields.empty() && fields.size() != count)
            throw std::invalid_argument("FluidGridWebGL upload component count does not match the mixture");
        for (const auto& field : fields)
            if (!field.empty() && field.size() != cell_count)
                throw std::invalid_argument("FluidGridWebGL upload component dimensions do not match the grid");
    };
    validate(state.liquid_mass, liquid_count);
    validate(state.gas_mass, gas_count);
    const size_t group_count = (liquid_count + gas_count + 3) / 4;
    while (components.size() > group_count) components.pop_back();
    while (components.size() < group_count) components.emplace_back(width, height);
    std::vector<float> packed(cell_count * 4);
    for (size_t group = 0; group < group_count; ++group)
    {
        auto& component = components[group];
        component.liquid_mask = component.gas_mask = {};
        std::fill(packed.begin(), packed.end(), 0.0f);
        for (size_t lane = 0; lane < 4; ++lane)
        {
            const size_t index = group * 4 + lane;
            if (index >= liquid_count + gas_count) break;
            const bool liquid = index < liquid_count;
            (liquid ? component.liquid_mask : component.gas_mask)[lane] = 1;
            const auto& fields = liquid ? state.liquid_mass : state.gas_mass;
            const size_t k = liquid ? index : index - liquid_count;
            if (fields.empty() || fields[k].empty()) continue;
            for (size_t i = 0; i < cell_count; ++i) packed[4 * i + lane] = fields[k][i];
        }
        glBindTexture(GL_TEXTURE_2D, component.mass.id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, packed.data());
    }
    density_valid = false;
}

void FluidGridWebGL::Impl::rebuildDensity()
{
    if (density_valid) return;
    if (components.empty()) run(fill, { &density });
    for (size_t first = 0; first < components.size(); first += 4)
    {
        std::array<std::array<float, 4>, 4> liquid_masks = {}, gas_masks = {};
        for (size_t j = 0; j < 4 && first + j < components.size(); ++j)
        {
            liquid_masks[j] = components[first + j].liquid_mask;
            gas_masks[j] = components[first + j].gas_mask;
        }
        glUseProgram(density_pass.id);
        glUniform4fv(density_pass.location("liquid_masks"), 4, liquid_masks[0].data());
        glUniform4fv(density_pass.location("gas_masks"), 4, gas_masks[0].data());
        sumGroups(density_pass, density, next_density, first);
    }
    density_valid = true;
}

void FluidGridWebGL::Impl::rebuildColor(const FluidGridWebGL& grid)
{
    if (components.empty()) run(fill, { &color }, {}, 0, 1);
    for (size_t first = 0; first < components.size(); first += 4)
    {
        std::array<std::array<float, 4>, 4> channels[3] = {};
        for (size_t group = 0; group < 4 && first + group < components.size(); ++group)
        for (size_t lane = 0; lane < 4; ++lane)
        {
            const size_t index = (first + group) * 4 + lane;
            if (index >= liquid_count + gas_count) break;
            const auto& component = index < liquid_count ? grid.liquid_mixture.components[index]
                : grid.gas_mixture.components[index - liquid_count];
            const auto& rgb = grid.substances[component.substance].color;
            for (size_t c = 0; c < 3; ++c)
                channels[c][group][lane] = float(rgb[c] / (component.density * grid.cell_size * grid.cell_size));
        }
        glUseProgram(color_pass.id);
        glUniform4fv(color_pass.location("red"), 4, channels[0][0].data());
        glUniform4fv(color_pass.location("green"), 4, channels[1][0].data());
        glUniform4fv(color_pass.location("blue"), 4, channels[2][0].data());
        sumGroups(color_pass, color, next_color, first);
    }
}

void FluidGridWebGL::Impl::sumGroups(Program& program, Texture& result, Texture& scratch, size_t first)
{
    // Four RGBA inputs cover sixteen ingredients per draw without exceeding sampler limits.
    const size_t count = std::min(size_t(4), components.size() - first);
    const Texture* inputs[4];
    for (size_t j = 0; j < 4; ++j) inputs[j] = &components[first + std::min(j, count - 1)].mass;
    program.integer("group_count", int(count));
    run(program, { &scratch }, { {"masses0", inputs[0]}, {"masses1", inputs[1]},
        {"masses2", inputs[2]}, {"masses3", inputs[3]}, {"previous", &result} }, 0, first != 0);
    result.swap(scratch);
}

void FluidGridWebGL::upload(const State& state)
{
#ifdef __EMSCRIPTEN__
    if (!emscripten_webgl_get_current_context())
        throw std::runtime_error("FluidGridWebGL requires a current graphics context");
    if (!emscripten_webgl_enable_extension(emscripten_webgl_get_current_context(), "EXT_color_buffer_float"))
        throw std::runtime_error("FluidGridWebGL requires EXT_color_buffer_float");
#else
    // numeric-only headless runs do not create a GL context; --snapshots enables one
    if (!glad_glGetString || !glGetString(GL_VERSION))
        throw std::runtime_error("FluidGridWebGL requires a current graphics context; use --snapshots when running headlessly");
#endif
    ComputeState guard;
    if (!gpu) gpu = std::make_unique<Impl>(width, height);
    gpu->configure(*this);
    auto upload = [&](Texture& field, const std::vector<float>& values)
    {
        if (values.empty()) { gpu->run(gpu->fill, { &field }); return; }
        // host-provided arrays must match the destination to avoid reading past their storage
        if (values.size() != size_t(field.width) * field.height)
            throw std::invalid_argument("FluidGridWebGL upload dimensions do not match the grid");
        glBindTexture(GL_TEXTURE_2D, field.id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, field.width, field.height, GL_RED, GL_FLOAT, values.data());
    };
    upload(gpu->liquid_fraction, state.liquid_fraction);
    upload(gpu->u, state.u);
    upload(gpu->v, state.v);
    upload(gpu->pressure, state.pressure);
    upload(gpu->previous_pressure, state.pressure);
    gpu->uploadComponents(*this, state);
    gpu->previous_solve_iterations = 7;
    gpu->cached_outflow = 0;
    gpu->velocity_stats_valid = state.u.empty() && state.v.empty();
    gpu->multigrid.useCPUCoarsePressure(cpu_pressure_cells);
}

void FluidGridWebGL::commitEdits(detail::CellEditBatch& batch)
{
    batch.prepare(width, height, cell_size * cell_size, liquid_mixture, gas_mixture);
    if (!gpu || gpu->liquid_count != liquid_mixture.components.size() || gpu->gas_count != gas_mixture.components.size())
        throw std::logic_error("Upload the configured fluid mixtures before committing edits");

    ComputeState guard;
    const auto& cells = batch.cells;
    for (size_t first = 0; first < cells.size();)
    {
        const int x = cells[first].x, y = cells[first].y;
        auto rowEnd = [&](size_t start) {
            size_t end = start + 1;
            while (end < cells.size() && cells[end].y == cells[start].y && cells[end].x == cells[end - 1].x + 1) ++end;
            return end;
        };
        size_t last = rowEnd(first);
        const int span_width = int(last - first);
        int span_height = 1;
        // Equal adjacent row spans form one rectangle, without uploading gaps in a shape.
        while (last < cells.size() && cells[last].x == x && cells[last].y == y + span_height)
        {
            const size_t end = rowEnd(last);
            if (end - last != size_t(span_width)) break;
            last = end;
            ++span_height;
        }

        const size_t count = last - first;
        auto& data = gpu->edit_upload;
        data.assign(count * (1 + 4 * gpu->components.size()), 0.0f);
        for (size_t i = 0; i < count; ++i)
        {
            const auto& cell = cells[first + i];
            data[i] = cell.component < gpu->liquid_count ? 1.0f : 0.0f;
            data[count + (cell.component / 4) * count * 4 + i * 4 + cell.component % 4] = float(cell.mass);
        }

        glBindTexture(GL_TEXTURE_2D, gpu->liquid_fraction.id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, span_width, span_height, GL_RED, GL_FLOAT, data.data());
        for (size_t group = 0; group < gpu->components.size(); ++group)
        {
            glBindTexture(GL_TEXTURE_2D, gpu->components[group].mass.id);
            glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, span_width, span_height, GL_RGBA, GL_FLOAT, data.data() + count * (1 + 4 * group));
        }
        first = last;
    }
    gpu->density_valid = false;
}

FluidGridWebGL::State FluidGridWebGL::download() const
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
    State state{ read(gpu->liquid_fraction), read(gpu->u), read(gpu->v), read(gpu->pressure) };
    const size_t cell_count = size_t(width) * height;
    state.liquid_mass.assign(gpu->liquid_count, std::vector<float>(cell_count));
    state.gas_mass.assign(gpu->gas_count, std::vector<float>(cell_count));
    for (size_t group = 0; group < gpu->components.size(); ++group)
    {
        const auto packed = gpu->runner.read(gpu->components[group].mass);
        for (size_t lane = 0; lane < 4; ++lane)
        {
            const size_t index = group * 4 + lane;
            if (index >= gpu->liquid_count + gpu->gas_count) break;
            auto& mass = index < gpu->liquid_count ? state.liquid_mass[index] : state.gas_mass[index - gpu->liquid_count];
            for (size_t i = 0; i < cell_count; ++i) mass[i] = packed[4 * i + lane];
        }
    }
    return state;
}

FluidGridWebGL::ProjectionResult FluidGridWebGL::step(double dt, int max_iterations, double divergence_tolerance)
{
    if (dt <= 0.0) return {};
    ComputeState guard;
    gpu->configure(*this);
    gpu->advect(dt);
    gpu->applyGravity(dt);
    return gpu->project(dt, max_iterations, divergence_tolerance);
}

void FluidGridWebGL::advect(double dt)
{
    if (dt <= 0.0) return;
    ComputeState guard;
    gpu->configure(*this);
    gpu->advect(dt);
}

void FluidGridWebGL::applyGravity(double dt)
{
    ComputeState guard;
    gpu->configure(*this);
    gpu->applyGravity(dt);
}

FluidGridWebGL::ProjectionResult FluidGridWebGL::project(double dt, int max_iterations, double divergence_tolerance)
{
    if (dt <= 0.0) return {};
    ComputeState guard;
    gpu->configure(*this);
    return gpu->project(dt, max_iterations, divergence_tolerance);
}

double FluidGridWebGL::maxDivergence() const
{
    ComputeState guard;
    gpu->configure(*this);
    return gpu->maxDivergence();
}

void FluidGridWebGL::Impl::buildTransportPrograms()
{
    const std::string geometry = common + R"(
        float liquidArea(vec2 normal, float offset, vec2 rect_min, vec2 rect_max)
        {
            vec2 size = rect_max - rect_min;
            float area = size.x * size.y;
            // measure the line from the rectangle's furthest liquid-side corner
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
        vec2 interfaceNormal(sampler2D liquid, ivec2 p)
        {
            // Use the nearest boundary cell when a neighbour lies outside the grid.
            ivec2 lo = max(p - ivec2(1), ivec2(0)), hi = min(p + ivec2(1), grid_size - 1);
            float top_left = at(liquid, lo), top = at(liquid, ivec2(p.x, lo.y)), top_right = at(liquid, ivec2(hi.x, lo.y));
            float left = at(liquid, ivec2(lo.x, p.y)), right = at(liquid, ivec2(hi.x, p.y));
            float bottom_left = at(liquid, ivec2(lo.x, hi.y)), bottom = at(liquid, ivec2(p.x, hi.y)), bottom_right = at(liquid, hi);

            // Column/row totals measure surface height rather than one cell's abrupt fraction jump.
            float left_sum = top_left + left + bottom_left, right_sum = top_right + right + bottom_right;
            float top_sum = top_left + top + top_right, bottom_sum = bottom_left + bottom + bottom_right;
            vec2 centered = 0.5 * vec2(left_sum - right_sum, top_sum - bottom_sum);

            // Youngs' weighted gradient supplies the alternative near corners and thin features.
            vec2 youngs = vec2(left_sum + left - right_sum - right, top_sum + top - bottom_sum - bottom);
            vec2 normal = centered;
            if (centered.x == 0.0 && centered.y == 0.0)
                normal = youngs;
            // MYC chooses the dominant height direction, then the steeper slope estimate.
            else if (abs(centered.x) <= abs(centered.y))
            {
                normal.y = centered.y > 0.0 ? 1.0 : -1.0;
                if (abs(youngs.x) > abs(centered.x * youngs.y)) normal = youngs;
            }
            else
            {
                normal.x = centered.x > 0.0 ? 1.0 : -1.0;
                if (abs(youngs.y) > abs(centered.y * youngs.x)) normal = youngs;
            }
            // scaling the normal leaves the interface unchanged and keeps float geometry well scaled
            float scale = max(abs(normal.x), abs(normal.y));
            return scale == 0.0 ? vec2(1, 0) : normal / scale;
        }
    )";
    runner.build(reconstruct, "rebuildInterfaces", geometry, R"(
        uniform sampler2D liquid;
        void main()
        {
            float fraction = at(liquid, cell());
            if (fraction <= 0.0 || fraction >= 1.0) { output0 = vec4(0); return; }
            vec2 normal = interfaceNormal(liquid, cell());
            output0 = vec4(normal, interfaceOffset(normal, fraction), 0);
        }
    )");
    runner.build(transfers, "rebuildTransfersX/Y", geometry, R"(
        uniform sampler2D liquid, interfaces, flow, density;
        void main()
        {
            ivec2 p = cell(), axis = mode == 0 ? ivec2(1, 0) : ivec2(0, 1);
            int coordinate = mode == 0 ? p.x : p.y, extent = mode == 0 ? grid_size.x : grid_size.y;
            if (coordinate == 0 || coordinate == extent) { output0 = vec4(0); return; }
            float fluid = at(flow, p) * time_step / cell_size;
            ivec2 donor = fluid > 0.0 ? p - axis : p;
            float fraction = at(liquid, donor), liquid_amount = 0.0;
            if (fraction >= 1.0) liquid_amount = abs(fluid);
            else if (fraction > 0.0 && fluid != 0.0)
            {
                // intersect the reconstructed liquid with the outgoing edge strip
                vec2 rect_min = vec2(0), rect_max = vec2(1);
                if (fluid > 0.0) rect_min = vec2(axis) * (1.0 - abs(fluid));
                else rect_max -= vec2(axis) * (1.0 - abs(fluid));
                vec3 geometry = texelFetch(interfaces, donor, 0).rgb;
                liquid_amount = liquidArea(geometry.xy, geometry.z, rect_min, rect_max);
            }
            // Float geometry can overestimate a vanishing sliver; an intersection cannot
            // exceed either the donor's phase volume or the outgoing strip's volume.
            float donor_liquid = clamp(fraction, 0.0, 1.0), strip = abs(fluid);
            liquid_amount = clamp(liquid_amount, max(0.0, strip - (1.0 - donor_liquid)), min(strip, donor_liquid));
            liquid_amount *= sign(fluid);
            // Sum the same donor ratios used by every ingredient, once for momentum.
            vec2 phases = texelFetch(density, donor, 0).rg;
            vec2 ratios = vec2(fraction > 0.0 ? liquid_amount / fraction : 0.0,
                fraction < 1.0 ? (fluid - liquid_amount) / (1.0 - fraction) : 0.0);
            float mass = dot(phases, ratios);
            output0 = vec4(liquid_amount, mass, 0, 0);
        }
    )");
    runner.build(freeze_weights, "split weights", common, R"(
        uniform sampler2D liquid, density;
        void main()
        {
            float f = at(liquid, cell());
            vec2 phases = texelFetch(density, cell(), 0).rg;
            bool selected = f > 0.5;
            output0 = vec4(selected ? 1.0 : 0.0, selected ? phases.r / f : phases.g / (1.0 - f), 0, 0);
        }
    )");
    const char* component_transport = R"(
        uniform sampler2D liquid, masses, corrections, transfers, flow, weights;
        uniform vec4 liquid_mask, gas_mask;
#ifdef HORIZONTAL
        OUT1 vec4 output1;
#endif
#ifdef SINGLE_GROUP
    #ifdef HORIZONTAL
        OUT2 vec4 output2;
        OUT3 vec4 output3;
    #else
        OUT1 vec4 output1;
        OUT2 vec4 output2;
    #endif
#endif
        vec4 transfer(ivec2 p, ivec2 axis)
        {
            int coordinate = mode == 0 ? p.x : p.y;
            int extent = mode == 0 ? grid_size.x : grid_size.y;
            if (coordinate == 0 || coordinate == extent) return vec4(0);
            float volume = at(flow, p) * time_step / cell_size;
            ivec2 donor = volume > 0.0 ? p - axis : p;
            float f = at(liquid, donor), liquid_volume = at(transfers, p);
            vec4 ratio = liquid_mask * (f > 0.0 ? liquid_volume / f : 0.0) +
                gas_mask * (f < 1.0 ? (volume - liquid_volume) / (1.0 - f) : 0.0);
            return texelFetch(masses, donor, 0) * ratio;
        }
        void main()
        {
            ivec2 p = cell(), axis = mode == 0 ? ivec2(1, 0) : ivec2(0, 1);
            float f = at(liquid, p);
#ifdef HORIZONTAL
            // Capture the unsplit correction while this cell's original mass is in registers.
            vec4 selected = f > 0.5 ? liquid_mask : gas_mask;
            vec4 weight = texelFetch(masses, p, 0) * selected / (f > 0.5 ? f : 1.0 - f);
            output1 = weight;
#else
            vec4 weight = texelFetch(corrections, p, 0);
#endif
            // Pair each split correction with its face flux to preserve a pure uniform phase.
            vec4 before = transfer(p, axis) - weight * (at(flow, p) * time_step / cell_size);
            vec4 after = transfer(p + axis, axis) - weight * (at(flow, p + axis) * time_step / cell_size);
            output0 = texelFetch(masses, p, 0) + (before - after);
#ifdef SINGLE_GROUP
            // One packed group contains the whole mixture: emit density and geometry together.
            vec2 phases = vec2(dot(output0, liquid_mask), dot(output0, gas_mask)) / (cell_size * cell_size);
            float split = at(weights, p);
            float before_volume = at(transfers, p) - split * (at(flow, p) * time_step / cell_size);
            float after_volume = at(transfers, p + axis) - split * (at(flow, p + axis) * time_step / cell_size);
            float fraction = f + (before_volume - after_volume);
    #ifdef HORIZONTAL
            output2 = vec4(phases, 0, 0);
            output3 = vec4(fraction);
    #else
            output1 = vec4(phases, 0, 0);
            output2 = vec4(fraction);
    #endif
#endif
        }
    )";
    runner.build(transport_components_x, "advect components X", common + "\n#define HORIZONTAL\n", component_transport);
    runner.build(transport_components_y, "advect components Y", common, component_transport);
    runner.build(transport_single_x, "advect single group X", common + "\n#define HORIZONTAL\n#define SINGLE_GROUP\n", component_transport);
    runner.build(transport_single_y, "advect single group Y", common + "\n#define SINGLE_GROUP\n", component_transport);
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
            float face_density = 0.5 * (densityAt(density, p - axis) + densityAt(density, p));
            float value = at(source, p);
            output0 = vec4(mode < 2 ? face_density * value : value / face_density);
        }
    )");
    runner.build(transport_momentum, "advectMomentumX/Y", common, R"(
        uniform sampler2D momentum, donor_velocity, original_velocity, flow, weights, transfers;
        float mass(ivec2 p) { return texelFetch(transfers, p, 0).g; }
        float massCorrection(ivec2 p, ivec2 axis)
        {
            float weight = texelFetch(weights, p, 0).g;
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
    runner.build(transport_liquid, "transport liquid", common, R"(
        uniform sampler2D liquid, transfers, flow, weights;
        void main()
        {
            ivec2 p = cell(), axis = mode == 0 ? ivec2(1, 0) : ivec2(0, 1);
            float weight = at(weights, p);
            // Subtract the split correction at each face, so pure liquid's equal fluxes cancel exactly.
            float before = at(transfers, p) - weight * (at(flow, p) * time_step / cell_size);
            float after = at(transfers, p + axis) - weight * (at(flow, p + axis) * time_step / cell_size);
            output0 = vec4(at(liquid, p) + (before - after));
        }
    )");

}

void FluidGridWebGL::Impl::rebuildInterfaces()
{
    run(reconstruct, { &interfaces }, { {"liquid", &liquid_fraction} });
}

void FluidGridWebGL::Impl::rebuildTransfersX(double dt)
{
    run(transfers, { &transfers_u }, { {"liquid", &liquid_fraction}, {"interfaces", &interfaces}, {"flow", &u}, {"density", &density} }, float(dt));
}

void FluidGridWebGL::Impl::rebuildTransfersY(double dt)
{
    run(transfers, { &transfers_v }, { {"liquid", &liquid_fraction}, {"interfaces", &interfaces}, {"flow", &v}, {"density", &density} }, float(dt), 1);
}

void FluidGridWebGL::Impl::advectComponents(double dt, int direction)
{
    const bool single_group = components.size() == 1;
    Program& program = single_group ? (direction == 0 ? transport_single_x : transport_single_y)
        : (direction == 0 ? transport_components_x : transport_components_y);
    for (auto& component : components)
    {
        glUseProgram(program.id);
        program.vector("liquid_mask", component.liquid_mask);
        program.vector("gas_mask", component.gas_mask);
        const std::initializer_list<Input> inputs = {
            {"liquid", &liquid_fraction}, {"masses", &component.mass}, {"corrections", direction == 0 ? &component.mass : &component.correction},
            {"transfers", direction == 0 ? &transfers_u : &transfers_v}, {"flow", direction == 0 ? &u : &v}, {"weights", &split_weights} };
        if (single_group && direction == 0)
            run(program, { &component.next_mass, &component.correction, &next_density, &next_liquid_fraction }, inputs, float(dt), direction);
        else if (single_group)
            run(program, { &component.next_mass, &next_density, &next_liquid_fraction }, inputs, float(dt), direction);
        else if (direction == 0)
            run(program, { &component.next_mass, &component.correction }, inputs, float(dt), direction);
        else
            run(program, { &component.next_mass }, inputs, float(dt), direction);
        component.mass.swap(component.next_mass);
    }
    if (single_group) density.swap(next_density);
    else run(transport_liquid, { &next_liquid_fraction }, {
        {"liquid", &liquid_fraction}, {"transfers", direction == 0 ? &transfers_u : &transfers_v},
        {"flow", direction == 0 ? &u : &v}, {"weights", &split_weights} }, float(dt), direction);
    liquid_fraction.swap(next_liquid_fraction);
    density_valid = single_group;
}

void FluidGridWebGL::Impl::rebuildMomentum()
{
    rebuildDensity();
    // Initialize donor velocities alongside momentum, without multiplying and dividing them by density.
    run(momentum, { &momentum_u, &next_u }, { {"density", &density}, {"source", &u} });
    run(momentum, { &momentum_v, &next_v }, { {"density", &density}, {"source", &v} }, 0, 1);
}

void FluidGridWebGL::Impl::recoverVelocity()
{
    rebuildDensity();
    run(momentum, { &next_u }, { {"density", &density}, {"source", &momentum_u} }, 0, 2);
    run(momentum, { &next_v }, { {"density", &density}, {"source", &momentum_v} }, 0, 3);
}

void FluidGridWebGL::Impl::advectMomentumX(double dt)
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

void FluidGridWebGL::Impl::advectMomentumY(double dt)
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

void FluidGridWebGL::Impl::advect(double dt)
{
    rebuildMomentum();
    if (!velocity_stats_valid && !std::isfinite(maxDivergence()))
        throw std::runtime_error("FluidGridWebGL encountered non-finite transport velocity");
    const double max_outflow = cached_outflow;
    const double max_step = max_outflow > 0.0 ? 0.5 * cell_size / max_outflow : dt;

    // u/v stay fixed throughout transport; next_u/v contain the evolving donor velocities
    for (double remaining = dt; remaining > 0.0;)
    {
        const double step = std::min(remaining, max_step);
        run(freeze_weights, { &split_weights }, { {"liquid", &liquid_fraction}, {"density", &density} });
        rebuildInterfaces();
        rebuildTransfersX(step);
        advectMomentumX(step);
        advectComponents(step, 0);
        recoverVelocity();

        rebuildInterfaces();
        rebuildTransfersY(step);
        advectMomentumY(step);
        advectComponents(step, 1);
        recoverVelocity();
        remaining -= step;
    }
    u.swap(next_u);
    v.swap(next_v);
    velocity_stats_valid = false;
}

}
