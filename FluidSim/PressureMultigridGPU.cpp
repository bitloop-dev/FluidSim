#include "PressureMultigridGPU.h"

namespace fluid::gpu_detail
{

PressureMultigridGPU::Level::Level(int w, int h, int s, bool finest) :
    width(w), height(h), spacing(s), coefficients(finest ? Texture() : Texture(w, h, 4)),
    rhs(finest ? Texture() : Texture(w, h)), solution(w, h), next_solution(w, h), residual(w, h)
{}

PressureMultigridGPU::PressureMultigridGPU(PassRunner& runner, int width, int height) : runner(runner)
{
    // D^-1 A has eigenvalues in [0,2]; target its upper spectrum with a fixed Chebyshev polynomial.
    for (int i = 0; i < smoothing_sweeps; ++i)
        smoothing_weights[i] = float(1.0 / (1.1 + 0.9 * std::cos(3.14159265358979323846 * (i + 0.5) / smoothing_sweeps)));
    int w = width, h = height, spacing = 1;
    for (;;)
    {
        levels.emplace_back(w, h, spacing, levels.empty());
        if (w * h <= 16) break;
        w = (w + 1) / 2; h = (h + 1) / 2; spacing *= 2;
    }
    for (size_t i = 0; i + 1 < levels.size(); ++i)
        buildTransfers(levels[i], levels[i + 1]);
    const int unknowns = std::max(1, w * h - 1);
    coarse_factor = Texture(unknowns, unknowns);

    const std::string geometry = common + R"(
        uniform ivec2 domain_size;
        uniform float spacing, mean_divisor;
        uniform sampler2D rhs_stats;
        float center(int i, float step, int extent) { return 0.5 * (float(i) * step + min(float(i + 1) * step, float(extent))); }
        float rhsAt(sampler2D rhs, ivec2 p)
        {
            float mean = 0.0;
            if (mean_divisor > 0.0)
            {
                float sum = at(rhs_stats, ivec2(0));
                mean = sum / mean_divisor;
                // Correct reciprocal rounding before removing the mean on odd grids.
                mean += (sum - mean * mean_divisor) / mean_divisor;
            }
            return at(rhs, p) - mean;
        }
        float product(sampler2D source, sampler2D coefficients, ivec2 p)
        {
            vec4 a = texelFetch(coefficients, p, 0);
            float c = at(source, p);
            return (a.x * (c - at(source, max(p - ivec2(1, 0), ivec2(0)))) + a.y * (c - at(source, min(p + ivec2(1, 0), grid_size - 1)))) +
                   (a.z * (c - at(source, max(p - ivec2(0, 1), ivec2(0)))) + a.w * (c - at(source, min(p + ivec2(0, 1), grid_size - 1))));
        }
    )";
    auto build = [&](Program& p, const char* name, const char* body) { runner.build(p, name, geometry, body); };
    build(coarsen, "multigrid coarse conductances", R"(
        uniform sampler2D source;
        float face(ivec2 p, int axis)
        {
            ivec2 fine_size = textureSize(source, 0);
            int coordinate = p[axis], extent = grid_size[axis];
            if (coordinate == 0 || coordinate == extent) return 0.0;
            int edge = 2 * coordinate;
            float step = spacing * 0.5;
            float before = (center(edge - 1, step, domain_size[axis]) - center(coordinate - 1, spacing, domain_size[axis])) /
                           (center(edge - 1, step, domain_size[axis]) - center(edge - 2, step, domain_size[axis]));
            float after = edge + 1 < fine_size[axis] ?
                (center(coordinate, spacing, domain_size[axis]) - center(edge, step, domain_size[axis])) /
                (center(edge + 1, step, domain_size[axis]) - center(edge, step, domain_size[axis])) : 0.0;
            int perpendicular = 1 - axis;
            float value = 0.0;
            // Conductances add in parallel; resistances add along each normal path.
            for (int j = 0; j < 2; ++j)
            {
                ivec2 q; q[axis] = edge - 1; q[perpendicular] = 2 * p[perpendicular] + j;
                if (q[perpendicular] >= fine_size[perpendicular]) continue;
                ivec2 direction = axis == 0 ? ivec2(1, 0) : ivec2(0, 1);
                int channel = axis == 0 ? 1 : 3;
                float resistance = before / texelFetch(source, q - direction, 0)[channel] + 1.0 / texelFetch(source, q, 0)[channel];
                if (after != 0.0) resistance += after / texelFetch(source, q + direction, 0)[channel];
                value += 1.0 / resistance;
            }
            return value;
        }
        void main()
        {
            ivec2 p = cell();
            output0 = vec4(face(p, 0), face(p + ivec2(1, 0), 0), face(p, 1), face(p + ivec2(0, 1), 1));
        }
    )");
    build(smooth_pair, "multigrid fused red black sweep", R"(
        uniform sampler2D source, rhs, coefficients;
        float firstColor(ivec2 p)
        {
            p = clamp(p, ivec2(0), grid_size - 1);
            vec4 a = texelFetch(coefficients, p, 0);
            float diagonal = (a.x + a.y) + (a.z + a.w), value = rhsAt(rhs, p);
            if (mode < 2)
                value += (a.x * at(source, max(p - ivec2(1, 0), ivec2(0))) + a.y * at(source, min(p + ivec2(1, 0), grid_size - 1))) +
                         (a.z * at(source, max(p - ivec2(0, 1), ivec2(0))) + a.w * at(source, min(p + ivec2(0, 1), grid_size - 1)));
            return diagonal > 0.0 ? value / diagonal : 0.0;
        }
        void main()
        {
            ivec2 p = cell();
            if ((p.x + p.y) % 2 == mode % 2) { output0 = vec4(firstColor(p)); return; }
            vec4 a = texelFetch(coefficients, p, 0);
            float diagonal = (a.x + a.y) + (a.z + a.w);
            // Recompute the new opposite-colour neighbours instead of issuing a second draw.
            float value = rhsAt(rhs, p) +
                (a.x * firstColor(p - ivec2(1, 0)) + a.y * firstColor(p + ivec2(1, 0))) +
                (a.z * firstColor(p - ivec2(0, 1)) + a.w * firstColor(p + ivec2(0, 1)));
            output0 = vec4(diagonal > 0.0 ? value / diagonal : 0.0);
        }
    )");
    build(polynomial_smooth, "multigrid polynomial smoothing", R"(
        uniform sampler2D source, rhs, coefficients;
        uniform float relaxation;
        void main()
        {
            ivec2 p = cell();
            vec4 a = texelFetch(coefficients, p, 0);
            float diagonal = (a.x + a.y) + (a.z + a.w);
            float value = rhsAt(rhs, p);
            if (mode == 0) value -= product(source, coefficients, p);
            output0 = vec4((mode == 0 ? at(source, p) : 0.0) + (diagonal > 0.0 ? relaxation * value / diagonal : 0.0));
        }
    )");
    build(residual, "multigrid residual", R"(
        uniform sampler2D source, rhs, coefficients;
        void main() { output0 = vec4(rhsAt(rhs, cell()) - product(source, coefficients, cell())); }
    )");
    build(restrict_pass, "multigrid transpose restriction", R"(
        uniform sampler2D source, maps;
        void main()
        {
            ivec2 p = cell(), base = p * 2 - 1, size = textureSize(source, 0);
            vec4 wx = texelFetch(maps, ivec2(p.x, 0), 0), wy = texelFetch(maps, ivec2(p.y, 1), 0);
            float value = 0.0;
            for (int y = 0; y < 4; ++y)
            {
                float horizontal = 0.0;
                for (int x = 0; x < 4; ++x)
                {
                    ivec2 q = base + ivec2(x, y);
                    if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) continue;
                    horizontal += wx[x] * at(source, q);
                }
                value += wy[y] * horizontal;
            }
            output0 = vec4(value);
        }
    )");
    build(prolong, "multigrid prolong correction", R"(
        uniform sampler2D source, coarse, maps;
        void main()
        {
            ivec2 p = cell();
            vec3 px = texelFetch(maps, ivec2(p.x, 0), 0).rgb, py = texelFetch(maps, ivec2(p.y, 1), 0).rgb;
            float a = (1.0 - px.z) * at(coarse, ivec2(px.x, py.x)) + px.z * at(coarse, ivec2(px.y, py.x));
            float b = (1.0 - px.z) * at(coarse, ivec2(px.x, py.y)) + px.z * at(coarse, ivec2(px.y, py.y));
            output0 = vec4(at(source, p) + (1.0 - py.z) * a + py.z * b);
        }
    )");
    build(factor, "multigrid coarse Cholesky", R"(
        uniform sampler2D coefficients;
        void main()
        {
            ivec2 target = cell();
            if (target.x > target.y) { output0 = vec4(0); return; }
            int n = grid_size.x * grid_size.y - 1;
            if (n == 0) { output0 = vec4(1); return; }
            float lower[225];
            // Each tiny output computes the required part of the same pinned factor.
            for (int row = 0; row <= target.y; ++row)
            {
                int index = row + 1;
                ivec2 p = ivec2(index % grid_size.x, index / grid_size.x);
                vec4 a = texelFetch(coefficients, p, 0);
                for (int col = 0; col <= row; ++col)
                {
                    float value = col == row ? (a.x + a.y) + (a.z + a.w) :
                        col == row - 1 && p.x > 0 ? -a.x : col == row - grid_size.x && p.y > 0 ? -a.z : 0.0;
                    for (int k = 0; k < col; ++k) value -= lower[row * 15 + k] * lower[col * 15 + k];
                    if (row == col)
                    {
                        // A negative pivot marks numerical failure without a CPU readback.
                        if (value <= 0.0 || !finite(value)) { output0 = vec4(-1); return; }
                        value = sqrt(value);
                    }
                    else value /= lower[col * 15 + col];
                    lower[row * 15 + col] = value;
                    if (row == target.y && col == target.x) { output0 = vec4(value); return; }
                }
            }
        }
    )");
    build(coarse_solve, "multigrid coarse solve", R"(
        uniform sampler2D rhs, factor;
        void main()
        {
            int count = grid_size.x * grid_size.y, n = count - 1;
            float values[15], mean = 0.0;
            for (int i = 0; i < count; ++i) mean += rhsAt(rhs, ivec2(i % grid_size.x, i / grid_size.x));
            mean /= float(count);
            for (int row = 0; row < n; ++row)
            {
                float diagonal = at(factor, ivec2(row, row));
                // An unusable factor drops only the coarse correction; fine-grid CG still checks accuracy.
                if (diagonal <= 0.0 || !finite(diagonal)) { output0 = vec4(0); return; }
                float value = rhsAt(rhs, ivec2((row + 1) % grid_size.x, (row + 1) / grid_size.x)) - mean;
                for (int col = 0; col < row; ++col) value -= at(factor, ivec2(col, row)) * values[col];
                values[row] = value / diagonal;
            }
            for (int row = n - 1; row >= 0; --row)
            {
                float value = values[row];
                for (int col = row + 1; col < n; ++col) value -= at(factor, ivec2(row, col)) * values[col];
                values[row] = value / at(factor, ivec2(row, row));
            }
            mean = 0.0;
            for (int i = 0; i < n; ++i) mean += values[i];
            mean /= float(count);
            int index = cell().y * grid_size.x + cell().x;
            output0 = vec4((index == 0 ? 0.0 : values[index - 1]) - mean);
        }
    )");
}

void PressureMultigridGPU::buildTransfers(Level& fine, const Level& coarse)
{
    const int fine_extent = std::max(fine.width, fine.height), coarse_extent = std::max(coarse.width, coarse.height);
    fine.prolong_map = Texture(fine_extent, 2, 4);
    fine.restrict_map = Texture(coarse_extent, 2, 4);
    std::vector<float> interpolation(size_t(fine_extent) * 8), restriction(size_t(coarse_extent) * 8);
    for (int axis = 0; axis < 2; ++axis)
    {
        const int size = axis == 0 ? fine.width : fine.height, parent_size = axis == 0 ? coarse.width : coarse.height;
        const int extent = axis == 0 ? levels.front().width : levels.front().height;
        auto center = [&](int i, int spacing) { return 0.5 * (i * spacing + std::min((i + 1) * spacing, extent)); };
        for (int f = 0; f < size; ++f)
        {
            const int parent = f / 2;
            const int first = std::max(0, parent - int(center(f, fine.spacing) < center(parent, coarse.spacing)));
            const int second = std::min(first + 1, parent_size - 1);
            const float weight = first == second ? 0.0f : float(std::clamp((center(f, fine.spacing) - center(first, coarse.spacing)) /
                (center(second, coarse.spacing) - center(first, coarse.spacing)), 0.0, 1.0));
            const int i = (axis * fine_extent + f) * 4;
            interpolation[i] = float(first); interpolation[i + 1] = float(second); interpolation[i + 2] = weight;
            // Store the exact transpose weights, for fine indices 2*c-1 through 2*c+2.
            restriction[(axis * coarse_extent + first) * 4 + f - (2 * first - 1)] += 1.0f - weight;
            if (weight != 0.0f) restriction[(axis * coarse_extent + second) * 4 + f - (2 * second - 1)] += weight;
        }
    }
    glBindTexture(GL_TEXTURE_2D, fine.prolong_map.id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fine_extent, 2, GL_RGBA, GL_FLOAT, interpolation.data());
    glBindTexture(GL_TEXTURE_2D, fine.restrict_map.id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, coarse_extent, 2, GL_RGBA, GL_FLOAT, restriction.data());
}

void PressureMultigridGPU::run(Program& program, int level, std::initializer_list<const Texture*> outputs,
                             std::initializer_list<PassRunner::Input> inputs, int mode, float relaxation)
{
    runner.bind(program, outputs, inputs);
    glUniform2i(program.location("grid_size"), levels[level].width, levels[level].height);
    glUniform2i(program.location("domain_size"), levels.front().width, levels.front().height);
    program.scalar("spacing", float(levels[level].spacing));
    program.scalar("mean_divisor", level == 0 ? float(levels.front().width * levels.front().height) : 0.0f);
    program.integer("mode", mode);
    program.scalar("relaxation", relaxation);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

const Texture& PressureMultigridGPU::coefficients(int level) const
{
    return level == 0 ? *fine_coefficients : levels[level].coefficients;
}

void PressureMultigridGPU::rebuild(const Texture& fine)
{
    fine_coefficients = &fine;
    for (int i = 1; i < int(levels.size()); ++i)
        run(coarsen, i, {&levels[i].coefficients}, {{"source", &coefficients(i - 1)}});
    run(factor, int(levels.size()) - 1, {&coarse_factor}, {{"coefficients", &coefficients(int(levels.size()) - 1)}});
}

void PressureMultigridGPU::smoothLevel(int index, const Texture& rhs, bool reverse, bool from_zero, int sweep)
{
    auto& level = levels[index];
    const bool polynomial = level.width * level.height > 65536;
    const int mode = polynomial ? int(from_zero) : int(reverse) + (from_zero ? 2 : 0);
    const float weight = polynomial ? smoothing_weights[reverse ? smoothing_sweeps - 1 - sweep : sweep] : 0.0f;
    run(polynomial ? polynomial_smooth : smooth_pair, index, {&level.next_solution}, {{"source", &level.solution}, {"rhs", &rhs},
        {"coefficients", &coefficients(index)}, {"rhs_stats", residual_stats}}, mode, weight);
    level.solution.swap(level.next_solution);
}

void PressureMultigridGPU::cycle(int index, const Texture& rhs)
{
    auto& fine = levels[index];
    if (index + 1 == int(levels.size()))
    {
        run(coarse_solve, index, {&fine.solution}, {{"rhs", &rhs}, {"factor", &coarse_factor}, {"rhs_stats", residual_stats}});
        return;
    }
    auto& coarse = levels[index + 1];
    smoothLevel(index, rhs, false, true, 0);
    for (int sweep = 1; sweep < smoothing_sweeps; ++sweep) smoothLevel(index, rhs, false, false, sweep);
    run(residual, index, {&fine.residual}, {{"source", &fine.solution}, {"rhs", &rhs},
        {"coefficients", &coefficients(index)}, {"rhs_stats", residual_stats}});
    run(restrict_pass, index, {&coarse.rhs}, {{"source", &fine.residual}, {"maps", &fine.restrict_map}});
    cycle(index + 1, coarse.rhs);
    run(prolong, index, {&fine.next_solution}, {{"source", &fine.solution}, {"coarse", &coarse.solution}, {"maps", &fine.prolong_map}});
    fine.solution.swap(fine.next_solution);
    for (int sweep = 0; sweep < smoothing_sweeps; ++sweep) smoothLevel(index, rhs, true, false, sweep);
}

const Texture& PressureMultigridGPU::apply(const Texture& residual, const Texture& stats)
{
    residual_stats = &stats;
    cycle(0, residual);
    return levels.front().solution;
}

}
