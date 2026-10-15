#pragma once
#include <bitloop.h>
#include <limits>
#include <numeric>

SIM_BEG;

using namespace bl;

struct FluidCell
{
    float pressure = 0.0f;
    float water_fraction = 0.0f;
    bool solid = false;
};


static inline double sampleScalar(
    const std::vector<double>& data,
    int width,
    int height,
    double x,
    double y)
{
    x = std::clamp(x, 0.0, double(width - 1));
    y = std::clamp(y, 0.0, double(height - 1));

    int x0 = int(x);
    int y0 = int(y);
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);

    double tx = x - x0;
    double ty = y - y0;

    auto at = [&](int x, int y) {
        return data[y * width + x];
    };

    double a = std::lerp(at(x0, y0), at(x1, y0), tx);
    double b = std::lerp(at(x0, y1), at(x1, y1), tx);

    return std::lerp(a, b, ty);
}

struct FluidGrid
{
    int width = 0;
    int height = 0;
    double cell_size = 1.0;

    DVec2 origin = {};

    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> next_u;
    std::vector<double> next_v;

    std::vector<double> dye;
    std::vector<double> next_dye;

    std::vector<double> pressure;

    double air_density = 0.01;
    double water_density = 100.0;
    double gravity_y = 100.0;

    std::vector<double> water_fraction;
    std::vector<double> next_water_fraction;
    std::vector<double> density;

    std::vector<double> inv_density_u;
    std::vector<double> inv_density_v;
    std::vector<double> inv_pressure_diagonal;

    FluidGrid(int w, int h, double cell_size) :
        width(w),
        height(h),
        cell_size(cell_size),
        u((w + 1)* h, 0.0),
        v(w* (h + 1), 0.0),
        next_u((w + 1)* h, 0.0),
        next_v(w* (h + 1), 0.0),
        dye(w* h, 0.0),
        next_dye(w* h, 0.0),
        water_fraction(w* h, 0.0),
        next_water_fraction(w* h, 0.0),
        density(w* h, 0.0),
        inv_density_u((w + 1)* h, 0.0),
        inv_density_v(w* (h + 1), 0.0),
        inv_pressure_diagonal(w* h, 0.0)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                dye[y * width + x] = x < width / 2 ? 1.0 : 0.0;
            }
        }

        initializePool();
    }

    void initializePool(double amplitude = 0.04)
    {
        std::fill(u.begin(), u.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);
        std::fill(pressure.begin(), pressure.end(), 0.0);
        std::fill(previous_pressure.begin(), previous_pressure.end(), 0.0);

        constexpr double pi = 3.14159265358979323846;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // A cosine surface starts a slosh and meets the side walls horizontally.
                const double surface =
                    x + height * (0.5 + amplitude * std::cos(pi * (x*3 + 0.5) / width));

                // Measure how much of this cell lies below the water surface.
                water_fraction[y * width + x] =
                    std::clamp(y + 1.0 - surface, 0.0, 1.0);
            }
        }
    }

    void advect(double dt)
    {
        const double step = dt / cell_size;

        Thread::forEachBatch(next_u, [&](int i0, int i1)
        {
            for (int i = i0; i < i1; ++i)
            {
                const int x = i % (width + 1);
                const int y = i / (width + 1);

                const double ux = u[i];
                const double vy = sampleScalar(v, width, height + 1, x - 0.5, y + 0.5);

                // Backtrace in this component's staggered grid coordinates.
                next_u[i] = sampleScalar(u, width + 1, height, x - ux * step, y - vy * step);
            }
        });

        Thread::forEachBatch(next_v, [&](int i0, int i1)
        {
            for (int i = i0; i < i1; ++i)
            {
                const int x = i % width;
                const int y = i / width;

                const double ux = sampleScalar(u, width + 1, height, x + 0.5, y - 0.5);
                const double vy = v[i];

                // Backtrace in this component's staggered grid coordinates.
                next_v[i] = sampleScalar(v, width, height + 1, x - ux * step, y - vy * step);
            }
        });

        u.swap(next_u);
        v.swap(next_v);
    }

    void advectDye(double dt)
    {
        Thread::forEachBatch(next_dye, [&](int i0, int i1)
        {
            for (int i = i0; i < i1; ++i)
            {
                const int x = i % width;
                const int y = i / width;
                DVec2 pos{
                    origin.x + (x + 0.5) * cell_size,
                    origin.y + (y + 0.5) * cell_size
                };

                DVec2 velocity = sampleVelocity(pos);
                DVec2 prev = pos - velocity * dt;

                double sample_x = (prev.x - origin.x) / cell_size - 0.5;
                double sample_y = (prev.y - origin.y) / cell_size - 0.5;

                next_dye[y * width + x] = sampleScalar(
                    dye, width, height, sample_x, sample_y
                );
            }
        });

        dye.swap(next_dye);
    }

    void advectWaterFraction(double dt)
    {
        double max_outflow = 0.0;

        for (int y = 0; y < height; ++y) 
        {
            for (int x = 0; x < width; ++x)
            {
                const double outflow =
                    std::max( u[uIndex(x + 1, y)], 0.0) +
                    std::max(-u[uIndex(x, y)],     0.0) +
                    std::max( v[vIndex(x, y + 1)], 0.0) +
                    std::max(-v[vIndex(x, y)],     0.0);

                max_outflow = std::max(max_outflow, outflow);
            }
        }

        // Limit transport so no cell exports more than half its volume per substep.
        const double max_step =
            max_outflow > 0.0 ? 0.5 * cell_size / max_outflow : dt;

        auto fluxX = [&](int x, int y) {
            if (x == 0 || x == width)
                return 0.0;

            const double speed = u[uIndex(x, y)];

            // The upstream cell supplies the water crossing this face.
            const int donor = y * width + (speed >= 0.0 ? x - 1 : x);
            return speed * water_fraction[donor];
        };

        auto fluxY = [&](int x, int y) {
            if (y == 0 || y == height)
                return 0.0;

            const double speed = v[vIndex(x, y)];
            const int donor = (speed >= 0.0 ? y - 1 : y) * width + x;
            return speed * water_fraction[donor];
        };

        double remaining = dt;

        while (remaining > 0.0) {
            const double step = std::min(remaining, max_step);

            Thread::forEachBatch(next_water_fraction, [&](int i0, int i1) {
                for (int i = i0; i < i1; ++i) {
                    const int x = i % width;
                    const int y = i / width;

                    // Each shared face removes exactly what it adds to its neighbor.
                    const double net_flux =
                        fluxX(x + 1, y) - fluxX(x, y) +
                        fluxY(x, y + 1) - fluxY(x, y);

                    next_water_fraction[i] =
                        water_fraction[i] - step / cell_size * net_flux;
                }
            });

            water_fraction.swap(next_water_fraction);
            remaining -= step;
        }
    }

    constexpr int uIndex(int x, int y) const
    {
        return y * (width + 1) + x;
    }

    constexpr int vIndex(int x, int y) const
    {
        return y * width + x;
    }

    constexpr DVec2 sampleVelocity(DVec2 pos) const
    {
        double x = (pos.x - origin.x) / cell_size;
        double y = (pos.y - origin.y) / cell_size;

        return {
            sampleScalar(u, width + 1, height, x, y - 0.5),
            sampleScalar(v, width, height + 1, x - 0.5, y)
        };
    }

    void applyGravity(double dt)
    {
        // Gravity accelerates both materials equally; skip the two outer wall rows.
        for (int i = width; i < width * height; ++i)
            v[i] += gravity_y * dt;
    }

    void addVortex(DVec2 center, double radius, double strength)
    {
        auto vortexVelocity = [&](DVec2 pos) -> DVec2 {
            double dx = pos.x - center.x;
            double dy = pos.y - center.y;

            double r2 = dx * dx + dy * dy;
            double falloff = std::exp(-r2 / (2.0 * radius * radius));

            return {
                -dy * strength * falloff,
                 dx * strength * falloff
            };
        };

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x <= width; ++x)
            {
                DVec2 pos{
                    origin.x + x * cell_size,
                    origin.y + (y + 0.5) * cell_size
                };

                u[uIndex(x, y)] += vortexVelocity(pos).x;
            }
        }

        for (int y = 0; y <= height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                DVec2 pos{
                    origin.x + (x + 0.5) * cell_size,
                    origin.y + y * cell_size
                };

                v[vIndex(x, y)] += vortexVelocity(pos).y;
            }
        }
    }

    struct ProjectionResult
    {
        int iterations = 0;
        double max_divergence = std::numeric_limits<double>::infinity();
        bool converged = false;
    };

    // divergence_tolerance is the maximum permitted cell divergence, in inverse seconds
    ProjectionResult project(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6)
    {
        if (dt <= 0.0 ||
            width <= 0 || height <= 0 ||
            max_iterations < 0)
        {
            return {};
        }

        const double pressure_scale = cell_size * cell_size / dt;

        auto index = [&](int x, int y) {
            return y * width + x;
        };

        const size_t count = size_t(width) * height;
        pressure.resize(count);
        previous_pressure.resize(count);
        pressure_rhs.resize(count);
        pressure_residual.resize(count);
        pressure_correction.resize(count);
        pressure_direction.resize(count);
        pressure_product.resize(count);

        rebuildPressureCoefficients();

        // prevent flow through the outer boundaries
        for (int y = 0; y < height; ++y) {
            u[uIndex(0, y)] = 0.0;
            u[uIndex(width, y)] = 0.0;
        }

        for (int x = 0; x < width; ++x) {
            v[vIndex(x, 0)] = 0.0;
            v[vIndex(x, height)] = 0.0;
        }

        // solve the positive pressure operator against negative, scaled divergence
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int i = index(x, y);

                double du = u[uIndex(x + 1, y)] - u[uIndex(x, y)];
                double dv = v[vIndex(x, y + 1)] - v[vIndex(x, y)];

                pressure_rhs[i] = -(du + dv) / cell_size * pressure_scale;
            }
        }

        // closed walls require a compatible right-hand side; pressure is defined up to a constant
        removeMean(pressure_rhs);
        ProjectionResult result;
        result.iterations = solvePressure(max_iterations, divergence_tolerance * pressure_scale);

        // correct horizontal velocities
        for (int y = 0; y < height; ++y) {
            for (int x = 1; x < width; ++x) {
                // if the cell on the right has higher pressure than the cell on the left,
                // increases velocity across that boundary towards the left
                double gradient = (
                    pressure[index(x, y)] -
                    pressure[index(x - 1, y)]
                ) / cell_size;

                const int face = uIndex(x, y);
                u[face] -= gradient * dt * inv_density_u[face];
            }
        }

        // correct vertical velocities
        for (int y = 1; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // if the cell on the bottom has higher pressure than the cell on the top,
                // increases velocity across that boundary towards the top
                double gradient = (
                    pressure[index(x, y)] -
                    pressure[index(x, y - 1)]
                    ) / cell_size;

                const int face = vIndex(x, y);
                v[face] -= gradient * dt * inv_density_v[face];
            }
        }

        result.max_divergence = maxDivergence();
        result.converged = result.max_divergence <= divergence_tolerance;
        return result;
    }

    double maxDivergence() const
    {
        double result = 0.0;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double du = u[uIndex(x + 1, y)] - u[uIndex(x, y)];
                double dv = v[vIndex(x, y + 1)] - v[vIndex(x, y)];

                double divergence = (du + dv) / cell_size;

                if (!std::isfinite(divergence))
                    return std::numeric_limits<double>::infinity();

                result = std::max(result, std::abs(divergence));
            }
        }

        return result;
    }

private:
    std::vector<double> previous_pressure;
    std::vector<double> pressure_rhs;
    std::vector<double> pressure_residual;
    std::vector<double> pressure_correction;
    std::vector<double> pressure_direction;
    std::vector<double> pressure_product;

    void rebuildPressureCoefficients()
    {
        // Convert the material fraction into a cell-averaged density.
        for (size_t i = 0; i < water_fraction.size(); ++i)
        {
            const double fraction = std::clamp(water_fraction[i], 0.0, 1.0);
            density[i] =
                fraction * water_density +
                (1.0 - fraction) * air_density;
        }

        // Zero wall weights exclude pressure corrections through the tank boundary.
        std::fill(inv_density_u.begin(), inv_density_u.end(), 0.0);
        std::fill(inv_density_v.begin(), inv_density_v.end(), 0.0);

        // Each shared face uses the reciprocal of the neighboring cells' average density.
        for (int y = 0; y < height; ++y)
        {
            for (int x = 1; x < width; ++x)
            {
                const int i = y * width + x;
                inv_density_u[uIndex(x, y)] = 2.0 / (density[i - 1] + density[i]);
            }
        }
        for (int y = 1; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int i = y * width + x;
                inv_density_v[vIndex(x, y)] = 2.0 / (density[i - width] + density[i]);
            }
        }

        // The matrix diagonal is the sum of the four shared face weights.
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const double diagonal =
                    inv_density_u[uIndex(x, y)] +
                    inv_density_u[uIndex(x + 1, y)] +
                    inv_density_v[vIndex(x, y)] +
                    inv_density_v[vIndex(x, y + 1)];
                inv_pressure_diagonal[y * width + x] = diagonal > 0.0 ? 1.0 / diagonal : 0.0;
            }
        }
    }

    static void removeMean(std::vector<double>& values)
    {
        const double mean = std::reduce(values.begin(), values.end(), 0.0) / double(values.size());
        for (double& value : values)
            value -= mean;
    }

    static double maxAbs(const std::vector<double>& values)
    {
        // Preserve non-finite failure detection while reducing absolute residuals.
        return std::transform_reduce(values.begin(), values.end(), 0.0,
            [](double a, double b) { return std::max(a, b); },
            [](double value) { return std::isfinite(value) ? std::abs(value) : std::numeric_limits<double>::infinity(); });
    }

    void applyPressureOperator(
        const std::vector<double>& input,
        std::vector<double>& output) const
    {
        // General stencil: used only for boundaries and very thin grids.
        auto applyCell = [&](int x, int y)
        {
            const int i = y * width + x;
            const double center = input[i];

            double value = 0.0;
            if (x > 0)          value += inv_density_u[uIndex(x, y)]     * (center - input[i - 1]);
            if (x + 1 < width)  value += inv_density_u[uIndex(x + 1, y)] * (center - input[i + 1]);
            if (y > 0)          value += inv_density_v[i]                * (center - input[i - width]);
            if (y + 1 < height) value += inv_density_v[i + width]        * (center - input[i + width]);

            output[i] = value;
        };

        // These grids have no interior.
        if (width < 3 || height < 3)
        {
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    applyCell(x, y);

            return;
        }

        // Keep the small pressure stencil on this thread; worker dispatch costs more here.
        const int stride = width;
        for (int y = 1; y < height - 1; ++y) {
            const double* row = input.data() + y * stride;
            const double* above = row - stride;
            const double* below = row + stride;
            const double* horizontal = inv_density_u.data() + y * (stride + 1);
            const double* up = inv_density_v.data() + y * stride;
            const double* down = up + stride;
            double* result = output.data() + y * stride;
            for (int x = 1; x < stride - 1; ++x) {
                const double center = row[x];
                result[x] =
                    (horizontal[x] * (center - row[x - 1]) + horizontal[x + 1] * (center - row[x + 1])) +
                    (up[x] * (center - above[x]) + down[x] * (center - below[x]));
            }
        }

        // Top and bottom, including corners.
        for (int x = 0; x < width; ++x)
        {
            applyCell(x, 0);
            applyCell(x, height - 1);
        }

        // Left and right, excluding the already processed corners.
        for (int y = 1; y < height - 1; ++y)
        {
            applyCell(0, y);
            applyCell(width - 1, y);
        }
    }

    struct ResidualStats
    {
        double sum = 0.0;
        double maximum = 0.0;
    };

    ResidualStats updatePressureAndResidual(double alpha)
    {
        double* pressure_values = pressure.data();
        double* residual_values = pressure_residual.data();
        const double* direction = pressure_direction.data();
        const double* product = pressure_product.data();
        const size_t count = pressure.size();
        double sums[4] = {}, maxima[4] = {};

        // Measure the new residual while updating it, with four independent accumulators.
        auto update = [&](size_t i) {
            pressure_values[i] += alpha * direction[i];
            residual_values[i] -= alpha * product[i];
            return residual_values[i];
        };
        size_t i = 0;
        for (; i + 3 < count; i += 4) {
            for (size_t lane = 0; lane < 4; ++lane) {
                const double value = update(i + lane);
                sums[lane] += value;
                maxima[lane] = std::max(maxima[lane], std::abs(value));
            }
        }
        ResidualStats stats{(sums[0] + sums[1]) + (sums[2] + sums[3]),
            std::max(std::max(maxima[0], maxima[1]), std::max(maxima[2], maxima[3]))};
        for (; i < count; ++i) {
            const double value = update(i);
            stats.sum += value;
            stats.maximum = std::max(stats.maximum, std::abs(value));
        }
        // Any non-finite residual also makes its sum non-finite.
        if (!std::isfinite(stats.sum)) stats.maximum = std::numeric_limits<double>::infinity();
        return stats;
    }

    double applyPreconditioner(const std::vector<double>& residual, std::vector<double>& correction, double residual_mean) const
    {
        // Project before and after diagonal scaling to preserve PCG symmetry.
        for (size_t i = 0; i < residual.size(); ++i)
            correction[i] = (residual[i] - residual_mean) * inv_pressure_diagonal[i];
        const double mean = std::reduce(correction.begin(), correction.end(), 0.0) / double(correction.size());

        // Accumulate residual dot correction while removing the correction's mean.
        double dots[4] = {};
        size_t i = 0;
        for (; i + 3 < residual.size(); i += 4) {
            for (size_t lane = 0; lane < 4; ++lane) {
                correction[i + lane] -= mean;
                dots[lane] += residual[i + lane] * correction[i + lane];
            }
        }
        double dot = (dots[0] + dots[1]) + (dots[2] + dots[3]);
        for (; i < residual.size(); ++i) {
            correction[i] -= mean;
            dot += residual[i] * correction[i];
        }
        return dot;
    }

    int solvePressure(int max_iterations, double residual_tolerance)
    {
        // A divergence-free field needs no correction, regardless of the previous pressure.
        if (maxAbs(pressure_rhs) <= residual_tolerance) {
            std::fill(pressure.begin(), pressure.end(), 0.0);
            std::fill(previous_pressure.begin(), previous_pressure.end(), 0.0);
            return 0;
        }

        // Predict pressure from the last two solves, then rebuild its residual for the current system.
        for (size_t i = 0; i < pressure.size(); ++i) {
            const double latest = pressure[i];
            pressure[i] = 2.0 * latest - previous_pressure[i];
            previous_pressure[i] = latest;
        }
        removeMean(pressure);
        applyPressureOperator(pressure, pressure_product);
        for (size_t i = 0; i < pressure.size(); ++i)
            pressure_residual[i] = pressure_rhs[i] - pressure_product[i];

        if (maxAbs(pressure_residual) <= residual_tolerance || max_iterations == 0)
            return 0;

        double residual_mean = std::reduce(pressure_residual.begin(), pressure_residual.end(), 0.0) / double(pressure_residual.size());
        double residual_dot_correction = applyPreconditioner(pressure_residual, pressure_correction, residual_mean);
        pressure_direction = pressure_correction;
        int iterations = 0;

        while (iterations < max_iterations && residual_dot_correction > 0.0 && std::isfinite(residual_dot_correction))
        {
            applyPressureOperator(pressure_direction, pressure_product);

            const double denominator = std::transform_reduce(pressure_direction.begin(), pressure_direction.end(), pressure_product.begin(), 0.0);
            if (denominator <= 0.0 || !std::isfinite(denominator))
                break;

            const double alpha = residual_dot_correction / denominator;
            if (!std::isfinite(alpha))
                break;

            const auto stats = updatePressureAndResidual(alpha);
            residual_mean = stats.sum / double(pressure_residual.size());

            ++iterations;

            bool restart = false;
            if (stats.maximum <= residual_tolerance)
            {
                // verify the true residual before accepting convergence; restart if recurrence drifted
                applyPressureOperator(pressure, pressure_product);

                for (size_t i = 0; i < pressure.size(); ++i)
                    pressure_residual[i] = pressure_rhs[i] - pressure_product[i];
                if (maxAbs(pressure_residual) <= residual_tolerance)
                    break;

                residual_mean = std::reduce(pressure_residual.begin(), pressure_residual.end(), 0.0) / double(pressure_residual.size());
                restart = true;
            }

            const double next_dot = applyPreconditioner(pressure_residual, pressure_correction, residual_mean);
            const double beta = restart ? 0.0 : next_dot / residual_dot_correction;

            if (!std::isfinite(beta))
                break;

            for (size_t i = 0; i < pressure.size(); ++i)
                pressure_direction[i] = pressure_correction[i] + beta * pressure_direction[i];

            residual_dot_correction = next_dot;
        }

        removeMean(pressure);
        return iterations;
    }
};

struct Particle
{
    DVec2 pos;

    Particle(double x, double y)
    {
        pos.x = x;
        pos.y = y;
    }
};

struct FluidSim_Scene : public Scene<FluidSim_Scene>
{
    CameraInfo       camera;
    CameraNavigator  navigator;

    /// ─────── Variables ───────
    int grid_w = 128;
    int grid_h = 128;

    double cell_size = 4.0;

    bool simulation_failed = false;

    double world_w = double(grid_w) * cell_size;
    double world_h = double(grid_h) * cell_size;
    double world_hypot = bl::hypot(world_w, world_h);

    FluidGrid fluid;

    bool show_particles = false;
    int particles_x = 50;
    int particles_y = 50;
    std::vector<Particle> particles;

    const char* fluid_shader_src = R"(
        #version 300 es
        precision highp float;

        uniform sampler2D u_water;
        
        in vec2 v_uv;
        out vec4 outColor;

        void main()
        {
            float water_fraction = texture(u_water, v_uv).r;

            vec3 air_color = vec3(0.015, 0.025, 0.06);
            vec3 water_color = vec3(0.1, 0.75, 0.95);

            outColor = vec4(mix(air_color, water_color, water_fraction), 1.0);
        }
    )";

    ShaderSurface surface;
    mutable GLuint water_texture = 0;
    mutable std::vector<unsigned char> water_pixels;
    void uploadWaterTexture() const;
    void renderFluidSurface() const;
#ifdef BITLOOP_DEV_MODE
    mutable CapturePreprocessor snapshot_preprocessor;
#endif

    /// ─────── Methods ───────
    

    /// ─────── Launch config ───────
    struct Config {};

    FluidSim_Scene(Config& info [[maybe_unused]])
        : fluid(grid_w, grid_h, cell_size)
    {
    }

    struct UI : BufferedInterfaceModel
    {
        using BufferedInterfaceModel::BufferedInterfaceModel;
        void sidebar() override;
        // void overlay() override; // ImGui viewport overlay

        /// ─────── Your UI-only variables ───────

    };

    /// ─────── Scene methods ───────
    void sceneStart() override;
    void sceneMounted(Viewport* viewport) override;
    void sceneUnmounted(Viewport* viewport) override;
    void sceneDestroy() override;
    void sceneProcess(double dt) override;

    /// ─────── Viewport methods ───────
    void viewportProcess(Viewport* ctx, double dt) override;
    void viewportDraw(Viewport* ctx) const override;

    /// ─────── Input handling ───────
    void onEvent(Event e) override; // receives all event types

    #ifdef BITLOOP_DEV_MODE
    void traceFrame(SimTraceFrame& frame) const override;
    bool captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const override;
    bool saveCheckpoint(SimCheckpointWriter& writer) const override;
    bool loadCheckpoint(const SimCheckpointReader& reader) override;
    #endif
};

struct FluidSim_Project : public Project<FluidSim_Project>
{
    static ProjectInfo info()
    {
        return ProjectInfo({ "New Projects", "FluidSim" });
    }

    struct UI : BufferedInterfaceModel
    {
        using BufferedInterfaceModel::BufferedInterfaceModel;
        void sidebar();
        // void overlay();
    };

    int viewport_count = 1;

    void projectPrepare(Layout& layout) override;
};

SIM_END;
