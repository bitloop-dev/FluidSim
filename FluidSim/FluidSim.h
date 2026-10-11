#pragma once
#include <bitloop.h>

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

    FluidGrid(int w, int h, double cell_size) :
        width(w),
        height(h),
        cell_size(cell_size),
        u((w + 1)* h, 0.0),
        v(w* (h + 1), 0.0),
        next_u((w + 1)* h, 0.0),
        next_v(w* (h + 1), 0.0),
        dye(w* h, 0.0),
        next_dye(w* h, 0.0)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                dye[y * width + x] = x < width / 2 ? 1.0 : 0.0;
            }
        }
    }

    void advect(double dt)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x <= width; x++)
            {
                DVec2 pos{
                    origin.x + x * cell_size,
                    origin.y + (y + 0.5) * cell_size
                };

                DVec2 vel = sampleVelocity(pos);
                DVec2 prev{
                    pos.x - vel.x * dt,
                    pos.y - vel.y * dt
                };

                next_u[uIndex(x, y)] = sampleVelocity(prev).x;
            }
        }

        for (int y = 0; y <= height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                DVec2 pos{
                    origin.x + (x + 0.5) * cell_size,
                    origin.y + y * cell_size
                };

                DVec2 vel = sampleVelocity(pos);
                DVec2 prev{
                    pos.x - vel.x * dt,
                    pos.y - vel.y * dt
                };

                next_v[vIndex(x, y)] = sampleVelocity(prev).y;
            }
        }

        u.swap(next_u);
        v.swap(next_v);
    }

    void advectDye(double dt)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; x++)
            {
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
        }

        dye.swap(next_dye);
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

    void project(double dt, int iterations = 200)
    {
        if (dt <= 0.0 || cell_size <= 0.0)
            return;

        auto index = [&](int x, int y) {
            return y * width + x;
        };

        std::vector<double> divergence(width * height, 0.0);
        std::vector<double> pressure(width * height, 0.0);

        double h = cell_size;

        // prevent flow through the outer boundaries
        for (int y = 0; y < height; ++y) {
            u[uIndex(0, y)] = 0.0;
            u[uIndex(width, y)] = 0.0;
        }

        for (int x = 0; x < width; ++x) {
            v[vIndex(x, 0)] = 0.0;
            v[vIndex(x, height)] = 0.0;
        }

        // calculate divergence
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int i = index(x, y);

                double du = u[uIndex(x + 1, y)] - u[uIndex(x, y)];
                double dv = v[vIndex(x, y + 1)] - v[vIndex(x, y)];

                divergence[i] = (du + dv) / h;
            }
        }

        // solve pressure using Gauss-Seidel
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int i = index(x, y);

                    double sum = 0.0;
                    int neighbours = 0;

                    if (x > 0) {
                        sum += pressure[index(x - 1, y)];
                        ++neighbours;
                    }

                    if (x < width - 1) {
                        sum += pressure[index(x + 1, y)];
                        ++neighbours;
                    }

                    if (y > 0) {
                        sum += pressure[index(x, y - 1)];
                        ++neighbours;
                    }

                    if (y < height - 1) {
                        sum += pressure[index(x, y + 1)];
                        ++neighbours;
                    }

                    if (neighbours > 0) {
                        double target = (
                            sum - h * h * divergence[i] / dt
                            ) / neighbours;

                        double omega = 1.8;

                        pressure[i] += omega * (target - pressure[i]);
                    }
                }
            }
        }

        // correct horizontal velocities
        for (int y = 0; y < height; ++y) {
            for (int x = 1; x < width; ++x) {
                double gradient = (
                    pressure[index(x, y)] -
                    pressure[index(x - 1, y)]
                    ) / h;

                u[uIndex(x, y)] -= gradient * dt;
            }
        }

        // correct vertical velocities
        for (int y = 1; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double gradient = (
                    pressure[index(x, y)] -
                    pressure[index(x, y - 1)]
                    ) / h;

                v[vIndex(x, y)] -= gradient * dt;
            }
        }
    }

    double maxDivergence() const
    {
        double result = 0.0;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double du = u[uIndex(x + 1, y)] - u[uIndex(x, y)];
                double dv = v[vIndex(x, y + 1)] - v[vIndex(x, y)];

                double divergence = (du + dv) / cell_size;

                result = std::max(result, std::abs(divergence));
            }
        }

        return result;
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

    double world_w = double(grid_w) * cell_size;
    double world_h = double(grid_h) * cell_size;
    double world_hypot = bl::hypot(world_w, world_h);

    FluidGrid fluid;

    bool show_particles = true;
    int particles_x = 50;
    int particles_y = 50;
    std::vector<Particle> particles;

    const char* fluid_shader_src = R"(
        #version 300 es
        precision highp float;

        uniform sampler2D u_dye;
        
        in vec2 v_uv;
        out vec4 outColor;

        void main()
        {
            float dye = texture(u_dye, v_uv).r;

            vec3 background = vec3(0.015, 0.025, 0.06);
            vec3 ink = vec3(0.1, 0.75, 0.95);

            outColor = vec4(mix(background, ink, dye), 1.0);
        }
    )";

    ShaderSurface surface;
    mutable GLuint dye_texture = 0;
    mutable std::vector<unsigned char> dye_pixels;
    void uploadDyeTexture() const;

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
