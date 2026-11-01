#pragma once
#include "AmrField2D.h"
#include "FluidLinearReconstruction.h"
#include "FluidMixture.h"
#include "SubstanceCatalogue.h"
#include <cmath>
#include <span>
#include <utility>

namespace fluid
{
// Non-owning observation of committed cells. The only spatial cache is a patch
// index per 2x2 finest group; no masses, momentum or pressure are resampled.
class FluidCellView
{
public:
    struct UniformFields
    {
        std::span<const double> liquidFraction;
        std::span<const std::vector<double>> liquidMass;
        std::span<const std::vector<double>> gasMass;
        std::span<const double> u;
        std::span<const double> v;
        std::span<const double> pressure;
    };

    struct MaterialPlane
    {
        std::vector<double> stencil;
        std::vector<LinearSlope> slopes;
        int count = 0;

        double mean(int n, int side = 0) const { return stencil[side * count + n]; }

        double value(int n, double dx, double dy) const { return mean(n) + dx * slopes[n].x + dy * slopes[n].y; }
    };

    struct Patch
    {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
        int scale = 1;

        AmrArray2D<const double> q;
        AmrArray2D<const double> pressure;
        AmrArray2D<const double> u;
        AmrArray2D<const double> v;
        AmrArray2D<const int> active;
    };

    struct Cell
    {
        const FluidCellView* view = nullptr;
        const Patch* patch = nullptr;
        int x = 0;
        int y = 0;

        explicit operator bool() const { return view != nullptr; }

        double size() const { return view->cellSize * patch->scale; }

        double cx() const { return (x + 0.5) * size(); }

        double cy() const { return (y + 0.5) * size(); }

        int index() const { return y * view->width + x; }

        bool liquid() const
        {
            if (!view->isUniform())
            {
                for (int k = 0; k < view->liquidComponents; ++k)
                {
                    if (density(k) != 0.0)
                        return true;
                }

                return false;
            }

            if (view->uniform.liquidFraction[index()] != 0.0)
                return true;

            for (const auto& mass : view->uniform.liquidMass)
            {
                if (mass[index()] != 0.0)
                    return true;
            }

            return false;
        }

        double density(int k) const
        {
            if (view->isUniform())
                return view->uniform.gasMass[k][index()] / (size() * size());

            return patch->q(x, y, k);
        }

        double density() const
        {
            double rho = 0.0;
            for (int k = 0; k < view->components(); ++k)
                rho += density(k);

            return rho;
        }

        double fraction(int k) const { return density(k) / view->mixture->components[k].density; }

        double face(int axis, int offset) const
        {
            if (view->isUniform())
            {
                return axis == 0 ? view->uniform.u[y * (view->width + 1) + x + offset]
                                 : view->uniform.v[(y + offset) * view->width + x];
            }

            return axis == 0 ? patch->u(x + offset, y) : patch->v(x, y + offset);
        }

        double velocity(int axis) const
        {
            if (view->isUniform())
                return 0.5 * (face(axis, 0) + face(axis, 1));

            return patch->q(x, y, view->components() + axis) / density();
        }

        double momentum(int axis) const
        {
            if (view->isUniform())
                return density() * velocity(axis);

            return patch->q(x, y, view->components() + axis);
        }

        double pressure() const
        {
            if (view->isUniform())
                return view->uniform.pressure[index()];

            return view->pressureInterval > 0.0 ? patch->pressure(x, y) / view->pressureInterval : 0.0;
        }
    };

    // The caller owns every field and keeps it alive for the observation.
    FluidCellView(int w, int h, double dx, const FluidMixture& material, const SubstanceCatalogue& substances,
        UniformFields fields, std::array<double, 2> force, const void* id)
        : width(w),
          height(h),
          cellSize(dx),
          mixture(&material),
          catalogue(&substances),
          uniform(fields),
          acceleration(force),
          identity(id)
    {
        patches.push_back({0, 0, width, height, 1});
        owners.assign(size_t((width + 1) / 2) * ((height + 1) / 2), 0);
    }

    // Non-owning view over an explicit native-state observation. The caller
    // keeps the patch arrays alive; no physical state is created or advanced.
    FluidCellView(int w, int h, double dx, const FluidMixture& material, const SubstanceCatalogue& substances,
        std::vector<Patch> nativePatches, std::array<double, 2> force, double interval, int liquids, const void* id)
        : width(w),
          height(h),
          cellSize(dx),
          mixture(&material),
          catalogue(&substances),
          acceleration(force),
          identity(id),
          pressureInterval(interval),
          liquidComponents(liquids),
          patches(std::move(nativePatches))
    {
        owners.assign(size_t((width + 1) / 2) * ((height + 1) / 2), -1);
        for (int index = 0; index < int(patches.size()); ++index)
        {
            const auto& p = patches[index];
            for (int y = p.y; y < p.y + p.height; y += 2)
            {
                for (int x = p.x; x < p.x + p.width; x += 2)
                {
                    if (p.active(x / p.scale, y / p.scale))
                        owners[(y / 2) * ((width + 1) / 2) + x / 2] = index;
                }
            }
        }
    }

    Cell at(int x, int y) const
    {
        if (x < 0 || y < 0 || x >= width || y >= height)
            return {};

        const auto& patch = patches[owners[(y / 2) * ((width + 1) / 2) + x / 2]];
        return {this, &patch, x / patch.scale, y / patch.scale};
    }

    int minimumScale(int x, int y, int scale) const
    {
        int stride = scale;
        for (int dy = 0; dy < scale; dy += 2)
        {
            for (int dx = 0; dx < scale; dx += 2)
                stride = std::min(stride, at(x + dx, y + dy).patch->scale);
        }

        return stride;
    }

    void reconstruct(MaterialPlane& plane, int x, int y, int scale, bool includeMomentum = false) const
    {
        plane.count = components() + (includeMomentum ? 2 : 0);
        plane.stencil.assign(5 * plane.count, 0.0);
        plane.slopes.resize(plane.count);
        constexpr int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        for (int side = 0; side < 5; ++side)
        {
            const int gx = std::clamp(x + scale * offsets[side][0], 0, width - scale);
            const int gy = std::clamp(y + scale * offsets[side][1], 0, height - scale);
            const auto source = at(gx, gy);
            if (source.patch->scale > scale)
            {
                // A smaller stencil footprint inside a larger cell uses that
                // cell's conservative plane, matching coarse/fine ghost fill.
                MaterialPlane parent;
                const int parentScale = source.patch->scale;
                reconstruct(parent, source.x * parentScale, source.y * parentScale, parentScale, includeMomentum);

                for (int n = 0; n < plane.count; ++n)
                {
                    plane.stencil[side * plane.count + n] =
                        parent.value(n, (gx + 0.5 * scale) / parentScale - source.x - 0.5,
                            (gy + 0.5 * scale) / parentScale - source.y - 0.5);
                }

                continue;
            }

            // Restrict actual samples, including groups crossing fine patch edges.
            const int stride = minimumScale(gx, gy, scale);
            const int samples = (scale / stride) * (scale / stride);

            for (int dy = 0; dy < scale; dy += stride)
            {
                for (int dx = 0; dx < scale; dx += stride)
                {
                    const auto c = at(gx + dx, gy + dy);
                    for (int n = 0; n < components(); ++n)
                        plane.stencil[side * plane.count + n] += c.density(n) / samples;
                    if (includeMomentum)
                    {
                        for (int d = 0; d < 2; ++d)
                        {
                            const int requested = d ? y + scale * offsets[side][1] : x + scale * offsets[side][0];
                            const bool reflected = requested < 0 || requested >= (d ? height : width);
                            plane.stencil[side * plane.count + components() + d] +=
                                (reflected ? -1.0 : 1.0) * c.momentum(d) / samples;
                        }
                    }
                }
            }
        }

        auto sample = [&](int dx, int dy, int n)
        {
            return plane.mean(n, dx < 0 ? 1 : dx > 0 ? 2 : dy < 0 ? 3 : dy > 0 ? 4 : 0);
        };

        const auto limiter = linearLimiter(components(), sample);

        for (int n = 0; n < components(); ++n)
            plane.slopes[n] = linearSlope(n, sample, limiter);
        if (includeMomentum)
        {
            const auto momentum = linearMomentumSlopes(components(), sample, limiter);
            for (int d = 0; d < 2; ++d)
                plane.slopes[components() + d] = momentum[d];
        }
    }

    double materialSample(int x, int y, int ingredient) const
    {
        const auto c = at(x, y);
        if (c.patch->scale == 1)
            return c.fraction(ingredient);

        MaterialPlane plane;
        const int scale = c.patch->scale;
        reconstruct(plane, c.x * scale, c.y * scale, scale);

        return plane.value(ingredient, (x + 0.5) / scale - c.x - 0.5, (y + 0.5) / scale - c.y - 0.5) /
               mixture->components[ingredient].density;
    }

    bool isUniform() const { return !uniform.liquidFraction.empty(); }

    int components() const { return int(mixture->components.size()); }

    int width;

    int height;
    double cellSize;
    const FluidMixture* mixture;
    const SubstanceCatalogue* catalogue;
    UniformFields uniform;
    std::array<double, 2> acceleration{};

    const void* identity;
    double pressureInterval = 0.0;
    int liquidComponents = 0;
    std::vector<Patch> patches;
    std::vector<int> owners;
};
}
