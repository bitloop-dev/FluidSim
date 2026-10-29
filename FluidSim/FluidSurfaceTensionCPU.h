#pragma once
#include "Adaptive/FluidInterfaceGeometry.h"
#include <array>
#include <span>
#include <limits>
#include <vector>

namespace fluid
{

// Double-precision counterpart of UniformWebGPU/FluidSurfaceTensionKernels.h.
// Three ordered sweeps: heights, local curvature, then missing-curvature completion.
struct FluidSurfaceTensionCPU
{
    struct Height
    {
        double x = 0., y = 0.;
        double orientation_x = 0., orientation_y = 0.;
    };
    struct Curvature
    {
        double value = 0.;
        bool valid = false;
    };

    int width, height;
    double cell_size = 1.;
    std::span<const double> liquid; // Rebound before every rebuild, including after grid copies/uploads.
    std::vector<Height> heights;
    std::vector<Curvature> raw, curvature;

    FluidSurfaceTensionCPU(int width, int height)
        : width(width), height(height), heights(size_t(width) * height),
          raw(heights.size()), curvature(heights.size()) {}

    static int mirror(int i, int extent)
    {
        const int period = 2 * extent;
        const int q = ((i % period) + period) % period;
        return q >= extent ? period - 1 - q : q;
    }

    double fraction(int x, int y) const
    {
        // Even reflection imposes the same neutral 90-degree wall contact angle.
        const double f = liquid[size_t(mirror(y, height)) * width + mirror(x, width)];
        // Transport roundoff leaves nominally pure double cells a few ulps from 0/1.
        // Treat those as pure for geometry only, or they create spurious height endpoints.
        constexpr double roundoff = 32. * std::numeric_limits<double>::epsilon();
        if (f <= roundoff) return 0.;
        if (f >= 1. - roundoff) return 1.;
        return f;
    }

    bool interfacial(int x, int y) const
    {
        const double f = fraction(x, y);
        if (f > 0. && f < 1.) return true;
        for (const double neighbour : {fraction(x - 1, y), fraction(x + 1, y), fraction(x, y - 1), fraction(x, y + 1)})
            if ((f >= 1. && neighbour <= 0.) || (f <= 0. && neighbour >= 1.)) return true;
        return false;
    }

    interface_geometry::Vec outward(int x, int y) const
    {
        auto n = interface_geometry::normal([&](int dx, int dy) { return fraction(x + dx, y + dy); });
        const double magnitude = std::max(std::abs(n.x), std::abs(n.y));
        n.x /= magnitude;
        n.y /= magnitude;
        const double length = std::sqrt(n.x * n.x + n.y * n.y);
        return {n.x / length, n.y / length};
    }

    std::array<double, 2> column(int x, int y, int axis_x, int axis_y) const
    {
        int start = -4;
        double base = fraction(x - 4 * axis_x, y - 4 * axis_y);
        bool have = base <= 0. || base >= 1.;
        base = base >= 1. ? 1. : 0.;
        double sum = 0., nearest = 10.;
        std::array<double, 2> result = {};
        for (int k = -3; k <= 4; ++k)
        {
            const double f = fraction(x + k * axis_x, y + k * axis_y);
            if (f <= 0. || f >= 1.)
            {
                const double pure = f >= 1. ? 1. : 0.;
                if (have && pure != base)
                {
                    const double h = start + .5 + (base == 1. ? sum : (k - start - 1) - sum);
                    if (std::abs(h) < nearest)
                    {
                        nearest = std::abs(h);
                        result = {h, base == 1. ? 1. : -1.};
                    }
                }
                start = k;
                base = pure;
                have = true;
                sum = 0.;
            }
            else
                sum += f;
        }
        return result;
    }

    void buildHeight(int x, int y)
    {
        auto& h = heights[size_t(y) * width + x];
        h = {};
        double low = 1., high = 0.;
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const double f = fraction(x + dx, y + dy);
            low = std::min(low, f);
            high = std::max(high, f);
        }
        if (low == high) return;
        const auto horizontal = column(x, y, 1, 0);
        const auto vertical = column(x, y, 0, 1);
        h = {horizontal[0], vertical[0], horizontal[1], vertical[1]};
    }

    Height heightAt(int x, int y) const
    {
        auto h = heights[size_t(mirror(y, height)) * width + mirror(x, width)];
        if (x < 0 || x >= width)
        {
            h.x = -h.x;
            h.orientation_x = -h.orientation_x;
        }
        if (y < 0 || y >= height)
        {
            h.y = -h.y;
            h.orientation_y = -h.orientation_y;
        }
        return h;
    }

    Curvature heightCurvature(int x, int y, bool horizontal) const
    {
        const int tangent_x = horizontal ? 0 : 1;
        const int tangent_y = horizontal ? 1 : 0;
        const auto a = heightAt(x - tangent_x, y - tangent_y);
        const auto b = heightAt(x, y);
        const auto c = heightAt(x + tangent_x, y + tangent_y);
        const double orientation = horizontal ? b.orientation_x : b.orientation_y;
        if (orientation == 0. || (horizontal ? a.orientation_x : a.orientation_y) != orientation ||
            (horizontal ? c.orientation_x : c.orientation_y) != orientation) return {};
        const double left = horizontal ? a.x : a.y;
        const double mid = horizontal ? b.x : b.y;
        const double right = horizontal ? c.x : c.y;
        const double slope = .5 * (right - left);
        const double second = (right - mid) - (mid - left);
        const double k = -orientation * second / std::pow(1. + slope * slope, 1.5);
        return {std::clamp(k, -1., 1.) / cell_size, true};
    }

    struct Fit
    {
        double s0 = 0., s1 = 0., s2 = 0., s3 = 0., s4 = 0.;
        double y0 = 0., y1 = 0., y2 = 0.;

        void add(double px, double py, interface_geometry::Vec n, double weight)
        {
            const double x = -px * n.y + py * n.x;
            const double y = px * n.x + py * n.y;
            const double x2 = x * x;
            s0 += weight;
            s1 += weight * x;
            s2 += weight * x2;
            s3 += weight * x2 * x;
            s4 += weight * x2 * x2;
            y0 += weight * y;
            y1 += weight * x * y;
            y2 += weight * x2 * y;
        }

        Curvature evaluate(double cell_size) const
        {
            if (s0 == 0.) return {};
            const double a = s2 - s1 * s1 / s0;
            const double b = s3 - s1 * s2 / s0;
            const double c = s4 - s2 * s2 / s0;
            const double d = y1 - s1 * y0 / s0;
            const double e = y2 - s2 * y0 / s0;
            const double determinant = a * c - b * b;
            // Thin fragments may supply coincident/collinear fit coordinates.
            if (determinant <= 1e-5 * std::max(a * c, 1e-12)) return {};
            const double quadratic = (a * e - b * d) / determinant;
            const double slope = (c * d - b * e) / determinant;
            const double k = -2. * quadratic / std::pow(1. + slope * slope, 1.5);
            // Same subcell-radius estimate limit as the GPU; never alters fractions.
            return {std::clamp(k, -2., 2.) / cell_size, true};
        }
    };

    void buildCurvature(int x, int y)
    {
        auto& k = raw[size_t(y) * width + x];
        k = {};
        if (!interfacial(x, y)) return;
        const auto n = outward(x, y);
        const bool horizontal = std::abs(n.x) > std::abs(n.y);
        k = heightCurvature(x, y, horizontal);
        if (!k.valid) k = heightCurvature(x, y, !horizontal);
        if (k.valid) return;
        Fit fit;
        for (int j = -1; j <= 1; ++j)
        {
            const auto hx = heightAt(x, y + j);
            const auto hy = heightAt(x + j, y);
            if (hx.orientation_x * n.x > 0.) fit.add(hx.x, j, n, 1.);
            if (hy.orientation_y * n.y > 0.) fit.add(j, hy.y, n, 1.);
        }
        k = fit.evaluate(cell_size);
    }

    std::array<double, 3> segment(int x, int y, interface_geometry::Vec n) const
    {
        const double offset = interface_geometry::offset(n, fraction(x, y));
        double low = 10., high = -10.;
        for (int i = 0; i <= 1; ++i)
        {
            if (std::abs(n.x) > 1e-7)
            {
                const double px = (offset - n.y * i) / n.x;
                if (px >= 0. && px <= 1.)
                {
                    const double s = -px * n.y + i * n.x;
                    low = std::min(low, s);
                    high = std::max(high, s);
                }
            }
            if (std::abs(n.y) > 1e-7)
            {
                const double py = (offset - n.x * i) / n.y;
                if (py >= 0. && py <= 1.)
                {
                    const double s = -i * n.y + py * n.x;
                    low = std::min(low, s);
                    high = std::max(high, s);
                }
            }
        }
        if (high <= low) return {};
        const double middle = .5 * (low + high);
        return {n.x * offset - n.y * middle - .5, n.y * offset + n.x * middle - .5, high - low};
    }

    void completeCurvature(int x, int y)
    {
        auto& k = curvature[size_t(y) * width + x];
        k = raw[size_t(y) * width + x];
        if (k.valid || !interfacial(x, y)) return;
        double sum = 0.;
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const auto neighbour = raw[size_t(mirror(y + dy, height)) * width + mirror(x + dx, width)];
            if (neighbour.valid)
            {
                sum += neighbour.value;
                ++count;
            }
        }
        if (count > 0)
        {
            k = {sum / count, true};
            return;
        }
        const auto n = outward(x, y);
        Fit fit;
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const double f = fraction(x + dx, y + dy);
            if (f <= 0. || f >= 1.) continue;
            const auto nq = outward(x + dx, y + dy);
            if (n.x * nq.x + n.y * nq.y <= 0.) continue;
            const auto s = segment(x + dx, y + dy, nq);
            if (s[2] > 0.) fit.add(s[0] + dx, s[1] + dy, n, s[2]);
        }
        k = fit.evaluate(cell_size);
    }

    double faceCurvature(int left, int right) const
    {
        const auto a = curvature[left], b = curvature[right];
        if (!a.valid) return b.valid ? b.value : 0.;
        return b.valid ? .5 * (a.value + b.value) : a.value;
    }
};
}
