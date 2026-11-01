#pragma once
#include <algorithm>
#include <cmath>

namespace fluid::interface_geometry
{
struct Vec
{
    double x;
    double y;
};

// MYC normal from the same 3x3 phase stencil used by the uniform liquid engine.
template <class Sample> Vec normal(const Sample& f)
{
    const double tl = f(-1, -1);
    const double t = f(0, -1);
    const double tr = f(1, -1);
    const double l = f(-1, 0);
    const double r = f(1, 0);
    const double bl = f(-1, 1);
    const double b = f(0, 1);
    const double br = f(1, 1);
    const double left = tl + l + bl;
    const double right = tr + r + br;
    const double top = tl + t + tr;
    const double bottom = bl + b + br;
    Vec centered{0.5 * (left - right), 0.5 * (top - bottom)};
    const Vec youngs{left + l - right - r, top + t - bottom - b};
    if (centered.x == 0.0 && centered.y == 0.0)
        return youngs.x == 0.0 && youngs.y == 0.0 ? Vec{1.0, 0.0} : youngs;

    if (std::abs(centered.x) <= std::abs(centered.y))
    {
        centered.y = std::copysign(1.0, centered.y);

        if (std::abs(youngs.x) > std::abs(centered.x * youngs.y))
            return youngs;
    }
    else
    {
        centered.x = std::copysign(1.0, centered.x);

        if (std::abs(youngs.y) > std::abs(centered.y * youngs.x))
            return youngs;
    }

    return centered;
}

template <class Vector> double offset(Vector n, double fraction)
{
    const double a = std::min(std::abs(n.x), std::abs(n.y));
    const double b = std::max(std::abs(n.x), std::abs(n.y));
    const double triangle = 0.5 * a / b;
    const double cut = fraction < triangle         ? std::sqrt(2.0 * a * b * fraction)
                       : fraction <= 1.0 - triangle ? b * fraction + 0.5 * a
                                                   : a + b - std::sqrt(2.0 * a * b * (1.0 - fraction));

    return std::min(n.x, 0.0) + std::min(n.y, 0.0) + cut;
}

template <class Vector> double area(Vector n, double offset, Vector lo, Vector hi)
{
    const double width = hi.x - lo.x;
    const double height = hi.y - lo.y;
    const double total = width * height;
    const double cut = offset - (n.x * (n.x >= 0.0 ? lo.x : hi.x) + n.y * (n.y >= 0.0 ? lo.y : hi.y));
    const double a = std::min(std::abs(n.x) * width, std::abs(n.y) * height);
    const double b = std::max(std::abs(n.x) * width, std::abs(n.y) * height);
    if (cut <= 0.0)
        return 0.0;

    if (cut >= a + b)
        return total;

    if (a == 0.0)
        return total * cut / b;

    if (cut < a)
        return total * cut * cut / (2.0 * a * b);

    if (cut < b)
        return total * (cut - 0.5 * a) / b;

    const double dry = a + b - cut;

    return total * (1.0 - dry * dry / (2.0 * a * b));
}
}
