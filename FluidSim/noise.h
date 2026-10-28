#pragma once

static inline double perlinNoise(double x, double y)
{
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const double fx = x - ix, fy = y - iy;

    auto gradient = [](int gx, int gy, double dx, double dy)
    {
        // Hash each lattice corner to a repeatable gradient direction.
        std::uint32_t hash = std::uint32_t(gx) * 0x9e3779b9u ^ std::uint32_t(gy) * 0x85ebca6bu ^ 0x12345678u;
        hash ^= hash >> 16;
        hash *= 0x7feb352du;
        hash ^= hash >> 15;
        hash *= 0x846ca68bu;
        hash ^= hash >> 16;
        constexpr double diagonal = 0.7071067811865476;
        constexpr double directions[8][2] = {
            {1, 0}, {-1, 0}, {0, 1}, {0, -1},
            {diagonal, diagonal}, {-diagonal, diagonal},
            {diagonal, -diagonal}, {-diagonal, -diagonal}
        };
        const auto& direction = directions[hash & 7u];
        return direction[0] * dx + direction[1] * dy;
    };

    // Quintic interpolation joins neighbouring cells with smooth derivatives.
    auto fade = [](double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); };
    const double sx = fade(fx), sy = fade(fy);
    const double top = std::lerp(gradient(ix, iy, fx, fy), gradient(ix + 1, iy, fx - 1.0, fy), sx);
    const double bottom = std::lerp(gradient(ix, iy + 1, fx, fy - 1.0), gradient(ix + 1, iy + 1, fx - 1.0, fy - 1.0), sx);
    return std::lerp(top, bottom, sy);
}