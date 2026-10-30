#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "../noise.h"

// The same initial gas pattern for both experiments; the catalogue supplies its colours.
static inline double perlinCloudShare(int x, int y, int width)
{
    const double scale = 6.0 / width;
    const double nx = (x + 0.5) * scale;
    const double ny = (y + 0.5) * scale;
    const double noise = (
        perlinNoise(nx, ny) +
        0.5 * perlinNoise(2.0 * nx, 2.0 * ny) +
        0.25 * perlinNoise(4.0 * nx, 4.0 * ny)) / 1.75;
    return std::clamp(0.15 + 2.5 * noise, 0.0, 1.0);
}
