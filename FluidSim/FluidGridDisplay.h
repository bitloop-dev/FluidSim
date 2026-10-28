#pragma once
#include "FluidGridBase.h"
#include "FluidGridWebGL.h"
#include <cstdint>
#include <span>
#include <vector>

namespace fluid
{

// Render-thread adapter for the CPU and GPU engines' catalogue colours.
class FluidGridDisplay
{
public:
    FluidGridDisplay() = default;
    FluidGridDisplay(const FluidGridDisplay&) = delete;
    FluidGridDisplay& operator=(const FluidGridDisplay&) = delete;

    std::uint32_t texture(const FluidGridBase& grid) const;
    // Prepared on the simulation owner; GL rendering consumes only derived RGBA.
    std::uint32_t texture(int width, int height, std::span<const float> rgba) const;
    std::uint32_t texture(const FluidGridWebGL& grid) const { return grid.colorTexture(); }
    void destroy();

private:
    mutable std::uint32_t cpu_texture = 0;
    mutable int width = 0, height = 0;
    mutable std::vector<float> pixels;
};

}
