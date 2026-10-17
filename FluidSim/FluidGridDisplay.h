#pragma once
#include "FluidGrid.h"
#include "FluidGridGPU.h"
#include <cstdint>
#include <vector>

namespace fluid
{

// render-thread adapter: upload CPU fractions, or use the GPU engine's existing texture
class FluidGridDisplay
{
public:
    FluidGridDisplay() = default;
    FluidGridDisplay(const FluidGridDisplay&) = delete;
    FluidGridDisplay& operator=(const FluidGridDisplay&) = delete;

    std::uint32_t texture(const FluidGrid& grid) const;
    std::uint32_t texture(const FluidGridGPU& grid) const { return grid.waterTexture(); }
    void destroy();

private:
    mutable std::uint32_t cpu_texture = 0;
    mutable int width = 0, height = 0;
    mutable std::vector<float> pixels;
};

}
