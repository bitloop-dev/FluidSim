#pragma once
#include "FluidGrid.h"

#include "FluidGridGPU.h"
#include <cstdint>
#include <vector>

namespace fluid
{
class AmrGas;
class FluidCoarseningPreview;

// Render-thread adapter for the CPU and GPU engines' catalogue colours.
class FluidGridDisplay
{
public:
    FluidGridDisplay() = default;
    FluidGridDisplay(const FluidGridDisplay&) = delete;
    FluidGridDisplay& operator=(const FluidGridDisplay&) = delete;

    std::uint32_t texture(const FluidGrid& grid) const;
#ifdef FLUIDSIM_HAS_AMR
    std::uint32_t texture(const AmrGas& grid) const;
    std::uint32_t resolutionTexture(const AmrGas& grid) const;
#endif
    std::uint32_t texture(const FluidCoarseningPreview& preview) const;
    std::uint32_t resolutionTexture(int width, int height) const;
    std::uint32_t texture(const FluidGridGPU& grid) const { return grid.colorTexture(); }
    void destroy();

private:
    std::uint32_t uploadPixels(int new_width, int new_height) const;
    mutable std::uint32_t cpu_texture = 0;
    mutable int width = 0, height = 0;
    mutable std::vector<float> pixels;
    mutable std::uint32_t preview_texture = 0;
    mutable int preview_width = 0, preview_height = 0;
    mutable std::uint64_t preview_revision = 0;
    mutable const FluidCoarseningPreview* preview_source = nullptr;
    mutable std::uint32_t resolution_texture = 0;
    mutable int resolution_width = 0, resolution_height = 0;
    mutable std::vector<std::uint8_t> resolution_scales;
    std::uint32_t uploadResolution(int w, int h, const std::vector<std::uint8_t>& scales) const;
};

}
