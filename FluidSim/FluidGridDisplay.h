#pragma once
#include "FluidGrid.h"
#include "FluidMaterialPixels.h"

#include <cstdint>
#include <span>
#include <vector>

namespace fluid
{
class FluidGridAdaptive;
class FluidCoarseningPreview;
struct GasLayout;

// Render-thread adapter; GPU images arrive from their simulation-thread owner.
class FluidGridDisplay
{
public:
    FluidGridDisplay() = default;
    FluidGridDisplay(const FluidGridDisplay&) = delete;
    FluidGridDisplay& operator=(const FluidGridDisplay&) = delete;

    std::uint32_t texture(const FluidGrid& grid) const;
    struct MaterialTextures { std::uint32_t fractions, liquid; };
    static FluidMaterialPixels materialPixels(const FluidGrid& grid);
    MaterialTextures materialTextures(const FluidGrid& grid) const;
    MaterialTextures materialTextures(const FluidMaterialPixels& material) const;
    std::uint32_t texture(const FluidMaterialPixels& material, const ElementCatalogue& elements,
                          const FluidMixture& mixture) const;
    // RGBA = catalogue colour, ElementId; slots follow the supplied mixture.
    std::uint32_t paletteTexture(const ElementCatalogue& elements, const FluidMixture& mixture) const;
#ifdef FLUIDSIM_HAS_AMR
    std::uint32_t texture(const FluidGridAdaptive& grid) const;
    static FluidMaterialPixels materialPixels(const FluidGridAdaptive& grid);
    MaterialTextures materialTextures(const FluidGridAdaptive& grid) const;
    std::uint32_t resolutionTexture(const FluidGridAdaptive& grid) const;
#endif
    std::uint32_t texture(const FluidCoarseningPreview& preview) const;
    std::uint32_t previewTexture(int width,int height,std::span<const std::uint8_t> labels,std::uint64_t revision,const void* source) const;
    std::uint32_t resolutionTexture(int width, int height) const;
    std::uint32_t resolutionTexture(int width,int height,const GasLayout& layout) const;
    std::uint32_t texture(int width, int height, std::span<const float> rgba) const;
    void destroy();

private:
    std::uint32_t uploadPixels(int new_width, int new_height, std::span<const float> rgba) const;
    mutable std::uint32_t cpu_texture = 0;
    mutable int width = 0, height = 0;
    mutable std::vector<float> pixels;
    mutable std::uint32_t fraction_texture = 0, liquid_texture = 0, palette_texture = 0;
    mutable int material_width = 0, material_height = 0, material_groups = 0, palette_width = 0;
    mutable std::uint32_t preview_texture = 0;
    mutable int preview_width = 0, preview_height = 0;
    mutable std::uint64_t preview_revision = 0;
    mutable const void* preview_source = nullptr;
    mutable std::uint32_t resolution_texture = 0;
    mutable int resolution_width = 0, resolution_height = 0;
    mutable std::vector<std::uint8_t> resolution_scales;
    std::uint32_t uploadResolution(int w, int h, const std::vector<std::uint8_t>& scales) const;
};

}
