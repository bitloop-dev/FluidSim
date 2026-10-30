#pragma once
#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace fluid {

// Derived display samples only. Each RGBA layer holds four ingredients in
// mixture order (liquid, then gas); one scalar liquid-coverage plane follows.
// One contiguous allocation lets WebGPU read both textures in one transfer.
struct FluidMaterialPixels {
    int width = 0, height = 0, ingredients = 0;
    std::vector<float> values;

    FluidMaterialPixels() = default;
    FluidMaterialPixels(int w, int h, int count) : width(w), height(h), ingredients(count) {}

    int groups() const { return std::max(1, (ingredients + 3) / 4); }
    std::size_t cellCount() const { return std::size_t(width) * height; }
    std::size_t valueCount() const { return cellCount() * (4 * groups() + 1); }
    std::size_t fractionIndex(std::size_t cell, int ingredient) const {
        return (std::size_t(ingredient / 4) * cellCount() + cell) * 4 + ingredient % 4;
    }
    std::span<const float> fractions() const { return {values.data(), cellCount() * 4 * groups()}; }
    std::span<const float> liquid() const { return {values.data() + cellCount() * 4 * groups(), cellCount()}; }
};

} // namespace fluid
