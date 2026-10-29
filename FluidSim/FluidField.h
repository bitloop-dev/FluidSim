#pragma once
#include <bitloop/core/types.h>
#include <array>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace fluid
{
namespace webgpu { class Compute; }
namespace detail { struct FluidFieldAccess; }

// Shared device and ordered command stream. Create/use/release on one owner thread.
class FluidWebGPUContext
{
public:
    FluidWebGPUContext();
    bool operator==(const FluidWebGPUContext&) const = default;

private:
    friend struct detail::FluidFieldAccess;
    std::shared_ptr<webgpu::Compute> commands;
};

// Cell-centred samples. Physical meaning (acceleration, velocity, etc.) belongs to
// the consumer. Copies of this handle share live data; transfers make new data.
class FluidField
{
public:
    struct Layout
    {
        int width;
        int height;
        double cell_size;
        bl::DVec2 origin = {};
        int channels = 1;
    };

    explicit FluidField(Layout layout); // CPU storage, initially zero.
    FluidField(Layout layout, const FluidWebGPUContext& context); // GPU storage, initially zero.
    const Layout& layout() const;
    bool isWebGPU() const;

    // CPU access only. Interleaved channels, with X varying fastest.
    std::span<float> values();
    std::span<const float> values() const;
    // Explicit data movement; write uploads and read blocks for readback on GPU fields.
    void write(std::span<const float> values);
    std::vector<float> read() const;
    FluidField upload(const FluidWebGPUContext& context) const; // CPU -> new GPU field.
    FluidField download() const; // New CPU copy.

private:
    friend struct detail::FluidFieldAccess;
    struct Storage;
    std::shared_ptr<Storage> storage;
};

// CPU producer; returns up to four channels at each world-space cell centre.
void generateFluidField(FluidField& field,
    const std::function<std::array<float, 4>(bl::DVec2)>& evaluate);

// Independent GPU producer. Define fn fieldValue(position: vec2f) -> vec4f;
// unused channels are ignored. fieldParameter(index) reads up to twelve vec4s.
class FluidFieldWebGPUGenerator
{
public:
    FluidFieldWebGPUGenerator(const FluidWebGPUContext& context, std::string_view shader);
    ~FluidFieldWebGPUGenerator();
    FluidFieldWebGPUGenerator(const FluidFieldWebGPUGenerator&) = delete;
    FluidFieldWebGPUGenerator& operator=(const FluidFieldWebGPUGenerator&) = delete;
    void generate(FluidField& field, std::span<const std::array<float, 4>> parameters = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
