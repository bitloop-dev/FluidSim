#pragma once
#include <cstdint>
#include <memory>

namespace fluid
{
class FluidAmbientMask;

// Render-thread palette adapter. Uploads only cell labels; opacity belongs to the scene.
class FluidAmbientMaskDisplay
{
public:
    FluidAmbientMaskDisplay();
    ~FluidAmbientMaskDisplay();
    FluidAmbientMaskDisplay(const FluidAmbientMaskDisplay&) = delete;
    FluidAmbientMaskDisplay& operator=(const FluidAmbientMaskDisplay&) = delete;
    std::uint32_t texture(const FluidAmbientMask& mask) const;
    void destroy();

private:
    struct Impl;
    mutable std::unique_ptr<Impl> impl;
};
}
