#include "FluidAmbientMaskDisplay.h"
#include "FluidAmbientMask.h"
#include "FluidGridGPUInternal.h"

namespace fluid
{
using namespace gpu_detail;

struct FluidAmbientMaskDisplay::Impl
{
    Texture labels, overlay;
    PassRunner runner;
    Program display;
    const FluidAmbientMask* source = nullptr;
    std::uint64_t revision = 0;
    Impl(int w, int h) : labels(w, h), overlay(w, h, 4)
    {
        runner.build(display, "CPU ambient mask overlay", common, R"(
            uniform sampler2D labels;
            void main()
            {
                int label = int(round(255.0 * at(labels, cell())));
                vec3 colour = vec3(0.65); // waiting
                if (label == 0) colour = vec3(0.2, 0.55, 1); // incompatible material
                if (label == 1) colour = vec3(1, 0.85, 0); // velocity / pressure departure
                if (label == 2) colour = vec3(1, 0.45, 0.1); // buffer
                if (label == 4) colour = vec3(0.7, 0.3, 0.85); // disconnected quiet patch
                if (label == 5) colour = vec3(0.15, 0.85, 0.3); // reservoir-connected candidate
                output0 = vec4(colour, 1.0);
            }
        )");
    }
};

FluidAmbientMaskDisplay::FluidAmbientMaskDisplay() = default;
FluidAmbientMaskDisplay::~FluidAmbientMaskDisplay() = default;
void FluidAmbientMaskDisplay::destroy() { impl.reset(); }

std::uint32_t FluidAmbientMaskDisplay::texture(const FluidAmbientMask& mask) const
{
    if (mask.cells.empty()) return 0;
    ComputeState guard;
    if (!impl || impl->labels.width != mask.width || impl->labels.height != mask.height)
        impl = std::make_unique<Impl>(mask.width, mask.height);
    auto& d = *impl;
    if (d.source != &mask || d.revision != mask.revision)
    {
        glBindTexture(GL_TEXTURE_2D, d.labels.id);
        // GL normalises each byte; the palette shader decodes the original label.
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mask.width, mask.height, GL_RED, GL_UNSIGNED_BYTE, mask.cells.data());
        d.runner.bind(d.display, {&d.overlay}, {{"labels", &d.labels}});
        glDrawArrays(GL_TRIANGLES, 0, 3);
        d.source = &mask;
        d.revision = mask.revision;
    }
    return d.overlay.id;
}

}
