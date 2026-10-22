#include "FluidAmbientMaskDisplayTests.h"
#include "FluidAmbientMaskDisplay.h"
#include "FluidAmbientMask.h"
#include "FluidGridGPUInternal.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace fluid::test
{
namespace
{
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void preservesGraphicsState()
{
    gpu_detail::ComputeState restore;
    FluidAmbientMaskDisplay observer;
    glEnable(GL_SCISSOR_TEST);
    glEnable(GL_BLEND);
    glViewport(3, 4, 17, 19);
    glPixelStorei(GL_PACK_ROW_LENGTH, 17);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 17);
    glActiveTexture(GL_TEXTURE3);
    FluidAmbientMask ambient;
    ambient.width = 7; ambient.height = 5; ambient.revision = 1;
    ambient.cells.assign(35, FluidAmbientMask::Candidate);
    observer.texture(ambient);
    GLint viewport[4], value;
    glGetIntegerv(GL_VIEWPORT, viewport);
    require(viewport[0] == 3 && viewport[1] == 4 && viewport[2] == 17 && viewport[3] == 19, "mask display changed viewport state");
    require(glIsEnabled(GL_BLEND) && glIsEnabled(GL_SCISSOR_TEST), "mask display changed graphics capabilities");
    glGetIntegerv(GL_PACK_ROW_LENGTH, &value);
    require(value == 17, "mask display changed pack state");
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &value);
    require(value == 17, "mask display changed unpack state");
    glGetIntegerv(GL_ACTIVE_TEXTURE, &value);
    require(value == GL_TEXTURE3, "mask display changed active texture");
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glActiveTexture(GL_TEXTURE0);
}

void ambientPaletteMatchesCells()
{
    gpu_detail::ComputeState guard;
    FluidAmbientMask mask;
    mask.width = 7; mask.height = 5; mask.revision = 1;
    for (int i = 0; i < 35; ++i) mask.cells.push_back(std::uint8_t(i % 6));
    FluidAmbientMaskDisplay observer;
    gpu_detail::PassRunner reader;
    auto read = [&] {
        gpu_detail::Texture view;
        view.id = observer.texture(mask); view.width = 7; view.height = 5; view.channels = 4;
        auto pixels = reader.read(view); view.id = 0; return pixels;
    };
    const float palette[6][3] = {{.2f,.55f,1}, {1,.85f,0}, {1,.45f,.1f}, {.65f,.65f,.65f}, {.7f,.3f,.85f}, {.15f,.85f,.3f}};
    auto pixels = read();
    for (int i = 0; i < 35; ++i) for (int c = 0; c < 3; ++c)
        require(std::abs(pixels[4 * i + c] - palette[i % 6][c]) < 1e-6, "ambient palette moved or changed a cell label");
    mask.cells[0] = FluidAmbientMask::Candidate; ++mask.revision;
    require(std::abs(read()[0] - .15f) < 1e-6, "ambient display cached a stale mask");
}
}

void ambientMaskDisplayRegressions()
{
    preservesGraphicsState();
    ambientPaletteMatchesCells();
    require(glGetError() == GL_NO_ERROR, "GL error during mask display checks");
    std::cout << "PASS ambient mask palette, cache and graphics state\n";
}
}
