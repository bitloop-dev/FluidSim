#include "FluidGridDisplay.h"
#include <bitloop/core/debug.h>
#include <bitloop/nanovgx/nano_shader_surface.h>
#include <algorithm>

namespace fluid
{

std::uint32_t FluidGridDisplay::texture(const FluidGrid& grid) const
{
    pixels.resize(grid.water_fraction.size());
    std::transform(grid.water_fraction.begin(), grid.water_fraction.end(), pixels.begin(),
                   [](double value) { return float(std::clamp(value, 0.0, 1.0)); });

    bl::GLStateGuard guard;
    glActiveTexture(GL_TEXTURE0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (!cpu_texture)
    {
        glGenTextures(1, &cpu_texture);
        glBindTexture(GL_TEXTURE_2D, cpu_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    else
        glBindTexture(GL_TEXTURE_2D, cpu_texture);

    if (width != grid.width || height != grid.height)
    {
        width = grid.width;
        height = grid.height;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    }
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, pixels.data());
    return cpu_texture;
}

void FluidGridDisplay::destroy()
{
    if (cpu_texture) glDeleteTextures(1, &cpu_texture);
    cpu_texture = 0;
    width = height = 0;
    pixels.clear();
}

}
