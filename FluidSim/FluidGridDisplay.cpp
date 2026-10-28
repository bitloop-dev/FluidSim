#include "FluidGridDisplay.h"
#include <bitloop/core/debug.h>
#include <bitloop/nanovgx/nano_shader_surface.h>

namespace fluid
{

std::uint32_t FluidGridDisplay::texture(const FluidGridBase& grid) const
{
    grid.writeColorPixels(pixels);
    return texture(grid.getWidth(), grid.getHeight(), pixels);
}

std::uint32_t FluidGridDisplay::texture(int new_width, int new_height, std::span<const float> rgba) const
{
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

    if (width != new_width || height != new_height)
    {
        width = new_width;
        height = new_height;

        // Allocate storage for four floating-point colour channels.
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F,
            width, height, 0,
            GL_RGBA, GL_FLOAT, nullptr);
    }

    // Upload the current colours into the existing texture.
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0,
        width, height,
        GL_RGBA, GL_FLOAT, rgba.data());

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
