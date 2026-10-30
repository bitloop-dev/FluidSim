#include "FluidGridDisplay.h"
#include <bitloop/core/debug.h>
#include <bitloop/nanovgx/nano_shader_surface.h>
#include <cmath>

namespace fluid
{

namespace
{
std::uint32_t uploadTexture(std::uint32_t& cpu_texture, int& width, int& height,
    int new_width, int new_height, GLenum format, const float* data)
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

        glTexImage2D(
            GL_TEXTURE_2D, 0, format == GL_RED ? GL_R32F : GL_RGBA32F,
            width, height, 0,
            format, GL_FLOAT, nullptr);
    }

    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0,
        width, height,
        format, GL_FLOAT, data);

    return cpu_texture;
}
}

std::uint32_t FluidGridDisplay::texture(const FluidGridBase& grid) const
{
    grid.writeColorPixels(pixels);
    return texture(grid.getWidth(), grid.getHeight(), pixels);
}

std::uint32_t FluidGridDisplay::texture(int w, int h, std::span<const float> rgba) const
{
    return uploadTexture(cpu_texture, width, height, w, h, GL_RGBA, rgba.data());
}

std::uint32_t FluidGridDisplay::proximityTexture(int w, int h, std::span<const float> rgba, float range) const
{
    // Continuous coverage avoids switching bubbles on/off at a phase threshold.
    // Keep coverage linear so a bubble split across cells retains its influence.
    proximity.resize(size_t(w) * h);
    for (size_t i = 0; i < proximity.size(); ++i)
        proximity[i] = std::clamp(1.f - rgba[4 * i + 3], 0.f, 1.f);

    // Separable exponential smoothing blends contributions from neighbouring
    // pockets, avoiding the ridges between nearest-bubble distance regions.
    const float decay = std::exp(-1.f / range);
    auto smooth = [&](int first, int count, int stride)
    {
        for (int n = 1; n < count; ++n)
            proximity[first + n * stride] = std::lerp(proximity[first + n * stride], proximity[first + (n - 1) * stride], decay);
        for (int n = count - 2; n >= 0; --n)
            proximity[first + n * stride] = std::lerp(proximity[first + n * stride], proximity[first + (n + 1) * stride], decay);
    };
    for (int y = 0; y < h; ++y) smooth(y * w, w, 1);
    for (int x = 0; x < w; ++x) smooth(x, h, w);
    return uploadTexture(proximity_texture, proximity_width, proximity_height, w, h, GL_RED, proximity.data());
}

void FluidGridDisplay::destroy()
{
    if (cpu_texture) glDeleteTextures(1, &cpu_texture);
    if (proximity_texture) glDeleteTextures(1, &proximity_texture);
    cpu_texture = 0;
    width = height = 0;
    pixels.clear();
    proximity_texture = 0;
    proximity_width = proximity_height = 0;
    proximity.clear();
}

}
