#include "FluidGridDisplay.h"
#include <bitloop/core/debug.h>
#include <bitloop/core/threads.h>
#include <bitloop/nanovgx/nano_shader_surface.h>
#include <algorithm>

namespace fluid
{

std::uint32_t FluidGridDisplay::texture(const FluidGrid& grid) const
{
    const int cell_count = grid.width * grid.height;
    const double cell_volume = grid.cell_size * grid.cell_size;

    pixels.resize(size_t(cell_count) * 4);

    bl::Thread::forEachBatch(cell_count, [&](int i0, int i1)
    {
        for (int i = i0; i < i1; ++i)
        {
            float* pixel = pixels.data() + size_t(i) * 4;
            pixel[0] = pixel[1] = pixel[2] = 0.0f;
            pixel[3] = 1.0f;

            auto addMixture = [&](const FluidMixture& mixture, const auto& masses)
            {
                for (size_t k = 0; k < mixture.components.size(); ++k)
                {
                    const auto& component = mixture.components[k];
                    const auto& color = grid.elements[component.element].color;

                    // Convert ingredient mass into its fraction of the whole cell.
                    const double volume_fraction =
                        masses[k][i] / (component.density * cell_volume);

                    for (int channel = 0; channel < 3; ++channel)
                        pixel[channel] += float(volume_fraction * color[channel]);
                }
            };

            addMixture(grid.liquid_mixture, grid.liquid_mass);
            addMixture(grid.gas_mixture, grid.gas_mass);
        }
    });

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
        GL_RGBA, GL_FLOAT, pixels.data());

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
