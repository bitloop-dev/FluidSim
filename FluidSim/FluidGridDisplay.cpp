#include "FluidGridDisplay.h"
#include "FluidCellView.h"
#include "FluidCoarseningPreview.h"
#include <bitloop/core/debug.h>
#include <bitloop/core/threads.h>
#include <bitloop/nanovgx/nano_shader_surface.h>
#include <algorithm>

namespace fluid
{

namespace {
void cellColour(const FluidGrid& grid, int i, float* pixel)
{
    const double cell_volume = grid.cell_size * grid.cell_size;
    pixel[0] = pixel[1] = pixel[2] = 0.0f;
    pixel[3] = 1.0f;
    auto addMixture = [&](const FluidMixture& mixture, const auto& masses) {
        for (size_t k = 0; k < mixture.components.size(); ++k)
        {
            const auto& component = mixture.components[k];
            const auto& color = grid.elements[component.element].color;
            const double fraction = masses[k][i] / (component.density * cell_volume);
            for (int channel = 0; channel < 3; ++channel)
                pixel[channel] += float(fraction * color[channel]);
        }
    };
    addMixture(grid.liquid_mixture, grid.liquid_mass);
    addMixture(grid.gas_mixture, grid.gas_mass);
}
}

std::uint32_t FluidGridDisplay::texture(const FluidGrid& grid) const
{
    const int count = grid.width * grid.height;
    pixels.resize(size_t(count) * 4);
    bl::Thread::forEachBatch(count, [&](int first, int last) {
        for (int i = first; i < last; ++i) cellColour(grid, i, pixels.data() + size_t(i) * 4);
    });
    return uploadPixels(grid.width, grid.height);
}

#ifdef FLUIDSIM_HAS_AMR
std::uint32_t FluidGridDisplay::texture(const AmrGas& grid) const {
    const FluidCellView view(grid);
    pixels.resize(size_t(view.width)*view.height*4);
    // Display the same bounded plane used by transport; these pixels are not
    // persistent fine material. Covered coarse cells contribute no second colour.
    const int patchCount=int(view.patches.size());
    bl::Thread::forEachBatch(patchCount,[&](int first,int last) {
        FluidCellView::MaterialPlane plane;
        for (int index=first;index<last;++index) {
            const auto& p=view.patches[index];
            for (int row=0; row<p.height/p.scale; ++row) for (int x=p.x; x<p.x+p.width; x+=p.scale) {
                const int y=p.y+row*p.scale;
                if (!p.active(x/p.scale,y/p.scale)) continue;
                if (p.scale>1) view.reconstruct(plane,x,y,p.scale);
                for (int dy=0; dy<p.scale; ++dy) for (int dx=0; dx<p.scale; ++dx) {
                    float colour[4]={0,0,0,1};
                    for (int k=0; k<view.components(); ++k) {
                        const auto& component=view.mixture->components[k];
                        const auto& c=(*view.catalogue)[component.element].color;
                        const double q=p.scale==1 ? p.q(x,y,k) : plane.value(k,(dx+.5)/p.scale-.5,(dy+.5)/p.scale-.5);
                        for (int channel=0; channel<3; ++channel) colour[channel]+=float(q/component.density*c[channel]);
                    }
                    std::copy_n(colour,4,pixels.data()+size_t((y+dy)*view.width+x+dx)*4);
                }
            }
        }
    },std::min({8,int(bl::Thread::threadCount()),patchCount}));
    return uploadPixels(view.width,view.height);
}
#endif

std::uint32_t FluidGridDisplay::uploadPixels(int new_width, int new_height) const
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
        GL_RGBA, GL_FLOAT, pixels.data());

    return cpu_texture;
}

std::uint32_t FluidGridDisplay::texture(const FluidCoarseningPreview& preview) const
{
    bl::GLStateGuard guard;
    glActiveTexture(GL_TEXTURE0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (!preview_texture)
    {
        glGenTextures(1, &preview_texture);
        glBindTexture(GL_TEXTURE_2D, preview_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    else glBindTexture(GL_TEXTURE_2D, preview_texture);
    if (preview_width != preview.width || preview_height != preview.height)
    {
        preview_width = preview.width; preview_height = preview.height;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, preview_width, preview_height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
        preview_source = nullptr;
    }
    if (preview_source != &preview || preview_revision != preview.revision)
    {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, preview_width, preview_height, GL_RED, GL_UNSIGNED_BYTE, preview.labels.data());
        preview_source = &preview; preview_revision = preview.revision;
    }
    return preview_texture;
}

std::uint32_t FluidGridDisplay::resolutionTexture(int width,int height) const {
    const int w=(width+1)/2,h=(height+1)/2;
    return uploadResolution(w,h,std::vector<std::uint8_t>(size_t(w)*h,1));
}
#ifdef FLUIDSIM_HAS_AMR
std::uint32_t FluidGridDisplay::resolutionTexture(const AmrGas& grid) const {
    const int w=(grid.finestWidth()+1)/2,h=(grid.finestHeight()+1)/2;
    std::vector<std::uint8_t> scales(size_t(w)*h,std::uint8_t(grid.layout().baseRatio));
    int scale=grid.layout().baseRatio;
    for (const auto& patches:grid.layout().refinedPatches) {
        scale/=2;
        for (const auto& p:patches)
            for (int y=p.y/2;y<(p.y+p.height)/2;++y) std::fill_n(scales.begin()+y*w+p.x/2,p.width/2,std::uint8_t(scale));
    }
    return uploadResolution(w,h,scales);
}
#endif
std::uint32_t FluidGridDisplay::uploadResolution(int w,int h,const std::vector<std::uint8_t>& scales) const {
    if (resolution_texture && resolution_width==w && resolution_height==h && resolution_scales==scales) return resolution_texture;
    bl::GLStateGuard guard;
    glActiveTexture(GL_TEXTURE0); glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    if (!resolution_texture)
    {
        glGenTextures(1,&resolution_texture); glBindTexture(GL_TEXTURE_2D,resolution_texture);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    }
    else glBindTexture(GL_TEXTURE_2D,resolution_texture);
    glTexImage2D(GL_TEXTURE_2D,0,GL_R8,w,h,0,GL_RED,GL_UNSIGNED_BYTE,scales.data());
    resolution_width=w; resolution_height=h; resolution_scales=scales;
    return resolution_texture;
}

void FluidGridDisplay::destroy()
{
    if (cpu_texture) glDeleteTextures(1, &cpu_texture);
    if (preview_texture) glDeleteTextures(1, &preview_texture);
    if (resolution_texture) glDeleteTextures(1,&resolution_texture);
    resolution_texture=0; resolution_width=resolution_height=0; resolution_scales.clear();
    preview_texture = 0; preview_width = preview_height = 0;
    preview_source = nullptr; preview_revision = 0;
    cpu_texture = 0;
    width = height = 0;
    pixels.clear();
}

}
