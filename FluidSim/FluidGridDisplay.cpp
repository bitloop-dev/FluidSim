#include "FluidGridDisplay.h"
#include "FluidCellView.h"
#include "FluidCoarseningPreview.h"
#include "FluidGridGPUInternal.h"
#include <bitloop/core/debug.h>
#include <bitloop/core/threads.h>
#include <bitloop/nanovgx/nano_shader_surface.h>
#include <algorithm>

namespace fluid
{

namespace {
void uploadFloatTexture(GLuint& texture, int width, int height, GLenum storage, GLenum format,
                        bool resized, const float* pixels)
{
    if (!texture) {
        glGenTextures(1,&texture);
        glBindTexture(GL_TEXTURE_2D,texture);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        resized=true;
    } else glBindTexture(GL_TEXTURE_2D,texture);
    if (resized) glTexImage2D(GL_TEXTURE_2D,0,storage,width,height,0,format,GL_FLOAT,nullptr);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,format,GL_FLOAT,pixels);
}

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

FluidMaterialPixels FluidGridDisplay::materialPixels(const FluidGrid& grid)
{
    FluidMaterialPixels result(grid.width,grid.height,int(grid.liquid_mixture.components.size()+grid.gas_mixture.components.size()));
    result.values.resize(result.valueCount());
    const double area=grid.cell_size*grid.cell_size;
    bl::Thread::forEachBatch(int(result.cellCount()),[&](int first,int last) {
        for (int i=first;i<last;++i) {
            int ingredient=0;
            auto add=[&](const FluidMixture& mixture,const auto& masses) {
                for (size_t k=0;k<mixture.components.size();++k,++ingredient)
                    result.values[result.fractionIndex(i,ingredient)]=float(masses[k][i]/(mixture.components[k].density*area));
            };
            add(grid.liquid_mixture,grid.liquid_mass); add(grid.gas_mixture,grid.gas_mass);
            result.values[result.fractions().size()+i]=float(grid.liquid_fraction[i]);
        }
    });
    return result;
}

FluidGridDisplay::MaterialTextures FluidGridDisplay::materialTextures(const FluidGrid& grid) const {
    return materialTextures(materialPixels(grid));
}

FluidGridDisplay::MaterialTextures FluidGridDisplay::materialTextures(const FluidMaterialPixels& material) const
{
    gpu_detail::ComputeState guard;
    glActiveTexture(GL_TEXTURE0);
    const bool resized=material_width!=material.width || material_height!=material.height || material_groups!=material.groups();
    uploadFloatTexture(fraction_texture,material.width,material.height*material.groups(),GL_RGBA32F,GL_RGBA,resized,material.fractions().data());
    uploadFloatTexture(liquid_texture,material.width,material.height,GL_R32F,GL_RED,resized,material.liquid().data());
    material_width=material.width; material_height=material.height; material_groups=material.groups();
    return {fraction_texture,liquid_texture};
}

std::uint32_t FluidGridDisplay::paletteTexture(const ElementCatalogue& elements,const FluidMixture& mixture) const
{
    const int width=std::max(1,int(mixture.components.size()));
    std::vector<float> palette(size_t(width)*4,0.f);
    for (size_t k=0;k<mixture.components.size();++k) {
        const auto id=mixture.components[k].element;
        const auto& color=elements[id].color;
        std::copy_n(color.data(),3,palette.data()+k*4);
        palette[k*4+3]=float(id);
    }
    gpu_detail::ComputeState guard;
    glActiveTexture(GL_TEXTURE0);
    uploadFloatTexture(palette_texture,width,1,GL_RGBA32F,GL_RGBA,palette_width!=width,palette.data());
    palette_width=width;
    return palette_texture;
}

std::uint32_t FluidGridDisplay::texture(const FluidMaterialPixels& material,const ElementCatalogue& elements,
                                      const FluidMixture& mixture) const
{
    pixels.assign(material.cellCount()*4,0.f);
    for (size_t i=0;i<material.cellCount();++i) {
        pixels[i*4+3]=1.f;
        for (int k=0;k<material.ingredients;++k) {
            const auto& color=elements[mixture.components[k].element].color;
            const float f=material.values[material.fractionIndex(i,k)];
            for (int c=0;c<3;++c) pixels[i*4+c]+=f*color[c];
        }
    }
    return uploadPixels(material.width,material.height,pixels);
}

std::uint32_t FluidGridDisplay::texture(const FluidGrid& grid) const
{
    const int count = grid.width * grid.height;
    pixels.resize(size_t(count) * 4);
    bl::Thread::forEachBatch(count, [&](int first, int last) {
        for (int i = first; i < last; ++i) cellColour(grid, i, pixels.data() + size_t(i) * 4);
    });
    return uploadPixels(grid.width, grid.height, pixels);
}

#ifdef FLUIDSIM_HAS_AMR
FluidMaterialPixels FluidGridDisplay::materialPixels(const FluidGridAdaptive& grid) {
    const FluidCellView view(grid);
    FluidMaterialPixels result(view.width,view.height,view.components());
    result.values.resize(result.valueCount());
    bl::Thread::forEachBatch(int(view.patches.size()),[&](int first,int last) {
        FluidCellView::MaterialPlane plane;
        for (int index=first;index<last;++index) {
            const auto& p=view.patches[index];
            for (int y=p.y;y<p.y+p.height;y+=p.scale) for (int x=p.x;x<p.x+p.width;x+=p.scale) {
                if (!p.active(x/p.scale,y/p.scale)) continue;
                if (p.scale>1) view.reconstruct(plane,x,y,p.scale);
                for (int dy=0;dy<p.scale;++dy) for (int dx=0;dx<p.scale;++dx) {
                    const size_t i=size_t(y+dy)*view.width+x+dx;
                    double liquid=0.;
                    for (int k=0;k<view.components();++k) {
                        const double q=p.scale==1 ? p.q(x,y,k) : plane.value(k,(dx+.5)/p.scale-.5,(dy+.5)/p.scale-.5);
                        const double fraction=q/view.mixture->components[k].density;
                        result.values[result.fractionIndex(i,k)]=float(fraction);
                        if (k<view.liquidComponents) liquid+=fraction;
                    }
                    result.values[result.fractions().size()+i]=float(liquid);
                }
            }
        }
    },std::min({8,int(bl::Thread::threadCount()),int(view.patches.size())}));
    return result;
}

FluidGridDisplay::MaterialTextures FluidGridDisplay::materialTextures(const FluidGridAdaptive& grid) const {
    return materialTextures(materialPixels(grid));
}

std::uint32_t FluidGridDisplay::texture(const FluidGridAdaptive& grid) const {
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
    return uploadPixels(view.width,view.height,pixels);
}
#endif

std::uint32_t FluidGridDisplay::texture(int width,int height,std::span<const float> rgba) const
{
    return uploadPixels(width,height,rgba);
}

std::uint32_t FluidGridDisplay::uploadPixels(int new_width, int new_height, std::span<const float> rgba) const
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

std::uint32_t FluidGridDisplay::texture(const FluidCoarseningPreview& preview) const
{
    return previewTexture(preview.width,preview.height,preview.labels,preview.revision,&preview);
}

std::uint32_t FluidGridDisplay::previewTexture(int w,int h,std::span<const std::uint8_t> labels,std::uint64_t revision,const void* source) const
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
    if (preview_width != w || preview_height != h)
    {
        preview_width = w; preview_height = h;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, preview_width, preview_height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
        preview_source = nullptr;
    }
    if (preview_source != source || preview_revision != revision)
    {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, preview_width, preview_height, GL_RED, GL_UNSIGNED_BYTE, labels.data());
        preview_source = source; preview_revision = revision;
    }
    return preview_texture;
}

std::uint32_t FluidGridDisplay::resolutionTexture(int width,int height) const {
    const int w=(width+1)/2,h=(height+1)/2;
    return uploadResolution(w,h,std::vector<std::uint8_t>(size_t(w)*h,1));
}
#ifdef FLUIDSIM_HAS_AMR
std::uint32_t FluidGridDisplay::resolutionTexture(const FluidGridAdaptive& grid) const {
    return resolutionTexture(grid.finestWidth(),grid.finestHeight(),grid.layout());
}
std::uint32_t FluidGridDisplay::resolutionTexture(int width,int height,const GasLayout& layout) const {
    const int w=(width+1)/2,h=(height+1)/2;
    std::vector<std::uint8_t> scales(size_t(w)*h,std::uint8_t(layout.baseRatio));
    int scale=layout.baseRatio;
    for (const auto& patches:layout.refinedPatches) {
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
    for (const auto texture:{fraction_texture,liquid_texture,palette_texture}) if (texture) glDeleteTextures(1,&texture);
    fraction_texture=liquid_texture=palette_texture=0;
    material_width=material_height=material_groups=palette_width=0;
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
