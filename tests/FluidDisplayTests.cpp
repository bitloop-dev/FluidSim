#include "FluidAmbientMaskDisplayTests.h"
#include "FluidGridDisplay.h"
#include "FluidGridGPU.h"
#include "FluidGridWebGL.h"
#include "FluidGridAdaptiveGPU.h"
#include "FluidShader.h"
#include "FluidLiquidDepth.h"
#include "FluidGridGPUInternal.h"
#include <bitloop/core/debug.h>
#include <bitloop/nanovgx/nano_shader_surface.h>
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace fluid;
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
void requireNear(float a,float b,float tolerance,const char* message) { require(std::abs(a-b)<=tolerance,message); }

std::vector<float> readTexture(GLuint texture,int width,int height,GLenum format=GL_RGBA,int channels=4) {
    glBindTexture(GL_TEXTURE_2D,texture);
    std::vector<float> values(size_t(width)*height*channels);
    glGetTexImage(GL_TEXTURE_2D,0,format,GL_FLOAT,values.data());
    return values;
}

struct Fixture {
    ElementCatalogue elements;
    FluidMixture liquid,gas,mixture;
    int liquids;
    Fixture(int count,int liquidCount) : liquids(liquidCount) {
        elements.add({"Unused",{1,1,1}});
        for (int k=0;k<count;++k)
            elements.add({"Ingredient "+std::to_string(k),{.1f+.03f*k,.7f-.02f*k,.2f+.01f*k}});
        for (int k=0;k<count;++k) {
            // Deliberately reverse IDs and reuse one element across phases.
            const auto id=ElementId(k==count-1 && count>4 ? count : count-k);
            const MixtureComponent c{id,.5+.3*k};
            (k<liquids?liquid:gas).components.push_back(c);
            mixture.components.push_back(c);
        }
    }
    FluidGrid::State state(int width,int height,double cellSize,bool independentGeometry) const {
        const int cells=width*height,count=int(mixture.components.size());
        FluidGrid::State s;
        s.liquid_fraction.resize(cells);
        s.liquid_mass.assign(liquids,std::vector<double>(cells));
        s.gas_mass.assign(count-liquids,std::vector<double>(cells));
        for (int i=0;i<cells;++i) {
            const double f=liquids ? .2+.6*double(i%width)/width : 0.;
            s.liquid_fraction[i]=f+(independentGeometry && liquids && i==1 ? .015 : 0.);
            std::vector<double> weights(count-liquids);
            double gasWeight=0.;
            for (int k=0;k<count-liquids;++k) {
                weights[k]=1.+.2*std::sin((k+1)*(double(i%width)/width+2.*double(i/width)/height));
                gasWeight+=weights[k];
            }
            for (int k=0;k<count;++k) {
                const double fraction=k<liquids ? f/liquids : (1.-f)*weights[k-liquids]/gasWeight;
                auto& mass=k<liquids ? s.liquid_mass[k] : s.gas_mass[k-liquids];
                mass[i]=fraction*mixture.components[k].density*cellSize*cellSize;
            }
        }
        return s;
    }
    template<class Grid> void configure(Grid& grid) const {
        grid.elements=elements; grid.liquid_mixture=liquid; grid.gas_mixture=gas;
    }
};

void shaderChecks(FluidGridDisplay& display,FluidGridDisplay::MaterialTextures textures,
                  const Fixture& fixture,const FluidMaterialPixels& expected) {
    gpu_detail::ComputeState guard;
    const int w=expected.width,h=expected.height;
    const auto palette=display.paletteTexture(fixture.elements,fixture.mixture);
    const auto records=readTexture(palette,std::max(1,expected.ingredients),1);
    for (int k=0;k<expected.ingredients;++k)
        require(records[k*4+3]==float(fixture.mixture.components[k].element),"palette lost mixture-to-element mapping");
    const auto resolution=display.resolutionTexture(w,h);
    auto bind=[&](const bl::ShaderSurface& shader) {
        shader.bindTexture2D("u_fractions",textures.fractions);
        shader.bindTexture2D("u_liquid",textures.liquid);
        shader.bindTexture2D("u_palette",palette);
        shader.bindTexture2D("u_resolution",resolution);
        shader.bindTexture2D("u_coarsening",resolution);
        shader.setUniform2f("u_grid_size",float(w),float(h));
        shader.setUniform2f("u_ingredient_counts",float(expected.ingredients),float(fixture.liquids));
        shader.setUniform1f("u_coarsening_opacity",0);
        shader.setUniform1f("u_mask_opacity",0);
    };
    bl::ShaderSurface shader;
    shader.setFragmentSource(fluid_shader_src);
    shader.render(w*3,h*3,bind);
    require(!shader.errors() && shader.texture(),"material GLSL failed to compile/render");
    auto rgb=readTexture(shader.texture(),w*3,h*3);
    for (int y=0;y<h*3;++y) for (int x=0;x<w*3;++x) {
        const size_t i=size_t(y/3)*w+x/3,dst=size_t(y*w*3+x)*4;
        for (int c=0;c<3;++c) {
            float color=0;
            for (int k=0;k<expected.ingredients;++k)
                color+=expected.values[expected.fractionIndex(i,k)]*fixture.elements[fixture.mixture.components[k].element].color[c];
            requireNear(rgb[dst+c],color,1.01f/255,"GLSL material colour differs from fraction-weighted reference");
        }
    }
    if (!expected.ingredients) return;
    // Exercise the actual shared helpers, including a neighbour at a boundary,
    // an ingredient beyond the first RGBA layer, and independent geometry.
    std::string probe=fluid_shader_src;
    probe.resize(probe.find("    void main()"));
    probe+=R"(
    void main() {
        ivec2 cell=ivec2(v_uv*u_grid_size);int k=int(u_ingredient_counts.x)-1;
        outColor=vec4(ingredientFraction(cell-ivec2(1,0),k),liquidFraction(cell),
                      float(ingredientElement(k))/32.0,liquidIngredient(k)?1.0:0.0);
    })";
    shader.setFragmentSource(probe);
    shader.render(w,h,bind);
    require(!shader.errors(),"material GLSL helper probe failed");
    const auto samples=readTexture(shader.texture(),w,h);
    const int k=expected.ingredients-1;
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        const size_t i=size_t(y)*w+x,neighbour=size_t(y)*w+std::max(0,x-1);
        requireNear(samples[i*4],expected.values[expected.fractionIndex(neighbour,k)],1.01f/255,"GLSL neighbour/layer addressing changed fraction");
        requireNear(samples[i*4+1],expected.liquid()[i],1.01f/255,"GLSL did not receive independent liquid geometry");
        requireNear(samples[i*4+2],float(fixture.mixture.components[k].element)/32,1.01f/255,"GLSL element ID differs from palette");
        require(samples[i*4+3]==float(k<fixture.liquids),"GLSL phase membership changed");
    }
}

void liquidDepthRegressions() {
    gpu_detail::ComputeState guard;
    constexpr int w=63,h=79; // Odd, rectangular image also checks jump bounds/coordinate packing.
    ElementCatalogue elements;
    const auto water=elements.add({"Water",{.1f,.35f,1.f}});
    const auto air=elements.add({"Air",{.015f,.025f,.06f}});
    FluidMixture mixture{{{water,1.},{air,1.}}};
    FluidMaterialPixels material(w,h,2);
    material.values.resize(material.valueCount());
    auto fill=[&](bool full) {
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
            const size_t i=size_t(y)*w+x;
            const float f=full || (y>=16 && (x-32)*(x-32)+(y-45)*(y-45)>16) ? 1.f : 0.f;
            material.values[material.fractionIndex(i,0)]=f;
            material.values[material.fractionIndex(i,1)]=1.f-f;
            material.values[material.fractions().size()+i]=f;
        }
    };
    fill(false);
    const auto original=material.values;
    FluidGridDisplay display;
    FluidLiquidDepth depth;
    bl::ShaderSurface shader;
    shader.setFragmentSource(fluid_shader_src);
    auto render=[&] {
        const auto textures=display.materialTextures(material);
        const auto seeds=depth.texture(w,h,textures.liquid);
        const auto palette=display.paletteTexture(elements,mixture);
        const auto resolution=display.resolutionTexture(w,h);
        shader.render(w,h,[&](const bl::ShaderSurface& program) {
            program.bindTexture2D("u_fractions",textures.fractions);
            program.bindTexture2D("u_liquid",textures.liquid);
            program.bindTexture2D("u_gas_seeds",seeds);
            program.bindTexture2D("u_palette",palette);
            program.bindTexture2D("u_resolution",resolution);
            program.bindTexture2D("u_coarsening",resolution);
            program.setUniform2f("u_grid_size",w,h);
            program.setUniform2f("u_ingredient_counts",2,1);
            program.setUniform1f("u_depth_shading",1);
            program.setUniform1f("u_depth_scale",14);
            program.setUniform1f("u_deep_brightness",.16f);
            program.setUniform1f("u_coarsening_opacity",0);
            program.setUniform1f("u_mask_opacity",0);
        });
        require(!shader.errors(),"liquid depth shader failed to compile");
        return readTexture(shader.texture(),w,h);
    };
    const auto image=render();
    auto channel=[&](int x,int y,int c) { return image[size_t(y*w+x)*4+c]; };
    for(int c=0;c<3;++c) {
        require(channel(3,18,c)>channel(3,32,c) && channel(3,32,c)>channel(3,58,c),"liquid does not darken away from gas");
        requireNear(channel(32,45,c),elements[air].color[c],1.01f/255,"depth shading changed a gas bubble");
    }
    require(channel(32,39,1)>channel(8,39,1)+.05f,"bubble gas does not illuminate nearby liquid");
    for(size_t i=0;i<material.cellCount();++i)
        require(image[i*4+3]==1.f,"liquid transparency illusion exposed the scene background");
    require(material.values==original,"depth renderer modified material input");
    fill(true);
    const auto full=render();
    for(size_t i=1;i<material.cellCount();++i) for(int c=0;c<4;++c)
        require(full[i*4+c]==full[c],"full tank lit up at a wall or retained stale gas seeds");
    require(full[2]<.2f,"full tank without gas did not use deep tint");
    depth.destroy(); display.destroy();
    std::cout<<"PASS opaque liquid depth falloff, bubble illumination, gas colour and full-tank reset\n";
}

void materialRegressions() {
    FluidGridDisplay display;
    for (int count:{18,7,4,0}) {
        Fixture f(count,std::min(2,count));
        FluidGrid cpu(7,3,.25); f.configure(cpu);
        const auto initial=f.state(7,3,.25,true); cpu.upload(initial);
        const auto expected=FluidGridDisplay::materialPixels(cpu);
        for (size_t i=0;i<expected.cellCount();++i) {
            require(expected.liquid()[i]==float(initial.liquid_fraction[i]),"CPU replaced tracked geometry by summed fractions");
            for (int k=0;k<count;++k) {
                const double mass=k<f.liquids ? initial.liquid_mass[k][i] : initial.gas_mass[k-f.liquids][i];
                requireNear(expected.values[expected.fractionIndex(i,k)],float(mass/(f.mixture.components[k].density*.25*.25)),0,"CPU fraction uses wrong density/area");
            }
        }
        auto textures=display.materialTextures(expected);
        require(readTexture(textures.fractions,7,3*expected.groups())==std::vector<float>(expected.fractions().begin(),expected.fractions().end()),"fraction texture changed values/row order");
        require(readTexture(textures.liquid,7,3,GL_RED,1)==std::vector<float>(expected.liquid().begin(),expected.liquid().end()),"liquid texture changed values/row order");
        shaderChecks(display,textures,f,expected);
        FluidGridGPU gpu(7,3,.25); f.configure(gpu); gpu.upload(convertFluidState<float>(initial));
        const auto before=gpu.download();
        const auto actual=gpu.materialPixels();
        require(actual.values.size()==expected.values.size(),"WebGPU display field size differs");
        for (size_t i=0;i<actual.values.size();++i) requireNear(actual.values[i],expected.values[i],2e-7f,"WebGPU fraction differs from CPU");
        const auto after=gpu.download();
        require(before.liquid_mass==after.liquid_mass && before.gas_mass==after.gas_mass && before.liquid_fraction==after.liquid_fraction &&
            before.u==after.u && before.v==after.v && before.pressure==after.pressure,"WebGPU display changed physical state");
        textures=display.materialTextures(actual); shaderChecks(display,textures,f,expected);
        FluidGridWebGL webgl(7,3,.25); f.configure(webgl); webgl.upload(convertFluidState<float>(initial));
        const auto glBefore=webgl.download();
        textures={webgl.fractionTexture(),webgl.liquidFractionTexture()};
        const auto fractions=readTexture(textures.fractions,7,3*expected.groups());
        for (size_t i=0;i<fractions.size();++i) requireNear(fractions[i],expected.values[i],2e-7f,"WebGL packed fraction differs from CPU");
        shaderChecks(display,textures,f,expected);
        const auto glAfter=webgl.download();
        require(glBefore.liquid_mass==glAfter.liquid_mass && glBefore.gas_mass==glAfter.gas_mass && glBefore.liquid_fraction==glAfter.liquid_fraction &&
            glBefore.u==glAfter.u && glBefore.v==glAfter.v && glBefore.pressure==glAfter.pressure,"WebGL display changed physical state");
    }
    for (bool wet:{false,true}) {
        Fixture f(7,wet?2:0);
        const GasLayout layout{4,{{{8,0,16,32}},{{12,4,8,24}}}};
        FluidGridAdaptive cpu(32,32,.25,f.elements,f.liquid,f.gas,layout,8);
        FluidGridAdaptiveGPU gpu(32,32,.25,f.elements,f.liquid,f.gas,layout,8);
        const auto initial=f.state(32,32,.25,false);
        cpu.upload(initial); gpu.upload(initial);
        const auto expected=FluidGridDisplay::materialPixels(cpu);
        const auto before=gpu.totals();
        const auto actual=gpu.materialPixels();
        const auto after=gpu.totals();
        require(before.ingredientMass==after.ingredientMass && before.momentum==after.momentum,"adaptive display changed physical totals");
        for (size_t i=0;i<actual.values.size();++i) requireNear(actual.values[i],expected.values[i],3e-6f,"adaptive bounded display fractions differ");
        shaderChecks(display,display.materialTextures(actual),f,expected);
        const auto oldColor=gpu.colorPixels();
        const auto newColor=readTexture(display.texture(actual,f.elements,f.mixture),32,32);
        for (size_t i=0;i<newColor.size();++i) requireNear(newColor[i],oldColor[i],3e-6f,"adaptive fraction display changed existing material colours");
    }
    display.destroy();
    std::cout<<"PASS material fields, GLSL helpers, CPU/WebGPU/WebGL parity, adaptive bounded reconstruction and unchanged state\n";
}
} // namespace

int main()
{
    SDL_Window* window = nullptr;
    SDL_GLContext context = nullptr;
    int result = 0;
    try {
        if (!SDL_Init(SDL_INIT_VIDEO)) throw std::runtime_error(SDL_GetError());
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
        window = SDL_CreateWindow("Fluid display tests",32,32,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN);
        if (!window) throw std::runtime_error(SDL_GetError());
        context = SDL_GL_CreateContext(window);
        if (!context || !gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) throw std::runtime_error("OpenGL test context unavailable");
        fluid::test::ambientMaskDisplayRegressions();
        materialRegressions();
        liquidDepthRegressions();
        fluid::FluidGridDisplay display;
        std::array<float,24> rgba{};
        for (size_t i=0;i<rgba.size();++i) rgba[i]=float(i)/32.f;
        const auto texture=display.texture(3,2,rgba);
        glBindTexture(GL_TEXTURE_2D,texture);
        std::array<float,24> output{};
        glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,output.data());
        if (output!=rgba) throw std::runtime_error("GPU colour bridge changed image values or row order");
        display.destroy();
        if (glGetError()!=GL_NO_ERROR) throw std::runtime_error("OpenGL display error");
        std::cout<<"PASS cached GPU colour bridge and ambient display\n";
    } catch (const std::exception& error) {
        std::cerr<<"FAIL "<<error.what()<<'\n'; result=1;
    }
    if (context) SDL_GL_DestroyContext(context);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
    return result;
}
