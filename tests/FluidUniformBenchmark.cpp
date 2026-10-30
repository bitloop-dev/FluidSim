#include "FluidGridGPU.h"
#include "FluidGridWebGL.h"
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
using Clock = std::chrono::steady_clock;

template<class Grid> void finish(Grid& grid)
{
    if constexpr (std::is_same_v<Grid,fluid::FluidGridGPU>) grid.synchronize();
    else glFinish();
}

template<class Grid> void benchmark(int n,int frames,bool profile,int cutoff,const std::string& image)
{
    const auto setupStarted=Clock::now();
    Grid grid(n,n,1.);
    const auto constructed=Clock::now();
    auto water=grid.elements.add({"Water",{.1f,.35f,1.f}});
    auto red=grid.elements.add({"Red",{.95f,.25f,.15f}});
    auto air=grid.elements.add({"Air",{.015f,.025f,.06f}});
    auto cloud=grid.elements.add({"Cloud",{0,1,1}});
    grid.liquid_mixture.components={{water,100.},{red,300.}};
    grid.gas_mixture.components={{air,.1},{cloud,.1}};
    grid.gas_damping_rate=2.;
    if constexpr (std::is_same_v<Grid,fluid::FluidGridWebGL>) grid.cpu_pressure_cells=cutoff;
    typename Grid::State state;
    state.liquid_fraction.resize(size_t(n)*n);
    state.liquid_mass.assign(2,std::vector<float>(size_t(n)*n));
    state.gas_mass.assign(2,std::vector<float>(size_t(n)*n));
    for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
        const int i=y*n+x;
        const float f=float(std::clamp(y+1.-(.7*n+.5*x),0.,1.));
        state.liquid_fraction[i]=f; state.liquid_mass[0][i]=100.f*f; state.gas_mass[0][i]=.1f*(1.f-f);
    }
    const double initialMass=std::accumulate(state.liquid_mass[0].begin(),state.liquid_mass[0].end(),0.);
    grid.upload(state); finish(grid);
    const auto uploaded=Clock::now();
    GLuint texture=0;
    glGenTextures(1,&texture); glBindTexture(GL_TEXTURE_2D,texture);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,n,n,0,GL_RGBA,GL_FLOAT,nullptr);
    double transport=0,forces=0,pressure=0,display=0,whole=0,maximumDiv=0,firstFrame=0;
    long long iterations=0;
    constexpr int warmup=4;
    auto time=[&](auto operation) {
        const auto started=Clock::now(); operation(); finish(grid);
        return std::chrono::duration<double,std::milli>(Clock::now()-started).count();
    };
    std::vector<float> pixels;
    for(int frame=0;frame<frames;++frame) {
        double a=0,b=0,c=0;
        const auto started=Clock::now();
        for(int step=0;step<5;++step) {
            fluid::FluidProjectionResult result;
            if(profile) {
                a+=time([&] { grid.advect(.016); });
                b+=time([&] { grid.dampGasVelocity(.016); grid.applyGravity(.016); });
                c+=time([&] { result=grid.project(.016,2000,1e-3); });
            } else result=grid.step(.016,2000,1e-3);
            if(!profile) finish(grid);
            if(!result.converged) throw std::runtime_error("Pressure projection failed");
            maximumDiv=std::max(maximumDiv,result.max_divergence);
            if(frame>=warmup) iterations+=result.iterations;
        }
        const auto shown=Clock::now();
        if constexpr (std::is_same_v<Grid,fluid::FluidGridGPU>) {
            pixels=grid.colorPixels();
            glBindTexture(GL_TEXTURE_2D,texture);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,n,n,GL_RGBA,GL_FLOAT,pixels.data());
        } else grid.colorTexture();
        glFinish();
        const double d=std::chrono::duration<double,std::milli>(Clock::now()-shown).count();
        if(frame==0) firstFrame=std::chrono::duration<double,std::milli>(Clock::now()-setupStarted).count();
        if(frame>=warmup) {
            transport+=a; forces+=b; pressure+=c; display+=d;
            whole+=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
        }
    }
    state=grid.download();
    uint64_t stateHash=14695981039346656037ull;
    auto hash=[&](const auto& values) { for(float value:values) { stateHash^=std::bit_cast<uint32_t>(value); stateHash*=1099511628211ull; } };
    hash(state.liquid_fraction); hash(state.u); hash(state.v); hash(state.pressure);
    for(const auto& values:state.liquid_mass) hash(values);
    for(const auto& values:state.gas_mass) hash(values);
    const double mass=std::accumulate(state.liquid_mass[0].begin(),state.liquid_mass[0].end(),0.);
    const double massError=std::abs(mass-initialMass)/initialMass;
    double volumeError=0.;
    for(size_t i=0;i<state.liquid_fraction.size();++i) {
        double volume=0.;
        for(size_t k=0;k<state.liquid_mass.size();++k) volume+=state.liquid_mass[k][i]/grid.liquid_mixture.components[k].density;
        for(size_t k=0;k<state.gas_mass.size();++k) volume+=state.gas_mass[k][i]/grid.gas_mixture.components[k].density;
        if(!std::isfinite(volume)) throw std::runtime_error("Nonfinite material");
        volumeError=std::max(volumeError,std::abs(volume-1.));
    }
    if(massError>1e-4) throw std::runtime_error("Ingredient mass drift exceeded uniform regression bound");
    if(!image.empty()) {
        if constexpr (std::is_same_v<Grid,fluid::FluidGridGPU>) pixels=grid.colorPixels();
        else {
            pixels.resize(size_t(n)*n*4); glBindTexture(GL_TEXTURE_2D,grid.colorTexture());
            glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,pixels.data());
        }
        std::ofstream output(image,std::ios::binary); output<<"P6\n"<<n<<' '<<n<<"\n255\n";
        for(size_t i=0;i<pixels.size();i+=4) for(int c=0;c<3;++c) output.put(char(std::lround(255.f*std::clamp(pixels[i+c],0.f,1.f))));
    }
    glDeleteTextures(1,&texture);
    const double count=frames-warmup;
    std::cout<<std::setprecision(9)<<"backend="<<(std::is_same_v<Grid,fluid::FluidGridGPU>?"webgpu":"webgl")
        <<",size="<<n<<",frames="<<frames<<",warmup="<<warmup<<",profile="<<profile
        <<",construction_ms="<<std::chrono::duration<double,std::milli>(constructed-setupStarted).count()
        <<",upload_ms="<<std::chrono::duration<double,std::milli>(uploaded-constructed).count()<<",first_frame_ms="<<firstFrame
        <<",cutoff="<<(std::is_same_v<Grid,fluid::FluidGridWebGL>?cutoff:0)
        <<",transport_ms="<<transport/count<<",forces_ms="<<forces/count<<",pressure_ms="<<pressure/count
        <<",display_ms="<<display/count<<",whole_ms="<<whole/count<<",iterations="<<iterations/count
        <<",max_div="<<maximumDiv<<",mass_error="<<massError<<",volume_error="<<volumeError<<",state_hash="<<stateHash<<'\n';
}
}

int main(int argc,char** argv)
{
    SDL_Window* window=nullptr; SDL_GLContext context=nullptr;
    int result=0;
    try {
        if(!SDL_Init(SDL_INIT_VIDEO)) throw std::runtime_error(SDL_GetError());
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3); SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,2);
        window=SDL_CreateWindow("Uniform backend benchmark",32,32,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN);
        context=window?SDL_GL_CreateContext(window):nullptr;
        if(!context || !gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) throw std::runtime_error("OpenGL unavailable");
        const std::string backend=argc>1?argv[1]:"webgpu";
        if(backend!="webgpu" && backend!="webgl") throw std::invalid_argument("backend must be webgpu or webgl");
        if(argc>2 && std::string(argv[2])!="profile" && std::string(argv[2])!="full") throw std::invalid_argument("mode must be profile or full");
        const bool profile=argc>2 && std::string(argv[2])=="profile";
        const int frames=argc>3?std::stoi(argv[3]):16,n=argc>4?std::stoi(argv[4]):512,cutoff=argc>5?std::stoi(argv[5]):1024;
        if(frames<=4) throw std::invalid_argument("frames must exceed the four warmup frames");
        const std::string image=argc>6?argv[6]:"";
        if(backend=="webgpu") benchmark<fluid::FluidGridGPU>(n,frames,profile,cutoff,image);
        else benchmark<fluid::FluidGridWebGL>(n,frames,profile,cutoff,image);
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; result=1; }
    if(context) SDL_GL_DestroyContext(context);
    if(window) SDL_DestroyWindow(window);
    SDL_Quit();
    return result;
}
