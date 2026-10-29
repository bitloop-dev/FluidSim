#include "FluidGrid.h"
#include "FluidGridWebGPU.h"
#include "FluidGridDisplay.h"
#include "UniformWebGPU/FluidGridGPU.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#ifndef __EMSCRIPTEN__
#include <SDL3/SDL.h>
#include <glad/glad.h>
#endif

using namespace fluid;
namespace
{
static_assert(std::is_base_of_v<FluidGridBase, FluidGridWebGPU> && !std::is_abstract_v<FluidGridWebGPU>);
static_assert(std::is_same_v<FluidGridWebGPU::State,FluidGridBase::State>);
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class A,class B> bool same(const A& a,const B& b)
{
    return a.liquid_fraction==b.liquid_fraction && a.u==b.u && a.v==b.v && a.pressure==b.pressure &&
           a.liquid_mass==b.liquid_mass && a.gas_mass==b.gas_mass;
}
struct Fixture
{
    ElementCatalogue elements;
    FluidMixture liquid,gas;
    Fixture()
    {
        liquid.components={{elements.add({"Water",{.1f,.35f,1.f}}),100.},
                           {elements.add({"Red",{.95f,.25f,.15f}}),300.}};
        gas.components={{elements.add({"Air",{.015f,.025f,.06f}}),.1},
                        {elements.add({"Cloud",{0.f,1.f,1.f}}),.1}};
    }
    FluidGridBase::State configure(FluidGridBase& grid) const
    {
        grid.configureMaterials(elements,liquid,gas);grid.setOrigin({-2.,7.});grid.setGravityY(100.);
        const int w=grid.getWidth(),h=grid.getHeight();const double area=grid.getCellSize()*grid.getCellSize();
        FluidGridBase::State state;
        state.liquid_fraction.resize(w*h);state.liquid_mass.assign(2,std::vector<double>(w*h));state.gas_mass.assign(2,std::vector<double>(w*h));
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)
        {
            const int i=y*w+x; const double fraction=std::clamp(y+1.-(x+h*(.5+.04*std::cos(3.141592653589793*(3*x+.5)/w))),0.,1.);
            const double red=x<w/4?1.:0.;const double cloud=.2+.1*std::sin(x*.13)*std::cos(y*.11);
            state.liquid_fraction[i]=fraction;
            state.liquid_mass[0][i]=100.*fraction*(1.-red)*area;state.liquid_mass[1][i]=300.*fraction*red*area;
            state.gas_mass[0][i]=.1*(1.-fraction)*(1.-cloud)*area;state.gas_mass[1][i]=.1*(1.-fraction)*cloud*area;
        }
        grid.upload(state);return state;
    }
    void configureNative(FluidGridGPU& grid,const FluidGridBase::State& initial) const
    {
        grid.elements=elements;grid.liquid_mixture=liquid;grid.gas_mixture=gas;grid.gravity_y=100.;
        grid.upload(convertFluidState<float>(initial));
    }
};
void selection()
{
    Fixture f;FluidGrid grid(32,32,2.);
    grid.configureMaterials(f.elements,f.liquid,f.gas);grid.setOrigin({3.,4.});grid.setGravityY(12.);
    grid.setDevice(FluidDevice::GPU);
    require(grid.getDevice()==FluidDevice::GPU && grid.getAdaptiveMode()==AdaptiveMode::DISABLED && !grid.requiresRenderThread(),"GPU selection/thread requirement incorrect");
    require(grid.getOrigin().x==3. && grid.getGravityY()==12. && grid.getLiquidMixture().components.size()==2,"Device selection discarded common configuration");
    grid.setAdaptiveMode(AdaptiveMode::MAX_4);
    require(grid.getDevice()==FluidDevice::GPU && grid.getAdaptiveMode()==AdaptiveMode::MAX_4,"GPU adaptive selection failed");
    grid.setDevice(FluidDevice::CPU);grid.setAdaptiveMode(AdaptiveMode::MAX_4);
    grid.setDevice(FluidDevice::GPU);
    require(grid.getDevice()==FluidDevice::GPU && grid.getAdaptiveMode()==AdaptiveMode::MAX_4,"Reverse-order GPU adaptive selection failed");
    grid.setAdaptiveMode(AdaptiveMode::DISABLED);grid.setDevice(FluidDevice::GPU);
    f.configure(grid);auto before=grid.download();grid.setDevice(FluidDevice::GPU);grid.setAdaptiveMode(AdaptiveMode::DISABLED);
    bool rejected=false;try{grid.setDevice(FluidDevice::CPU);}catch(const std::logic_error&){rejected=true;}
    require(rejected && same(before,grid.download()),"Live device selection reset state");
    FluidGrid moved=std::move(grid);moved.destroy();moved.setDevice(FluidDevice::CPU);f.configure(moved);
    require(moved.step(.001,2000,1e-8).converged,"Explicit restart from GPU to CPU failed");
    std::cout<<"PASS GPU configuration, both selection orders, no-op/live selection, move and explicit restart\n";
}
void commonParity()
{
    Fixture f;FluidGridWebGPU backend(33,25,2.5);FluidGridBase& api=backend;
    auto initial=f.configure(api);FluidGridGPU native(33,25,2.5);f.configureNative(native,initial);
    auto expected=[&]{return convertFluidState<double>(native.download());};
    require(same(api.download(),expected()),"Common upload did not preserve native float initialization");
    auto before=api.download();auto zero=api.step(0.);
    require(zero.converged && zero.iterations==0 && same(before,api.download()),"Zero-time common call changed state");
    for(int step=0;step<24;++step)
    {
        const auto a=api.step(.004,2000,1e-3),b=native.step(.004,2000,1e-3);
        require(a.converged && b.converged && a.iterations==b.iterations && a.max_divergence==b.max_divergence,"Common WebGPU stepping changed native projection");
        require(same(api.download(),expected()),"Common WebGPU step changed native float fields");
    }
    auto edit=api.beginEdit();auto nativeEdit=native.beginEdit();
    edit.reserve(4);
    edit.setCell(3,2,Phase::Liquid,0);edit.setCell(4,2,Phase::Liquid,1);edit.setCell(4,2,Phase::Gas,3);
    nativeEdit.setCell(3,2,Phase::Liquid,0);nativeEdit.setCell(4,2,Phase::Liquid,1);nativeEdit.setCell(4,2,Phase::Gas,3);
    before=api.download();require(same(before,expected()),"Staged edit mutated state");edit.commit();nativeEdit.commit();
    require(edit.empty() && same(api.download(),expected()),"Common edit changed native sparse replacement/history");
    std::vector<float> pixels(33*25*4);const auto storage=pixels.data();
    api.writeColorPixels(pixels);require(pixels==native.colorPixels(),"Common display changed native palette output");
    api.writeColorPixels(pixels);require(pixels.data()==storage,"Colour observation discarded reusable caller storage");
    require(same(api.download(),expected()),"Colour observation changed physical fields");
    auto edited=api.download();edit.setCell(1,1,Phase::Gas,2);edit.setCell(-1,1,Phase::Gas,2);
    bool rejected=false;try{edit.commit();}catch(const std::out_of_range&){rejected=true;}
    require(rejected && edit.size()==2 && same(edited,api.download()),"Rejected common GPU batch lost atomicity");
    const auto failure=api.step(.01,0,1e-12);
    const auto rawFailure=native.step(.01,0,1e-12);
    require(!failure.converged && failure.iterations==rawFailure.iterations && same(api.download(),expected()),"Common API hid native iteration-cap failure");
    api.destroy();api.upload(initial);native.destroy();f.configureNative(native,initial);
    require(same(api.download(),expected()),"GPU destroy/upload reset changed native initialization");
    api.configureMaterials({}, {}, {});api.upload({});api.writeColorPixels(pixels);
    for(size_t i=0;i<pixels.size();++i)require(pixels[i]==(i%4==3?1.f:0.f),"Empty palette left stale common colours");
    std::cout<<"PASS polymorphic float-state parity, zero/reset/failure, sparse edits/atomicity and reusable colours\n";
}
void evolvingParity()
{
    Fixture f;FluidGrid grid(512,512,1.);grid.setDevice(FluidDevice::GPU);FluidGridBase& api=grid;
    auto initial=f.configure(api);FluidGridGPU native(512,512,1.);f.configureNative(native,initial);
    for(int frame=0;frame<200;++frame)
    {
        for(int substep=0;substep<2;++substep)
        {
            const auto a=api.step(.01,2000,1e-3),b=native.step(.01,2000,1e-3);
            require(a.converged && b.converged && a.iterations==b.iterations && a.max_divergence==b.max_divergence,"Evolving common/native projection differs");
        }
        if(frame==0 || (frame+1)%40==0)
        {
            const auto state=api.download(),expected=convertFluidState<double>(native.download());
            require(same(state,expected),"Evolving common wrapper changed native fields");
            double drift=0.,volume_error=0.;
            auto budget=[&](const auto& fields,const auto& original)
            {
                for(size_t k=0;k<fields.size();++k)
                {
                    const double total=std::accumulate(fields[k].begin(),fields[k].end(),0.);
                    const double start=std::accumulate(original[k].begin(),original[k].end(),0.);
                    drift=std::max(drift,std::abs(total-start)/std::max(1.,start));
                    for(double mass:fields[k])require(std::isfinite(mass),"Non-finite evolving ingredient mass");
                }
            };
            budget(state.liquid_mass,initial.liquid_mass);budget(state.gas_mass,initial.gas_mass);
            for(size_t i=0;i<state.liquid_fraction.size();++i)
            {
                const double liquid=state.liquid_mass[0][i]/100.+state.liquid_mass[1][i]/300.;
                const double gas=(state.gas_mass[0][i]+state.gas_mass[1][i])/.1;
                volume_error=std::max({volume_error,std::abs(liquid-state.liquid_fraction[i]),std::abs(gas-(1.-state.liquid_fraction[i]))});
            }
            require(drift<1e-4 && volume_error<.002,"Evolving float mass/phase-volume budget exceeded");
            std::cout<<"frame="<<frame+1<<", exact_native_parity=1, mass_drift="<<drift<<", phase_volume_error="<<volume_error<<'\n';
        }
    }
    std::cout<<"PASS 512x512 first 200 frames, 2 x .01-second steps per frame, exact native/common state/result parity\n";
}
void checkpointParity()
{
    for (bool extra_component : {false, true})
    {
        Fixture fixture;
        FluidGrid original(33, 25, 2.5);
        original.setDevice(FluidDevice::GPU);
        auto initial = fixture.configure(original);
        if (extra_component)
        {
            fixture.gas.components.push_back({fixture.elements.add({"Extra gas", {.7f, .2f, .4f}}), .1});
            initial.gas_mass.push_back(initial.gas_mass[1]);
            for (size_t i = 0; i < initial.gas_mass[1].size(); ++i)
            {
                initial.gas_mass[1][i] *= .6;
                initial.gas_mass[2][i] *= .4;
            }
            original.configureMaterials(fixture.elements, fixture.liquid, fixture.gas);
            original.upload(initial);
        }
        original.setGravityY(73.);
        original.setOrigin({-7., 4.});
        for (int tick = 0; tick < 50; ++tick)
            require(original.step(.004, 2000, .01).converged, "Checkpoint warmup failed");
        if (extra_component)
        {
            auto edit = original.beginEdit();
            edit.setCell(3, 2, Phase::Liquid, fixture.liquid.components[0].element);
            edit.setCell(4, 2, Phase::Gas, fixture.gas.components[2].element);
            edit.commit(); // Capture with stale derived density/colour caches as well as warm pressure history.
        }

        const auto before = original.download();
        FluidCheckpoint saved;
        require(original.saveCheckpoint(saved), "Uniform WebGPU checkpoint unsupported");
        require(same(before, original.download()), "Saving a checkpoint changed physical state");
        const size_t groups = extra_component ? 2 : 1;
        require(saved.data.size() == ((3 + 4 * groups) * 33 * 25 + 34 * 25 + 33 * 26) * sizeof(float), "Checkpoint contains the wrong native fields");
        saved.metadata = bl::JSON::json::parse(saved.metadata.dump()); // Exercise the on-disk numeric round trip.

        FluidGrid resumed(33, 25, 2.5);
        resumed.setDevice(FluidDevice::GPU);
        require(resumed.loadCheckpoint(saved), "Loading uniform WebGPU checkpoint failed");
        require(same(before, resumed.download()), "Checkpoint restore changed physical fields");
        FluidCheckpoint observed;
        resumed.saveCheckpoint(observed);
        require(saved.metadata == observed.metadata && saved.data == observed.data, "Checkpoint restore lost native configuration/history");
        bool rejected = false;
        try { resumed.setDevice(FluidDevice::CPU); }
        catch (const std::logic_error&) { rejected = true; }
        require(rejected, "Checkpoint load did not mark the wrapper initialized");

        for (int tick = 0; tick < 200; ++tick)
        {
            const auto a = original.step(.004, 2000, .01);
            const auto b = resumed.step(.004, 2000, .01);
            require(a.converged && b.converged && a.iterations == b.iterations && a.max_divergence == b.max_divergence,
                "Resumed pressure solve differs from uninterrupted solve");
            require(same(original.download(), resumed.download()), "Resumed trajectory differs from uninterrupted state");
        }
        original.saveCheckpoint(saved);
        resumed.saveCheckpoint(observed);
        require(saved.metadata == observed.metadata && saved.data == observed.data, "Continued checkpoint history differs");
        std::vector<float> a, b;
        original.writeColorPixels(a);
        resumed.writeColorPixels(b);
        require(a == b, "Resumed material colours differ");

        for (int corruption = 0; corruption < 8; ++corruption)
        {
            auto invalid = saved;
            switch (corruption)
            {
            case 0: invalid.metadata["version"] = 999; break;
            case 1: invalid.metadata["backend"] = "another-engine"; break;
            case 2: invalid.metadata["width"] = 34; break;
            case 3: invalid.metadata["scalar"] = "float64"; break;
            case 4: invalid.metadata["byte_order"] = "unsupported"; break;
            case 5: invalid.data.pop_back(); break;
            case 6: invalid.metadata["liquid_mixture"][0]["element"] = 999; break;
            case 7: invalid.metadata.erase("cached_outflow"); break;
            }
            rejected = false;
            try { resumed.loadCheckpoint(invalid); }
            catch (const std::exception&) { rejected = true; }
            resumed.saveCheckpoint(observed);
            require(rejected && saved.metadata == observed.metadata && saved.data == observed.data,
                "Rejected checkpoint mutated the live grid");
        }
        std::cout << "PASS checkpoint exact state/history/configuration/colour continuation for200 ticks, ingredients="
                  << (extra_component ? 5 : 4) << ", eight incompatible/corrupt loads rejected without mutation\n";
    }

    Fixture fixture;
    FluidGrid setup(17, 13, 1.);
    auto initial = fixture.configure(setup);
    FluidGridGPU native(17, 13, 1.);
    fixture.configureNative(native, initial);
    native.gas_damping_rate = .3;
    native.gravity_y = 11.;
    native.pressure_preconditioner = FluidGridGPU::PressurePreconditioner::Diagonal;
    FluidCheckpoint checkpoint;
    native.saveCheckpoint(checkpoint); // Initial state, before the first pressure solve.
    FluidGridGPU loaded(17, 13, 1.);
    loaded.loadCheckpoint(checkpoint);
    require(loaded.gas_damping_rate == .3 && loaded.gravity_y == 11. &&
        loaded.pressure_preconditioner == FluidGridGPU::PressurePreconditioner::Diagonal, "Checkpoint lost native physical/solver settings");
    for (int tick = 0; tick < 200; ++tick)
    {
        const auto a = native.step(.004, 2000, .01);
        const auto b = loaded.step(.004, 2000, .01);
        require(a.converged && b.converged && a.iterations == b.iterations && a.max_divergence == b.max_divergence &&
            same(native.download(), loaded.download()), "Initial-state checkpoint changed diagonal/damped evolution");
    }
    std::cout << "PASS initial-state checkpoint, diagonal solver and nonzero damping exact continuation for200 ticks\n";

    for (const auto device : {FluidDevice::CPU, FluidDevice::GPU})
    for (const auto mode : {AdaptiveMode::DISABLED, AdaptiveMode::MAX_4})
    {
        if (device == FluidDevice::GPU && mode == AdaptiveMode::DISABLED) continue;
        FluidGrid unsupported(32, 24, 1.);
        unsupported.setDevice(device);
        unsupported.setAdaptiveMode(mode);
        FluidCheckpoint checkpoint;
        require(!unsupported.saveCheckpoint(checkpoint) && !unsupported.loadCheckpoint(checkpoint), "Unimplemented backend silently accepted checkpointing");
    }
}

#ifndef __EMSCRIPTEN__
void workerImageToRender()
{
    std::vector<float> pixels;std::exception_ptr error;
    // Every device operation, including destruction, stays on this worker.
    std::thread owner([&]
    {
        try
        {
            FluidGrid grid(16,16,1.);grid.setDevice(FluidDevice::GPU);Fixture f;f.configure(grid);
            require(grid.step(.002,2000,1e-3).converged,"Worker GPU step failed");grid.writeColorPixels(pixels);grid.destroy();
        }
        catch(...){error=std::current_exception();}
    });
    owner.join();if(error)std::rethrow_exception(error);
    if(!SDL_Init(SDL_INIT_VIDEO))throw std::runtime_error(SDL_GetError());
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,2);
    auto window=SDL_CreateWindow("Fluid WebGPU image handoff",32,32,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN);
    if(!window)throw std::runtime_error(SDL_GetError());
    auto context=SDL_GL_CreateContext(window);if(!context)throw std::runtime_error(SDL_GetError());
    SDL_GL_MakeCurrent(window,context);require(gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)!=0,"GL loader failed");
    FluidGridDisplay display;auto texture=display.texture(16,16,pixels);
    std::vector<float> observed(pixels.size());glBindTexture(GL_TEXTURE_2D,texture);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,observed.data());
    require(observed==pixels && glGetError()==GL_NO_ERROR,"Rendered colour handoff differs from worker-derived image");
    display.destroy();SDL_GL_DestroyContext(context);SDL_DestroyWindow(window);SDL_Quit();
    std::cout<<"PASS worker-owned WebGPU lifecycle and pixel-exact derived-image handoff to GL render owner\n";
}
#endif
}
int main(int argc,char** argv)
{
    try
    {
        std::cout<<std::unitbuf;
        if(argc>1 && std::string(argv[1])=="--evolving"){evolvingParity();return 0;}
        selection();commonParity();checkpointParity();
#ifndef __EMSCRIPTEN__
        workerImageToRender();
#endif
        return 0;
    }
    catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}
}
