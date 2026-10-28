#include "FluidGrid.h"
#include "FluidGridAdaptiveCPU.h"
#include "Adaptive/FluidGridAdaptive.h"
#include "Adaptive/FluidCellView.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <type_traits>

using namespace fluid;
namespace
{
static_assert(std::is_base_of_v<FluidGridBase, FluidGridAdaptiveCPU> && !std::is_abstract_v<FluidGridAdaptiveCPU>);
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Fixture
{
    ElementCatalogue elements;
    FluidMixture gas, liquid;
    Fixture(bool mixed = false, double contrast = 1.)
    {
        gas.components = {{elements.add({"Air", {.1f, .2f, .3f}}), .1},
                          {elements.add({"Cloud", {0.f, 1.f, 1.f}}), .1 * contrast}};
        if (mixed) liquid.components = {{elements.add({"Water", {0.f, 0.f, 1.f}}), 100.}};
    }
    FluidGridBase::State configure(FluidGridBase& grid, bool moving = true) const
    {
        grid.configureMaterials(elements, liquid, gas); grid.setGravityY(0.);
        const int w = grid.getWidth(), h = grid.getHeight();
        const double area = grid.getCellSize() * grid.getCellSize();
        FluidGridBase::State s;
        s.liquid_fraction.resize(w*h); s.gas_mass.assign(2, std::vector<double>(w*h));
        if (!liquid.components.empty()) s.liquid_mass.assign(1, std::vector<double>(w*h));
        s.u.resize((w+1)*h); s.v.resize(w*(h+1));
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
        {
            const int i = y*w+x;
            const double share = .2 + .6 * std::exp(-(std::pow(x-w*.35,2) + std::pow(y-h*.45,2))/48.);
            const double wet = liquid.components.empty() ? 0. : std::clamp(y - .6*h - .15*x, 0., 1.);
            s.liquid_fraction[i] = wet;
            if (!s.liquid_mass.empty()) s.liquid_mass[0][i] = wet * 100. * area;
            s.gas_mass[0][i] = (1.-wet)*(1.-share)*gas.components[0].density*area;
            s.gas_mass[1][i] = (1.-wet)*share*gas.components[1].density*area;
        }
        auto psi = [&](int x, int y)
        {
            if (!moving || !x || !y || x==w || y==h) return 0.;
            return 10.*std::exp(-(std::pow(x-w*.5,2)+std::pow(y-h*.5,2))/180.);
        };
        for(int y=0;y<h;++y)for(int x=0;x<=w;++x)s.u[y*(w+1)+x]=(psi(x,y+1)-psi(x,y))/grid.getCellSize();
        for(int y=0;y<=h;++y)for(int x=0;x<w;++x)s.v[y*w+x]=-(psi(x+1,y)-psi(x,y))/grid.getCellSize();
        grid.upload(s); return s;
    }
};
void compareField(const AmrField2D<double>& a, const AmrField2D<double>& b, int components)
{
    for(int p=0;p<a.size();++p)
    {
        const auto x=a.const_array(p), y=b.const_array(p); const auto& box=a.box(p);
        for(int j=box.smallEnd(1);j<=box.bigEnd(1);++j) for(int i=box.smallEnd(0);i<=box.bigEnd(0);++i)
            for(int n=0;n<components;++n)
                require(std::abs(x(i,j,n)-y(i,j,n)) <= 1e-10 * (1.+std::abs(x(i,j,n))), "Common API changed a native adaptive field");
    }
}
void compareNative(const FluidGridAdaptive& a, const FluidGridAdaptive& b)
{
    require(a.layout()==b.layout(), "Backend schedule changed the native layout policy");
    for(size_t i=0;i<a.levels().size();++i)
    {
        const auto& x=*a.levels()[i]; const auto& y=*b.levels()[i];
        compareField(x.conserved,y.conserved,a.ingredientCount()+2);
        compareField(x.pressure,y.pressure,1);
        for(int d=0;d<2;++d)compareField(x.velocity[d],y.velocity[d],1);
    }
}
void manualParity()
{
    for(bool mixed : {false,true})
    {
        Fixture f(mixed,3.); FluidAdaptiveSettings settings; settings.automatic_refinement=false;
        FluidGridAdaptiveCPU backend(32,32,1.,4,settings); FluidGridBase& api=backend;
        auto initial=f.configure(api); backend.setGravityY(.8);
        FluidGridAdaptive raw(32,32,1.,f.elements,f.liquid,f.gas,{}); raw.upload(initial);
        compareNative(raw,backend.native());
        const auto before=api.download(); require(api.step(0.).converged,"Zero-time step failed");
        require(before.gas_mass==api.download().gas_mass && backend.refinementChecks()==0,"Zero-time step ran selection or changed state");
        for(int i=0;i<24;++i)
        {
            const auto expected=raw.step(.01,1e-8,0.,{0.,.8},2000).projection;
            const auto actual=api.step(.01,2000,1e-8);
            require(expected.converged && actual.converged,"Adaptive polymorphic step failed");
            compareNative(raw,backend.native());
        }
        require(backend.refinementChecks()==0 && backend.native().layout()==GasLayout{},"Automatic refinement OFF changed layout");
        auto raster=api.download(); std::vector<float> pixels; api.writeColorPixels(pixels);
        require(raster.u.size()==33*32 && raster.v.size()==32*33 && pixels.size()==32*32*4,"Common observation dimensions incorrect");
        compareNative(raw,backend.native());
        const auto failure=api.step(.1,0,1e-12);
        require(!failure.converged && failure.iterations==0,"Adaptive API ignored the caller iteration cap or hid failure");
    }
    std::cout << "PASS adaptive step-only native parity, gas/liquid, OFF, observation, zero-time and iteration cap\n";
}
void saveResolution(const std::filesystem::path& directory, const FluidGridAdaptive& grid, int frame)
{
    if(directory.empty())return;
    std::filesystem::create_directories(directory);
    FluidCellView view(grid);
    std::ofstream out(directory/('f'+std::to_string(frame)+".ppm"),std::ios::binary);
    out<<"P6\n"<<view.width<<' '<<view.height<<"\n255\n";
    for(int y=0;y<view.height;++y)for(int x=0;x<view.width;++x)
    {
        const int scale=view.at(x,y).patch->scale;
        const unsigned char fine[3]={245,175,55}, medium[3]={50,170,230}, coarse[3]={55,65,100};
        out.write(reinterpret_cast<const char*>(scale==1?fine:scale==2?medium:coarse),3);
    }
}
void automaticParity(const std::filesystem::path& directory)
{
    Fixture f; FluidAdaptiveSettings settings;
    FluidGridAdaptiveCPU backend(96,96,1.,4,settings); auto initial=f.configure(backend);
    FluidGridAdaptive raw(96,96,1.,f.elements,f.gas,{}); raw.upload(initial);
    FluidCoarseningPreview selection; double time=0., next=0.; int checks=0, changes=0;
    auto old=raw.layout(); const auto mass0=raw.totals().ingredientMass;
    for(int frame=0;frame<200;++frame)
    {
        if(time>=next)
        {
            selection.update(raw,settings.tolerances,settings.stability,time,settings.prediction_horizon,4);
            if(raw.regrid(selection.proposeLayout(raw)))require(raw.project(1e-8,0.,{},2000).converged,"Raw restoration failed");
            next=time+settings.check_interval; ++checks;
        }
        require(raw.step(.02,1e-8,0.,{},2000).projection.converged,"Raw evolution failed");
        require(backend.step(.02,2000,1e-8).converged,"Automatic common step failed");
        time+=.02; compareNative(raw,backend.native());
        if(raw.layout()!=old){++changes;old=raw.layout();}
        require(backend.refinementChecks()==checks,"Backend checked refinement outside the agreed cadence");
        if(frame==0 || frame==31 || frame==99 || frame==199)saveResolution(directory,backend.native(),frame);
    }
    require(changes>0 && raw.activeCellCount()<96*96,"Automatic test never exercised real coarse/fine state");
    const auto totals=backend.native().totals();
    for(size_t k=0;k<mass0.size();++k)require(std::abs(totals.ingredientMass[k]/mass0[k]-1.)<1e-10,"Automatic remap lost component mass");
    auto raster=backend.download();
    for(size_t k=0;k<mass0.size();++k)
        require(std::abs(std::accumulate(raster.gas_mass[k].begin(),raster.gas_mass[k].end(),0.)/totals.ingredientMass[k]-1.)<1e-11,"Diagnostic raster changed conserved mass");
    std::vector<float> pixels; backend.writeColorPixels(pixels);
    for(size_t i=0;i<raster.gas_mass[0].size();++i)for(int c=0;c<3;++c)
    {
        float expected=0.; for(size_t k=0;k<f.gas.components.size();++k)
            expected+=float(raster.gas_mass[k][i]/f.gas.components[k].density*f.elements[f.gas.components[k].element].color[c]);
        require(std::abs(pixels[4*i+c]-expected)<1e-6,"Display differs from the bounded material reconstruction");
    }
    // Sparse edits in a quiet coarse region must refine locally, replace only
    // the requested finest cells, and keep the existing projected face field.
    auto edit=backend.beginEdit(); const auto before=backend.native().totals().ingredientMass;
    const double removedAir=raster.gas_mass[0][4*96+4];
    const double removedCloud=raster.gas_mass[1][4*96+4];
    edit.setCell(4,4,Phase::Gas,0); edit.setCell(4,4,Phase::Gas,1); edit.commit();
    require(FluidCellView(backend.native()).at(4,4).patch->scale==1,"Fine edit silently overwrote a coarse cell");
    const auto edited=backend.download();
    require(edited.gas_mass[0][4*96+4]==0. && edited.gas_mass[1][4*96+4]==.1,"Adaptive last-write replacement incorrect");
    const auto after=backend.native().totals().ingredientMass;
    require(std::abs(after[0]-before[0]+removedAir)<1e-10 && std::abs(after[1]-before[1]+removedCloud-.1)<1e-10,"Sparse adaptive edit changed unselected material budget");
    edit.setCell(3,3,Phase::Gas,0); edit.setCell(-1,3,Phase::Gas,0);
    bool rejected=false; try{edit.commit();}catch(const std::out_of_range&){rejected=true;}
    require(rejected && edit.size()==2 && backend.download().gas_mass==edited.gas_mass,"Invalid adaptive batch lost atomicity");
    require(backend.step(.02,2000,1e-8).converged,"Edited adaptive state could not advance");
    std::cout<<"PASS 200-frame native scheduling parity, checks="<<checks<<", layout_changes="<<changes<<", active_cells="<<raw.activeCellCount()<<", conservation, bounded colour and sparse edits\n";
}
void selectionAndLifecycle()
{
    for(auto mode:{AdaptiveMode::MAX_2,AdaptiveMode::MAX_4,AdaptiveMode::MAX_8,AdaptiveMode::MAX_16,AdaptiveMode::MAX_32})
    {
        FluidGrid grid(32,32,2.); Fixture f; grid.configureMaterials(f.elements,{},f.gas); grid.setOrigin({2.,3.}); grid.setGravityY(0.);
        grid.setAdaptiveMode(mode);
        require(grid.getAdaptiveMode()==mode && grid.getOrigin().x==2. && grid.getGasMixture().components.size()==2,"Mode selection lost configuration");
        f.configure(grid,false); FluidGridBase& api=grid;
        require(api.step(.02,2000,1e-8).converged,"Facade adaptive backend did not run");
        auto before=api.download(); grid.setAdaptiveMode(mode);
        require(before.gas_mass==api.download().gas_mass,"Same-mode selection reset physical state");
        FluidAdaptiveSettings off; off.automatic_refinement=false; grid.setAdaptiveSettings(off);
        require(api.step(.02,2000,1e-8).converged,"Runtime quality policy update failed");
        FluidGrid moved=std::move(grid); require(moved.step(.02,2000,1e-8).converged,"Moved facade lost adaptive configuration dispatch");
        moved.destroy(); moved.setAdaptiveMode(AdaptiveMode::DISABLED); f.configure(moved,false);
        require(moved.step(.02).converged,"Explicit destroy/reinitialize to uniform failed");
    }
    std::cout<<"PASS all MAX modes through FluidGridBase, configuration, live policy, move and explicit restart\n";
}
void liquidSourceInCoarseGas()
{
    Fixture f(true); FluidGridAdaptiveCPU backend(96,96,1.);
    auto initial=f.configure(backend,false);
    std::fill(initial.liquid_fraction.begin(),initial.liquid_fraction.end(),0.);
    std::fill(initial.liquid_mass[0].begin(),initial.liquid_mass[0].end(),0.);
    for(auto& mass:initial.gas_mass)std::fill(mass.begin(),mass.end(),.05);
    backend.upload(initial); FluidGridBase& api=backend;
    require(api.step(.32,2000,1e-8).converged && api.step(.32,2000,1e-8).converged,"Quiet mixed-mixture setup failed");
    require(FluidCellView(backend.native()).at(4,4).patch->scale==4,"Liquid source test did not start in coarse gas");
    const auto before=backend.native().totals().ingredientMass;
    auto edit=api.beginEdit();edit.setCell(4,4,Phase::Liquid,2);edit.commit();
    const auto state=api.download();const int i=4*96+4;
    require(state.liquid_fraction[i]==1. && state.liquid_mass[0][i]==100. && state.gas_mass[0][i]==0. && state.gas_mass[1][i]==0.,"Liquid source did not replace one finest gas cell");
    const auto after=backend.native().totals().ingredientMass;
    require(std::abs(after[0]-before[0]-100.)<1e-10 && std::abs(after[1]-before[1]+.05)<1e-10 && std::abs(after[2]-before[2]+.05)<1e-10,"Liquid source changed unselected budgets");
    backend.setGravityY(100.);
    require(api.step(.02,2000,1e-8).converged && FluidCellView(backend.native()).at(4,4).patch->scale==1,"Automatic policy lost a newly edited liquid cell");
    require(backend.native().totals().finite,"Liquid edit produced non-finite evolving state");
    std::cout<<"PASS common liquid source in coarse gas, local refinement, component budgets and gravity evolution\n";
}
}
int main(int argc,char** argv)
{
    try
    {
        bl::Thread::setMaxThreads(8); manualParity();
        automaticParity(argc>1?std::filesystem::path(argv[1]):std::filesystem::path{});
        selectionAndLifecycle(); liquidSourceInCoarseGas(); return 0;
    }
    catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
