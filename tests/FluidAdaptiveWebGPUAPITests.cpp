#include "FluidGrid.h"
#include "FluidGridAdaptiveWebGPU.h"
#include "AdaptiveWebGPU/FluidGridAdaptiveGPU.h"
#include "AdaptiveWebGPU/FluidCoarseningGPU.h"
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
static_assert(std::is_base_of_v<FluidGridBase,FluidGridAdaptiveWebGPU> && !std::is_abstract_v<FluidGridAdaptiveWebGPU>);
static_assert(std::is_same_v<FluidGridAdaptiveWebGPU::State,FluidGridBase::State>);
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
bool same(const FluidGridBase::State& a,const FluidGridBase::State& b)
{
    return a.liquid_mass==b.liquid_mass && a.gas_mass==b.gas_mass && a.liquid_fraction==b.liquid_fraction && a.u==b.u && a.v==b.v && a.pressure==b.pressure;
}
struct Fixture
{
    SubstanceCatalogue substances;FluidMixture liquid,gas;
    Fixture(bool mixed=false,double contrast=1.)
    {
        gas.components={{substances.add({"Air",{.1f,.2f,.3f}}),.1},{substances.add({"Cloud",{0.f,1.f,1.f}}),.1*contrast}};
        if(mixed)liquid.components={{substances.add({"Water",{0.f,0.f,1.f}}),100.}};
    }
    FluidGridBase::State configure(FluidGridBase& grid,bool moving=true,bool wet=true) const
    {
        grid.configureMaterials(substances,liquid,gas);grid.setGravityY(0.);
        const int w=grid.getWidth(),h=grid.getHeight();const double area=grid.getCellSize()*grid.getCellSize();
        FluidGridBase::State s;s.liquid_fraction.resize(w*h);s.gas_mass.assign(2,std::vector<double>(w*h));
        if(!liquid.components.empty())s.liquid_mass.assign(1,std::vector<double>(w*h));
        s.u.resize((w+1)*h);s.v.resize(w*(h+1));
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)
        {
            const int i=y*w+x;const double share=moving?.2+.6*std::exp(-(std::pow(x-w*.35,2)+std::pow(y-h*.45,2))/48.):.5;
            const double f=!liquid.components.empty() && wet?std::clamp(y-.6*h-.15*x,0.,1.):0.;
            s.liquid_fraction[i]=f;if(!s.liquid_mass.empty())s.liquid_mass[0][i]=f*100.*area;
            s.gas_mass[0][i]=(1.-f)*(1.-share)*gas.components[0].density*area;
            s.gas_mass[1][i]=(1.-f)*share*gas.components[1].density*area;
        }
        auto psi=[&](int x,int y){if(!moving || !x || !y || x==w || y==h)return 0.;return 10.*std::exp(-(std::pow(x-w*.5,2)+std::pow(y-h*.5,2))/180.);};
        for(int y=0;y<h;++y)for(int x=0;x<=w;++x)s.u[y*(w+1)+x]=(psi(x,y+1)-psi(x,y))/grid.getCellSize();
        for(int y=0;y<=h;++y)for(int x=0;x<w;++x)s.v[y*w+x]=-(psi(x+1,y)-psi(x,y))/grid.getCellSize();
        grid.upload(s);return s;
    }
};
void compareNative(const FluidGridAdaptiveGPU& a,const FluidGridAdaptiveGPU& b)
{
    require(a.layout()==b.layout(),"Common adaptive GPU scheduling changed layout");
    for(size_t lev=0;lev<a.levels().size();++lev)
    {
        const auto& x=*a.levels()[lev];const auto& y=*b.levels()[lev];
        const auto q=a.commands().read(x.conserved),r=b.commands().read(y.conserved);
        for(const auto& p:x.qLayout.patches)forCells(p.box,[&](int i,int j)
        {
            for(int n=0;n<a.ingredientCount()+2;++n)require(q[p.offset+p.index(i,j)+n*p.plane]==r[p.offset+p.index(i,j)+n*p.plane],"Common adaptive GPU changed native conserved fields");
        });
        require(a.commands().read(x.pressure)==b.commands().read(y.pressure),"Common adaptive GPU changed compensated pressure");
        for(int d=0;d<2;++d)require(a.commands().read(x.velocity[d])==b.commands().read(y.velocity[d]),"Common adaptive GPU changed native MAC faces");
    }
}
void manualParity()
{
    for(bool mixed:{false,true})
    {
        Fixture f(mixed,3.);FluidAdaptiveSettings off;off.automatic_refinement=false;
        FluidGridAdaptiveWebGPU backend(32,32,1.,4,off);FluidGridBase& api=backend;
        auto initial=f.configure(api);backend.setGravityY(.8);
        FluidGridAdaptiveGPU raw(32,32,1.,f.substances,f.liquid,f.gas,{});raw.upload(initial);
        compareNative(raw,backend.native());auto before=api.download();api.step(0.,2000,1e-3);
        require(same(before,api.download()) && backend.refinementChecks()==0,"Zero-time adaptive GPU call changed state or selected layout");
        for(int frame=0;frame<24;++frame)
        {
            const auto a=api.step(.01,2000,1e-3),b=raw.step(.01,1e-3,0.,{0.,.8},2000).projection;
            require(a.converged && b.converged && a.iterations==b.iterations && a.max_divergence==b.max_divergence,"Common adaptive GPU step changed native result");
        }
        compareNative(raw,backend.native());std::vector<float> pixels(32*32*4);auto* storage=pixels.data();api.writeColorPixels(pixels);
        require(pixels==raw.colorPixels() && storage==pixels.data(),"Common adaptive colour output differs or discards caller storage");
        compareNative(raw,backend.native());auto raster=api.download();
        require(raster.u.size()==33*32 && raster.v.size()==32*33,"Adaptive common raster dimensions incorrect");
        const auto& level=*backend.native().levels().front();
        const auto pressure=backend.native().commands().read(level.pressure);
        const int plane=level.pressure.width*level.pressure.height;
        for(const auto& p:level.pressureLayout.patches)forCells(p.box,[&](int x,int y)
        {
            const int at=p.offset+p.index(x,y);
            const double expected=(double(pressure[at])+pressure[at+plane])/backend.native().pressureInterval();
            require(raster.pressure[y*32+x]==expected,"Common readback lost compensated pressure or physical units");
        });
        require(backend.refinementChecks()==0 && backend.native().layout()==GasLayout{},"Refinement OFF changed the fine layout");
        const auto oldQ=backend.native().commands().read(level.conserved);
        const auto oldU=backend.native().commands().read(level.velocity[0]),oldV=backend.native().commands().read(level.velocity[1]);
        const auto& patch=level.qLayout.patches.front();const int at=patch.offset+patch.index(7,8),nc=backend.native().ingredientCount();
        double rho=0.;for(int k=0;k<nc;++k)rho+=oldQ[at+k*patch.plane];
        auto flowingEdit=api.beginEdit();flowingEdit.setCell(7,8,Phase::Gas,1);flowingEdit.commit();
        const auto newQ=backend.native().commands().read(level.conserved);const double replacement=float(f.gas.components[1].density);
        for(int d=0;d<2;++d)
            require(std::abs(newQ[at+(nc+d)*patch.plane]/replacement-oldQ[at+(nc+d)*patch.plane]/rho)<1e-6,"Fine material replacement changed local cell velocity");
        require(pressure==backend.native().commands().read(level.pressure) && oldU==backend.native().commands().read(level.velocity[0]) && oldV==backend.native().commands().read(level.velocity[1]),"Fine edit changed projected faces or pressure history");
        const auto failed=api.step(.1,0,1e-12);require(!failed.converged && failed.iterations==0,"Adaptive WebGPU ignored caller iteration cap");
        api.destroy();api.upload(initial);require(backend.refinementChecks()==0 && backend.simulationTime()==0.,"Adaptive GPU restart retained schedule");
    }
    std::cout<<"PASS gas/mixed-liquid native step parity, OFF, zero/reset/failure, compensated state and reusable colours\n";
}
void saveLayout(const std::filesystem::path& path,const FluidGridAdaptiveGPU& grid,int frame)
{
    if(path.empty())return;std::filesystem::create_directories(path);
    const int w=grid.finestWidth(),h=grid.finestHeight();std::vector<int> scale(w*h,0);
    for(const auto& level:grid.levels())
    {
        const int size=int(std::lround(level->geometry.cell_size/grid.finestCellSize()));
        const auto mask=grid.commands().read(level->active);
        for(const auto& patch:level->cellLayout.patches)forCells(patch.box,[&](int x,int y)
        {
            if(!mask[patch.offset+patch.index(x,y)])return;
            for(int dy=0;dy<size;++dy)for(int dx=0;dx<size;++dx)scale[(y*size+dy)*w+x*size+dx]=size;
        });
    }
    std::ofstream out(path/("frame"+std::to_string(frame)+".ppm"),std::ios::binary);out<<"P6\n"<<w<<' '<<h<<"\n255\n";
    for(int size:scale){const unsigned char fine[3]={245,175,55},medium[3]={50,170,230},coarse[3]={55,65,100};out.write(reinterpret_cast<const char*>(size==1?fine:size==2?medium:coarse),3);}
}
void automaticParity(const std::filesystem::path& path)
{
    Fixture f;FluidAdaptiveSettings settings;FluidGridAdaptiveWebGPU backend(96,96,1.,4,settings);
    auto initial=f.configure(backend);FluidGridAdaptiveGPU raw(96,96,1.,f.substances,f.gas,{});raw.upload(initial);
    FluidCoarseningGPU selection;double time=0.,next=0.;int checks=0,changes=0;auto previous=raw.layout();const auto mass0=raw.totals().ingredientMass;
    for(int frame=0;frame<200;++frame)
    {
        if(time>=next)
        {
            selection.update(raw,settings.tolerances,settings.stability,time,settings.prediction_horizon,4);
            if(raw.regrid(selection.proposeLayout(raw)))require(raw.project(1e-4,0.,{},2000).converged,"Native policy restoration failed");
            ++checks;next=time+settings.check_interval;
        }
        auto a=backend.step(.02,2000,1e-4),b=raw.step(.02,1e-4,0.,{},2000).projection;
        require(a.converged && b.converged && a.iterations==b.iterations && a.max_divergence==b.max_divergence,"200-frame adaptive native/common result differs");
        time+=.02;require(backend.refinementChecks()==checks && raw.layout()==backend.native().layout(),"Automatic cadence/policy differs from source control");
        if(raw.layout()!=previous){++changes;previous=raw.layout();}
        if(frame==0 || frame==31 || frame==99 || frame==199){compareNative(raw,backend.native());saveLayout(path,backend.native(),frame+1);}
    }
    require(changes>0 && raw.activeCellCount()<96*96,"Automatic common fixture did not exercise coarse/fine state");
    const auto totals=backend.native().totals();auto raster=backend.download();std::vector<float> pixels;backend.writeColorPixels(pixels);
    for(size_t k=0;k<mass0.size();++k)
    {
        require(std::abs(totals.ingredientMass[k]/mass0[k]-1.)<1e-5,"Evolving adaptive GPU component drift exceeded budget");
        require(std::abs(std::accumulate(raster.gas_mass[k].begin(),raster.gas_mass[k].end(),0.)/totals.ingredientMass[k]-1.)<1e-10,"Reconstructed common raster changed native ingredient mass");
    }
    for(size_t i=0;i<96*96;++i)for(int channel=0;channel<3;++channel)
    {
        double color=0.;for(size_t k=0;k<f.gas.components.size();++k)color+=raster.gas_mass[k][i]/f.gas.components[k].density*f.substances[f.gas.components[k].substance].color[channel];
        require(std::abs(pixels[4*i+channel]-color)<2e-5,"Common raster differs from native bounded material display");
    }
    compareNative(raw,backend.native());
    std::cout<<"PASS 200 evolving frames, checks="<<checks<<", layout_changes="<<changes<<", active_cells="<<raw.activeCellCount()<<", native fields/results, conservative raster and bounded colours\n";
}
void editsAndModes()
{
    Fixture f(true);FluidGridAdaptiveWebGPU backend(96,96,2.5);auto initial=f.configure(backend,false,false);FluidGridBase& api=backend;
    require(api.step(.32,2000,1e-3).converged && api.step(.32,2000,1e-3).converged,"Quiet mixed-mixture setup failed");
    require(backend.native().layout()==GasLayout{4,{}},"Source edit fixture did not start wholly coarse");
    auto before=backend.native().totals().ingredientMass;auto edit=api.beginEdit();
    edit.setCell(4,4,Phase::Gas,0);edit.setCell(4,4,Phase::Liquid,2);edit.commit();
    const auto state=api.download();const int i=4*96+4;
    require(edit.empty() && state.liquid_fraction[i]==1. && state.liquid_mass[0][i]==625. && state.gas_mass[0][i]==0. && state.gas_mass[1][i]==0.,"Adaptive GPU finest-cell replacement or duplicate ordering failed");
    const auto after=backend.native().totals().ingredientMass;
    require(std::abs(after[0]-before[0]-625.)<1e-6 && std::abs(after[1]-before[1]+.3125)<1e-6 && std::abs(after[2]-before[2]+.3125)<1e-6,"Adaptive GPU edit changed unselected material budget");
    edit.setCell(3,3,Phase::Gas,0);edit.setCell(-1,3,Phase::Gas,0);bool rejected=false;
    try{edit.commit();}catch(const std::out_of_range&){rejected=true;}
    require(rejected && edit.size()==2 && same(state,api.download()),"Adaptive GPU invalid batch was not atomic");
    backend.setGravityY(100.);require(api.step(.02,2000,1e-3).converged && backend.native().totals().finite,"Edited liquid could not evolve under gravity");
    for(auto mode:{AdaptiveMode::MAX_2,AdaptiveMode::MAX_4,AdaptiveMode::MAX_8,AdaptiveMode::MAX_16,AdaptiveMode::MAX_32})
    {
        FluidGrid grid(32,32,1.);grid.setAdaptiveMode(mode);grid.setDevice(FluidDevice::GPU);
        Fixture gas;gas.configure(grid,false);grid.setOrigin({2.,3.});FluidGridBase& common=grid;
        require(common.step(.02,2000,1e-3).converged && grid.getDevice()==FluidDevice::GPU && grid.getAdaptiveMode()==mode,"GPU/MAX wrapper selection failed");
        auto frozen=grid.getAdaptiveSettings();frozen.automatic_refinement=false;grid.setAdaptiveSettings(frozen);
        auto old=common.download();grid.setDevice(FluidDevice::GPU);grid.setAdaptiveMode(mode);require(same(old,common.download()),"Same GPU/MAX selection reset state");
        FluidGrid moved=std::move(grid);require(moved.step(.02,2000,1e-3).converged,"Moved adaptive GPU wrapper lost policy dispatch");
        moved.destroy();moved.setAdaptiveMode(AdaptiveMode::DISABLED);gas.configure(moved,false);require(moved.step(.01,2000,1e-3).converged,"Explicit adaptive-to-uniform GPU restart failed");
    }
    std::cout<<"PASS coarse-gas liquid source, budgets, atomicity, gravity, all GPU/MAX modes, live policy, move and explicit restart\n";
}
}
int main(int argc,char** argv)
{
    try{std::cout<<std::unitbuf;manualParity();automaticParity(argc>1?std::filesystem::path(argv[1]):std::filesystem::path{});editsAndModes();return 0;}
    catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
