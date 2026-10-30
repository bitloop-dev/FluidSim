#include "FluidCheckpointTestHelpers.h"
#include "UniformWebGPU/FluidGridGPU.h"
#include "FluidGrid.h"
#include "FluidTestSetup.h"
#include "UniformWebGPU/FluidSurfaceTensionKernels.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace fluid;
namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
template<class State>
bool same(const State& a,const State& b)
{
    return a.liquid_fraction==b.liquid_fraction && a.u==b.u && a.v==b.v && a.pressure==b.pressure &&
        a.liquid_mass==b.liquid_mass && a.gas_mass==b.gas_mass;
}
FluidGridGPU::State disk(int w,int h,double radius,bool bubble=false,double cx=-1,double cy=-1,double aspect=1)
{
    if(cx<0) cx=w*.5+.13;
    if(cy<0) cy=h*.5-.29;
    FluidGridGPU::State state;
    state.liquid_fraction.resize(size_t(w)*h);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
    {
        double area=0;
        // Integrate the circle-cell intersection, rather than initializing a pixel staircase.
        for(int row=0;row<128;++row)
        {
            const double dy=(y+(row+.5)/128.-cy)*aspect;
            if(std::abs(dy)>=radius) continue;
            const double extent=std::sqrt(radius*radius-dy*dy)*aspect;
            area+=std::max(0.,std::min(x+1.,cx+extent)-std::max(double(x),cx-extent))/128.;
        }
        state.liquid_fraction[size_t(y)*w+x]=float(bubble?1.-area:area);
    }
    return state;
}
template<class State>
double maximumSpeed(const State& state)
{
    double result=0;
    for(const auto* field:{&state.u,&state.v}) for(const auto v:*field)
    {
        require(std::isfinite(v),"Nonfinite capillary velocity"); result=std::max(result,std::abs(double(v)));
    }
    return result;
}
template<class State>
void health(const State& state,double initial_mass)
{
    const double mass=std::accumulate(state.liquid_mass[0].begin(),state.liquid_mass[0].end(),0.);
    require(std::abs(mass-initial_mass)<1e-4*std::max(1.,initial_mass),"Capillarity lost liquid ingredient mass");
    for(size_t i=0;i<state.liquid_fraction.size();++i)
    {
        const double f=state.liquid_fraction[i];
        require(std::isfinite(f)&&f>=-1e-6&&f<=1.000001,"Capillary fraction outside bounds");
        require(std::abs(state.liquid_mass[0][i]/100.-f)<.002 && std::abs(state.gas_mass[0][i]/.1-(1.-f))<.002,
            "Capillary local phase volume mismatch");
    }
}
template<class Grid>
void upload(Grid& grid, const FluidGridGPU::State& state)
{
    test::uploadTestMaterials(grid, convertFluidState<typename Grid::Scalar>(state));
}
template<class Grid>
void flat()
{
    for(int kind=0;kind<4;++kind)
    {
        Grid grid(17,13,1); grid.gravity_y=0; grid.setSurfaceTension(10000);
        FluidGridGPU::State state; state.liquid_fraction.resize(17*13);
        for(int y=0;y<13;++y) for(int x=0;x<17;++x)
            state.liquid_fraction[y*17+x]=kind<2?float(kind):float(kind==2?x<8:y<6);
        upload(grid,state);
        const auto before=grid.download();
        for(int i=0;i<8;++i) require(grid.step(.005,2000,1e-4).converged,"Flat capillary projection failed");
        const auto after=grid.download();
        require(after.liquid_fraction==before.liquid_fraction && maximumSpeed(after)==0,"Flat/pure interface acquired capillary motion");
    }
    std::cout<<"PASS pure phases and grid-aligned flat interfaces\n";
}
template<class Grid>
void circle(double sigma,double tolerance,double radius,int steps,bool bubble=false,bool wall=false)
{
    const int size=64;
    const double cx=wall?0.:size*.5+.13, cy=size*.5-.29;
    Grid grid(size,size,1); grid.gravity_y=0; grid.setSurfaceTension(sigma);
    upload(grid,disk(size,size,radius,bubble,cx,cy));
    const auto initial=grid.download();
    const double mass=std::accumulate(initial.liquid_mass[0].begin(),initial.liquid_mass[0].end(),0.);
    double peak=0;
    for(int i=0;i<steps;++i)
    {
        const auto r=grid.step(.005,2000,tolerance);
        require(r.converged,"Circular capillary projection failed");
        if(i%20==0 || i+1==steps) { const auto s=grid.download(); health(s,mass); peak=std::max(peak,maximumSpeed(s)); }
    }
    const auto state=grid.download();
    double pin=0,pout=0,ni=0,no=0;
    for(int y=0;y<size;++y) for(int x=0;x<size;++x)
    {
        const double r=std::hypot(x+.5-cx,y+.5-cy);
        if(r<radius-2) { pin+=state.pressure[y*size+x]; ni++; }
        if(r>radius+3) { pout+=state.pressure[y*size+x]; no++; }
    }
    const double jump=pin/ni-pout/no,expected=sigma/radius;
    std::cout<<"circle sigma="<<sigma<<" tolerance="<<tolerance<<" radius="<<radius<<" bubble="<<bubble<<" wall="<<wall
        <<" jump="<<jump<<" expected="<<expected<<" peak_speed="<<peak<<std::endl;
    require(std::abs(jump-expected)<.08*expected,"2D Laplace pressure jump is inaccurate");
    if(tolerance<=.001) require(peak<.15*std::sqrt(sigma/(100.*radius)),"Large parasitic capillary currents");
}
void geometryParity()
{
    webgpu::Compute compute;
    webgpu::SurfaceTensionKernels kernels;
    kernels.build(compute);
    double worst = 0.;
    for (const auto [w, h] : {std::pair{64, 64}, std::pair{33, 25}, std::pair{1, 13}, std::pair{13, 1}})
    for (int kind = 0; kind < 4; ++kind)
    {
        auto state = disk(w, h, 6, kind == 1, kind == 2 ? 0. : -1.);
        if (kind == 3)
            for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
                state.liquid_fraction[y * w + x] = float(std::clamp(y + 1. - .25 * x - h * .4, 0., 1.));
        auto liquid = compute.field(w, h);
        auto heights = compute.field(w, h, 4);
        auto raw = compute.field(w, h, 2);
        auto completed = compute.field(w, h, 2);
        compute.write(liquid, state.liquid_fraction);
        webgpu::Parameters settings;
        settings.grid = {w, h, 0, 0};
        settings.value[0] = .5f;
        compute.run(kernels.heights, {&heights}, {{"liquid", &liquid}}, settings);
        compute.run(kernels.curvature, {&raw}, {{"liquid", &liquid}, {"height_field", &heights}}, settings);
        compute.run(kernels.complete, {&completed}, {{"liquid", &liquid}, {"raw_curvature", &raw}}, settings);
        const auto gpu = compute.read(completed);
        const std::vector<double> values(state.liquid_fraction.begin(), state.liquid_fraction.end());
        FluidSurfaceTensionCPU cpu(w, h);
        cpu.liquid = values;
        cpu.cell_size = .5;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) cpu.buildHeight(x, y);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) cpu.buildCurvature(x, y);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) cpu.completeCurvature(x, y);
        for (size_t i = 0; i < values.size(); ++i)
        {
            require(cpu.curvature[i].valid == (gpu[2 * i + 1] != 0), "CPU/GPU curvature validity differs");
            worst = std::max(worst, std::abs(cpu.curvature[i].value - gpu[2 * i]));
        }
    }
    std::cout << "CPU/GPU geometry max curvature difference=" << worst << '\n';
    require(worst < 2e-4, "CPU/GPU curvature estimates differ");
}


void subdivisionAndResume()
{
    FluidGridGPU a(32,32,1),b(32,32,1); a.gravity_y=b.gravity_y=0;
    a.setSurfaceTension(40000); b.setSurfaceTension(40000);
    const auto state=disk(32,32,6,false,-1,-1,1.1);
    test::uploadTestMaterials(a,state); test::uploadTestMaterials(b,state);
    const double dt=.025; const int count=int(std::ceil(dt/a.capillaryStepLimit()));
    FluidStepTimings timings;
    const auto result=a.step(dt,2000,.001,&timings);
    require(result.converged&&result.physical_steps==count,"Capillary physics step count is incorrect");
    int iterations=0;
    for(int i=0;i<count;++i) { auto r=b.step(dt/count,2000,.001); require(r.converged,"Manual capillary substep failed"); iterations+=r.iterations; }
    require(same(a.download(),b.download())&&result.iterations==iterations,"Automatic capillary subdivision differs from explicit complete steps");
    require(timings.transport_ms>0 && timings.forces_ms>0 && timings.pressure_ms>0,"Capillary stage timing was not aggregated");
    FluidCheckpoint saved; a.saveCheckpoint(saved);
    FluidGridGPU resumed(32,32,1); resumed.loadCheckpoint(saved);
    require(resumed.getSurfaceTension()==40000,"Checkpoint lost surface tension");
    for(int i=0;i<30;++i)
    {
        require(a.step(.005,2000,.001).converged&&resumed.step(.005,2000,.001).converged,"Resumed projection failed");
        require(same(a.download(),resumed.download()),"Surface-tension resume changed trajectory");
    }
    a.setSurfaceTension(0); a.saveCheckpoint(saved); makeLegacyCheckpoint(saved, 1);
    resumed.loadCheckpoint(saved); require(resumed.getSurfaceTension()==0,"Legacy checkpoint did not disable tension");
    require(same(a.download(),resumed.download()),"Legacy checkpoint changed native state");
    FluidGrid unsupported(8,8,1);
    unsupported.setAdaptiveMode(AdaptiveMode::MAX_2);
    bool rejected=false; try { unsupported.setSurfaceTension(1); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"Unsupported backend silently accepted capillarity");
    std::cout<<"PASS full-step subdivision, iteration accounting, exact resume and legacy/unsupported handling\n";
}

void cpuIntegration()
{
    FluidGridCPU automatic(32, 32, 1);
    automatic.gravity_y = 0;
    automatic.setSurfaceTension(40000);
    upload(automatic, disk(32, 32, 6, false, -1, -1, 1.1));
    FluidField acceleration({32, 32, 1, {}, 2});
    generateFluidField(acceleration, [](bl::DVec2 p)
    {
        return std::array<float, 4>{float(.01 * p.y), float(-.01 * p.x), 0, 0};
    });
    automatic.setAccelerationField(acceleration);
    auto manual = automatic;
    const double dt = .025;
    const int count = int(std::ceil(dt / automatic.capillaryStepLimit()));
    FluidStepTimings timings;
    const auto result = automatic.step(dt, 2000, .001, &timings);
    int iterations = 0;
    for (int i = 0; i < count; ++i)
    {
        const auto part = manual.step(dt / count, 2000, .001);
        require(part.converged, "CPU manual capillary substep failed");
        iterations += part.iterations;
    }
    require(result.converged && count > 1 && result.physical_steps == count && result.iterations == iterations &&
        same(automatic.download(), manual.download()), "CPU full-step subdivision differs from explicit steps");
    require(timings.transport_ms > 0 && timings.forces_ms > 0 && timings.pressure_ms > 0,
        "CPU capillary timings were not aggregated");
    auto copied = automatic; // Copying cached geometry must not retain another grid's fractions.
    require(copied.step(.005, 2000, .001).converged && automatic.step(.005, 2000, .001).converged &&
        same(copied.download(), automatic.download()), "CPU copied capillary state diverged");

    automatic.clearAccelerationField();
    automatic.setSurfaceTension(0);
    automatic.gravity_y = 10;
    auto original = automatic;
    for (int i = 0; i < 20; ++i)
    {
        // Original zero-tension operations, including existing pressure history.
        original.advect(.005);
        original.applyGravity(.005);
        const auto reference = original.project(.005, 2000, .001);
        const auto current = automatic.step(.005, 2000, .001);
        require(reference.converged && current.converged && reference.iterations == current.iterations &&
            current.physical_steps == 1 && same(original.download(), automatic.download()),
            "Disabled CPU tension changed the original numerical path");
    }
    const auto before = automatic.download();
    require(automatic.step(0).physical_steps == 0 && same(before, automatic.download()), "Zero-time capillarity advanced state");
    bool rejected = false;
    try { automatic.setSurfaceTension(-1); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && automatic.getSurfaceTension() == 0, "Invalid CPU tension changed the coefficient");
    automatic.setSurfaceTension(40000);
    const auto failed = automatic.step(.025, 0, 1e-12);
    require(!failed.converged && failed.physical_steps == 1, "CPU continued capillary substeps after solver failure");

    FluidGrid api(8, 8, 1);
    api.setSurfaceTension(20000);
    api.setDevice(FluidDevice::GPU);
    require(api.getSurfaceTension() == 20000, "CPU-to-GPU selection lost surface tension");
    api.setDevice(FluidDevice::CPU);
    require(api.getSurfaceTension() == 20000, "GPU-to-CPU selection lost surface tension");
    rejected = false;
    try { api.setAdaptiveMode(AdaptiveMode::MAX_2); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && api.getAdaptiveMode() == AdaptiveMode::DISABLED && api.getSurfaceTension() == 20000,
        "Adaptive selection silently discarded surface tension");
    std::cout << "PASS CPU subdivision, field composition, copying, zero path, timings, failure and API selection\n";
}
template<class Grid>
void deformationAndMotion()
{
    auto moments=[](const typename Grid::State& s,int w)
    {
        double volume=0,x=0,y=0,xx=0,yy=0;
        for(size_t i=0;i<s.liquid_fraction.size();++i)
        {
            const double f=s.liquid_fraction[i],cx=double(i%w)+.5,cy=double(i/w)+.5;
            volume+=f; x+=f*cx; y+=f*cy; xx+=f*cx*cx; yy+=f*cy*cy;
        }
        return std::array<double,3>{x/volume,xx/volume-x*x/(volume*volume),yy/volume-y*y/(volume*volume)};
    };
    Grid ellipse(48,48,1); ellipse.gravity_y=0; ellipse.setSurfaceTension(40000);
    upload(ellipse,disk(48,48,6,false,-1,-1,1.15));
    auto initial=ellipse.download(); const auto before=moments(initial,48);
    const double mass=std::accumulate(initial.liquid_mass[0].begin(),initial.liquid_mass[0].end(),0.);
    for(int i=0;i<80;++i) require(ellipse.step(.005,2000,.001).converged,"Deforming droplet projection failed");
    const auto state=ellipse.download(); const auto after=moments(state,48); health(state,mass);
    std::cout<<"ellipse variance difference "<<before[1]-before[2]<<" -> "<<after[1]-after[2]<<std::endl;
    require(std::abs(after[1]-after[2])<.8*std::abs(before[1]-before[2]),"Surface tension did not restore an elongated droplet");

    Grid moving(96,64,1); moving.gravity_y=0; moving.setSurfaceTension(10000);
    auto input=disk(96,64,6,false,32.13,31.71);
    auto window=[](double x,double extent)
    {
        const double edge=std::min(x,extent-x);
        return edge>=12?1.:std::pow(std::sin(3.14159265358979323846*.5*edge/12.),2);
    };
    auto psi=[&](double x,double y) { return 10.*(y-32)*window(x,96)*window(y,64); };
    input.u.resize(97*64); input.v.resize(96*65);
    for(int y=0;y<64;++y) for(int x=0;x<=96;++x) input.u[y*97+x]=float(psi(x,y+1)-psi(x,y));
    for(int y=0;y<=64;++y) for(int x=0;x<96;++x) input.v[y*96+x]=float(psi(x,y)-psi(x+1,y));
    upload(moving,input);
    initial=moving.download(); const double moving_mass=std::accumulate(initial.liquid_mass[0].begin(),initial.liquid_mass[0].end(),0.);
    for(int i=0;i<200;++i) require(moving.step(.005,2000,.001).converged,"Moving droplet projection failed");
    const auto moved=moving.download(); health(moved,moving_mass);
    const auto location=moments(moved,96);
    std::cout<<"moving droplet centroid_x="<<location[0]<<std::endl;
    require(location[0]>35 && location[1]+location[2]<60,"Moving droplet failed to travel coherently");
}
}
int main(int argc,char** argv)
{
    try
    {
        std::cout<<std::unitbuf;
        geometryParity();
        if(argc==6) { circle<FluidGridGPU>(std::stod(argv[1]),std::stod(argv[2]),std::stod(argv[3]),std::stoi(argv[4]),std::stoi(argv[5])!=0); return 0; }
        flat<FluidGridGPU>();
        circle<FluidGridGPU>(10000,.001,8,200);
        circle<FluidGridGPU>(10000,.001,8,200,true);
        circle<FluidGridGPU>(10000,.001,8,200,false,true);
        subdivisionAndResume();
        deformationAndMotion<FluidGridGPU>();
        std::cout << "CPU uniform capillarity\n";
        flat<FluidGridCPU>();
        circle<FluidGridCPU>(10000,.001,8,200);
        circle<FluidGridCPU>(10000,.001,8,200,true);
        circle<FluidGridCPU>(10000,.001,8,200,false,true);
        deformationAndMotion<FluidGridCPU>();
        cpuIntegration();
        std::cout<<"PASS surface tension\n"; return 0;
    }
    catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<std::endl; return 1; }
}
