#include "FluidCoarseningPreview.h"
#include "FluidCellView.h"
#include "AmrGasRuntime.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numbers>
#include <stdexcept>

using fluid::FluidGridAdaptive;
using Preview=fluid::FluidCoarseningPreview;
using Clock=std::chrono::steady_clock;
namespace {
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
double perlinNoise(double x, double y)
{
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const double fx = x - ix, fy = y - iy;

    auto gradient = [](int gx, int gy, double dx, double dy)
    {
        // Hash each lattice corner to a repeatable gradient direction.
        std::uint32_t hash = std::uint32_t(gx) * 0x9e3779b9u ^ std::uint32_t(gy) * 0x85ebca6bu ^ 0x12345678u;
        hash ^= hash >> 16;
        hash *= 0x7feb352du;
        hash ^= hash >> 15;
        hash *= 0x846ca68bu;
        hash ^= hash >> 16;
        constexpr double diagonal = 0.7071067811865476;
        constexpr double directions[8][2] = {
            {1, 0}, {-1, 0}, {0, 1}, {0, -1},
            {diagonal, diagonal}, {-diagonal, diagonal},
            {diagonal, -diagonal}, {-diagonal, -diagonal}
        };
        const auto& direction = directions[hash & 7u];
        return direction[0] * dx + direction[1] * dy;
    };

    // Quintic interpolation joins neighbouring cells with smooth derivatives.
    auto fade = [](double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); };
    const double sx = fade(fx), sy = fade(fy);
    const double top = std::lerp(gradient(ix, iy, fx, fy), gradient(ix + 1, iy, fx - 1.0, fy), sx);
    const double bottom = std::lerp(gradient(ix, iy + 1, fx, fy - 1.0), gradient(ix + 1, iy + 1, fx - 1.0, fy - 1.0), sx);
    return std::lerp(top, bottom, sy);
}

struct Fixture {
    fluid::ElementCatalogue catalogue;
    fluid::FluidMixture mixture;
    Fixture(double contrast=1.) {
        mixture.components={{catalogue.add({"Air"}),.1},{catalogue.add({"Cloud"}),.1*contrast}};
    }
    FluidGridAdaptive::State state(int n,double world,double speed,bool cloudy=true) const {
        FluidGridAdaptive::State s;
        const double h=world/n, pi=std::numbers::pi;
        s.gas_mass.assign(2,std::vector<double>(n*n));
        s.u.resize((n+1)*n); s.v.resize(n*(n+1));
        for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
            const double nx=(x+.5)*6./n, ny=(y+.5)*6./n;
            const double noise=(perlinNoise(nx,ny)+.5*perlinNoise(2*nx,2*ny)+.25*perlinNoise(4*nx,4*ny))/1.75;
            const double f=cloudy?std::clamp(.15+2.5*noise,0.,1.):.5;
            s.gas_mass[0][y*n+x]=mixture.components[0].density*h*h*(1-f);
            s.gas_mass[1][y*n+x]=mixture.components[1].density*h*h*f;
        }
        auto psi=[&](int x,int y) {
            if (x==0 || y==0 || x==n || y==n) return 0.;
            return speed*world/pi*std::pow(std::sin(pi*x/n)*std::sin(pi*y/n),2);
        };
        for (int y=0;y<n;++y) for (int x=0;x<=n;++x) s.u[y*(n+1)+x]=(psi(x,y+1)-psi(x,y))/h;
        for (int y=0;y<=n;++y) for (int x=0;x<n;++x) s.v[y*n+x]=-(psi(x+1,y)-psi(x,y))/h;
        return s;
    }
};
void liquidTranslation() {
    Fixture f;constexpr int n=64;
    const fluid::FluidMixture liquid{{{f.catalogue.add({"Water"}),100.}}};
    auto state=f.state(n,n,0.,false);
    state.liquid_mass.assign(1,std::vector<double>(n*n));state.liquid_fraction.resize(n*n);
    for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
        const bool wet=x>=24 && x<32;const int i=y*n+x;
        state.liquid_fraction[i]=wet;state.liquid_mass[0][i]=wet?100.:0.;
        if(wet)for(auto& mass:state.gas_mass)mass[i]=0.;
    }
    FluidGridAdaptive grid(n,n,1.,f.catalogue,liquid,f.mixture,{4,{{{8,0,48,n}},{{16,0,32,n}}}});
    grid.upload(state);const auto initial=grid.totals();
    // Prescribed interior translation isolates the geometric operator. Closed
    // end walls are far from the slab; their pure-gas split corrections cancel.
    for(const auto& level:grid.levels()) {
        const int end=level->geometry.Domain().bigEnd(0)+1;
        for(int patchIndex=0;patchIndex<level->velocity[0].size();++patchIndex) {
            auto u=level->velocity[0].array(patchIndex);
            fluid::forCells(level->velocity[0].box(patchIndex),[=](int i,int j) {u(i,j)=i==0 || i==end?0.:1.;});
        }
        for(int patchIndex=0;patchIndex<level->conserved.size();++patchIndex) {
            auto q=level->conserved.array(patchIndex);
            fluid::forCells(level->conserved.box(patchIndex),[=](int i,int j) {q(i,j,3)=q(i,j)+q(i,j,1)+q(i,j,2);});
        }
    }
    for(int step=0;step<8;++step)grid.transport(.25);
    fluid::FluidCellView view(grid);
    for(int y=0;y<n;++y)for(int x=0;x<n;++x)
        require(std::abs(view.at(x,y).fraction(0)-(x>=26 && x<34?1.:0.))<1e-11,"Geometric liquid slab smeared during exact translation");
    const auto total=grid.totals();
    for(size_t k=0;k<initial.ingredientMass.size();++k)
        require(std::abs(total.ingredientMass[k]/initial.ingredientMass[k]-1.)<1e-11,"Three-level phase transport lost ingredient mass");
    std::cout<<"liquid_translation,PASS,sharp_surface,shared_fluxes,conservation\n";
}
void liquidSliverRoundoff() {
    Fixture f;constexpr int n=32;
    const fluid::FluidMixture liquid{{{f.catalogue.add({"Water"}),100.}}};
    auto s=f.state(n,n,0.,false);
    // Liquid subtraction has a few ulps of error relative to the independently
    // stored gas mass, as it does after many conservative phase updates.
    s.liquid_mass.assign(1,std::vector<double>(n*n,100.*(1.-8e-16)));
    s.liquid_fraction.assign(n*n,1.-8e-16);
    for(auto& mass:s.gas_mass)std::fill(mass.begin(),mass.end(),.1*5e-18);
    FluidGridAdaptive grid(n,n,1.,f.catalogue,liquid,f.mixture,{});grid.upload(s);
    const auto initial=grid.totals();
    for(const auto& level:grid.levels())for(int patchIndex=0;patchIndex<level->velocity[0].size();++patchIndex) {
        auto u=level->velocity[0].array(patchIndex);
        fluid::forCells(level->velocity[0].box(patchIndex),[=](int i,int j){u(i,j)=i==0 || i==n?0.:1.;});
    }
    for(int step=0;step<16;++step) {
        grid.transport(.2);const auto total=grid.totals();
        require(total.finite && total.minMassDensity>=0.,"Minority-phase roundoff exported more than its mass");
        for(int k=1;k<3;++k)require(std::abs(total.ingredientMass[k]/initial.ingredientMass[k]-1.)<1e-12,"Tiny gas ingredient was discarded");
    }
    std::cout<<"liquid_sliver,PASS,positive_conservative_tiny_gas_without_clipping\n";
}
void transportTimeStep() {
    constexpr int n=64;
    constexpr double h=.5;
    // Two separated, elongated vortices put the largest X/Y speeds in
    // different places. Discrete streamfunction differences close every cell.
    auto psi=[](int x,int y) {
        auto vortex=[](int x,int y,int x0,int y0,int w,int h) {
            if(x<=x0 || x>=x0+w || y<=y0 || y>=y0+h)return 0.;
            return 30.*std::pow(std::sin(std::numbers::pi*(x-x0)/w)*std::sin(std::numbers::pi*(y-y0)/h),2);
        };
        return vortex(x,y,6,10,24,8)+vortex(x,y,42,32,8,24);
    };
    for(bool twoPhase:{false,true})for(int ratio:{1,4}) {
        Fixture f(100.);
        const fluid::FluidMixture liquid=twoPhase ? fluid::FluidMixture{{{f.catalogue.add({"Water"}),100.}}} : fluid::FluidMixture{};
        auto state=f.state(n,n*h,0.);
        if(twoPhase) {
            state.liquid_mass.assign(1,std::vector<double>(n*n));state.liquid_fraction.resize(n*n);
            for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
                const int i=y*n+x;const double fraction=std::clamp(y+1.-(.25*n+.45*x),0.,1.);
                state.liquid_fraction[i]=fraction;state.liquid_mass[0][i]=100.*h*h*fraction;
                for(auto& mass:state.gas_mass)mass[i]*=1.-fraction;
            }
        }
        FluidGridAdaptive grid(n,n,h,f.catalogue,liquid,f.mixture,{ratio,{}},16);grid.upload(state);
        require(std::isinf(grid.stableTimeStep()),"Still transport should need no CFL subdivisions");
        auto& level=*grid.levels()[0];
        for(int d=0;d<2;++d)for(int patchIndex=0;patchIndex<level.velocity[d].size();++patchIndex) {
            const auto u=level.velocity[d].array(patchIndex);
            fluid::forCells(level.velocity[d].box(patchIndex),[=](int i,int j) {
                u(i,j)=(d==0 ? psi(i*ratio,(j+1)*ratio)-psi(i*ratio,j*ratio) :
                    psi(i*ratio,j*ratio)-psi((i+1)*ratio,j*ratio))/(ratio*h);
            });
        }
        require(grid.maxDivergence()<1e-11,"CFL fixture must have projected faces");
        const double oldLimit=.45*ratio*h/(level.velocity[0].norm0(0)+level.velocity[1].norm0(0));
        const double dt=grid.stableTimeStep();
        require(dt>1.2*oldLimit,"Distant velocity extrema still restrict the local timestep");
        for(auto& velocity:level.velocity)velocity.multiply(4.,0,1);
        require(std::abs(grid.stableTimeStep()*4./dt-1.)<1e-12,"Transport limit must scale with actual face speed");
        for(auto& velocity:level.velocity)velocity.multiply(.25,0,1);
        const auto initial=grid.totals();
        for(int step=0;step<24;++step) {
            grid.transport(dt);const auto total=grid.totals();
            require(total.finite && total.minMassDensity>=-1e-12,"Local CFL lost material positivity");
            require(total.maxVolumeError<1e-10,"Local CFL lost volume closure");
            for(size_t k=0;k<initial.ingredientMass.size();++k)
                require(std::abs(total.ingredientMass[k]/initial.ingredientMass[k]-1.)<1e-11,"Local CFL lost ingredient mass");
        }
        std::cout<<"transport_cfl,phases="<<(twoPhase?2:1)<<",cell_ratio="<<ratio<<",limit_gain="<<dt/oldLimit<<",PASS\n";
    }
    Fixture f;
    FluidGridAdaptive outgoing(n,n,h,f.catalogue,f.mixture,{},16);
    outgoing.upload(f.state(n,n*h,0.));
    // A deliberately non-projected field checks that all four outgoing faces
    // count, even when signs oppose. Do not advance this diagnostic field.
    for(int d=0;d<2;++d)for(int patchIndex=0;patchIndex<outgoing.levels()[0]->velocity[d].size();++patchIndex) {
        const auto u=outgoing.levels()[0]->velocity[d].array(patchIndex);
        fluid::forCells(outgoing.levels()[0]->velocity[d].box(patchIndex),[=](int i,int j) {
            u(i,j)=((d==0 && j==16 && i==17) || (d==1 && i==16 && j==17)) ? 8. :
                    (i==16 && j==16 ? -8. : 0.);
        });
    }
    require(std::abs(outgoing.stableTimeStep()/(.45*h/32.)-1.)<1e-12,"Opposing outgoing faces were cancelled or omitted");
    FluidGridAdaptive joined(n,n,h,f.catalogue,f.mixture,{2,{{{16,16,32,32}}}},16);
    joined.upload(f.state(n,n*h,0.));
    auto& fine=*joined.levels()[1];
    // Two opposing fine faces have zero signed coarse average. Preserve the
    // old join restriction rather than incorrectly treating that face as still.
    for(int patchIndex=0;patchIndex<fine.velocity[0].size();++patchIndex) {
        const auto u=fine.velocity[0].array(patchIndex);
        fluid::forCells(fine.velocity[0].box(patchIndex),[=](int i,int j) {u(i,j)=i==16?(j%2?12.:-12.):0.;});
    }
    require(std::abs(joined.stableTimeStep()/(.45*h/12.)-1.)<1e-12,"Coarse/fine reconstruction protection changed");
    std::cout<<"transport_cfl,opposing_faces_and_join_guard,PASS\n";
}
void subcycledLiquidStep() {
    Fixture f; constexpr int n=64;
    const fluid::FluidMixture liquid{{{f.catalogue.add({"Water"}),100.}}};
    auto s=f.state(n,n,12.,false);
    s.liquid_mass.assign(1,std::vector<double>(n*n));s.liquid_fraction.resize(n*n);
    for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
        const int i=y*n+x;const double fraction=std::clamp(y+1.-(.45*n+.2*x),0.,1.);
        s.liquid_fraction[i]=fraction;s.liquid_mass[0][i]=100.*fraction;
        for(auto& mass:s.gas_mass)mass[i]*=1.-fraction;
    }
    for(const fluid::GasLayout layout:{fluid::GasLayout{},fluid::GasLayout{4,{{{8,8,48,48}},{{16,16,32,32}}}}}) {
        FluidGridAdaptive grid(n,n,1.,f.catalogue,liquid,f.mixture,layout);grid.upload(s);
        const auto initial=grid.totals();
        constexpr double dt=.12,tolerance=1e-9;
        require(dt>2.*grid.stableTimeStep(),"Subcycled liquid fixture needs several transport steps");
        int transportSteps=0;
        for(int i=0;i<16;++i) {
            const auto result=grid.step(dt,tolerance,.2,{0.,10.});
            transportSteps+=result.substeps;
            require(result.projection.converged,"Subcycled liquid projection failed");
            require(grid.pressureInterval()==dt,"Pressure impulse does not represent the physical update interval");
            const auto total=grid.totals();
            require(total.finite && total.minMassDensity>=-1e-12,"Frozen-face liquid transport produced invalid material");
            require(total.maxVolumeError<1e-8,"Frozen-face liquid transport lost volume closure");
            for(size_t k=0;k<initial.ingredientMass.size();++k)
                require(std::abs(total.ingredientMass[k]/initial.ingredientMass[k]-1.)<1e-8,"Subcycled liquid transport lost ingredient mass");
        }
        require(transportSteps>32,"Subcycled liquid fixture did not exercise frozen-face transport");
        const auto before=grid.totals();
        require(grid.step(0.,tolerance,.2,{0.,10.}).substeps==0,"Zero timestep advanced transport");
        require(grid.totals().ingredientMass==before.ingredientMass && grid.totals().momentum==before.momentum,
            "Zero timestep changed conserved state");
        std::cout<<"liquid_subcycling,levels="<<grid.levels().size()<<",transport_steps="<<transportSteps<<",physical_steps=16,PASS\n";
    }
}
void liquidSelectionAndTransport(int steps=100) {
    Fixture f;
    const fluid::FluidMixture liquid{{{f.catalogue.add({"Water"}),100.},{f.catalogue.add({"Dense liquid"}),300.}}};
    constexpr int n=128;
    auto s=f.state(n,n,0.,false);
    s.liquid_mass.assign(2,std::vector<double>(n*n));s.liquid_fraction.resize(n*n);
    for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
        const double fraction=std::clamp(y+1.-(.65*n+.35*x),0.,1.);
        const int i=y*n+x;s.liquid_fraction[i]=fraction;
        s.liquid_mass[0][i]=100.*fraction*.75;s.liquid_mass[1][i]=300.*fraction*.25;
        for(auto& gas:s.gas_mass)gas[i]*=1.-fraction;
    }
    FluidGridAdaptive grid(n,n,1.,f.catalogue,liquid,f.mixture,{});grid.upload(s);
    const auto initial=grid.totals();Preview preview;bool coarsened=false;
    for(int step=0;step<steps;++step) {
        if(step%10==0) {
            preview.update(grid,{},{1.,0.},step*.008,.16,4);
            if(grid.regrid(preview.proposeLayout(grid)))require(grid.project(1e-9).converged,"Liquid remap projection failed");
            coarsened|=grid.layout().baseRatio>1;
            const fluid::FluidCellView view(grid);
            for(const auto& p:view.patches)for(int y=p.y;y<p.y+p.height;y+=p.scale)for(int x=p.x;x<p.x+p.width;x+=p.scale)
                if(view.at(x,y).patch==&p && view.at(x,y).liquid())require(p.scale==1,"Liquid lost finest resolution");
        }
        require(grid.step(.008,1e-9,.2,{0.,10.}).projection.converged,"Coupled liquid/gas step failed");
        const auto total=grid.totals();
        require(total.finite && total.minMassDensity>=-1e-12,"Liquid transport produced invalid material");
        require(total.maxVolumeError<1e-8,"Liquid/gas volume closure failed");
        for(size_t k=0;k<total.ingredientMass.size();++k)
            require(std::abs(total.ingredientMass[k]/initial.ingredientMass[k]-1.)<1e-8,"Liquid/gas conservation failed");
    }
    require(coarsened,"Liquid fixture never exercised coarse/fine coupling");
    std::cout<<"liquid_amr,PASS,geometric_transport,multiple_ingredients,1000_to_3000_density_contrast,gravity,conservation,fine_liquid\n";
}
int fineArea(const fluid::GasLayout& layout,int n) {
    if (layout.baseRatio==1) return n*n;
    if ((layout.baseRatio >> layout.refinedPatches.size())!=1) return 0;
    int area=0; for (const auto& p:layout.refinedPatches.back()) area+=p.width*p.height; return area;
}
bool fineAt(const fluid::GasLayout& layout,int x,int y) {
    if (layout.baseRatio==1) return true;
    if ((layout.baseRatio >> layout.refinedPatches.size())!=1) return false;
    for (const auto& p:layout.refinedPatches.back()) if (x>=p.x && x<p.x+p.width && y>=p.y && y<p.y+p.height) return true;
    return false;
}
void selectionTraceIsObservation(int maximum=4) {
    Fixture f; constexpr int n=128;
    FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,{});
    auto state=f.state(n,n,.25,false);
    for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
        const double cloud=std::hypot(x-40.,y-64.)<6?.9:.1;
        state.gas_mass[0][y*n+x]=.1*(1-cloud); state.gas_mass[1][y*n+x]=.1*cloud;
    }
    grid.upload(state);
    Preview plain,observed; Preview::Trace trace;
    bool merged=false, sawWaiting=false, sawDetail=false;
    for (int i=0;i<32;++i) {
        plain.update(grid,{},{},i*.08,.4,maximum);
        observed.update(grid,{},{},i*.08,.4,maximum,&trace);
        require(plain.labels==observed.labels && plain.statistics==observed.statistics,"Tracing changed selection/history");
        const auto layout=plain.proposeLayout(grid);
        require(layout==observed.proposeLayout(grid,&trace),"Tracing changed proposed patches");
        require((2<<trace.levels.size())==2*maximum,"Trace lost assessment scales");
        for (const auto& level:trace.levels)
            require(level.tile==std::max(16,level.scale) && level.nested.size()==size_t(level.tiles_x)*level.tiles_y,
                "Trace lost per-level patch tile units");
        for (const auto& level:trace.levels) for (const auto& c:level.cells) {
            sawWaiting|=c.raw==Preview::Candidate && c.decision==Preview::Detail;
            sawDetail|=c.evaluated && c.error.score({})>1;
        }
        if (grid.regrid(layout)) { merged=true; require(grid.project(1e-9).converged,"Trace fixture remap failed"); }
        require(grid.step(.08,1e-9).projection.converged,"Trace fixture evolution failed");
    }
    require(merged && sawWaiting && sawDetail,"Trace fixture missed merging, dwell or error requests");
    std::cout<<"selection_trace,PASS,identical_proposals_and_history\n";
}
void selectionCases() {
    Fixture f; constexpr int n=96;
    FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,{}); auto state=f.state(n,n,0.,false); grid.upload(state);
    Preview p; Preview::Tolerances t; Preview::Stability stability;
    p.update(grid,t,stability,0,.4,2);
    require(p.proposeLayout(grid).baseRatio==1,"Initial quiet dwell must retain fine data");
    p.update(grid,t,stability,.32,.4,2);
    require(p.counts[Preview::Candidate]==n*n/4,"Quiet physical and FAB boundaries must be coarsenable");
    require(p.proposeLayout(grid)==fluid::GasLayout{2,{}},"Uniform quiet field must become wholly coarse");
    // A diagonal stationary material band crossing tile/FAB edges, including walls.
    for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
        const double share=std::abs(x-y)<2 ? .9 : .1;
        state.gas_mass[0][y*n+x]=.1*(1-share); state.gas_mass[1][y*n+x]=.1*share;
    }
    grid.upload(state); p.reset(); p.update(grid,t,{1.,0.},0,.4,2);
    auto proposed=p.proposeLayout(grid); int area=fineArea(proposed,n);
    require(area>0 && area<n*n,"Diagonal detail should leave some coarsened regions");
    for (int y=0;y<p.height;++y) for (int x=0;x<p.width;++x)
        if (p.labels[y*p.width+x]!=Preview::Candidate) require(fineAt(proposed,2*x,2*y),"Cluster dropped a fine request");
    grid.regrid(proposed); require(grid.project(1e-9).converged,"Diagonal remap projection failed");
    p.reset(); p.update(grid,t,stability,0,.4,2); p.update(grid,t,stability,.32,.4,2);
    const auto again=p.proposeLayout(grid);
    require(fineArea(again,n)<n*n,"Coarse/fine boundary must not force blanket refinement");
    // New resolved detail in an existing coarse region is observable without hidden fine samples.
    grid.regrid({2,{}}); grid.upload(f.state(n,n,0.,false));
    for (const auto& level:grid.levels()) for(int fab=0;fab<level->conserved.size();++fab) {
        auto q=level->conserved.array(fab); const auto b=level->conserved.box(fab);
        for (int y=b.smallEnd(1);y<=b.bigEnd(1);++y) for (int x=b.smallEnd(0);x<=b.bigEnd(0);++x)
            if (x>=20 && x<24 && y>=20 && y<24) { q(x,y,0)=.01; q(x,y,1)=.09; }
    }
    p.reset(); p.update(grid,t,stability,0,.4,2);
    require(p.counts[Preview::Refine]>0 && fineArea(p.proposeLayout(grid),n)>0,"New coarse material feature must request fine cells");
    // No-op proposals stay no-ops, and a relaxed tolerance can release the feature.
    grid.regrid(p.proposeLayout(grid)); grid.project(1e-9);
    t.ingredient=.5; p.update(grid,t,stability,.4,.4,2); p.update(grid,t,stability,.8,.4,2);
    require(p.proposeLayout(grid)==fluid::GasLayout{2,{}},"Relaxed tolerance and dwell should release all refinement");
    require(grid.regrid(p.proposeLayout(grid)),"Release should change the layout"); grid.project(1e-9);
    p.update(grid,t,stability,1.2,.4,2);
    require(!grid.regrid(p.proposeLayout(grid)),"Unchanged quiet selection must not rebuild");
    std::cout<<"selection,PASS,diagonal_fine_fraction="<<double(area)/(n*n)<<'\n';
}
void staticSelectionSettles() {
    Fixture f;
    for(int n:{512,1024}) {
    FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,{}); grid.upload(f.state(n,n,0.));
    Preview p; int lateChanges=0,changes=0;
    for (int i=0;i<20;++i) {
        p.update(grid,{},{},i*.32,.4,2);
        if (grid.regrid(p.proposeLayout(grid))) {
            grid.project(1e-9); ++changes; if (i>=10) ++lateChanges;
        }
    }
    std::cout<<"stationary,n="<<n<<",changes="<<changes<<",late_changes="<<lateChanges<<'\n';
    require(lateChanges==0,"Stationary field must not keep merging and refining the same features");
    }
}
void materialSelectionConsistency() {
    Fixture f; constexpr int n=96;
    // Smooth interior affine gradients are exactly represented. Reflecting
    // walls can flatten the normal slope and legitimately retain fine cells.
    for (const fluid::GasLayout initial: {fluid::GasLayout{},fluid::GasLayout{2,{}},fluid::GasLayout{2,{{{32,32,32,32}}}}}) {
        auto state=f.state(n,n,0.,false);
        for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
            const double share=.1+.004*(x+y+1);
            state.gas_mass[0][y*n+x]=.1*(1-share); state.gas_mass[1][y*n+x]=.1*share;
        }
        FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,initial); grid.upload(state);
        Preview p; Preview::Tolerances t; t.ingredient=1e-10;
        p.update(grid,t,{1.,0.},0,0,2);
        for(int y=2;y<p.height-2;++y) for(int x=2;x<p.width-2;++x)
            require(p.labels[y*p.width+x]==Preview::Candidate || p.labels[y*p.width+x]==Preview::Coarse,"Affine material should be represented through FAB/level joins");
    }
    // A step aligned exactly with a coarse face has no internal fine variation,
    // but its resolved neighbour difference must prevent an immediate re-refine.
    for(bool startCoarse:{false,true}) {
        auto state=f.state(n,n,0.,false);
        for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
            const double share=x<n/2?.1:.9;
            state.gas_mass[0][y*n+x]=.1*(1-share); state.gas_mass[1][y*n+x]=.1*share;
        }
        FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,startCoarse?fluid::GasLayout{2,{}}:fluid::GasLayout{}); grid.upload(state);
        Preview p; int lateChanges=0;
        for(int check=0;check<32;++check) {
            p.update(grid,{},{},check*.32,.4,2);
            const auto layout=p.proposeLayout(grid);
            require(fineAt(layout,n/2-2,n/2) && fineAt(layout,n/2,n/2),"An aligned resolved edge must retain/request refinement on both sides");
            if(grid.regrid(layout)) { grid.project(1e-9); if(check>=8) ++lateChanges; }
        }
        require(lateChanges==0,"An aligned stationary material step must not oscillate between coarse and fine");
        require(fineArea(grid.layout(),n)<n*n,"Protecting a material step must leave its uniform surroundings coarse");
    }
    std::cout<<"material_selection,PASS,affine_resolution_and_wall_parity,aligned_step_settles\n";
}
void paddingDoesNotPropagate() {
    Fixture f; constexpr int n=128;
    const fluid::GasLayout layouts[]={{2,{{{48,48,16,16}}}},{2,{{{0,0,16,16},{96,96,16,16}}}}};
    for(const auto& initial:layouts) {
        FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,initial); grid.upload(f.state(n,n,0.,false));
        Preview p;
        // Exercise repeated checks within and beyond the quiet dwell. No flow
        // or material feature can justify spreading these initial fine patches.
        for(int i=0;i<=40;++i) {
            p.update(grid,{},{},i*.08,.4,2);
            const auto proposal=p.proposeLayout(grid);
            require(fineArea(proposal,n)<=fineArea(grid.layout(),n),"Quiet retention must not grow fine coverage");
            for(int y=0;y<n;y+=2) for(int x=0;x<n;x+=2)
                require(!fineAt(proposal,x,y) || fineAt(grid.layout(),x,y),"Quiet padding must not move into coarse cells");
            if(i==0) require(proposal==initial,"New fine cells still receive their quiet dwell");
            if(i>=4) require(proposal==fluid::GasLayout{2,{}},"Quiet patches must release after the dwell");
            if(grid.regrid(proposal)) require(grid.project(1e-9).converged,"Quiet remap projection failed");
        }
    }
    // A real feature still gets padding, and newly padded quiet cells cannot
    // propagate that padding after the feature has disappeared.
    const fluid::GasLayout initial{2,{{{48,48,16,16}}}};
    FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,initial); auto state=f.state(n,n,0.,false);
    for(int y=48;y<64;++y) for(int x=48;x<64;++x) {
        const double share=x%2?.9:.1;
        state.gas_mass[0][y*n+x]=.1*(1-share); state.gas_mass[1][y*n+x]=.1*share;
    }
    grid.upload(state); Preview p; p.update(grid,{},{},0,.4,2);
    const auto buffered=p.proposeLayout(grid);
    require(fineAt(buffered,46,54) && fineAt(buffered,64,54),"Measured fine detail must retain its spatial buffer");
    grid.regrid(buffered); grid.project(1e-9); grid.upload(f.state(n,n,0.,false));
    for(int i=1;i<=32;++i) {
        p.update(grid,{},{},i*.32,.4,2); const auto proposal=p.proposeLayout(grid);
        for(int y=0;y<n;y+=2) for(int x=0;x<n;x+=2)
            require(!fineAt(proposal,x,y) || fineAt(buffered,x,y),"Former feature padding must not reproduce itself");
        if(i>=2) require(proposal==fluid::GasLayout{2,{}},"Former detail and padding must settle to coarse");
        if(grid.regrid(proposal)) grid.project(1e-9);
    }
    std::cout<<"padding,PASS,quiet_retention_without_growth,real_detail_buffered\n";
}
void reconstructionContract() {
    constexpr int offsets[5][2]={{0,0},{-1,0},{1,0},{0,-1},{0,1}};
    for(double contrast:{1.,10.,1000.}) for(int tick=0;tick<64;++tick) for(bool constant:{false,true}) {
        std::array<std::array<double,6>,5> values{};
        std::array<double,2> low{1e30,1e30},high{-1e30,-1e30};
        for(int side=0;side<5;++side) {
            const int x=offsets[side][0],y=offsets[side][1];
            const double share=.5+.06*x+.04*y;
            values[side][2]=.1*(1-share);values[side][3]=.1*contrast*share;
            const double rho=values[side][2]+values[side][3];
            for(int d=0;d<2;++d) {
                const double velocity=(d?-.3:.7)+(constant?0.:.4*std::sin(.17*tick+(d?y:x)*.8));
                values[side][4+d]=rho*velocity;low[d]=std::min(low[d],velocity);high[d]=std::max(high[d],velocity);
            }
        }
        auto sample=[&](int x,int y,int n){return values[x<0?1:x>0?2:y<0?3:y>0?4:0][n];};
        const auto limiter=fluid::linearLimiter(4,sample);
        require(std::abs(limiter.x-1.)<1e-13 && std::abs(limiter.y-1.)<1e-13,"Moving momentum extrema flattened smooth material");
        const auto momentum=fluid::linearMomentumSlopes(4,sample,limiter);
        std::array<double,6> average{};
        for(double y:{-.5,-.25,.25,.5})for(double x:{-.5,-.25,.25,.5}) {
            double rho=0.,volume=0.;
            for(int n=0;n<4;++n) {
                const auto slope=fluid::linearSlope(n,sample,limiter);const double q=sample(0,0,n)+x*slope.x+y*slope.y;
                average[n]+=q/16.;rho+=q;volume+=q/(n<2?100.:n==2?.1:.1*contrast);
                require(q>=0.,"Reconstruction made a negative ingredient");
            }
            require(std::abs(volume-1.)<1e-13,"Reconstruction broke the mixture volume identity");
            for(int d=0;d<2;++d) {
                const double m=sample(0,0,4+d)+x*momentum[d].x+y*momentum[d].y;average[4+d]+=m/16.;
                require(m/rho>=low[d]-1e-13 && m/rho<=high[d]+1e-13,"Reconstruction overshot neighbouring velocity bounds");
                if(constant)require(std::abs(m/rho-(d?-.3:.7))<1e-13,"Refinement changed constant velocity across density variation");
            }
        }
        for(int n=0;n<6;++n)require(std::abs(average[n]-sample(0,0,n))<1e-12,"Reconstruction changed a conserved cell average");
    }
    std::cout<<"reconstruction,PASS,moving_extrema,volume,momentum,constant_velocity,velocity_bounds\n";
}
void representationContract() {
    constexpr int n=64;
    for(int scale:{2,4,8,16,32}) for(double contrast:{1.,10.}) for(int pattern=0;pattern<4;++pattern) {
        Fixture f(contrast); auto state=f.state(n,n,0.,false);
        for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
            const double share=pattern==0?.5:pattern==1?.1+.006*(x+y+1):pattern==2?.5+.3*std::sin(.21*x)*std::cos(.17*y):(x+y)%2?.9:.1;
            state.gas_mass[0][y*n+x]=.1*(1-share); state.gas_mass[1][y*n+x]=.1*contrast*share;
        }
        FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,{}); grid.upload(state);
        std::vector<Preview::Error> estimates;
        for(int y=0;y<n;y+=scale) for(int x=0;x<n;x+=scale) estimates.push_back(Preview::estimate(grid,x,y,scale));
        grid.regrid({scale,{}});
        fluid::FluidCellView coarse(grid);
        double underestimate=0.,excess=0.,materialLoss=0.;
        std::vector<double> displayed(2*n*n);
        for(int y=0;y<n;y+=scale) for(int x=0;x<n;x+=scale) {
            double loss=0.,densityLoss=0.,meanDensity=0.;
            for(int dy=0;dy<scale;++dy) for(int dx=0;dx<scale;++dx) {
                const int i=(y+dy)*n+x+dx;
                double densityError=0.;
                for(int k=0;k<2;++k) {
                    displayed[2*i+k]=coarse.materialSample(x+dx,y+dy,k)*f.mixture.components[k].density;
                    const double error=displayed[2*i+k]-state.gas_mass[k][i];
                    loss=std::max(loss,std::abs(error)/f.mixture.components[k].density);
                    densityError+=error; meanDensity+=state.gas_mass[k][i]/(scale*scale);
                }
                densityLoss=std::max(densityLoss,std::abs(densityError));
            }
            const auto e=estimates[(y/scale)*(n/scale)+x/scale];
            underestimate=std::max({underestimate,loss-e.ingredient,densityLoss/meanDensity-e.density});
            excess=std::max({excess,e.ingredient-loss,e.density-densityLoss/meanDensity});
            materialLoss=std::max(materialLoss,loss);
        }
        const int child=scale/2;
        grid.regrid({child,{}});
        fluid::FluidCellView refined(grid);
        for (int y=0;y<n;y+=child) for (int x=0;x<n;x+=child) for (int k=0;k<2;++k) {
            double expected=0.;
            for (int dy=0;dy<child;++dy) for (int dx=0;dx<child;++dx) expected+=displayed[2*((y+dy)*n+x+dx)+k]/(child*child);
            require(std::abs(refined.at(x,y).density(k)-expected)<1e-12,"Display and adjacent-level refinement must reconstruct the same physical material");
        }
        std::cout<<"representation,scale="<<scale<<",contrast="<<contrast<<",pattern="<<pattern<<",underestimate="<<underestimate
            <<",neighbour_excess="<<excess<<",reconstructed_material_loss="<<materialLoss<<'\n';
        require(underestimate<1e-12,"Material indicator must cover actual loss to the bounded coarse plane");
        if(pattern==0 || pattern==3) require(excess<1e-12,"Constant and alternating material losses should be exact");
        if(pattern==1) require(std::abs(materialLoss-.006*(scale-1))<1e-12,"Reflecting corners retain the expected affine loss");
    }
}
void threeSizeSelection() {
    Fixture f; constexpr int n=192;
    auto state=f.state(n,n,0.,false);
    for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
        double share=.1+.4*std::exp(-((x-144.5)*(x-144.5)+(y-144.5)*(y-144.5))/32.);
        if (x>=32 && x<48 && y>=32 && y<48) share=(x+y)%2?.9:.1;
        state.gas_mass[0][y*n+x]=.1*(1-share); state.gas_mass[1][y*n+x]=.1*share;
    }
    FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,{}); grid.upload(state);
    Preview p; p.update(grid,{},{1.,0.},0,.4,4);
    const auto layout=p.proposeLayout(grid);
    require(layout.baseRatio==4 && layout.refinedPatches.size()==2,"Separated detail must produce all three physical sizes");
    grid.regrid(layout); require(grid.project(1e-9).converged,"Three-size selection projection failed");
    fluid::FluidCellView view(grid);
    require(view.at(40,40).patch->scale==1 && view.at(144,144).patch->scale==2 && view.at(96,16).patch->scale==4,
        "Fine detail, broader curvature and quiet space must choose appropriate sizes");
    // Every finest patch retains the actual 2x2 stencil required by FillPatch.
    for (const auto& patch:layout.refinedPatches.back())
        for (int y=std::max(0,patch.y-4);y<std::min(n,patch.y+patch.height+4);y+=2)
            for (int x=std::max(0,patch.x-4);x<std::min(n,patch.x+patch.width+4);x+=2)
                require(view.at(x,y).patch->scale<=2,"Fine patch lacks nested coarse stencil");
    // A known finest feature can travel beyond its current intermediate parent.
    // Its swept request must remain finest even where the current cells are 4x4.
    for (const auto& level:grid.levels()) { level->velocity[0].setVal(10.); level->velocity[1].setVal(0.); }
    p.reset(); p.update(grid,{},{1.,0.},0,4.,4);
    require(view.at(96,40).patch->scale==4 && fineAt(p.proposeLayout(grid),96,40),"Lookahead lost a finest request inside a 4x4 region");
    grid.upload(f.state(n,n,0.,false)); p.reset();
    for (int i=0;i<8;++i) {
        p.update(grid,{},{},i*.32,.4,4);
        if (grid.regrid(p.proposeLayout(grid))) require(grid.project(1e-9).converged,"Quiet release projection failed");
    }
    require(grid.layout()==fluid::GasLayout{4,{}},"Quiet nested patches must disappear without padding feedback");
    grid.upload(state); p.reset(); p.update(grid,{},{},0,.4,4);
    require(!p.proposeLayout(grid).refinedPatches.empty(),"Newly resolved coarse curvature must request 2x2 cells");
    std::cout<<"three_sizes,PASS,detail_scale_1,curvature_scale_2,quiet_scale_4,nesting_and_release\n";
}
void fourSizeSelection() {
    Fixture f; constexpr int n=256;
    auto state=f.state(n,n,0.,false);
    for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
        double share=.1+.4*std::exp(-((x-192.5)*(x-192.5)+(y-48.5)*(y-48.5))/32.)
                        +.4*std::exp(-((x-192.5)*(x-192.5)+(y-192.5)*(y-192.5))/256.);
        if(x>=32 && x<48 && y>=32 && y<48)share=(x+y)%2?.9:.1;
        state.gas_mass[0][y*n+x]=.1*(1-share);state.gas_mass[1][y*n+x]=.1*share;
    }
    FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,{});grid.upload(state);
    const auto initial=grid.totals(); Preview p; Preview::Trace trace;
    p.update(grid,{},{1.,0.},0,.4,8,&trace);
    const auto layout=p.proposeLayout(grid,&trace);
    require(layout.baseRatio==8 && layout.refinedPatches.size()==3,"Separated features must select all four sizes");
    grid.regrid(layout);
    require(grid.project(1e-9).converged,"Four-size projection failed");
    const fluid::FluidCellView view(grid);
    std::cout<<"four_size_samples,"<<view.at(40,40).patch->scale<<','<<view.at(192,48).patch->scale<<','
             <<view.at(192,192).patch->scale<<','<<view.at(80,160).patch->scale<<'\n';
    require(view.at(40,40).patch->scale==1 && view.at(192,48).patch->scale==2 &&
            view.at(192,192).patch->scale==4 && view.at(80,160).patch->scale==8,"Four-size feature selection lost its scale");
    for(size_t lev=0;lev<layout.refinedPatches.size();++lev) {
        const int parent=layout.baseRatio>>lev,buffer=2*parent;
        for(const auto& patch:layout.refinedPatches[lev])
            for(int y=std::max(0,patch.y-buffer);y<std::min(n,patch.y+patch.height+buffer);y+=2)
                for(int x=std::max(0,patch.x-buffer);x<std::min(n,patch.x+patch.width+buffer);x+=2)
                    require(view.at(x,y).patch->scale<=parent,"Child patch lacks a valid nested parent stencil");
    }
    // A finest feature can request cells beyond both intermediate levels.
    for(const auto& level:grid.levels()){level->velocity[0].setVal(10.);level->velocity[1].setVal(0.);}
    p.reset();p.update(grid,{},{1.,0.},0,7.,8);
    require(view.at(112,40).patch->scale==8 && fineAt(p.proposeLayout(grid),112,40),"Lookahead lost a finest request inside 8x8 cells");
    grid.upload(f.state(n,n,0.,false));p.reset();
    for(int i=0;i<8;++i) {
        p.update(grid,{},{},i*.32,.4,8);
        if(grid.regrid(p.proposeLayout(grid)))require(grid.project(1e-9).converged,"Four-size quiet release failed");
    }
    require(grid.layout()==fluid::GasLayout{8,{}},"Quiet four-level hierarchy must release without padding feedback");
    // Exercise the multi-level refinement bridge, then ordinary coupled steps.
    grid.upload(state);grid.regrid({});grid.project(1e-9);grid.regrid(layout);grid.project(1e-9);
    for(int i=0;i<16;++i)require(grid.step(.016,1e-9,0.,{.1,-.07}).projection.converged,"Four-level coupled transport failed");
    const auto totals=grid.totals();
    for(size_t k=0;k<initial.ingredientMass.size();++k)
        require(std::abs(totals.ingredientMass[k]/initial.ingredientMass[k]-1)<1e-11,"Four-level transfer or transport lost mass");
    require(totals.finite && totals.minMassDensity>=0,"Four-level state invalid");
    std::cout<<"four_sizes,PASS,selection,nesting,lookahead,release,8_to_1_bridge,conservative_evolution\n";
}
void deeperSelection() {
    Fixture f; constexpr int n=512;
    for (int maximum:{16,32}) {
        auto state=f.state(n,n,.25,false);
        for (int y=80;y<96;++y) for (int x=80;x<96;++x) {
            const double share=(x+y)%2?.9:.1;
            state.gas_mass[0][y*n+x]=.1*(1-share);state.gas_mass[1][y*n+x]=.1*share;
        }
        FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,{});grid.upload(state);
        const auto initial=grid.totals();Preview p;Preview::Trace trace;
        p.update(grid,{},{1.,0.},0,.4,maximum,&trace);
        const auto layout=p.proposeLayout(grid,&trace);
        require(layout.baseRatio==maximum && (maximum>>layout.refinedPatches.size())==1,"Deep selection lost finest or coarsest level");
        grid.regrid(layout);require(grid.project(1e-9).converged,"Deep selection projection failed");
        const fluid::FluidCellView view(grid);
        require(view.at(88,88).patch->scale==1 && view.at(n-32,n-32).patch->scale==maximum,"Deep selection lost localized detail or quiet background");
        for(size_t lev=0;lev<layout.refinedPatches.size();++lev) {
            const int parent=maximum>>lev;
            bool used=false;
            for(const auto& patch:view.patches) if(patch.scale==parent)
                for(int y=patch.y;y<patch.y+patch.height;y+=parent)for(int x=patch.x;x<patch.x+patch.width;x+=parent)
                    used|=view.at(x,y).patch->scale==parent;
            require(used,"Deep hierarchy skipped an active intermediate resolution");
            for(const auto& patch:layout.refinedPatches[lev]) {
                require(patch.x%parent==0 && patch.y%parent==0 && patch.width%parent==0 && patch.height%parent==0,
                    "Refined patch does not align with whole parent cells");
                for(int y=std::max(0,patch.y-2*parent);y<std::min(n,patch.y+patch.height+2*parent);y+=parent)
                    for(int x=std::max(0,patch.x-2*parent);x<std::min(n,patch.x+patch.width+2*parent);x+=parent)
                        require(view.at(x,y).patch->scale<=parent,"Deep child patch lacks its parent interpolation stencil");
            }
        }
        for(const auto& level:grid.levels()){level->velocity[0].setVal(10.);level->velocity[1].setVal(0.);}
        p.reset();p.update(grid,{},{1.,0.},0,24.,maximum);
        require(view.at(320,88).patch->scale==maximum && fineAt(p.proposeLayout(grid),320,88),"Finest lookahead request lost across deep levels");
        grid.upload(f.state(n,n,0.,false));p.reset();
        for(int i=0;i<8;++i) {
            p.update(grid,{},{},i*.32,.4,maximum);
            if(grid.regrid(p.proposeLayout(grid)))require(grid.project(1e-9).converged,"Deep quiet release failed");
        }
        require(grid.layout()==fluid::GasLayout{maximum,{}},"Quiet deep hierarchy did not release");
        // Reconstruct through every missing intermediate level, then evolve
        // an actual mixed hierarchy rather than testing only uniform coarse.
        grid.upload(state);grid.regrid({});grid.project(1e-9);grid.regrid(layout);grid.project(1e-9);
        for(int i=0;i<8;++i)require(grid.step(.016,1e-9,0.,{.1,-.07}).projection.converged,"Deep coupled transport failed");
        const auto totals=grid.totals();
        for(size_t k=0;k<initial.ingredientMass.size();++k)
            require(std::abs(totals.ingredientMass[k]/initial.ingredientMass[k]-1)<1e-11,"Deep transfer or transport lost mass");
        require(totals.finite && totals.minMassDensity>=0 && totals.maxVolumeError<1e-8,"Deep state invalid");

        // Only the first quadrant changes; a truncated 64-bit footprint can
        // miss this while the later samples remain identical.
        FluidGridAdaptive history(n,n,1.,f.catalogue,f.mixture,{2,{{{64,64,16,16}}}});history.upload(f.state(n,n,0.,false));
        p.reset();p.update(history,{},{1.,.24},0,0,maximum);
        p.update(history,{},{1.,.24},.32,0,maximum,&trace);
        const int changed=(64/maximum)*trace.levels.back().width+64/maximum,unchanged=changed+1;
        require(trace.levels.back().cells[changed].decision==Preview::Candidate,"History fixture did not settle");
        history.regrid({2,{}});
        p.update(history,{},{1.,.24},.4,0,maximum,&trace);
        require(trace.levels.back().cells[changed].decision==Preview::Detail && trace.levels.back().cells[changed].quiet_since==.4 &&
                trace.levels.back().cells[unchanged].decision==Preview::Candidate,"Deep footprint change failed to reset only affected history");
        p.update(history,{},{1.,.24},.8,0,maximum,&trace);
        require(trace.levels.back().cells[changed].decision==Preview::Candidate,"Changed deep footprint could not settle again");
        std::cout<<"deep_sizes,maximum="<<maximum<<",PASS,alignment,nesting,lookahead,release,refinement_bridge,conservation,history\n";
    }
}
void velocityRepresentationContract() {
    constexpr int n=96;
    for(int scale:{2,4,8,16,32}) for(double contrast:{1.,10.}) for(int pattern=0;pattern<6;++pattern) {
        const int child=scale/2;
        Fixture f(contrast); auto state=f.state(n,n,0.,false);
        for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
            const double share=pattern==1?.5:.5+.3*std::sin(.07*x)*std::cos(.11*y);
            state.gas_mass[0][y*n+x]=.1*(1-share); state.gas_mass[1][y*n+x]=.1*contrast*share;
        }
        FluidGridAdaptive grid(n,n,1.,f.catalogue,f.mixture,{child,{}}); grid.upload(state);
        auto velocity=[&](double x,double y,int d) {
            if(pattern==0)return d?-.2:.4;
            if(pattern==1 || pattern==5)return d?-.01*(x-n*.5):.01*(y-n*.5);
            if(pattern==2)return d?-2.*std::sin(2.*std::numbers::pi*x/n)*std::pow(std::sin(std::numbers::pi*y/n),2):
                                  2.*std::sin(2.*std::numbers::pi*y/n)*std::pow(std::sin(std::numbers::pi*x/n),2);
            if(pattern==3)return ((int(x/child)+int(y/child))%2?.3:-.3)*(d?-.5:1.);
            return 0.;
        };
        // Raw reconstruction fixture: prescribed cell momentum and MAC faces.
        // Projection/evolution are checked separately on physical circulation.
        for(const auto& l:grid.levels()) for(int fab=0;fab<l->conserved.size();++fab) {
            auto q=l->conserved.array(fab); const auto b=l->conserved.box(fab);
            for(int y=b.smallEnd(1);y<=b.bigEnd(1);++y)for(int x=b.smallEnd(0);x<=b.bigEnd(0);++x)
                for(int d=0;d<2;++d)q(x,y,2+d)=(q(x,y,0)+q(x,y,1))*velocity((x+.5)*child,(y+.5)*child,d);
            for(int d=0;d<2;++d) {
                const auto face=l->velocity[d].array(fab); const auto box=l->velocity[d].box(fab);
                for(int y=box.smallEnd(1);y<=box.bigEnd(1);++y)for(int x=box.smallEnd(0);x<=box.bigEnd(0);++x)
                    face(x,y)=pattern==4?((d?x:y)%2?.3:-.3):velocity((x+(d?.5:0.))*child,(y+(d?0.:.5))*child,d);
            }
        }
        std::vector<double> before(2*n*n),estimates(n*n/(scale*scale));
        double oldMax=0.,estimateMax=0.;
        {
            fluid::FluidCellView view(grid);
            for(int y=0;y<n;y+=child)for(int x=0;x<n;x+=child)for(int d=0;d<2;++d)before[2*(y*n+x)+d]=view.at(x,y).velocity(d);
            for(int y=0;y<n;y+=scale)for(int x=0;x<n;x+=scale) {
                const auto e=Preview::estimate(grid,x,y,scale); estimates[(y/scale)*(n/scale)+x/scale]=e.velocity;
                if(x>=scale && y>=scale && x<n-scale && y<n-scale)estimateMax=std::max(estimateMax,e.velocity);
                double rho=0.,mean[2]{};
                for(int dy=0;dy<scale;dy+=child)for(int dx=0;dx<scale;dx+=child) {const auto c=view.at(x+dx,y+dy);rho+=c.density();for(int d=0;d<2;++d)mean[d]+=c.momentum(d);}
                for(auto& u:mean)u/=rho;
                for(int dy=0;dy<scale;dy+=child)for(int dx=0;dx<scale;dx+=child)for(int d=0;d<2;++d) {
                    const auto c=view.at(x+dx,y+dy); oldMax=std::max(oldMax,std::abs(c.velocity(d)-mean[d]));
                    for(int side=0;side<2;++side)oldMax=std::max(oldMax,std::abs(c.face(d,side)-mean[d]));
                }
            }
        }
        grid.regrid({scale,{}}); grid.regrid({child,{}});
        double loss=0.,under=0.,interiorLoss=0.;
        const fluid::FluidCellView after(grid);
        for(int y=0;y<n;y+=child)for(int x=0;x<n;x+=child)for(int d=0;d<2;++d) {
            const double error=std::abs(after.at(x,y).velocity(d)-before[2*(y*n+x)+d]);
            loss=std::max(loss,error); under=std::max(under,error-estimates[(y/scale)*(n/scale)+x/scale]);
            if(x>=scale && y>=scale && x<n-scale && y<n-scale)interiorLoss=std::max(interiorLoss,error);
        }
        std::cout<<"velocity_representation,scale="<<scale<<",contrast="<<contrast<<",pattern="<<pattern<<",actual_loss="<<loss
            <<",interior_loss="<<interiorLoss<<",estimate_interior="<<estimateMax<<",old_variation="<<oldMax<<",underestimate="<<under<<'\n';
        require(under<2e-12,"Velocity indicator underestimated actual adjacent-level momentum reconstruction loss");
        if(pattern==0)require(interiorLoss<2e-12,"Constant velocity lost across differing-density remap");
        if(pattern==1 && contrast==1.)require(interiorLoss<2e-12 && estimateMax<2e-12,"Interior affine motion should be reconstructible");
        if(pattern==3 || pattern==4)require(estimateMax>=.29,"Unresolved momentum or independent MAC faces escaped the indicator");
    }
}
void smoothCoarseningEvolution() {
    constexpr int n=256; Fixture f;
    const auto state=f.state(n,n,2.);
    for(int scale:{2,4}) {
        FluidGridAdaptive fine(n,n,1.,f.catalogue,f.mixture,{}),coarse(n,n,1.,f.catalogue,f.mixture,{});
        fine.upload(state); coarse.upload(state);
        Preview preview; Preview::Trace trace;
        preview.update(fine,{},{},0.,0.,scale,&trace);
        double estimate=0.; for(const auto& c:trace.levels.back().cells) estimate=std::max(estimate,c.error.velocity);
        coarse.regrid({scale,{}}); require(coarse.project(1e-3).converged,"Smooth coarsening projection failed");
        const double initialMass=coarse.totals().ingredientMass[0];
        for(int step=0;step<=25;++step) {
            if(step==0 || step==1 || step==25) {
                const fluid::FluidCellView a(fine),b(coarse);
                fluid::FluidCellView::MaterialPlane plane;
                double cellError=0.,faceError=0.,eligibleError=0.;
                for(int y=0;y<n;y+=scale)for(int x=0;x<n;x+=scale) {
                    b.reconstruct(plane,x,y,scale,true);
                    const bool eligible=trace.levels.back().cells[(y/scale)*(n/scale)+x/scale].error.velocity<=.05;
                    for(int dy=0;dy<scale;++dy)for(int dx=0;dx<scale;++dx) {
                        const double px=(dx+.5)/scale-.5,py=(dy+.5)/scale-.5;
                        const double rho=plane.value(0,px,py)+plane.value(1,px,py);
                        for(int d=0;d<2;++d) {
                            const double error=std::abs(a.at(x+dx,y+dy).velocity(d)-plane.value(2+d,px,py)/rho);
                            cellError=std::max(cellError,error);
                            if(eligible)eligibleError=std::max(eligibleError,error);
                        }
                    }
                    for(int d=0;d<2;++d)for(int side=0;side<2;++side) {
                        double mean=0.;
                        for(int j=0;j<scale;++j)mean+=a.at(x+(d?j:side*(scale-1)),y+(d?side*(scale-1):j)).face(d,side)/scale;
                        faceError=std::max(faceError,std::abs(mean-b.at(x,y).face(d,side)));
                    }
                }
                std::cout<<"smooth_coarsening,scale="<<scale<<",time="<<step*.016<<",initial_estimate="<<estimate
                    <<",reconstructed_velocity_error="<<cellError<<",eligible_velocity_error="<<eligibleError<<",coarse_face_flux_error="<<faceError<<'\n';
                // A merge estimate is instantaneous, not an accumulated bound.
                // Check against the full refine threshold after evolution; the
                // half-threshold only decides when it is quiet enough to merge.
                require(eligibleError<Preview::Tolerances{}.velocity && faceError<Preview::Tolerances{}.velocity,
                    "Accepted smooth flow exceeded the refine threshold within one check interval");
            }
            if(step==25)break;
            require(fine.step(.016,1e-3).projection.converged && coarse.step(.016,1e-3).projection.converged,"Smooth evolution failed projection");
        }
        require(std::abs(coarse.totals().ingredientMass[0]/initialMass-1)<1e-11,"Coarsening/evolution lost conserved material");
    }
}
struct Measurement {
    double total=0,physics=0,selection=0,remap=0,peakRemap=0,leaves=0,allocated=0;
    double massError=0,volumeError=0,divergence=0,minDensity=1.; int changes=0;
    std::vector<double> cloud,velocity;
};
Measurement run(int n,int steps,double world,bool adaptive,bool health,double contrast=1.,double ingredientTolerance=.02,int maxCellScale=2) {
    Fixture f(contrast); FluidGridAdaptive grid(n,n,world/n,f.catalogue,f.mixture,{});
    auto initialState=f.state(n,world,health?.5:2.);
    if (health) for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
        const double r=std::hypot((x+.5)/n-.35,(y+.5)/n-.35);
        const double cloud=.02+.9*std::exp(-r*r/(2*.07*.07)), area=world*world/(n*n);
        initialState.gas_mass[0][y*n+x]=area*f.mixture.components[0].density*(1-cloud);
        initialState.gas_mass[1][y*n+x]=area*f.mixture.components[1].density*cloud;
    }
    grid.upload(initialState,health?1e-9:1e-3);
    const auto initial=grid.totals(); Preview p; Measurement m;
    Preview::Tolerances tolerances; tolerances.ingredient=ingredientTolerance;
    const double tolerance=health?1e-9:1e-3;
    constexpr double dt=.016;
    // Same timing contract as the scene: assess every four 0.08-second frames,
    // with a one-frame margin in the motion sweep and a 0.24-second quiet dwell.
    if (adaptive) p.update(grid,tolerances,{},0,.4,maxCellScale);
    const int warmup=health?0:20;
    for (int i=0;i<steps+warmup;++i) {
        const auto start=Clock::now();
        auto result=grid.step(dt,tolerance,double(.2f));
        require(result.projection.converged,"Automatic trajectory projection failed");
        double physics=elapsed(start),selection=0,remap=0;
        if (adaptive && (i+1)%20==0) {
            const auto scan=Clock::now(); p.update(grid,tolerances,{},(i+1)*dt,.4,maxCellScale);
            auto layout=p.proposeLayout(grid); selection=elapsed(scan);
            const auto transfer=Clock::now();
            if (grid.regrid(std::move(layout))) {
                const auto restoration=grid.project(tolerance);
                require(restoration.converged,"Automatic trajectory remap restoration failed");
                result.projection.max_divergence=std::max(result.projection.max_divergence,restoration.max_divergence);
                remap=elapsed(transfer);
                if (i>=warmup) ++m.changes;
            }
        }
        const double total=elapsed(start);
        if (i>=warmup) {
            m.total+=total; m.physics+=physics; m.selection+=selection; m.remap+=remap; m.peakRemap=std::max(m.peakRemap,remap);
            m.leaves+=double(grid.activeCellCount())/(n*n); m.allocated+=double(grid.allocatedCellCount())/(n*n);
        }
        // Whole committed-state checks are outside timed work; per-tick for the
        // behavioral gate, selection boundaries for the throughput comparison.
        if (health || (i+1)%20==0 || i==steps+warmup-1) {
            const auto after=grid.totals();
            require(after.finite && after.minMassDensity>=-1e-12,"Automatic trajectory nonfinite or negative mass");
            for (int k=0;k<2;++k) m.massError=std::max(m.massError,std::abs(after.ingredientMass[k]/initial.ingredientMass[k]-1));
            m.volumeError=std::max(m.volumeError,after.maxVolumeError);
            m.minDensity=std::min(m.minDensity,after.minMassDensity);
            m.divergence=std::max(m.divergence,grid.maxDivergence());
            require(m.massError<1e-10 && m.volumeError<(health?1e-7:.02) && m.divergence<=tolerance*1.001,"Automatic trajectory invariant budget exceeded");
        }
    }
    fluid::FluidCellView view(grid); m.cloud.resize(n*n); m.velocity.resize(2*n*n);
    for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
        const auto c=view.at(x,y); m.cloud[y*n+x]=view.materialSample(x,y,1);
        for (int d=0;d<2;++d) m.velocity[2*(y*n+x)+d]=c.velocity(d);
    }
    std::cout<<std::setprecision(10)<<n<<','<<(adaptive?"automatic":"uniform")<<','<<contrast<<','<<steps<<','
        <<m.total/steps<<','<<m.physics/steps<<','<<m.selection/steps<<','<<m.remap/steps<<','<<m.peakRemap<<','<<m.changes<<','
        <<m.leaves/steps<<','<<m.allocated/steps<<','<<m.massError<<','<<m.volumeError<<','<<m.divergence<<",PASS\n";
    return m;
}
void compare(const Measurement& uniform,const Measurement& adaptive,bool check=false) {
    double cloud=0,velocity=0,peakCloud=0;
    for (size_t i=0;i<uniform.cloud.size();++i) { const double error=std::abs(uniform.cloud[i]-adaptive.cloud[i]); cloud+=error; peakCloud=std::max(peakCloud,error); }
    for (size_t i=0;i<uniform.velocity.size();++i) velocity+=std::abs(uniform.velocity[i]-adaptive.velocity[i]);
    std::cout<<"comparison,time_reduction="<<1-adaptive.total/uniform.total<<",cloud_L1="<<cloud/uniform.cloud.size()
        <<",cloud_Linf="<<peakCloud<<",velocity_L1="<<velocity/uniform.velocity.size()<<'\n';
    if (check) require(cloud/uniform.cloud.size()<.01 && peakCloud<.15 && velocity/uniform.velocity.size()<.005,
        "Automatic transport exceeded the localized-cloud comparison budget");
}
}
int main(int argc,char** argv) {
    try {
        bl::Thread::setMaxThreads(8); fluid::AmrGasRuntime runtime;
        std::cout<<"n,mode,contrast,steps,total_ms,physics_ms,selection_ms,remap_ms,peak_remap_ms,changes,active_fraction,allocated_fraction,mass_error,volume_error,divergence,status\n";

        if(argc>1 && std::string(argv[1])=="reconstruction") { reconstructionContract(); representationContract(); velocityRepresentationContract(); return 0; }
        if(argc>1 && std::string(argv[1])=="velocity") { velocityRepresentationContract(); smoothCoarseningEvolution(); return 0; }
        if(argc>1 && std::string(argv[1])=="cfl") { transportTimeStep(); subcycledLiquidStep(); return 0; }
        if(argc>1 && (std::string(argv[1])=="liquid" || std::string(argv[1])=="liquid-short")) { liquidTranslation(); liquidSliverRoundoff(); subcycledLiquidStep(); liquidSelectionAndTransport(std::string(argv[1])=="liquid-short"?96:100); return 0; }
        if(argc>1 && std::string(argv[1])=="depth") { deeperSelection(); return 0; }
        if (argc>1 && std::string(argv[1])=="selection") {
            selectionTraceIsObservation(); selectionTraceIsObservation(32); representationContract(); velocityRepresentationContract(); threeSizeSelection(); fourSizeSelection(); deeperSelection(); return 0;
        }
        if (argc>1) {
            const int n=std::stoi(argv[1]); const bool reverse=argc>2 && std::string(argv[2])=="reverse";
            const double ingredientTolerance=argc>3?std::stod(argv[3]):.02;
            Measurement uniform,adaptive;
            if (reverse) { adaptive=run(n,625,256.,true,false,1.,ingredientTolerance); uniform=run(n,625,256.,false,false); }
            else { uniform=run(n,625,256.,false,false); adaptive=run(n,625,256.,true,false,1.,ingredientTolerance); }
            compare(uniform,adaptive);
        } else {

            selectionTraceIsObservation();
            selectionTraceIsObservation(8);
            liquidTranslation(); liquidSliverRoundoff(); subcycledLiquidStep(); liquidSelectionAndTransport();
            transportTimeStep();
            selectionCases();
            staticSelectionSettles();
            materialSelectionConsistency();
            paddingDoesNotPropagate();
            reconstructionContract();
            representationContract();
            velocityRepresentationContract();
            smoothCoarseningEvolution();
            threeSizeSelection();
            fourSizeSelection();
            deeperSelection();
            // 24 seconds spans three nominal world/speed transit times, 75
            // selection opportunities and the initial damping transient.
            const auto uniform=run(64,1500,4.,false,true);
            const auto adaptive=run(64,1500,4.,true,true);
            compare(uniform,adaptive,true);
            require(adaptive.changes>=3 && adaptive.leaves/1500<.9,"Behavioral gate must exercise initial and subsequent automatic remaps");
            run(64,1500,4.,true,true,10.);
            run(128,1500,8.,true,true,1.,.02,4);
            run(128,1500,8.,true,true,10.,.02,4);
        }
    } catch (const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
