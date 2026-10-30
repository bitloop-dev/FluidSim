#include "../FluidSim/FluidGridAdaptive.h"
#include "amrex/AmrGas.h"
#include "amrex/AmrGasRuntime.h"
#include <AMReX_Cluster.H>
#include <AMReX_BoxIterator.H>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>

using fluid::FluidGridAdaptive;
using Upstream=fluid::reference::AmrGas;
namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
fluid::reference::GasLayout upstreamLayout(const fluid::GasLayout& layout) {
    fluid::reference::GasLayout out{layout.baseRatio,{}};
    for(const auto& level:layout.refinedPatches) {
        auto& patches=out.refinedPatches.emplace_back();
        for(const auto& p:level)patches.push_back({p.x,p.y,p.width,p.height});
    }
    return out;
}
double difference(const fluid::AmrField2D<double>& own,const amrex::MultiFab& ref,int ghosts=0) {
    require(own.size()==ref.size(),"Patch splitting differs");
    double maximum=0.;
    for(int p=0;p<own.size();++p) {
        const auto a=own.const_array(p);const auto b=ref.const_array(p);
        const auto box=own.box(p).grown(ghosts);
        fluid::forCells(box,own.nComp(),[&](int x,int y,int n) {
            require(std::isfinite(a(x,y,n))&&std::isfinite(b(x,y,0,n)),"Nonfinite reference comparison");
            maximum=std::max(maximum,std::abs(a(x,y,n)-b(x,y,0,n))/std::max(1.,std::abs(b(x,y,0,n))));
        });
    }
    return maximum;
}
void match(fluid::AmrField2D<double>& own,const amrex::MultiFab& ref) {
    for(int p=0;p<own.size();++p) {
        const auto a=own.array(p);const auto b=ref.const_array(p);
        fluid::forCells(own.box(p),own.nComp(),[&](int x,int y,int n){a(x,y,n)=b(x,y,0,n);});
    }
}
void matchState(FluidGridAdaptive& own,const Upstream& ref) {
    for(int lev=0;lev<int(own.levels().size());++lev) {
        auto& a=*own.levels()[lev];const auto& b=*ref.levels()[lev];
        match(a.conserved,b.conserved);match(a.pressure,b.pressure);
        for(int d=0;d<2;++d)match(a.velocity[d],b.velocity[d]);
    }
}
double stateError(const FluidGridAdaptive& own,const Upstream& ref,bool faces=true,int ghosts=0) {
    double error=0.;
    for(int lev=0;lev<int(own.levels().size());++lev) {
        const auto& a=*own.levels()[lev];const auto& b=*ref.levels()[lev];
        error=std::max(error,difference(a.conserved,b.conserved,ghosts));
        if(faces)for(int d=0;d<2;++d)error=std::max(error,difference(a.velocity[d],b.velocity[d]));
    }
    return error;
}
void clusters() {
    std::mt19937 random(723);
    for(int trial=0;trial<100;++trial) {
        std::vector<std::array<int,2>> points;
        std::vector<amrex::IntVect> upstream;
        for(int y=0;y<32;++y)for(int x=0;x<32;++x)if(random()%100<unsigned(5+trial%90)) {
            points.push_back({x,y});upstream.emplace_back(x,y);
        }
        auto own=fluid::clusterAmrTags(points,.7);
        amrex::ClusterList reference(upstream.data(),upstream.size());reference.new_chop(.7);
        const auto boxes=reference.boxArray();
        require(own.size()==boxes.size(),"Cluster count differs");
        for(int i=0;i<int(own.size());++i) {
            const auto& b=boxes[i];
            require(own[i]==fluid::AmrBox2D{b.smallEnd(0),b.smallEnd(1),b.length(0),b.length(1)},"Cluster coordinates/order differ");
        }
    }
    for(int width:{16,48,64,72,96,144,256})for(int maximum:{16,32,64}) {
        amrex::BoxArray original(amrex::Box(amrex::IntVect(8,16),amrex::IntVect(width+7,111)));original.maxSize(maximum);
        auto own=fluid::splitAmrBoxes({{8,16,width,96}},maximum);
        require(own.size()==original.size(),"Patch size count differs");
        for(int i=0;i<int(own.size());++i) {
            const auto& b=original[i];
            require(own[i]==fluid::AmrBox2D{b.smallEnd(0),b.smallEnd(1),b.length(0),b.length(1)},"Patch size/order differs");
        }
    }
    std::cout<<"clustering_and_patch_splitting PASS\n";
}
void engine(bool liquid,double contrast,int ingredients) {
    constexpr int n=128;
    fluid::ElementCatalogue catalogue;
    fluid::FluidMixture gas,water;
    for(int k=0;k<ingredients;++k)gas.components.push_back({catalogue.add({"gas"}),k==0?.1:.1*contrast});
    if(liquid)water.components.push_back({catalogue.add({"water"}),100.});
    FluidGridAdaptive::State state;
    state.gas_mass.assign(ingredients,std::vector<double>(n*n));
    if(liquid){state.liquid_mass.assign(1,std::vector<double>(n*n));state.liquid_fraction.resize(n*n);}
    state.u.resize((n+1)*n);state.v.resize(n*(n+1));
    auto psi=[](int x,int y) {return x==0||y==0||x==n||y==n?0.:3.*std::pow(std::sin(std::numbers::pi*x/n)*std::sin(std::numbers::pi*y/n),2);};
    for(int y=0;y<n;++y)for(int x=0;x<=n;++x)state.u[y*(n+1)+x]=psi(x,y+1)-psi(x,y);
    for(int y=0;y<=n;++y)for(int x=0;x<n;++x)state.v[y*n+x]=psi(x,y)-psi(x+1,y);
    for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
        const int i=y*n+x;
        const double f=liquid?std::clamp(y+1.-(.45*n+.3*x),0.,1.):0.;
        if(liquid){state.liquid_mass[0][i]=100.*f;state.liquid_fraction[i]=f;}
        double total=0.;for(int k=0;k<ingredients;++k)total+=1.+.8*std::sin(.09*x+.13*y+k);
        for(int k=0;k<ingredients;++k)state.gas_mass[k][i]=(1.-f)*gas.components[k].density*(1.+.8*std::sin(.09*x+.13*y+k))/total;
    }
    const std::vector<fluid::GasLayout> layouts{{},{4,{{{0,16,112,112}},{{16,32,80,80}}}},{32,{}},
        {32,{{{0,0,128,128}},{{0,0,128,128}},{{0,0,128,128}},{{8,8,112,112}},{{24,24,80,80}}}}, {}};
    FluidGridAdaptive own(n,n,1.,catalogue,water,gas,{},32);
    Upstream ref(n,n,1.,catalogue,water,gas,{},32);
    own.upload(state,1e-10);ref.upload(state,1e-10);
    double projection=stateError(own,ref),transfer=0.,transport=0.,ghost=0.;
    require(projection<2e-8,"Initial projection differs");
    for(const auto& layout:layouts) {
        matchState(own,ref);
        const bool changed=own.regrid(layout);
        require(changed==ref.regrid(upstreamLayout(layout)),"No-op regrid differs");
        transfer=std::max(transfer,stateError(own,ref,false));
        require(transfer<2e-11,"Conservative remap differs");
        matchState(own,ref);
        require(own.project(1e-9).converged&&ref.project(1e-9).converged,"Restoration failed");
        for(int step=0;step<3;++step) {
            matchState(own,ref);
            // Poison unused storage before a fresh fill; no stale ghost may participate.
            for(int lev=0;lev<int(own.levels().size());++lev) {
                auto& a=own.levels()[lev]->conserved;auto& b=ref.levels()[lev]->conserved;
                for(int p=0;p<a.size();++p) {
                    const auto q=a.array(p);const auto r=b.array(p);const auto box=a.box(p);
                    fluid::forCells(box.grown(2),a.nComp(),[&](int x,int y,int k){if(!box.contains(x,y))q(x,y,k)=r(x,y,0,k)=std::numeric_limits<double>::quiet_NaN();});
                }
            }
            require(own.project(1e-9,.016,{.3,-.2}).converged&&ref.project(1e-9,.016,{.3,-.2}).converged,"Projection failed");
            projection=std::max(projection,stateError(own,ref));
            // Only ghost values are compared here; projection changes valid momentum.
            for(int lev=0;lev<int(own.levels().size());++lev) {
                const auto& a=own.levels()[lev]->conserved;const auto& b=ref.levels()[lev]->conserved;
                for(int p=0;p<a.size();++p) {
                    const auto q=a.const_array(p);const auto r=b.const_array(p);const auto box=a.box(p);
                    fluid::forCells(box.grown(2),a.nComp(),[&](int x,int y,int k){if(!box.contains(x,y)) {
                        require(std::isfinite(q(x,y,k))&&std::isfinite(r(x,y,0,k)),"Ghost fill left poison");
                        ghost=std::max(ghost,std::abs(q(x,y,k)-r(x,y,0,k))/std::max(1.,std::abs(r(x,y,0,k))));
                    }});
                }
            }
            matchState(own,ref);
            const double dt=std::min(.016,.5*ref.stableTimeStep());
            own.transport(dt);ref.transport(dt);
            transport=std::max(transport,stateError(own,ref,false));
            require(transport<2e-10,"Transport differs on identical inputs");
        }
    }
    require(projection<2e-7&&ghost<2e-12,"Projection or boundary fill differs");
    std::cout<<"engine liquid="<<liquid<<" contrast="<<contrast<<" ingredients="<<ingredients<<" remap="<<transfer<<" ghost="<<ghost<<" transport="<<transport<<" projection="<<projection<<" PASS\n";
}
}
int main() {
    try {
        fluid::reference::AmrGasRuntime runtime;
        clusters();
        engine(false,1.,2);engine(false,100.,7);engine(true,1.,2);engine(true,100.,7);
    } catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
