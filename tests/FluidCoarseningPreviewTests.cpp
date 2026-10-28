#include "FluidCoarseningPreview.h"
#include "FluidCellView.h"
#ifdef FLUIDSIM_HAS_AMR
#include "AmrGasRuntime.h"
#endif
#include <bitloop/core/threads.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>


using fluid::FluidGridCPU;
using Preview = fluid::FluidCoarseningPreview;
constexpr Preview::Stability instant{1.0,0.0};
namespace
{
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void expectNear(double a, double b, const char* message) { require(std::abs(a-b) < 1e-10, message); }
FluidGridCPU::State configure(FluidGridCPU& g, double density_b = 0.1)
{
    const auto a = g.elements.add({"A", {0,0,0}}), b = g.elements.add({"B", {0,1,1}}), w = g.elements.add({"Liquid", {1,0,0}});
    g.gas_mixture.components = {{a,0.1},{b,density_b}};
    g.liquid_mixture.components = {{w,100}};
    g.gravity_y = 0;
    const int n = g.width*g.height;
    const double area = g.cell_size*g.cell_size;
    FluidGridCPU::State s;
    s.liquid_fraction.assign(n,0); s.liquid_mass.assign(1,std::vector<double>(n,0));
    s.gas_mass = {std::vector<double>(n,0.05*area),std::vector<double>(n,0.5*density_b*area)};
    s.pressure.assign(n,0); s.u.assign((g.width+1)*g.height,0); s.v.assign(g.width*(g.height+1),0);
    g.upload(s);
    return s;
}
void detailCases()
{
    FluidGridCPU g(16,16,0.7); auto s = configure(g);
    Preview preview; Preview::Tolerances t;
    std::fill(s.u.begin(),s.u.end(),20); std::fill(s.v.begin(),s.v.end(),-8);
    for (int y=0;y<16;++y) for (int x=0;x<16;++x) s.pressure[y*16+x]=1000+2*x*0.7-3*y*0.7;
    g.upload(s); preview.update(g,t,instant,0);
    require(preview.labels[3*preview.width+3]==Preview::Candidate,"uniform fast flow and linear pressure are coarsenable");
    expectNear(Preview::estimate(g,6,6).score(t),0,"affine pressure has no gradient detail");
    auto linear=s;
    for (int y=0;y<16;++y) for (int x=0;x<16;++x) {
        const double f=.2+.02*x+.01*y;
        linear.gas_mass[0][y*16+x]=f*.1*.49; linear.gas_mass[1][y*16+x]=(1-f)*.1*.49;
    }
    g.upload(linear);
    expectNear(Preview::estimate(g,6,6).ingredient,.015,"flat coarse material loses half the fine-cell gradient along each axis");
    auto shifted=s;
    for (auto& v:shifted.u) v+=100;
    for (auto& p:shifted.pressure) p+=2000;
    g.upload(shifted);
    expectNear(Preview::estimate(g,6,6).score(t),0,"velocity and pressure offsets do not create detail");

    for (int y=0;y<16;++y) for (int x=0;x<=16;++x) s.u[y*17+x]=20+(x%2 ? 0.5 : -0.5);
    g.upload(s);
    require(Preview::estimate(g,6,6).velocity>=0.5,"internal alternating faces cannot disappear in an average");
    std::fill(s.u.begin(),s.u.end(),20);
    for (int y=0;y<16;++y) for (int x=0;x<16;++x) s.pressure[y*16+x]=((x+y)%2 ? 1.0 : -1.0);
    g.upload(s);
    expectNear(Preview::estimate(g,6,6).pressure_gradient,2/0.7,"checkerboard pressure and physical cell size");

    std::fill(s.pressure.begin(),s.pressure.end(),0);
    for (int y=0;y<16;++y) for (int x=0;x<16;++x)
    {
        const double f=(x+y)%2 ? 0.6 : 0.4;
        s.gas_mass[0][y*16+x]=f*0.1*0.49; s.gas_mass[1][y*16+x]=(1-f)*0.1*0.49;
    }
    g.upload(s); const auto e=Preview::estimate(g,6,6);
    expectNear(e.ingredient,0.1,"equal-density ingredient detail must survive classification");
    expectNear(e.density,0,"colour detail need not imply density detail");
    preview.update(g,t,instant,0); require(preview.labels[3*preview.width+3]==Preview::Detail,"quiet ingredient pocket stays detailed");
    t.ingredient=0.11; preview.update(g,t,instant,0);
    require(preview.labels[3*preview.width+3]==Preview::Candidate,"tolerance explicitly allows detail loss");

    FluidGridCPU dense(16,16,0.7); auto ds=configure(dense,10);
    for (int y=0;y<16;++y) for (int x=0;x<16;++x)
    {
        const double f=(x+y)%2 ? 0.6 : 0.4;
        ds.gas_mass[0][y*16+x]=f*0.1*0.49; ds.gas_mass[1][y*16+x]=(1-f)*10*0.49;
    }
    dense.upload(ds); t.ingredient=0.5; preview.update(dense,t,instant,0);
    require(Preview::estimate(dense,6,6).density>0.19,"relative density detail across different ingredients");
    require(preview.labels[3*preview.width+3]==Preview::Detail,"density criterion independent of ingredient tolerance");
}
void uniformProtectionAndStability() {
    FluidGridCPU grid(16,16,1); auto state=configure(grid); Preview p;
    p.update(grid,{},{},0); require(p.counts[Preview::Candidate]==0 && p.statistics.waiting>0,"quiet dwell starts pending");
    p.update(grid,{},{},.24); require(p.counts[Preview::Candidate]>0,"quiet dwell matures");
    const auto labels=p.labels; p.update(grid,{},{},.24); require(p.labels==labels,"paused time cannot age history");
    state.liquid_fraction[5*16+5]=1e-14; state.liquid_mass[0][5*16+5]=1e-12; grid.upload(state);
    p.update(grid,{},{},.24); require(p.labels[3*p.width+3]==Preview::LiquidBuffer,"liquid halo revokes candidate immediately");
    require(p.labels[0]==Preview::Unassessed,"physical boundary remains unknown");
    p.reset(); require(p.labels.empty(),"disabled preview releases visible labels");
}
#ifdef FLUIDSIM_HAS_AMR
void nativeViews() {
    fluid::AmrGasRuntime runtime;
    FluidGridCPU source(32,32,.7); const auto state=configure(source);
    fluid::FluidGridAdaptive grid(32,32,.7,source.elements,source.gas_mixture,{2,{{{8,8,16,16}}}},8);
    grid.upload(state); Preview p; Preview::Tolerances t;
    const auto before=grid.totals();
    fluid::FluidCellView view(grid);
    require(view.at(0,0).patch->scale==2 && view.at(10,10).patch->scale==1,"view selects finest active level");
    expectNear(view.at(0,0).fraction(1),.5,"native density interpreted as cell fraction");
    require(grid.activeCellCount()==448 && grid.allocatedCellCount()==512,"active count excludes covered coarse storage");
    // Give the impulse a physical interval, then write a manufactured affine
    // pressure and committed momentum directly into native valid cells.
    grid.project(1e-9,.02);
    for (const auto& level:grid.levels()) for (int fab=0;fab<level->conserved.size();++fab) {
        auto q=level->conserved.array(fab), pressure=level->pressure.array(fab);
        const auto& box=level->conserved.box(fab); const double h=level->geometry.CellSize(0);
        for (int y=box.smallEnd(1);y<=box.bigEnd(1);++y) for (int x=box.smallEnd(0);x<=box.bigEnd(0);++x)
            pressure(x,y)=.02*(1000.+2.*(x+.5)*h-3.*(y+.5)*h);
    }
    p.update(grid,t,instant,0,0.,2);
    require(p.counts[Preview::Refine]==0,"affine physical pressure has no refinement detail across FAB/level joins");
    expectNear(fluid::FluidCellView(grid).at(10,10).pressure(),1000.+2*10.5*.7-3*10.5*.7,"impulse divided by physical interval");
    // Native cell velocity must read committed momentum, not stale ghosts or
    // just the averaged face value. A checkerboard survives that distinction.
    for (const auto& level:grid.levels()) for (int fab=0;fab<level->conserved.size();++fab) {
        auto q=level->conserved.array(fab); const auto& box=level->conserved.box(fab);
        for (int y=box.smallEnd(1);y<=box.bigEnd(1);++y) for (int x=box.smallEnd(0);x<=box.bigEnd(0);++x)
            q(x,y,2)=(q(x,y,0)+q(x,y,1))*(x%2?1.:-1.);
    }
    p.update(grid,t,instant,.1,0.,2);
    require(p.counts[Preview::Detail]>0 && p.counts[Preview::Refine]>0,"committed momentum detail is visible");
    const auto totals=grid.totals(); p.update(grid,t,instant,.2,1.,2);
    require(grid.totals().ingredientMass==totals.ingredientMass && grid.totals().momentum==totals.momentum,"preview does not mutate conserved state");
    // Known fine ingredient detail moves into coarse cells, with no recursive
    // propagation from held requests. Keep faces and cell momentum consistent;
    // their disagreement is itself a velocity-detail trigger.
    grid.upload(state); grid.project(1e-9,.02);
    for (const auto& level:grid.levels()) {
        level->velocity[0].setVal(4.); level->velocity[1].setVal(0.);
        for (int fab=0;fab<level->conserved.size();++fab) {
            auto q=level->conserved.array(fab); const auto& box=level->conserved.box(fab);
            for (int y=box.smallEnd(1);y<=box.bigEnd(1);++y) for (int x=box.smallEnd(0);x<=box.bigEnd(0);++x) {
                if (level->geometry.CellSize(0)<=.7) {
                    q(x,y,0)=x%2?.09:.01; q(x,y,1)=.1-q(x,y,0);
                }
                q(x,y,2)=4.*(q(x,y,0)+q(x,y,1)); q(x,y,3)=0.;
            }
        }
    }
    p.reset(); p.update(grid,t,instant,0,1.,2);
    require(p.labels[8*p.width+13]==Preview::Approaching,"fine feature warns coarse cells in forward direction");
    require(p.labels[8*p.width+2]!=Preview::Approaching,"feature does not sweep backwards");
    for (int y=4;y<12;++y) for (int x=4;x<12;++x) require(p.labels[y*p.width+x]!=Preview::Approaching,"prediction does not mark fine cells");
    grid.upload(state); p.reset(); p.update(grid,t,{},0,0.,2); p.update(grid,t,{},.24,0.,2);
    require(p.counts[Preview::Candidate]>0,"native fine merge candidates mature");
    grid.regrid({2,{{{4,4,16,16}}}}); grid.project(1e-9); p.update(grid,t,{},.32,0.,2);
    require(p.counts[Preview::Candidate]>0 && p.statistics.waiting>0,"layout change preserves overlap history and delays newly fine cells");
    require(grid.totals().ingredientMass==before.ingredientMass,"uniform mixture regrid conserves exact mass");
    grid.regrid({}); grid.project(1e-9); p.update(grid,t,instant,.4,0.,2);
    require(p.counts[Preview::Coarse]==0,"uniform endpoint replaces native coarse labels");
}
#endif
}
int main() {
    try {
        bl::Thread::setMaxThreads(8);
        detailCases(); uniformProtectionAndStability();
#ifdef FLUIDSIM_HAS_AMR
        nativeViews();
#endif
        std::cout<<"PASS uniform detail/liquid protection, native active views, pressure units, momentum, prediction, history and observer parity\n";
    } catch (const std::exception& error) { std::cerr<<"FAIL "<<error.what()<<'\n'; return 1; }
}
