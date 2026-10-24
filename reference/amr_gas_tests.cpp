#include "amrex/AmrGas.h"
#include "amrex/AmrGasInterpolation.h"
#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_BoxList.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PhysBCFunct.H>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace amrex;
using fluid::reference::AmrGas;
using Clock = std::chrono::steady_clock;
constexpr double pi = std::numbers::pi;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

fluid::reference::GasLayout coarsePatches(int n, const std::vector<fluid::reference::GasPatch>& patches) {
    BoxList coarse;
    for (const auto& p : patches) coarse.push_back(Box(IntVect(p.x, p.y), IntVect(p.x + p.width - 1, p.y + p.height - 1)));
    const auto fine = amrex::complementIn(Box(IntVect(0), IntVect(n - 1)), coarse);
    fluid::reference::GasLayout result{2, {{}}};
    for (const auto& box : fine) result.refinedPatches[0].push_back({box.smallEnd(0), box.smallEnd(1), box.length(0), box.length(1)});
    return result;
}

fluid::reference::GasLayout makeLayout(int n, const std::string& name, int phase = 0) {
    if (name == "uniform") return {};
    if (name == "coarse") return {2, {}};
    if (name == "coarse4") return {4, {}};
    if (name == "half4") return {4, {{{n/4,n/4,n/2,n/2}}}};
    if (name == "nested") return {4, {{{n/8,n/8,3*n/4,3*n/4}}, {{n/4,n/4,n/2,n/2}}}};
    if (name == "nested-buffer") return {4, {{{n/8,n/8,3*n/4,3*n/4}}, {{n/8+4,n/8+4,3*n/4-8,3*n/4-8}}}};
    if (name == "nested-edge") return {4, {{{0,0,3*n/4,3*n/4}}, {{0,0,n/2,n/2}}}};
    if (name == "nested-four") return {4,
        {{{0,0,n/2,3*n/4},{n/2,0,n/2,3*n/4}},
         {{n/8,n/8,n/4,n/4},{5*n/8,n/8,n/4,n/4},{n/8,n/2,n/4,n/8},{5*n/8,n/2,n/4,n/8}}}};
    if (name == "nested-changing") {
        if (phase%4==0) return makeLayout(n,"nested");
        if (phase%4==1) return makeLayout(n,"nested-four");
        if (phase%4==2) return makeLayout(n,"nested-edge");
        return {4, {{{n/4,n/4,3*n/4,3*n/4}}, {{n/2,n/2,n/2,n/2}}}};
    }
    if (name == "production-changing") {
        if (phase%4==0) return {2,{{{n/4,n/4,n/2,n/2}}}};
        if (phase%4==1) return {2,{{{n/8,n/4,n/4,n/2},{5*n/8,n/4,n/4,n/2}}}};
        if (phase%4==2) return makeLayout(n,"four");
        return {2,{{{n/16,n/8,n/4,n/2},{9*n/16,3*n/8,n/4,n/2}}}};
    }
    if (name == "sparse-changing") {
        if (phase % 4 == 0) return makeLayout(n, "patch");
        if (phase % 4 == 1) return makeLayout(n, "four");
        if (phase % 4 == 2) return makeLayout(n, "edge");
        return {2, {{{n/8,n/8,n/2,n/2}}}};
    }
    if (name == "matched-one" || (name == "changing" && phase % 3 == 0)) return coarsePatches(n, {{n/8,n/4,3*n/4,n/2}});
    if (name == "matched-two" || (name == "changing" && phase % 3 == 1))
        return coarsePatches(n, {{n/16,n/4,3*n/8,n/2},{9*n/16,n/4,3*n/8,n/2}});
    if (name == "matched-four" || name == "changing")
        return coarsePatches(n, {{n/16,n/16,3*n/8,n/4},{9*n/16,n/16,3*n/8,n/4}, {n/16,9*n/16,3*n/8,n/4},{9*n/16,9*n/16,3*n/8,n/4}});
    if (name == "scene-changing") {
        if (phase % 4 == 0) return coarsePatches(n, {{n/8,n/8,3*n/4,3*n/4}});
        if (phase % 4 == 1) return coarsePatches(n, {{n/8,n/8,n/4,3*n/4},{5*n/8,n/8,n/4,3*n/4}});
        if (phase % 4 == 2) return coarsePatches(n, {{n/8,n/8,n/4,n/4},{5*n/8,n/8,n/4,n/4},{n/8,5*n/8,n/4,n/4},{5*n/8,5*n/8,n/4,n/4}});
        return coarsePatches(n, {{n/16,n/16,n/2,n/4},{7*n/16,7*n/16,n/2,n/2}});
    }
    if (name == "edge") return {2, {{{0, n / 4, n / 2, n / 2}}}};
    if (name == "patch") return {2, {{{n / 4, n / 4, n / 2, n / 2}}}};
    fluid::reference::GasLayout result{2, {{}}};
    for (int y : {n / 8, 5 * n / 8}) for (int x : {n / 8, 5 * n / 8}) result.refinedPatches[0].push_back({x, y, n / 4, n / 4});
    return result;
}

std::pair<fluid::ElementCatalogue, fluid::FluidMixture> materials(int count, double contrast) {
    fluid::ElementCatalogue catalogue;
    fluid::FluidMixture mixture;
    for (int k = 0; k < count; ++k) mixture.components.push_back({catalogue.add({"Gas " + std::to_string(k)}), 0.1 * (k == 1 ? contrast : 1.)});
    return {catalogue, mixture};
}

AmrGas::State initialState(int n, double world, double speed, const fluid::FluidMixture& mixture, bool sharp = false) {
    AmrGas::State state;
    const double h = world / n;
    const int count = static_cast<int>(mixture.components.size());
    state.gas_mass.assign(count, std::vector<double>(n * n));
    state.u.resize((n + 1) * n); state.v.resize(n * (n + 1));
    for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
        const double px = (x + 0.5) / n, py = (y + 0.5) / n;
        const double cloud = sharp ? (std::hypot(px - 0.35, py - 0.35) < 0.16 ? 1. : 0.)
                                   : 0.5 + 0.4 * std::cos(2 * pi * px) * std::cos(2 * pi * py);
        for (int k = 0; k < count; ++k) {
            const double fraction = k == 1 ? cloud : (1. - cloud) / (count - 1);
            state.gas_mass[k][y * n + x] = mixture.components[k].density * h * h * fraction;
        }
    }
    auto psi = [&](int x, int y) {
        if (x == 0 || y == 0 || x == n || y == n) return 0.;
        return speed * world / pi * std::pow(std::sin(pi * x / n) * std::sin(pi * y / n), 2);
    };
    for (int y = 0; y < n; ++y) for (int x = 0; x <= n; ++x) state.u[y * (n + 1) + x] = (psi(x, y + 1) - psi(x, y)) / h;
    for (int y = 0; y <= n; ++y) for (int x = 0; x < n; ++x) state.v[y * n + x] = -(psi(x + 1, y) - psi(x, y)) / h;
    return state;
}

double massError(const fluid::reference::GasTotals& before, const fluid::reference::GasTotals& after) {
    double error = 0.;
    for (int k = 0; k < before.ingredientMass.size(); ++k)
        error = std::max(error, std::abs(after.ingredientMass[k] - before.ingredientMass[k]) / before.ingredientMass[k]);
    return error;
}

double sample(const MultiFab& mf, int i, int j, int n = 0) {
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
        if (mfi.validbox().contains(IntVect(i, j))) return mf.const_array(mfi)(i, j, 0, n);
    throw std::runtime_error("Requested a nonexistent fine sample");
}

void hierarchyEndpoints(int n) {
    auto [catalogue, mixture] = materials(7, 100.);
    const auto state = initialState(n, n, 0., mixture, true);
    for (const std::string name : {"uniform","coarse","coarse4","half4","nested","nested-edge","nested-four"}) {
        const auto layout = makeLayout(n, name);
        AmrGas uploaded(n,n,1.,catalogue,mixture,layout,16), remapped(n,n,1.,catalogue,mixture,{},16);
        uploaded.upload(state); remapped.upload(state); remapped.regrid(layout);
        long long allocated = 1LL*n*n/(layout.baseRatio*layout.baseRatio), active = allocated;
        for (int lev=1; lev<=layout.refinedPatches.size(); ++lev) {
            const int ratio = layout.baseRatio >> lev;
            for (const auto& patch : layout.refinedPatches[lev-1]) {
                const long long cells = 1LL*patch.width*patch.height/(ratio*ratio);
                allocated += cells; active += cells-cells/4;
            }
        }
        for (const auto* grid : {&uploaded, &remapped}) {
            require(grid->levels().size()==layout.refinedPatches.size()+1 && grid->allocatedCellCount()==allocated &&
                    grid->activeCellCount()==active, "Hierarchy contains an unexpected physical level or cell count");
            long long area = 0;
            for (int lev=0; lev<grid->levels().size(); ++lev) {
                const int ratio = layout.baseRatio >> lev;
                const auto& level = *grid->levels()[lev];
                require(level.geometry.CellSize(0)==ratio,"Hierarchy level has the wrong physical cell width");
                for (MFIter mfi(level.conserved); mfi.isValid(); ++mfi) for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                    const int x=bit()[0], y=bit()[1];
                    if (level.active.const_array(mfi)(x,y,0)) area+=ratio*ratio;
                    for (int c=0; c<mixture.components.size(); ++c) {
                        double expected=0.;
                        for (int j=y*ratio; j<(y+1)*ratio; ++j) for (int i=x*ratio; i<(x+1)*ratio; ++i)
                            expected+=state.gas_mass[c][j*n+i]/(ratio*ratio);
                        require(std::abs(level.conserved.const_array(mfi)(x,y,0,c)-expected)<1.e-13,
                                "Upload or base coarsening discarded original finer averages");
                    }
                }
            }
            require(area==1LL*n*n,"Active hierarchy does not cover the physical domain once");
        }
    }
    std::cout<<"hierarchy,uniform_1_2_4_and_nested_upload_remap,PASS\n";
}

void boundaryFillMatchesReference(int n) {
    fluid::reference::GasLinearInterpolater interpolation;
    double worst = 0.;
    for (int ingredients : {2, 7}) for (double contrast : {1., 100.}) for (bool twoPhase : {false,true}) {
        auto [catalogue, mixture] = materials(ingredients, contrast);
        fluid::FluidMixture liquid,gas=mixture;
        if(twoPhase) { liquid.components.push_back(gas.components.front());gas.components.erase(gas.components.begin()); }
        AmrGas grid(n, n, 1., catalogue, liquid, gas, {}, 16);
        auto state=[&] {
            auto s=initialState(n,n,0.,mixture);
            if(twoPhase) {
                s.liquid_fraction.resize(n*n);
                for(int i=0;i<n*n;++i)s.liquid_fraction[i]=s.gas_mass[0][i]/mixture.components[0].density;
                s.liquid_mass.push_back(std::move(s.gas_mass.front()));s.gas_mass.erase(s.gas_mass.begin());
            }
            return s;
        };
        const int fields = ingredients + 2;
        Vector<BCRec> boundary(fields);
        for (int c = 0; c < fields; ++c) for (int d = 0; d < 2; ++d) {
            const int bc = c == ingredients + d ? BCType::reflect_odd : BCType::reflect_even;
            boundary[c].setLo(d, bc); boundary[c].setHi(d, bc);
        }
        grid.upload(state());
        const std::vector<fluid::reference::GasLayout> layouts{{}, makeLayout(n,"four"), makeLayout(n,"four"),
            {2,{{{0,0,n/2,n/2},{3*n/4,3*n/4,n/4,n/4}}}}, {2,{{{0,0,n,n}}}}, {2,{}}, makeLayout(n,"edge"),
            makeLayout(n,"nested"), makeLayout(n,"nested-buffer"), makeLayout(n,"nested-four"), makeLayout(n,"nested-edge"), makeLayout(n,"half4"),
            {4,{{{0,0,n,n}},{{0,0,n,n}}}}, {4,{}}, {}};
        for (const auto& layout : layouts) {
            grid.regrid(layout);
            // Re-upload and repeated edits must refresh values without changing
            // the workspace; topology changes must replace it with the hierarchy.
            grid.upload(state());
            for (int edit = 0; edit < 3; ++edit) {
                for (const auto& level : grid.levels()) for (MFIter mfi(level->conserved); mfi.isValid(); ++mfi) {
                    const auto q = level->conserved.array(mfi);
                    for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                        const int x=bit()[0], y=bit()[1];
                        double sum=0., rho=0.;
                        for (int c=0; c<ingredients; ++c) { q(x,y,0,c)=1.1+std::sin(.31*x+.17*(c+1)*y+edit); sum+=q(x,y,0,c); }
                        for (int c=0; c<ingredients; ++c) { q(x,y,0,c)*=mixture.components[c].density/sum; rho+=q(x,y,0,c); }
                        q(x,y,0,ingredients)=rho*std::sin(.19*y+edit);
                        q(x,y,0,ingredients+1)=rho*std::cos(.23*x-edit);
                    }
                }
                // Freeze the oracle's inputs before filling in place, so an
                // accidental owned-cell overwrite cannot contaminate both sides.
                Vector<MultiFab> source(grid.levels().size());
                for(int lev=0;lev<grid.levels().size();++lev) {
                    auto& q=grid.levels()[lev]->conserved;
                    source[lev].define(q.boxArray(),q.DistributionMap(),fields,0);
                    MultiFab::Copy(source[lev],q,0,0,fields,0);
                    q.setBndry(std::numeric_limits<double>::quiet_NaN());
                }
                // A loose projection fills ghosts and faces without a state update.
                require(grid.project(1.e6).converged,"Boundary-fill probe failed");
                CpuBndryFuncFab function(nullptr);
                for (int lev=0; lev<grid.levels().size(); ++lev) {
                    auto& fine=*grid.levels()[lev];
                    MultiFab expected(fine.conserved.boxArray(),fine.conserved.DistributionMap(),fields,2);
                    expected.setVal(std::numeric_limits<double>::quiet_NaN());
                    PhysBCFunct<CpuBndryFuncFab> fineBC(fine.geometry,boundary,function);
                    if (lev==0) FillPatchSingleLevel(expected,0.,{&source[lev]},{0.},0,0,fields,fine.geometry,fineBC,0);
                    else {
                        auto& coarse=*grid.levels()[lev-1];
                        PhysBCFunct<CpuBndryFuncFab> coarseBC(coarse.geometry,boundary,function);
                        for (int first=0; first<fields;) {
                            const int count=first==0 ? (contrast==1. ? ingredients : fields) : 2;
                            auto* mapper=&interpolation;
                            FillPatchTwoLevels(expected,0.,{&source[lev-1]},{0.},{&source[lev]},{0.},first,first,count,
                                coarse.geometry,fine.geometry,coarseBC,first,fineBC,first,IntVect(2),mapper,boundary,first);
                            first+=count;
                        }
                    }
                    for (MFIter mfi(expected); mfi.isValid(); ++mfi) {
                        const auto actual=fine.conserved.const_array(mfi), reference=expected.const_array(mfi);
                        for (BoxIterator bit(mfi.fabbox()); bit.ok(); ++bit) for (int c=0; c<fields; ++c) {
                            const double a=actual(bit()[0],bit()[1],0,c), b=reference(bit()[0],bit()[1],0,c);
                            require(std::isfinite(a) && std::isfinite(b),"Boundary fill left an unfilled cell");
                            if(mfi.validbox().contains(bit()))require(a==b,"Boundary fill changed an owned cell");
                            worst=std::max(worst,std::abs(a-b));
                            require(std::abs(a-b)<1.e-12*std::max(1.,std::abs(b)),"Reused boundary fill differs from upstream interpolation");
                        }
                    }
                }
            }
        }
    }
    std::cout<<"boundary_fill,max_error,"<<worst<<",PASS\n";
}

void conservativeRegrid(int n) {
    auto [catalogue, mixture] = materials(7, 100.);
    AmrGas grid(n, n, 16. / n, catalogue, mixture, {});
    grid.upload(initialState(n, 16., 2., mixture, true));
    double worstMass = 0., worstMomentum = 0., worstVolume = 0., restorationImpulse = 0.;
    int changes = 0;
    for (int cycle = 0; cycle < 3; ++cycle) for (const std::string name : {"patch", "four", "edge", "coarse", "uniform", "matched-four", "scene-changing",
        "nested", "nested-four", "nested-edge", "half4", "coarse4", "uniform", "nested", "coarse", "nested-four", "coarse4", "coarse", "uniform"}) {
        const auto before = grid.totals();
        const auto target = makeLayout(n, name, cycle);
        struct Cell { int x, y, ratio; std::vector<double> values; };
        std::vector<Cell> overlap;
        for (int oldLev = 0; oldLev < grid.levels().size(); ++oldLev) {
            const int ratio = grid.layout().baseRatio >> oldLev;
            const auto& old = grid.levels()[oldLev]->conserved;
            const auto& active = grid.levels()[oldLev]->active;
            for (MFIter mfi(old); mfi.isValid(); ++mfi) for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                const int x = bit()[0], y = bit()[1];
                const auto covers = [&](const auto& patches) {
                    return std::any_of(patches.begin(), patches.end(), [&](const auto& p) {
                        return x*ratio >= p.x && y*ratio >= p.y && x*ratio < p.x+p.width && y*ratio < p.y+p.height;
                    });
                };
                bool retained = false;
                for (int lev=0; lev<=target.refinedPatches.size(); ++lev) if ((target.baseRatio >> lev)==ratio)
                    retained = (lev==0 || covers(target.refinedPatches[lev-1])) &&
                        (lev==target.refinedPatches.size() || !covers(target.refinedPatches[lev]));
                if (retained && active.const_array(mfi)(x,y,0)) {
                    Cell cell{x, y, ratio, {}};
                    for (int k = 0; k < grid.ingredientCount() + 2; ++k) cell.values.push_back(old.const_array(mfi)(x, y, 0, k));
                    overlap.push_back(std::move(cell));
                }
            }
        }
        require(grid.regrid(target), "Expected a changed regrid");
        ++changes;
        const auto after = grid.totals();
        const double error = std::hypot(after.momentum[0] - before.momentum[0], after.momentum[1] - before.momentum[1]);
        worstMass = std::max(worstMass, massError(before, after)); worstMomentum = std::max(worstMomentum, error);
        worstVolume = std::max(worstVolume, after.maxVolumeError);
        require(after.finite && after.minMassDensity >= -1.e-13 && massError(before, after) < 1.e-12 && error < 1.e-10,
                "Raw regrid failed mass, momentum, finite or positivity gate");
        for (const auto& cell : overlap) for (int lev=0; lev<grid.levels().size(); ++lev) if ((target.baseRatio >> lev)==cell.ratio)
            for (int k = 0; k < cell.values.size(); ++k)
                require(sample(grid.levels()[lev]->conserved, cell.x, cell.y, k) == cell.values[k], "Regrid changed retained same-resolution state");
        const auto restored = grid.project(1.e-9);
        require(restored.converged, "Regrid pressure restoration failed");
        const auto projected = grid.totals();
        restorationImpulse = std::max(restorationImpulse, std::hypot(projected.momentum[0] - after.momentum[0], projected.momentum[1] - after.momentum[1]));
        const auto* address = grid.levels()[0].get();
        auto shuffled = target;
        for (auto& patches : shuffled.refinedPatches) std::reverse(patches.begin(), patches.end());
        require(!grid.regrid(shuffled) && grid.levels()[0].get() == address, "No-op regrid rebuilt or changed state");
        require(grid.step(.016, 1.e-9).projection.converged, "Post-regrid step failed");
    }
    std::cout << "regrid,endpoints_overlap_noop,changes=" << changes << ",mass_error=" << worstMass << ",momentum_error=" << worstMomentum
              << ",volume_error=" << worstVolume << ",max_pressure_impulse=" << restorationImpulse << ",PASS\n";
}

void independentIngredients(int n) {
    // Three independent fractions expose per-component limiting that preserves
    // each mass but breaks their local volume sum, even during refinement at rest.
    const double fractions[3][3]={{.1,.2,.7},{.2,.3,.5},{.4,.1,.5}};
    for (double contrast : {1.,100.}) {
        auto [catalogue,mixture]=materials(3,contrast);
        auto state=initialState(n,16.,0.,mixture);
        const double area=std::pow(16./n,2);
        for (int y=0;y<n;++y) for (int x=0;x<n;++x) for (int k=0;k<3;++k)
            state.gas_mass[k][y*n+x]=mixture.components[k].density*area*fractions[(x/2+y/2)%3][k];
        AmrGas grid(n,n,16./n,catalogue,mixture,{2,{}});
        grid.upload(state);
        // Inspect raw remap before wall pressure changes the uniform velocity.
        for (const auto& level:grid.levels()) for (MFIter mfi(level->conserved);mfi.isValid();++mfi) {
            const auto q=level->conserved.array(mfi);
            for (BoxIterator bit(mfi.validbox());bit.ok();++bit) {
                const int x=bit()[0],y=bit()[1];
                const double rho=q(x,y,0,0)+q(x,y,0,1)+q(x,y,0,2);
                q(x,y,0,3)=.7*rho; q(x,y,0,4)=-.4*rho;
            }
        }
        const auto initial=grid.totals();
        for (const std::string name : {"uniform","coarse","four","edge","nested","coarse4","uniform","half4","nested-four"}) {
            require(grid.regrid(makeLayout(n,name)),"Independent-ingredient regrid did not change");
            const auto totals=grid.totals();
            require(totals.maxVolumeError<1e-13 && totals.minMassDensity>=0. && massError(initial,totals)<1e-12,
                    "Refinement broke independent ingredient volume closure");
            for (const auto& level:grid.levels()) for (MFIter mfi(level->conserved);mfi.isValid();++mfi) {
                const auto q=level->conserved.const_array(mfi);
                for (BoxIterator bit(mfi.validbox());bit.ok();++bit) {
                    const int x=bit()[0],y=bit()[1];
                    const double rho=q(x,y,0,0)+q(x,y,0,1)+q(x,y,0,2);
                    require(std::abs(q(x,y,0,3)/rho-.7)<1e-13 && std::abs(q(x,y,0,4)/rho+.4)<1e-13,
                            "Independent-ingredient remap changed uniform velocity");
                }
            }
        }
        state=initialState(n,16.,2.,mixture);
        for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
            const double a=.25+.1*std::sin(2*pi*(x+.5)/n), b=.35+.1*std::cos(2*pi*(y+.5)/n);
            const double f[3]={a,b,1-a-b};
            for (int k=0;k<3;++k) state.gas_mass[k][y*n+x]=mixture.components[k].density*area*f[k];
        }
        grid.upload(state);
        const auto before=grid.totals();
        double worstVolume=0.,worstMass=0.;
        for (int step=0;step<1500;++step) {
            if (step%125==0 && grid.regrid(makeLayout(n,"production-changing",step/125)))
                require(grid.project(1e-9).converged,"Independent-ingredient restoration failed");
            require(grid.step(.016,1e-9).projection.converged,"Independent-ingredient transport projection failed");
            const auto totals=grid.totals();
            worstVolume=std::max(worstVolume,totals.maxVolumeError); worstMass=std::max(worstMass,massError(before,totals));
            require(totals.finite && totals.minMassDensity>=-1e-13 && worstVolume<2e-6 && worstMass<2e-11,
                    "Independent-ingredient evolution lost volume, positivity or mass");
        }
        std::cout << "independent_ingredients,contrast=" << contrast << ",seconds=24,volume_error=" << worstVolume
                  << ",mass_error=" << worstMass << ",PASS\n";
    }
}

void ingredientLabelsPreserveFlow(int n) {
    auto [catalogue,mixture]=materials(3,1.);
    auto flat=initialState(n,16.,2.,mixture), coloured=flat;
    const double mass=.1*std::pow(16./n,2);
    for (int y=0;y<n;++y) for (int x=0;x<n;++x) {
        const double a=.25+.1*std::sin(2*pi*(x+.5)/n), b=.35+.1*std::cos(2*pi*(y+.5)/n);
        const double f[3]={a,b,1-a-b};
        for (int k=0;k<3;++k) {
            flat.gas_mass[k][y*n+x]=mass/3.;
            coloured.gas_mass[k][y*n+x]=mass*f[k];
        }
    }
    AmrGas uniformColour(n,n,16./n,catalogue,mixture,makeLayout(n,"four"));
    AmrGas varyingColour(n,n,16./n,catalogue,mixture,makeLayout(n,"four"));
    uniformColour.upload(flat,1e-11); varyingColour.upload(coloured,1e-11);
    double error=0.;
    for (int step=0;step<32;++step) {
        if (step%8==0) for (auto* grid : {&uniformColour,&varyingColour})
            if (grid->regrid(makeLayout(n,"production-changing",step/8)))
                require(grid->project(1e-11).converged,"Colour-label restoration failed");
        for (auto* grid : {&uniformColour,&varyingColour})
            require(grid->step(.016,1e-11).projection.converged,"Colour-label step failed");
        for (int lev=0;lev<uniformColour.levels().size();++lev) for (int d=0;d<2;++d) {
            const auto& a=uniformColour.levels()[lev]->velocity[d];
            const auto& b=varyingColour.levels()[lev]->velocity[d];
            for (MFIter mfi(a);mfi.isValid();++mfi) for (BoxIterator bit(mfi.validbox());bit.ok();++bit)
                error=std::max(error,std::abs(a.const_array(mfi)(bit()[0],bit()[1],0)-b.const_array(mfi)(bit()[0],bit()[1],0)));
        }
    }
    require(error<1e-7,"Equal-density ingredient labels changed bulk flow");
    std::cout << "ingredient_labels,velocity_error=" << error << ",PASS\n";
}

double initialFaceError(const AmrGas& grid, const AmrGas::State& initial, int n) {
    double maximum = 0.;
    for (int lev = 0; lev < grid.levels().size(); ++lev) {
        const auto& level = *grid.levels()[lev];
        const int ratio = grid.layout().baseRatio >> lev;
        for (int d = 0; d < 2; ++d) for (MFIter mfi(level.velocity[d]); mfi.isValid(); ++mfi) {
            const auto u = level.velocity[d].const_array(mfi);
            for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                const int i = bit()[0], j = bit()[1];
                double expected = 0.;
                for (int s = 0; s < ratio; ++s)
                    expected += (d == 0 ? initial.u[(j * ratio + s) * (n + 1) + i * ratio] : initial.v[j * ratio * n + i * ratio + s]) / ratio;
                maximum = std::max(maximum, std::abs(u(i, j, 0) - expected));
            }
        }
    }
    return maximum;
}

struct InterfaceMetrics { double fluxMismatch = 0., contrast = 0., cloudFlux = 0.; };
InterfaceMetrics interfaces(const AmrGas& grid) {
    InterfaceMetrics result;
    if (grid.levels().size() < 2) return result;
    for (int lev=1; lev<grid.levels().size(); ++lev) {
    const auto& coarse = *grid.levels()[lev-1]; const auto& fine = *grid.levels()[lev];
    const int ratio = grid.layout().baseRatio >> lev;
    const double h = fine.geometry.CellSize(0);
    const auto& patches = grid.layout().refinedPatches[lev-1];
    for (const auto& p : patches) for (int d = 0; d < 2; ++d) {
        const int lo = (d == 0 ? p.x : p.y)/ratio, hi = (d == 0 ? p.x + p.width : p.y + p.height)/ratio;
        const int tlo = (d == 0 ? p.y : p.x)/ratio, thi = (d == 0 ? p.y + p.height : p.x + p.width)/ratio;
        for (int side : {lo, hi}) for (int t = tlo; t < thi; t += 2) {
            const int i = d == 0 ? side : t, j = d == 1 ? side : t;
            const int outsideX = i - (d == 0 && side == lo), outsideY = j - (d == 1 && side == lo);
            if (!fine.geometry.Domain().contains(IntVect(outsideX, outsideY)) ||
                std::any_of(patches.begin(), patches.end(), [&](const auto& neighbour) {
                    return outsideX*ratio >= neighbour.x && outsideY*ratio >= neighbour.y && outsideX*ratio < neighbour.x + neighbour.width && outsideY*ratio < neighbour.y + neighbour.height;
                })) continue;
            const double a = sample(fine.velocity[d], i, j), b = sample(fine.velocity[d], i + (d == 1), j + (d == 0));
            result.fluxMismatch = std::max(result.fluxMismatch, std::abs(sample(coarse.velocity[d], i / 2, j / 2) - 0.5 * (a + b)));
            result.contrast = std::max(result.contrast, std::abs(a - b));
            for (int offset = 0; offset < 2; ++offset)
                result.cloudFlux += std::abs(sample(fine.flux[d], i + offset * (d == 1), j + offset * (d == 0), 1)) * h;
        }
    }
    }
    return result;
}

void equilibrium(int n, const std::string& layout) {
    auto [catalogue, mixture] = materials(7, 1.);
    AmrGas grid(n, n, 1. / n, catalogue, mixture, makeLayout(n, layout));
    grid.upload(initialState(n, 1., 0., mixture));
    auto before = grid.totals();
    double divergence = 0.;
    for (int step = 0; step < 20; ++step) {
        const auto result = grid.step(0.016, 1.e-10, 0., {0.3, -0.2});
        require(result.projection.converged, "Equilibrium projection failed");
        divergence = std::max(divergence, result.projection.max_divergence);
    }
    auto after = grid.totals();
    require(after.maxSpeed < 2.e-9 && massError(before, after) < 1.e-12 && after.maxVolumeError < 1.e-9, "Body-force equilibrium drifted");
    std::cout << "equilibrium," << layout << ",speed=" << after.maxSpeed << ",divergence=" << divergence
              << ",mass_error=" << massError(before, after) << ",volume_error=" << after.maxVolumeError << ",PASS\n";
}

void translatingMaterialAccuracy() {
    // Isolate the transport operator with prescribed constant interior velocity.
    // The compact cloud never reaches the closed walls; compare exact Gaussian
    // cell averages, not point samples. The full step/projection has separate tests.
    double previous=0.;
    for (int n:{32,64,128}) {
        auto [catalogue,mixture]=materials(2,1.);
        constexpr double world=4.,speedX=.4,speedY=.2,end=1.5,sigma=.18;
        const double h=world/n;
        auto gaussian=[&](int i,double center) {
            return sigma*std::sqrt(pi/2.)/h*(std::erf(((i+1)*h-center)/(std::sqrt(2.)*sigma))-std::erf((i*h-center)/(std::sqrt(2.)*sigma)));
        };
        auto state=initialState(n,world,0.,mixture);
        for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
            const double fraction=.02+.8*gaussian(x,1.4)*gaussian(y,1.4);
            state.gas_mass[0][y*n+x]=.1*h*h*(1-fraction); state.gas_mass[1][y*n+x]=.1*h*h*fraction;
        }
        AmrGas grid(n,n,h,catalogue,mixture,{}); grid.upload(state);
        for(const auto& level:grid.levels()) {
            for(MFIter mfi(level->conserved);mfi.isValid();++mfi) {
                const auto q=level->conserved.array(mfi);
                for(BoxIterator it(mfi.validbox());it.ok();++it) { q(it()[0],it()[1],0,2)=.1*speedX; q(it()[0],it()[1],0,3)=.1*speedY; }
            }
            for(int d=0;d<2;++d) for(MFIter mfi(level->velocity[d]);mfi.isValid();++mfi) {
                const auto u=level->velocity[d].array(mfi);
                for(BoxIterator it(mfi.validbox());it.ok();++it) u(it()[0],it()[1],0)=(it()[d]==0 || it()[d]==n)?0.:(d?speedY:speedX);
            }
        }
        const auto before=grid.totals(); double time=0.;
        while(time<end) { const double dt=std::min(end-time,.2*h/(speedX+speedY)); grid.transport(dt); time+=dt; }
        const auto after=grid.totals(); require(after.finite && after.minMassDensity>=-1e-12 && massError(before,after)<1e-12,"Manufactured transport failed conservation/positivity");
        double error=0.; int samples=0;
        for(int y=n/4;y<3*n/4;++y) for(int x=n/4;x<3*n/4;++x) {
            const double exact=.02+.8*gaussian(x,1.4+speedX*end)*gaussian(y,1.4+speedY*end);
            error+=std::abs(sample(grid.levels()[0]->conserved,x,y,1)/.1-exact); ++samples;
        }
        error/=samples;
        std::cout<<"linear_transport,n="<<n<<",L1="<<error<<",convergence_ratio="<<(previous?previous/error:0.)<<'\n';
        if(previous) require(previous/error>2.5,"Smooth prescribed transport must converge faster than first order");
        previous=error;
    }
}

void conservativeTransport(int n, const std::string& layout) {
    auto [catalogue, mixture] = materials(7, 100.);
    AmrGas grid(n, n, 16. / n, catalogue, mixture, makeLayout(n, layout));
    grid.upload(initialState(n, 16., 2., mixture, true));
    const auto before = grid.totals();
    const double dt = grid.stableTimeStep();
    grid.transport(dt);
    const auto after = grid.totals();
    const double momentumError = std::hypot(after.momentum[0] - before.momentum[0], after.momentum[1] - before.momentum[1]);
    const double mass = std::accumulate(before.ingredientMass.begin(), before.ingredientMass.end(), 0.);
    require(momentumError < 1.e-12 * mass && massError(before, after) < 1.e-12 && after.minMassDensity >= -1.e-13,
            "Shared transport flux lost mass, momentum or positivity at CFL limit");
    std::cout << "transport," << layout << ",dt=" << dt << ",momentum_error=" << momentumError << ",mass_error=" << massError(before, after)
              << ",min_density=" << after.minMassDensity << ",PASS\n";
}

double materialChange(const AmrGas& grid, const AmrGas::State& initial, int n, double world) {
    double change = 0.;
    const double fineArea = std::pow(world / n, 2), intrinsic = grid.gasMixture().components[1].density;
    for (int lev = 0; lev < grid.levels().size(); ++lev) {
        const auto& level = *grid.levels()[lev];
        const int ratio = grid.layout().baseRatio >> lev;
        const double area = fineArea * ratio * ratio;
        for (MFIter mfi(level.conserved); mfi.isValid(); ++mfi) {
            const auto q = level.conserved.const_array(mfi);
            const auto mask = level.active.const_array(mfi);
            for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                const int i = bit()[0], j = bit()[1];
                if (!mask(i, j, 0)) continue;
                double mass = 0.;
                for (int y = j * ratio; y < (j + 1) * ratio; ++y) for (int x = i * ratio; x < (i + 1) * ratio; ++x)
                    mass += initial.gas_mass[1][y * n + x];
                change += std::abs(q(i, j, 0, 1) * area - mass) / intrinsic;
            }
        }
    }
    return change / (world * world);
}

void run(int n, const std::string& layout, int count, double contrast, int steps, double world, double tolerance, double damping, bool check, int maxGrid) {
    auto [catalogue, mixture] = materials(count, contrast);
    const auto initial = initialState(n, world, 2., mixture, contrast != 1.);
    const auto setupStart = Clock::now();
    AmrGas grid(n, n, world / n, catalogue, mixture, makeLayout(n, layout), maxGrid);
    grid.upload(initial, tolerance);
    const double initialError = initialFaceError(grid, initial, n);
    const double setupMs = std::chrono::duration<double, std::milli>(Clock::now() - setupStart).count();
    const auto before = grid.totals();
    double fullMs = 0., transportMs = 0., projectMs = 0., maxDiv = 0., peakMs = 0., crossing = 0., worstMass = 0., worstVolume = 0., worstMin = before.minMassDensity;
    double peakSpeed = before.maxSpeed, peakEnergyRatio = 1.;
    double regridMs = 0., peakRegridMs = 0., rawMomentumError = 0.;
    int iterations = 0, substeps = 0, changes = 0;
    const int warmup = check ? 0 : 20;
    for (int s = -warmup; s < steps; ++s) {
        const auto start = Clock::now();
        if ((layout == "changing" || layout == "scene-changing" || layout == "sparse-changing" || layout == "production-changing" || layout == "nested-changing") && s > 0 && s % 125 == 0) {
            const auto rawBefore = check ? grid.totals() : fluid::reference::GasTotals{};
            require(grid.regrid(makeLayout(n, layout, s / 125)), "Moving layout did not change");
            if (check) {
                const auto rawAfter = grid.totals();
                const double error = std::hypot(rawAfter.momentum[0] - rawBefore.momentum[0], rawAfter.momentum[1] - rawBefore.momentum[1]);
                rawMomentumError = std::max(rawMomentumError, error);
                require(massError(rawBefore, rawAfter) < 1.e-12 && error < 1.e-10, "Moving regrid lost conserved state");
            }
            const auto restored = grid.project(tolerance);
            require(restored.converged, "Moving pressure restoration failed");
            iterations += restored.iterations; maxDiv = std::max(maxDiv, restored.max_divergence);
            const double cost = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            regridMs += cost; peakRegridMs = std::max(peakRegridMs, cost); ++changes;
        }
        auto result = grid.step(0.016, tolerance, damping);
        const double cost = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        require(result.projection.converged, "Step projection failed");
        maxDiv = std::max(maxDiv, result.projection.max_divergence);
        if (s >= 0) {
            fullMs += cost; transportMs += result.transportMs; projectMs += result.projectionMs; peakMs = std::max(peakMs, cost);
            iterations += result.projection.iterations; substeps += result.substeps;
        }
        if (check) {
            const auto totals = grid.totals();
            worstMass = std::max(worstMass, massError(before, totals));
            worstVolume = std::max(worstVolume, totals.maxVolumeError);
            worstMin = std::min(worstMin, totals.minMassDensity);
            peakSpeed = std::max(peakSpeed, totals.maxSpeed);
            peakEnergyRatio = std::max(peakEnergyRatio, totals.kineticEnergy / before.kineticEnergy);
            require(totals.finite && totals.minMassDensity >= -1.e-13 && worstMass < 2.e-11, "Non-finite, negative or non-conserved ingredients");
            require(totals.maxVolumeError < 2.e-6, "Gas volume closure drifted");
            const auto seam = interfaces(grid);
            require(seam.fluxMismatch < 1.e-14, "Coarse/fine face averaging failed");
            crossing += 0.016 * seam.cloudFlux;
        }
    }
    const auto after = grid.totals();
    const auto seam = interfaces(grid);
    const double changed = materialChange(grid, initial, n, world);
    require(after.finite && after.minMassDensity >= -1.e-13 && massError(before, after) < 2.e-11, "Final ingredient conservation failed");
    require(after.maxSpeed > 0. && after.kineticEnergy < before.kineticEnergy * 1.01, "Flow froze or gained excessive energy");
    require(changed > 1.e-7, "Cloud material did not evolve");
    if (check && steps >= 500 && grid.levels().size() > 1) require(crossing > 0.1 * before.ingredientMass[1] && seam.contrast > 1.e-7, "Insufficient boundary transport or tied subfaces");
    std::cout << n << ',' << layout << ',' << count << ',' << contrast << ',' << steps << ',' << world << ',' << tolerance << ',' << damping << ','
              << grid.activeCellCount() << ',' << grid.allocatedCellCount() << ',' << setupMs << ',' << fullMs / steps << ',' << transportMs / steps << ','
              << projectMs / steps << ',' << peakMs << ',' << double(iterations) / steps << ',' << double(substeps) / steps << ',' << maxDiv << ','
              << massError(before, after) << ',' << after.maxVolumeError << ',' << after.minMassDensity << ',' << after.maxSpeed << ','
              << after.kineticEnergy / before.kineticEnergy << ',' << seam.fluxMismatch << ',' << seam.contrast << ','
              << crossing / before.ingredientMass[1] << ',' << worstMass << ',' << worstVolume << ',' << worstMin << ',' << initialError << ',' << changed << ','
              << peakSpeed << ',' << peakEnergyRatio << ',' << regridMs / steps << ',' << peakRegridMs << ',' << changes << ',' << rawMomentumError << ",PASS\n";
}
}

int main(int argc, char* argv[]) {
    amrex::SetVerbose(0);
    amrex::Initialize(argc, argv);
    int code = 0;
    try {
        ParmParse pp;
        int n = 64, steps = 1500, count = 2, maxGrid = 64, reverse = 0;
        double contrast = 1., tolerance = 1.e-9, world = 16., damping = 0.;
        std::string mode = "check", layout = "all";
        pp.query("mode", mode); pp.query("layout", layout); pp.query("n_cell", n); pp.query("steps", steps);
        pp.query("components", count); pp.query("contrast", contrast); pp.query("tolerance", tolerance); pp.query("world", world);
        pp.query("damping", damping); pp.query("max_grid_size", maxGrid);
        pp.query("reverse", reverse);
        const bool check = mode == "check";
        std::cout << std::setprecision(12);
        std::cout << "n,layout,components,contrast,steps,world,tolerance,damping,active_cells,allocated_cells,setup_ms,mean_step_ms,transport_ms,projection_ms,peak_step_ms,iterations,substeps,max_divergence,mass_error,volume_error,min_density,max_speed,energy_ratio,flux_error,subface_contrast,cloud_boundary_transfers,worst_mass_error,worst_volume_error,worst_min_density,initial_face_error,cloud_L1_change,peak_speed,peak_energy_ratio,regrid_ms,peak_regrid_ms,layout_changes,raw_momentum_error,status\n";
        if (check) { translatingMaterialAccuracy(); hierarchyEndpoints(n); boundaryFillMatchesReference(n); conservativeRegrid(n); ingredientLabelsPreserveFlow(n); independentIngredients(n); }
        std::vector<std::string> layouts{"uniform", "patch", "four", "edge", "coarse", "matched-one", "matched-two", "matched-four", "changing", "scene-changing", "sparse-changing", "production-changing",
            "coarse4", "half4", "nested", "nested-edge", "nested-four", "nested-changing"};
        if (reverse) std::reverse(layouts.begin(), layouts.end());
        for (const std::string& name : layouts)
          if ((layout == "all" && (name == "uniform" || name == "patch" || name == "four" || (check && (name == "edge" || name == "coarse" || name == "changing" || name == "scene-changing" || name == "sparse-changing" || name == "production-changing")))) ||
              (layout == "all" && check && (name == "coarse4" || name == "half4" || name == "nested" || name == "nested-edge" || name == "nested-four" || name == "nested-changing")) ||
              ((layout == "matched" || layout == "costs") && (name == "uniform" || name == "matched-one" || name == "matched-two" || name == "matched-four" || name == "changing" || name == "scene-changing")) ||
              (layout == "costs" && (name == "patch" || name == "four" || name == "sparse-changing" || name == "production-changing")) || layout == name) {
            if (check) { equilibrium(n, name); conservativeTransport(n, name); }
            run(n, name, count, contrast, steps, world, tolerance, damping, check, maxGrid);
            if (check && contrast == 1.) run(n, name, 7, 100., steps, world, tolerance, damping, true, maxGrid);
            std::cout << std::flush;
        }
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; code = 1; }
    amrex::Finalize();
    return code;
}
