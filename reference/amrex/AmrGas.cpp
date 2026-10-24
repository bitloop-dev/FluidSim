#include "AmrGas.h"
#include "AmrGasInterpolation.h"
#include "FluidInterfaceGeometry.h"
#include <AMReX_BoxIterator.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_GMRES_MLMG.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace fluid::reference {
using namespace amrex;
using Clock = std::chrono::steady_clock;
namespace {
GasLinearInterpolater gasLinearInterpolation;
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
void sortPatches(GasLayout& layout) {
    for (auto& patches : layout.refinedPatches) std::sort(patches.begin(), patches.end(), [](const GasPatch& a, const GasPatch& b) {
        return std::tie(a.y, a.x, a.height, a.width) < std::tie(b.y, b.x, b.height, b.width);
    });
}
}

AmrGas::AmrGas(int width, int height, double cellSize, ElementCatalogue catalogue, FluidMixture gasMixture, GasLayout layout, int maxGridSize)
    : AmrGas(width,height,cellSize,std::move(catalogue),{},std::move(gasMixture),std::move(layout),maxGridSize) {}

AmrGas::AmrGas(int width, int height, double cellSize, ElementCatalogue catalogue, FluidMixture liquidMixture, FluidMixture gasMixture, GasLayout layout, int maxGridSize)
    : width(width), height(height), components(static_cast<int>(liquidMixture.components.size()+gasMixture.components.size())),
      maxGridSize(maxGridSize), liquidComponents(static_cast<int>(liquidMixture.components.size())), cellSize(cellSize),
      catalogue(std::move(catalogue)), mixture(liquidMixture), liquidPhase(std::move(liquidMixture)),
      gasPhase(std::move(gasMixture)), gridLayout(std::move(layout)) {
    mixture.components.insert(mixture.components.end(),gasPhase.components.begin(),gasPhase.components.end());
    equalIngredientDensities=std::all_of(mixture.components.begin(),mixture.components.end(),[&](const auto& c) {
        return c.density==mixture.components.front().density;
    });
    sortPatches(gridLayout);
    const int count = 1 + static_cast<int>(gridLayout.refinedPatches.size());
    const int fields = components + 2;
    RealBox physical({0., 0.}, {width * cellSize, height * cellSize});
    Array<int, 2> periodic{0, 0};
    Vector<Geometry> geometry;
    Vector<Array<MultiFab*, 2>> faceVelocity;
    Vector<Array<MultiFab const*, 2>> faceBeta;
    for (int lev = 0; lev < count; ++lev) {
        const int ratio = gridLayout.baseRatio >> lev;
        Box domain(IntVect(0, 0), IntVect(width / ratio - 1, height / ratio - 1));
        BoxArray grids(domain);
        if (lev != 0) {
            Vector<Box> boxes;
            for (const auto& p : gridLayout.refinedPatches[lev - 1])
                boxes.emplace_back(IntVect(p.x / ratio, p.y / ratio), IntVect((p.x + p.width) / ratio - 1, (p.y + p.height) / ratio - 1));
            grids = BoxArray(boxes.data(), static_cast<int>(boxes.size()));
        }
        grids.maxSize(maxGridSize);
        DistributionMapping mapping(grids);
        data.emplace_back(std::make_unique<Level>());
        auto& level = *data.back();
        level.geometry.define(domain, &physical, 0, periodic.data());
        level.conserved.define(grids, mapping, fields, 2);
        level.slopes.define(grids, mapping, 2 * components, 1);
        if(liquidComponents) {
            // Fraction, gas volume, interface normal and offset; derived only.
            level.phaseGeometry.define(grids,mapping,5,1);
            level.splitWeights.define(grids,mapping,fields,0);
        }
        level.pressure.define(grids, mapping, 1, 1);
        level.pressure.setVal(0.);
        level.divergence.define(grids, mapping, 1, 0);
        for (int d = 0; d < 2; ++d) {
            const auto faces = amrex::convert(grids, IntVect::TheDimensionVector(d));
            level.velocity[d].define(faces, mapping, 1, 0);
            level.beta[d].define(faces, mapping, 1, 0);
            level.beta[d].setVal(1.);
            level.flux[d].define(faces, mapping, fields, 0);
            level.predictedVelocity[d].define(faces, mapping, 1, 0);
        }
        geometry.push_back(level.geometry);
        faceVelocity.push_back(GetArrOfPtrs(level.velocity));
        faceBeta.push_back(GetArrOfConstPtrs(level.beta));
        allocatedCells += grids.numPts();
    }
    for (int lev = 0; lev < count; ++lev) {
        auto& level = *data[lev];
        const auto& grids = level.conserved.boxArray();
        const auto& mapping = level.conserved.DistributionMap();
        if (lev + 1 < count) level.active = makeFineMask(grids, mapping, data[lev + 1]->conserved.boxArray(), IntVect(2), 1, 0);
        else { level.active.define(grids, mapping, 1, 0); level.active.setVal(1); }
        activeCells += grids.numPts();
        if (lev + 1 < count) activeCells -= data[lev + 1]->conserved.boxArray().numPts() / 4;
    }
    boundary.resize(fields);
    for (int n = 0; n < fields; ++n) for (int d = 0; d < 2; ++d) {
        const int bc = n == components + d ? BCType::reflect_odd : BCType::reflect_even;
        boundary[n].setLo(d, bc); boundary[n].setHi(d, bc);
    }
    boundaryFill.resize(count - 1);
    for (int lev = 1; lev < count; ++lev) {
        const auto& fine = *data[lev];
        // Both field groups use the same linear stencil. Retain its
        // boundary buffers/communication layout, never the sampled state.
        const auto& info = FabArrayBase::TheFPinfo(fine.conserved, fine.conserved, IntVect(2), gasLinearInterpolation.BoxCoarsener(IntVect(2)),
                                                  fine.geometry, data[lev - 1]->geometry, nullptr);
        if (!info.ba_crse_patch.empty()) {
            auto& buffer = boundaryFill[lev - 1];
            buffer = std::make_unique<BoundaryFill>();
            buffer->coarse.define(info.ba_crse_patch, info.dm_patch, fields, 0, MFInfo(), *info.fact_crse_patch);
            buffer->fine.define(info.ba_fine_patch, info.dm_patch, fields, 0, MFInfo(), *info.fact_fine_patch);
        }
    }
    projector = std::make_unique<Hydro::MacProjector>(faceVelocity, faceBeta, geometry);
    const Array<LinOpBCType, 2> walls{LinOpBCType::Neumann, LinOpBCType::Neumann};
    projector->setDomainBC(walls, walls);
    projector->setVerbose(0);
    // GMRES uses Hydro's operator directly, so establish its homogeneous
    // level boundaries before preparing the multigrid preconditioner.
    for (int lev = 0; lev < count; ++lev) projector->getLinOp().setLevelBC(lev, nullptr);
    projector->getMLMG().setMaxIter(liquidComponents ? 2000 : 200);
    projector->getMLMG().setThrowException(true);
}

AmrGas::~AmrGas() = default;

bool AmrGas::regrid(GasLayout layout) {
    sortPatches(layout);
    if (layout == gridLayout) return false;
    restrictState();
    AmrGas next(width, height, cellSize, catalogue, liquidPhase, gasPhase, std::move(layout), maxGridSize);
    const int fields = components + 2;
    CpuBndryFuncFab function(nullptr);
    auto refine = [&](MultiFab& target, const Geometry& geometry, MultiFab& coarse, const Geometry& coarseGeometry, int ratio) {
        PhysBCFunct<CpuBndryFuncFab> coarseBC(coarseGeometry, boundary, function), fineBC(geometry, boundary, function);
        MultiFab* oldFine = nullptr;
        for (int oldLev = 0; oldLev < data.size(); ++oldLev)
            if ((gridLayout.baseRatio >> oldLev) == ratio) oldFine = &data[oldLev]->conserved;
        // FillPatch copies retained fine cells exactly and interpolates only newly refined cells from restricted real coarse state.
        // Equal-density labels must not flatten momentum. With density contrast,
        // one common limiter also preserves m=rho*u across constant-velocity refinement.
        for (int first=0;first<fields;) {
            const int count=first==0 ? (equalIngredientDensities ? components : fields) : 2;
            auto* interpolation=&gasLinearInterpolation;
            if (oldFine) {
                FillPatchTwoLevels(target, IntVect(0), 0., {&coarse}, {0.}, {oldFine}, {0.}, first, first, count,
                                   coarseGeometry, geometry, coarseBC, first, fineBC, first, IntVect(2), interpolation, boundary, first);
            } else {
                InterpFromCoarseLevel(target, IntVect(0), 0., coarse, first, first, count, coarseGeometry, geometry,
                                     coarseBC, first, fineBC, first, IntVect(2), interpolation, boundary, first);
            }
            first+=count;
        }
    };
    // A finer new base may need an intermediate full-domain resolution. This
    // transfer-only bridge retains old overlap at each 2:1 join; no fine state
    // survives outside the new physical hierarchy.
    std::unique_ptr<MultiFab> bridge;
    Geometry bridgeGeometry;
    MultiFab* baseSource = &data[0]->conserved;
    const Geometry* baseGeometry = &data[0]->geometry;
    for (int ratio = gridLayout.baseRatio / 2; ratio > next.gridLayout.baseRatio; ratio /= 2) {
        const auto domain = amrex::refine(baseGeometry->Domain(), 2);
        BoxArray grids(domain); grids.maxSize(maxGridSize);
        auto refined = std::make_unique<MultiFab>(grids, DistributionMapping(grids), fields, 0);
        auto geometry = amrex::refine(*baseGeometry, 2);
        refine(*refined, geometry, *baseSource, *baseGeometry, ratio);
        bridge = std::move(refined);
        bridgeGeometry = geometry;
        baseSource = bridge.get(); baseGeometry = &bridgeGeometry;
    }
    for (int lev = 0; lev < next.data.size(); ++lev) {
        auto& target = *next.data[lev];
        const int ratio = next.gridLayout.baseRatio >> lev;
        if (ratio >= gridLayout.baseRatio) {
            // Restrict the original state, not a newly coarsened parent. In
            // particular 1->4->2 must retain the original 2-cell averages.
            if (ratio == gridLayout.baseRatio) target.conserved.ParallelCopy(data[0]->conserved, 0, 0, fields);
            else average_down(data[0]->conserved, target.conserved, 0, fields, IntVect(ratio / gridLayout.baseRatio));
        } else if (lev == 0) {
            refine(target.conserved, target.geometry, *baseSource, *baseGeometry, ratio);
        } else {
            auto& coarse = *next.data[lev - 1];
            refine(target.conserved, target.geometry, coarse.conserved, coarse.geometry, ratio);
        }
    }
    next.restrictState();
    data.swap(next.data);
    projector.swap(next.projector);
    boundaryFill.swap(next.boundaryFill);
    std::swap(gridLayout, next.gridLayout);
    std::swap(activeCells, next.activeCells);
    std::swap(allocatedCells, next.allocatedCells);
    lastPressureInterval = 0.;
    return true;
}

void AmrGas::averageFaces(Array<MultiFab, 2> Level::* field) {
    for (int lev = static_cast<int>(data.size()) - 1; lev > 0; --lev)
        average_down_faces(GetArrOfConstPtrs(data[lev].get()->*field), GetArrOfPtrs(data[lev - 1].get()->*field), IntVect(2), data[lev - 1]->geometry);
}

void AmrGas::restrictState() {
    for (int lev = static_cast<int>(data.size()) - 1; lev > 0; --lev)
        average_down(data[lev]->conserved, data[lev - 1]->conserved, 0, components + 2, IntVect(2));
}

void AmrGas::fillGhosts() {
    CpuBndryFuncFab function(nullptr);
    for (int lev = 0; lev < data.size(); ++lev) {
        auto& fine = *data[lev];
        PhysBCFunct<CpuBndryFuncFab> fineBC(fine.geometry, boundary, function);
        if (lev > 0 && boundaryFill[lev - 1]) {
            auto& coarse = *data[lev - 1];
            PhysBCFunct<CpuBndryFuncFab> coarseBC(coarse.geometry, boundary, function);
            auto& buffer = *boundaryFill[lev - 1];
            buffer.coarse.setDomainBndry(std::numeric_limits<double>::quiet_NaN(), coarse.geometry);
            FillPatchSingleLevel(buffer.coarse, 0., {&coarse.conserved}, {0.}, 0, 0, components + 2, coarse.geometry, coarseBC, 0);
            // Match regrid's ingredient/constant-velocity interpolation contract.
            for (int first=0;first<components+2;) {
                const int count=first==0 ? (equalIngredientDensities ? components : components+2) : 2;
                auto* interpolation=&gasLinearInterpolation;
                FillPatchInterp(buffer.fine, first, buffer.coarse, first, count, IntVect(0), coarse.geometry, fine.geometry,
                                fine.geometry.Domain(), IntVect(2), interpolation, boundary, first);
                first+=count;
            }
            fine.conserved.ParallelCopyToGhost(buffer.fine, 0, 0, components + 2, IntVect(0), IntVect(2));
        }
        // With source == destination, AMReX exchanges only ghosts, then applies
        // physical BCs. Same-level data takes precedence over coarse fill.
        FillPatchSingleLevel(fine.conserved, 0., {&fine.conserved}, {0.}, 0, 0, components + 2, fine.geometry, fineBC, 0);
    }
}

void AmrGas::upload(const State& state, double tolerance) {
    // A gas-only instance must not silently discard the other phase's mass.
    if (!liquidComponents && std::any_of(state.liquid_fraction.begin(), state.liquid_fraction.end(), [](double value) { return value != 0.; }))
        throw std::invalid_argument("Liquid input requires a configured liquid mixture");
    if(!liquidComponents) for (const auto& mass : state.liquid_mass)
        if (std::any_of(mass.begin(), mass.end(), [](double value) { return value != 0.; }))
            throw std::invalid_argument("AmrGas cannot discard liquid ingredient mass");
    const double fineArea = cellSize * cellSize;
    for (int lev = 0; lev < data.size(); ++lev) {
        auto& level = *data[lev];
        const int ratio = gridLayout.baseRatio >> lev;
        for (MFIter mfi(level.conserved); mfi.isValid(); ++mfi) {
            const auto q = level.conserved.array(mfi);
            for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                const int i = bit()[0], j = bit()[1];
                for (int n = 0; n < components + 2; ++n) q(i, j, 0, n) = 0.;
                for (int y = j * ratio; y < (j + 1) * ratio; ++y) for (int x = i * ratio; x < (i + 1) * ratio; ++x) {
                    const auto index = y * width + x;
                    double rho = 0.;
                    for (int n = 0; n < components; ++n) {
                        const double density = (n<liquidComponents ? state.liquid_mass[n][index] : state.gas_mass[n-liquidComponents][index]) / fineArea;
                        q(i, j, 0, n) += density / (ratio * ratio);
                        rho += density;
                    }
                    const double u = state.u.empty() ? 0. : 0.5 * (state.u[y * (width + 1) + x] + state.u[y * (width + 1) + x + 1]);
                    const double v = state.v.empty() ? 0. : 0.5 * (state.v[y * width + x] + state.v[(y + 1) * width + x]);
                    q(i, j, 0, components) += rho * u / (ratio * ratio);
                    q(i, j, 0, components + 1) += rho * v / (ratio * ratio);
                }
            }
        }
        level.pressure.setVal(0.);
    }
    restrictState();
    if (!project(tolerance).converged) throw std::runtime_error("Initial fluid projection did not converge");
}

FluidProjectionResult AmrGas::project(double tolerance, double dt, std::array<double, 2> acceleration) {
    lastPressureInterval = dt;
    if(dt>0.)lastAcceleration=acceleration;
    fillGhosts();
    Vector<Array<MultiFab const*, 2>> beta;
    Vector<MultiFab*> phi;
    const int nc = components;
    for (auto& entry : data) {
        auto& level = *entry;
        const auto domain = level.geometry.Domain();
        for (int d = 0; d < 2; ++d) {
            const int di = d == 0, dj = d == 1;
            const double impulse = dt * acceleration[d];
#pragma omp parallel
            for (MFIter mfi(level.velocity[d], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const auto q = level.conserved.const_array(mfi);
                const auto u = level.velocity[d].array(mfi), b = level.beta[d].array(mfi);
                ParallelFor(mfi.tilebox(), [=](int i, int j, int k) noexcept {
                    double left = 0., right = 0.;
                    for (int n = 0; n < nc; ++n) { left += q(i - di, j - dj, k, n); right += q(i, j, k, n); }
                    b(i, j, k) = 2. / (left + right);
                    const int coordinate = d == 0 ? i : j;
                    const bool wall = coordinate == domain.smallEnd(d) || coordinate == domain.bigEnd(d) + 1;
                    u(i, j, k) = wall ? 0. : 0.5 * (q(i - di, j - dj, k, nc + d) / left + q(i, j, k, nc + d) / right) + impulse;
                });
            }
            level.velocity[d].OverrideSync();
        }
        beta.push_back(GetArrOfConstPtrs(level.beta));
        phi.push_back(&level.pressure);
    }
    // Match MacProjector's composite predictor before testing the actual divergence contract.
    averageFaces(&Level::velocity);
    const double predictedDivergence = maxDivergence();
    const bool needsProjection = predictedDivergence > tolerance;
    int iterations = 0;
    if (needsProjection) {
        for (auto& entry : data) for (int d = 0; d < 2; ++d) MultiFab::Copy(entry->predictedVelocity[d], entry->velocity[d], 0, 0, 1, 0);
        {
            projector->updateBeta(beta);
            if (equalIngredientDensities) {
                // Without material density contrast, standalone multigrid already
                // converges in a few cycles and avoids Krylov's extra vector work.
                projector->project(phi, 0., tolerance);
                iterations = projector->getMLMG().getNumIters();
            } else {
                Vector<MultiFab const*> rhs;
                Vector<Array<MultiFab*, 2>> pressureFlux;
                for (auto& entry : data) {
                    // Hydro's operator is -D(beta G); maxDivergence already computed D(U*).
                    entry->divergence.mult(-1.);
                    rhs.push_back(&entry->divergence);
                    pressureFlux.push_back(GetArrOfPtrs(entry->velocity));
                }
                GMRESMLMG solver(projector->getMLMG());
                solver.setMaxIters(liquidComponents ? 2000 : 200);
                // This absolute L2 target is stricter than the requested maximum
                // divergence. The corrected composite faces are checked below too.
                solver.solve(phi, rhs, 0., tolerance);
                iterations = solver.getNumIters();
                // Hydro fills coarse/fine pressure boundaries before taking gradients.
                // Reuse velocity storage for the correction, then restore the predictor.
                projector->getFluxes(pressureFlux, phi, MLMG::Location::FaceCenter);
                for (auto& entry : data) for (int d = 0; d < 2; ++d)
                    MultiFab::Add(entry->velocity[d], entry->predictedVelocity[d], 0, 0, 1, 0);
                averageFaces(&Level::velocity);
            }
        }
    } else {
        for (auto& entry : data) entry->pressure.setVal(0.);
        if (dt * acceleration[0] == 0. && dt * acceleration[1] == 0.) return {0, predictedDivergence, true};
    }
    for (auto& entry : data) {
        auto& level = *entry;
        const auto domain = level.geometry.Domain();
#pragma omp parallel
        for (MFIter mfi(level.conserved, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto q = level.conserved.array(mfi);
            const auto px = level.predictedVelocity[0].const_array(mfi), py = level.predictedVelocity[1].const_array(mfi);
            const auto ux = level.velocity[0].const_array(mfi), uy = level.velocity[1].const_array(mfi);
            const double ax = dt * acceleration[0], ay = dt * acceleration[1];
            ParallelFor(mfi.tilebox(), [=](int i, int j, int k) noexcept {
                double rho = 0.;
                for (int n = 0; n < nc; ++n) rho += q(i, j, k, n);
                // The cell velocity receives only the averaged MAC correction, as in incflo's cell projection.
                // Body-force faces vanish on closed walls; averaging force and pressure together preserves rest equilibrium.
                const double wx = 0.5 * ((i != domain.smallEnd(0)) + (i != domain.bigEnd(0)));
                const double wy = 0.5 * ((j != domain.smallEnd(1)) + (j != domain.bigEnd(1)));
                // Projected and predicted faces are both averaged down; their difference is the same composite correction.
                const double cx = needsProjection ? 0.5 * ((ux(i, j, k) - px(i, j, k)) + (ux(i + 1, j, k) - px(i + 1, j, k))) : 0.;
                const double cy = needsProjection ? 0.5 * ((uy(i, j, k) - py(i, j, k)) + (uy(i, j + 1, k) - py(i, j + 1, k))) : 0.;
                q(i, j, k, nc) += rho * (cx + wx * ax);
                q(i, j, k, nc + 1) += rho * (cy + wy * ay);
            });
        }
    }
    restrictState();
    FluidProjectionResult result;
    result.iterations = iterations;
    result.max_divergence = needsProjection ? maxDivergence() : predictedDivergence;
    result.converged = std::isfinite(result.max_divergence) && result.max_divergence <= 1.01 * tolerance;
    return result;
}

void AmrGas::transport(double dt) {
    if (dt == 0.) return;
    if(liquidComponents) { transportPhases(dt); return; }
    fillGhosts();
    const int fields = components + 2;
    const int nc=components;
    const bool equalDensity=equalIngredientDensities;
    for (auto& entry : data) {
        auto& level = *entry;
        const double halfDt=.5*dt/level.geometry.CellSize(0);
#pragma omp parallel
        for (MFIter mfi(level.slopes); mfi.isValid(); ++mfi) {
            const auto q=level.conserved.const_array(mfi); const auto slopes=level.slopes.array(mfi);
            ParallelFor(amrex::grow(mfi.validbox(),1),[=](int i,int j,int k) noexcept {
                const int count=equalDensity ? nc : fields;
                auto sample=[&](int dx,int dy,int n) { return q(i+dx,j+dy,k,n); };
                const auto limiter=linearLimiter(count,sample);
                for (int n=0; n<nc; ++n) {
                    const auto s=linearSlope(n,sample,limiter);
                    slopes(i,j,k,n)=s.x; slopes(i,j,k,nc+n)=s.y;
                }
            });
        }
        for (int d = 0; d < 2; ++d) {
            const int di = d == 0, dj = d == 1;
#pragma omp parallel
            for (MFIter mfi(level.flux[d], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const auto q = level.conserved.const_array(mfi), u = level.velocity[d].const_array(mfi);
                const auto slopes=level.slopes.const_array(mfi); const auto flux = level.flux[d].array(mfi);
                ParallelFor(mfi.tilebox(), [=](int i, int j, int k) noexcept {
                    const double speed=u(i,j,k);
                    const bool left=speed>=0.;
                    const int x=i-(left?di:0), y=j-(left?dj:0);
                    double rho=0.; for (int n=0; n<nc; ++n) rho+=q(x,y,k,n);
                    // Trace the donor plane back half a step in both directions.
                    // The normal projected face speed keeps that point inside
                    // the donor; transverse tracing is bounded by the same box.
                    const double normal=(left?.5:-.5)-halfDt*speed;
                    const double transverse=std::clamp(-halfDt*q(x,y,k,nc+1-d)/rho,-.5,.5);
                    const double dx=di?normal:transverse, dy=dj?normal:transverse;
                    double massFlux=0.;
                    for (int n=0; n<nc; ++n) {
                        const double transfer=speed*(q(x,y,k,n)+dx*slopes(x,y,k,n)+dy*slopes(x,y,k,nc+n));
                        flux(i,j,k,n)=transfer; massFlux+=transfer;
                    }
                    // Preserve the existing donor-velocity momentum scheme,
                    // using exactly the mass transported by the ingredients.
                    for (int axis=0; axis<2; ++axis) flux(i,j,k,nc+axis)=massFlux*q(x,y,k,nc+axis)/rho;
                });
            }
            level.flux[d].OverrideSync();
        }
    }
    // Same-timestep reflux: each coarse boundary uses the sum of the actual fine ingredient AND momentum fluxes.
    averageFaces(&Level::flux);
    for (auto& entry : data) {
        auto& level = *entry;
        const double scale = dt / level.geometry.CellSize(0);
#pragma omp parallel
        for (MFIter mfi(level.conserved, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto q = level.conserved.array(mfi);
            const auto fx = level.flux[0].const_array(mfi), fy = level.flux[1].const_array(mfi);
            ParallelFor(mfi.tilebox(), fields, [=](int i, int j, int k, int n) noexcept {
                q(i, j, k, n) -= scale * (fx(i + 1, j, k, n) - fx(i, j, k, n) + fy(i, j + 1, k, n) - fy(i, j, k, n));
            });
        }
    }
    restrictState();
}

void AmrGas::transportPhases(double dt) {
    const int nc=components,nl=liquidComponents,fields=nc+2;
    const auto* material=mixture.components.data();
    // Freeze the Weymouth-Yue split correction before either directional pass.
    // Its X/Y contributions cancel for a divergence-free advecting field.
    for(auto& entry:data) {
        auto& level=*entry;
#pragma omp parallel
        for(MFIter mfi(level.conserved,TilingIfNotGPU());mfi.isValid();++mfi) {
            const auto q=level.conserved.const_array(mfi);const auto w=level.splitWeights.array(mfi);
            ParallelFor(mfi.tilebox(),[=](int i,int j,int k) noexcept {
                double liquid=0.,gas=0.,rho=0.;
                for(int n=0;n<nc;++n) {
                    (n<nl?liquid:gas)+=q(i,j,k,n)/material[n].density;rho+=q(i,j,k,n);
                }
                const bool selected=liquid>.5; const double volume=selected?liquid:gas;
                double mass=0.;
                for(int n=0;n<nc;++n) {
                    w(i,j,k,n)=((n<nl)==selected)?q(i,j,k,n)/volume:0.;mass+=w(i,j,k,n);
                }
                for(int d=0;d<2;++d)w(i,j,k,nc+d)=mass*q(i,j,k,nc+d)/rho;
            });
        }
    }
    for(int d=0;d<2;++d) {
        fillGhosts();
        const int di=d==0,dj=d==1;
        for(auto& entry:data) {
            auto& level=*entry;
            const double scale=dt/level.geometry.CellSize(0);
#pragma omp parallel
            for(MFIter mfi(level.phaseGeometry);mfi.isValid();++mfi) {
                const auto q=level.conserved.const_array(mfi);const auto geometry=level.phaseGeometry.array(mfi);
                ParallelFor(amrex::grow(mfi.validbox(),1),[=](int i,int j,int k) noexcept {
                    double f=0.,gas=0.;
                    for(int n=0;n<nl;++n)f+=q(i,j,k,n)/material[n].density;
                    for(int n=nl;n<nc;++n)gas+=q(i,j,k,n)/material[n].density;
                    geometry(i,j,k,0)=f;geometry(i,j,k,1)=gas;
                    // Reconstruct the smaller phase directly. Subtracting a
                    // nearly full liquid flux can overwhelm a tiny gas sliver.
                    const bool liquid=f<=gas;const double minority=liquid?f:gas;
                    auto fraction=[&](int dx,int dy) {
                        double value=0.;for(int n=liquid?0:nl;n<(liquid?nl:nc);++n)value+=q(i+dx,j+dy,k,n)/material[n].density;
                        return value;
                    };
                    if(minority>0.) {
                        const auto normal=interface_geometry::normal(fraction);
                        geometry(i,j,k,2)=normal.x;geometry(i,j,k,3)=normal.y;
                        geometry(i,j,k,4)=interface_geometry::offset(normal,minority);
                    }
                });
            }
#pragma omp parallel
            for(MFIter mfi(level.flux[d],TilingIfNotGPU());mfi.isValid();++mfi) {
                const auto q=level.conserved.const_array(mfi),u=level.velocity[d].const_array(mfi);
                const auto geometry=level.phaseGeometry.const_array(mfi);const auto flux=level.flux[d].array(mfi);
                ParallelFor(mfi.tilebox(),[=](int i,int j,int k) noexcept {
                    const double speed=u(i,j,k);
                    if(speed==0.) { for(int n=0;n<fields;++n)flux(i,j,k,n)=0.;return; }
                    const bool positive=speed>0.;const int x=i-(positive?di:0),y=j-(positive?dj:0);
                    const double f=geometry(x,y,k,0),g=geometry(x,y,k,1),total=speed*scale;
                    const bool smallLiquid=f<=g;const double minority=smallLiquid?f:g;
                    double smallTransfer=0.;
                    if(minority>0.) {
                        using interface_geometry::Vec;
                        const double strip=std::abs(total),lo=positive?1.-strip:0.,hi=positive?1.:strip;
                        const Vec a{di?lo:0.,dj?lo:0.},b{di?hi:1.,dj?hi:1.};
                        const Vec normal{geometry(x,y,k,2),geometry(x,y,k,3)};
                        const double area=interface_geometry::area(normal,geometry(x,y,k,4),a,b);
                        // Same geometric roundoff bound as the uniform PLIC flux;
                        // neither phase may export more than its donor contains.
                        smallTransfer=std::copysign(std::clamp(area,std::max(0.,strip-(1.-minority)),std::min(strip,minority)),speed);
                    }
                    const double liquid=smallLiquid?smallTransfer:total-smallTransfer;
                    const double gas=smallLiquid?total-smallTransfer:smallTransfer;
                    double rho=0.,massFlux=0.;
                    for(int n=0;n<nc;++n) {
                        const double volume=n<nl?f:g;
                        const double transfer=volume>0. ? q(x,y,k,n)*((n<nl?liquid:gas)/volume)/scale : 0.;
                        flux(i,j,k,n)=transfer;massFlux+=transfer;rho+=q(x,y,k,n);
                    }
                    for(int axis=0;axis<2;++axis)flux(i,j,k,nc+axis)=massFlux*q(x,y,k,nc+axis)/rho;
                });
            }
            level.flux[d].OverrideSync();
        }
        // A coarse face uses exactly the fine phase/ingredient/momentum flux.
        for(int lev=int(data.size())-1;lev>0;--lev)
            average_down_faces(data[lev]->flux[d],data[lev-1]->flux[d],IntVect(2),data[lev-1]->geometry);
        for(auto& entry:data) {
            auto& level=*entry;const double scale=dt/level.geometry.CellSize(0);
#pragma omp parallel
            for(MFIter mfi(level.conserved,TilingIfNotGPU());mfi.isValid();++mfi) {
                const auto q=level.conserved.array(mfi);const auto flux=level.flux[d].const_array(mfi);
                const auto u=level.velocity[d].const_array(mfi),w=level.splitWeights.const_array(mfi);
                ParallelFor(mfi.tilebox(),fields,[=](int i,int j,int k,int n) noexcept {
                    const double volume=scale*(u(i+di,j+dj,k)-u(i,j,k));
                    q(i,j,k,n)+=scale*(flux(i,j,k,n)-flux(i+di,j+dj,k,n))+w(i,j,k,n)*volume;
                });
            }
        }
        restrictState();
    }
}

double AmrGas::stableTimeStep() const {
    double rate = 0.;
    if (data.size() == 1) {
        const auto& level = *data.front();
        // Bound what one donor can export, not unrelated global X/Y extrema.
        // The .45 margin protects a bounded plane up to twice its cell average.
#pragma omp parallel reduction(max:rate)
        for (MFIter mfi(level.conserved, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto u = level.velocity[0].const_array(mfi), v = level.velocity[1].const_array(mfi);
            const auto box = mfi.tilebox();
            for (int y = box.smallEnd(1); y <= box.bigEnd(1); ++y)
            for (int x = box.smallEnd(0); x <= box.bigEnd(0); ++x) {
                const double outflow = std::max(u(x+1,y,0),0.) + std::max(-u(x,y,0),0.) +
                                       std::max(v(x,y+1,0),0.) + std::max(-v(x,y,0),0.);
                rate = std::max(rate, outflow);
            }
        }
        rate /= level.geometry.CellSize(0);
    } else {
        // Keep the established join bound: coarse interpolation followed by
        // reconstruction can supply four times the coarse donor average, and
        // signed coarse velocities can hide opposing fine-face outflows.
        for (const auto& entry : data) {
            const auto& level = *entry;
            rate = std::max(rate, (level.velocity[0].norm0(0) + level.velocity[1].norm0(0)) / level.geometry.CellSize(0));
        }
    }
    return rate > 0. ? 0.45 / rate : std::numeric_limits<double>::infinity();
}

GasStepResult AmrGas::step(double dt, double tolerance, double damping, std::array<double, 2> acceleration) {
    GasStepResult result;
    if (dt == 0.) {
        const double divergence = maxDivergence();
        result.projection = {0, divergence, std::isfinite(divergence) && divergence <= tolerance};
        return result;
    }
    // Advect with the previous projected faces for the whole physical step.
    // Transport updates conserved material/momentum, not these face velocities.
    auto start = Clock::now();
    const double transportLimit = stableTimeStep();
    double remaining = dt;
    while (remaining > 0.) {
        const double subDt = std::min(remaining, transportLimit);
        transport(subDt);
        ++result.substeps;
        remaining -= subDt;
    }
    if (damping != 0.) for (auto& level : data) {
        if(!liquidComponents)level->conserved.mult(std::exp(-damping * dt),components,2,0);
        else {
            const int nl=liquidComponents,nc=components;
#pragma omp parallel
            for(MFIter mfi(level->conserved,TilingIfNotGPU());mfi.isValid();++mfi) {
                const auto q=level->conserved.array(mfi);
                ParallelFor(mfi.tilebox(),[=](int i,int j,int k) noexcept {
                    for(int n=0;n<nl;++n)if(q(i,j,k,n)!=0.)return;
                    const double factor=std::exp(-damping*dt);
                    q(i,j,k,nc)*=factor;q(i,j,k,nc+1)*=factor;
                });
            }
        }
    }
    result.transportMs = elapsed(start);
    // Only now rebuild the predictor and project it, using the updated density
    // and the full force impulse. Every transport substep used projected faces.
    start = Clock::now();
    result.projection = project(tolerance, dt, acceleration);
    result.projectionMs = elapsed(start);
    return result;
}

double AmrGas::maxDivergence() {
    double maximum = 0.;
    for (auto& entry : data) {
        auto& level = *entry;
        computeDivergence(level.divergence, GetArrOfConstPtrs(level.velocity), level.geometry);
#pragma omp parallel reduction(max:maximum)
        for (MFIter mfi(level.divergence, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto div = level.divergence.const_array(mfi);
            const auto mask = level.active.const_array(mfi);
            for (BoxIterator bit(mfi.tilebox()); bit.ok(); ++bit) if (mask(bit()[0], bit()[1], 0)) {
                const double value = div(bit()[0], bit()[1], 0);
                maximum = std::isfinite(value) ? std::max(maximum, std::abs(value)) : std::numeric_limits<double>::infinity();
            }
        }
    }
    return maximum;
}

GasTotals AmrGas::totals() const {
    GasTotals result;
    result.ingredientMass.resize(components);
    result.minMassDensity = std::numeric_limits<double>::infinity();
    for (const auto& entry : data) {
        const auto& level = *entry;
        const double area = std::pow(level.geometry.CellSize(0), 2);
        for (MFIter mfi(level.conserved); mfi.isValid(); ++mfi) {
            const auto q = level.conserved.const_array(mfi);
            const auto mask = level.active.const_array(mfi);
            for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                const int i = bit()[0], j = bit()[1];
                if (!mask(i, j, 0)) continue;
                double rho = 0., volume = 0.;
                for (int n = 0; n < components; ++n) {
                    const double value = q(i, j, 0, n);
                    result.finite &= std::isfinite(value);
                    result.ingredientMass[n] += value * area;
                    result.minMassDensity = std::min(result.minMassDensity, value);
                    rho += value;
                    volume += value / mixture.components[n].density;
                }
                const double mx = q(i, j, 0, components), my = q(i, j, 0, components + 1);
                result.finite &= std::isfinite(mx) && std::isfinite(my);
                result.momentum[0] += mx * area; result.momentum[1] += my * area;
                result.kineticEnergy += 0.5 * (mx * mx + my * my) / rho * area;
                result.maxSpeed = std::max(result.maxSpeed, std::hypot(mx, my) / rho);
                result.maxVolumeError = std::max(result.maxVolumeError, std::abs(volume - 1.));
            }
        }
    }
    return result;
}
}
