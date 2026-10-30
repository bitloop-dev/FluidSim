#include "FluidGridAdaptive.h"
#include "AmrPressure2D.h"
#include "FluidLinearReconstruction.h"
#include "FluidInterfaceGeometry.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fluid {
using Clock = std::chrono::steady_clock;
namespace {
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
}

// Adapter storage follows the existing FAB order; only pressure fields cross this boundary.
struct FluidGridAdaptive::PressureWorkspace
{
    AmrPressure2D solver;
    AmrPressure2D::Field pressure, rhs;
    AmrPressure2D::Faces beta, flux;

    static std::vector<AmrPressure2D::Level> hierarchy(const std::vector<std::unique_ptr<Level>>& data)
    {
        std::vector<AmrPressure2D::Level> levels;
        for (const auto& entry : data)
        {
            const auto& domain = entry->geometry.Domain();
            auto& level = levels.emplace_back(domain.length(0), domain.length(1), entry->geometry.CellSize(0));
            const auto& boxes = entry->boxes;
            for (int i = 0; i < boxes.size(); ++i)
            {
                const auto& box = boxes[i];
                level.boxes.push_back({box.smallEnd(0), box.smallEnd(1), box.length(0), box.length(1)});
            }
        }
        return levels;
    }

    explicit PressureWorkspace(const std::vector<std::unique_ptr<Level>>& data)
        : solver(hierarchy(data)), pressure(solver.makeField()), rhs(solver.makeField()),
          beta{solver.makeField(), solver.makeField()}, flux{solver.makeField(), solver.makeField()} {}

    int project(std::vector<std::unique_ptr<Level>>& data, double tolerance, int maxIterations, bool equalDensity)
    {
        for (int lev = 0; lev < int(data.size()); ++lev)
        {
            auto& level = *data[lev];
#pragma omp parallel for
            for (int patchIndex=0;patchIndex<level.pressure.size();++patchIndex) {
                const auto& patch = solver.patches(lev)[patchIndex];
                const auto p = level.pressure.const_array(patchIndex), div = level.divergence.const_array(patchIndex);
                const auto& box = patch.box;
                for (int y = box.y; y < box.y + box.height; ++y)
                    for (int x = box.x; x < box.x + box.width; ++x)
                    {
                        const int i = patch.index(x, y);
                        pressure[lev][i] = p(x, y);
                        rhs[lev][i] = -div(x, y);
                    }
                for (int d = 0; d < 2; ++d)
                {
                    const auto b = level.beta[d].const_array(patchIndex);
                    for (int y = box.y; y < box.y + box.height + (d == 1); ++y)
                        for (int x = box.x; x < box.x + box.width + (d == 0); ++x)
                            beta[d][lev][patch.index(x, y)] = b(x, y);
                }
            }
        }
        solver.setCoefficients(beta);
        const auto result = equalDensity ? solver.solveMultigrid(pressure,rhs,tolerance,maxIterations) : solver.solve(pressure, rhs, tolerance, maxIterations);
        if (!result.converged) throw std::runtime_error("Extracted pressure solver did not converge");
        solver.fluxes(flux, pressure);
        for (int lev = 0; lev < int(data.size()); ++lev)
        {
            auto& level = *data[lev];
#pragma omp parallel for
            for (int patchIndex=0;patchIndex<level.pressure.size();++patchIndex) {
                const auto& patch = solver.patches(lev)[patchIndex];
                const auto p = level.pressure.array(patchIndex);
                const auto& box = patch.box;
                for (int y = box.y; y < box.y + box.height; ++y)
                    for (int x = box.x; x < box.x + box.width; ++x)
                        p(x, y) = pressure[lev][patch.index(x, y)];
                for (int d = 0; d < 2; ++d)
                {
                    const auto u = level.velocity[d].array(patchIndex);
                    const auto predicted = level.predictedVelocity[d].const_array(patchIndex);
                    for (int y = box.y; y < box.y + box.height + (d == 1); ++y)
                        for (int x = box.x; x < box.x + box.width + (d == 0); ++x)
                            u(x, y) = predicted(x, y) + flux[d][lev][patch.index(x, y)];
                }
            }
        }
        return result.iterations;
    }
};

FluidGridAdaptive::FluidGridAdaptive(int width, int height, double cellSize, ElementCatalogue catalogue, FluidMixture gasMixture, GasLayout layout, int maxGridSize)
    : FluidGridAdaptive(width,height,cellSize,std::move(catalogue),{},std::move(gasMixture),std::move(layout),maxGridSize) {}

FluidGridAdaptive::FluidGridAdaptive(int width, int height, double cellSize, ElementCatalogue catalogue, FluidMixture liquidMixture, FluidMixture gasMixture, GasLayout layout, int maxGridSize)
    : width(width), height(height), components(static_cast<int>(liquidMixture.components.size()+gasMixture.components.size())),
      maxGridSize(maxGridSize), liquidComponents(static_cast<int>(liquidMixture.components.size())), cellSize(cellSize),
      catalogue(std::move(catalogue)), mixture(liquidMixture), liquidPhase(std::move(liquidMixture)),
      gasPhase(std::move(gasMixture)), gridLayout(std::move(layout)) {
    mixture.components.insert(mixture.components.end(),gasPhase.components.begin(),gasPhase.components.end());
    equalIngredientDensities=std::all_of(mixture.components.begin(),mixture.components.end(),[&](const auto& c) {
        return c.density==mixture.components.front().density;
    });
    normalizeGasLayout(gridLayout);
    auto hierarchy=AmrHierarchy2D(width,height,cellSize,gridLayout,maxGridSize);
    const int count = 1 + int(gridLayout.refinedPatches.size()), fields=components+2;
    for(int lev=0;lev<count;++lev) {
        data.push_back(std::make_unique<Level>());
        auto& level=*data.back();
        level.geometry=hierarchy.levels[lev].geometry;
        level.boxes=std::move(hierarchy.levels[lev].boxes);
        level.conserved.define(level.boxes,fields,2);
        level.slopes.define(level.boxes,2*components,1);
        if(liquidComponents){level.phaseGeometry.define(level.boxes,5,1);level.splitWeights.define(level.boxes,fields,0);}
        level.pressure.define(level.boxes,1,1);
        level.divergence.define(level.boxes,1,0);
        level.active.define(level.boxes,1,0);level.active.setVal(1);
        for(int d=0;d<2;++d) {
            auto faces=level.boxes;for(auto& b:faces)b=b.faces(d);
            level.velocity[d].define(faces,1,0);level.beta[d].define(faces,1,0);level.beta[d].setVal(1.);
            level.flux[d].define(faces,fields,0);level.predictedVelocity[d].define(faces,1,0);
            level.faceSync[d]=std::make_unique<AmrFaceSync2D>(level.boxes,d);
            if(lev)level.faceTransfer[d]=std::make_unique<AmrTransfer2D>(level.boxes,data[lev-1]->boxes,2,d);
        }
        level.boundary=std::make_unique<AmrBoundary2D>(level.conserved,level.geometry,lev?&data[lev-1]->conserved:nullptr);
        if(lev) {
            auto& coarse=*data[lev-1];
            level.cellTransfer=std::make_unique<AmrTransfer2D>(level.boxes,coarse.boxes);
            for(int p=0;p<coarse.active.size();++p)for(const auto& b:level.boxes) {
                const auto mask=coarse.active.array(p);
                forCells(intersect(coarse.boxes[p],b.coarse(2)),[&](int x,int y){mask(x,y)=0;});
            }
        }
    }
    activeCells=hierarchy.activeCells;allocatedCells=hierarchy.allocatedCells;
    pressureWorkspace=std::make_unique<PressureWorkspace>(data);
}

FluidGridAdaptive::~FluidGridAdaptive() = default;

bool FluidGridAdaptive::regrid(GasLayout layout) {
    normalizeGasLayout(layout);
    if(layout==gridLayout)return false;
    restrictState();fillGhosts();
    FluidGridAdaptive next(width,height,cellSize,catalogue,liquidPhase,gasPhase,std::move(layout),maxGridSize);
    const int fields=components+2;
    auto refine=[&](AmrField2D<double>& fine,const AmrField2D<double>& coarse,int ratio) {
        refineAmrField(fine,coarse,components);
        for(int lev=0;lev<int(data.size());++lev)if((gridLayout.baseRatio>>lev)==ratio)
            copyAmrOverlap(fine,data[lev]->conserved);
    };
    std::unique_ptr<AmrField2D<double>> bridge;
    const auto* source=&data[0]->conserved;
    for(int ratio=gridLayout.baseRatio/2;ratio>next.gridLayout.baseRatio;ratio/=2) {
        auto refined=std::make_unique<AmrField2D<double>>();
        const AmrGeometry2D geometry{width/ratio,height/ratio,cellSize*ratio};
        refined->define(splitAmrBoxes({geometry.Domain()},maxGridSize),fields,2);
        refine(*refined,*source,ratio);
        AmrBoundary2D(*refined,geometry,nullptr).fill(*refined,nullptr,components);
        bridge=std::move(refined);source=bridge.get();
    }
    for(int lev=0;lev<int(next.data.size());++lev) {
        auto& target=*next.data[lev];
        const int ratio=next.gridLayout.baseRatio>>lev;
        if(ratio>=gridLayout.baseRatio) {
            if(ratio==gridLayout.baseRatio)copyAmrOverlap(target.conserved,data[0]->conserved);
            else AmrTransfer2D(data[0]->boxes,target.boxes,ratio/gridLayout.baseRatio).average(data[0]->conserved,target.conserved);
        } else refine(target.conserved,lev?next.data[lev-1]->conserved:*source,ratio);
        target.boundary->fill(target.conserved,lev?&next.data[lev-1]->conserved:nullptr,components);
    }
    next.restrictState();
    data.swap(next.data);pressureWorkspace.swap(next.pressureWorkspace);
    std::swap(gridLayout,next.gridLayout);std::swap(activeCells,next.activeCells);std::swap(allocatedCells,next.allocatedCells);
    lastPressureInterval=0.;
    return true;
}

void FluidGridAdaptive::averageFaces(std::array<AmrField2D<double>,2> Level::* field) {
    for(int lev=int(data.size())-1;lev>0;--lev)for(int d=0;d<2;++d)
        data[lev]->faceTransfer[d]->average((data[lev].get()->*field)[d],(data[lev-1].get()->*field)[d]);
}
void FluidGridAdaptive::restrictState() {
    for(int lev=int(data.size())-1;lev>0;--lev)data[lev]->cellTransfer->average(data[lev]->conserved,data[lev-1]->conserved);
}
void FluidGridAdaptive::fillGhosts() {
    for(int lev=0;lev<int(data.size());++lev)
        data[lev]->boundary->fill(data[lev]->conserved,lev?&data[lev-1]->conserved:nullptr,components);
}

void FluidGridAdaptive::upload(const State& state, double tolerance) {
    // A gas-only instance must not silently discard the other phase's mass.
    if (!liquidComponents && std::any_of(state.liquid_fraction.begin(), state.liquid_fraction.end(), [](double value) { return value != 0.; }))
        throw std::invalid_argument("Liquid input requires a configured liquid mixture");
    if(!liquidComponents) for (const auto& mass : state.liquid_mass)
        if (std::any_of(mass.begin(), mass.end(), [](double value) { return value != 0.; }))
            throw std::invalid_argument("FluidGridAdaptive cannot discard liquid ingredient mass");
    const double fineArea = cellSize * cellSize;
    for (int lev = 0; lev < data.size(); ++lev) {
        auto& level = *data[lev];
        const int ratio = gridLayout.baseRatio >> lev;
        for (int patchIndex=0;patchIndex<level.conserved.size();++patchIndex) {
            const auto q = level.conserved.array(patchIndex);
            for(int j=level.conserved.box(patchIndex).y;j<=level.conserved.box(patchIndex).bigEnd(1);++j) for(int i=level.conserved.box(patchIndex).x;i<=level.conserved.box(patchIndex).bigEnd(0);++i) {
                for (int n = 0; n < components + 2; ++n) q(i, j, n) = 0.;
                for (int y = j * ratio; y < (j + 1) * ratio; ++y) for (int x = i * ratio; x < (i + 1) * ratio; ++x) {
                    const auto index = y * width + x;
                    double rho = 0.;
                    for (int n = 0; n < components; ++n) {
                        const double density = (n<liquidComponents ? state.liquid_mass[n][index] : state.gas_mass[n-liquidComponents][index]) / fineArea;
                        q(i, j, n) += density / (ratio * ratio);
                        rho += density;
                    }
                    const double u = state.u.empty() ? 0. : 0.5 * (state.u[y * (width + 1) + x] + state.u[y * (width + 1) + x + 1]);
                    const double v = state.v.empty() ? 0. : 0.5 * (state.v[y * width + x] + state.v[(y + 1) * width + x]);
                    q(i, j, components) += rho * u / (ratio * ratio);
                    q(i, j, components + 1) += rho * v / (ratio * ratio);
                }
            }
        }
        level.pressure.setVal(0.);
    }
    restrictState();
    if (!project(tolerance).converged) throw std::runtime_error("Initial fluid projection did not converge");
}

FluidProjectionResult FluidGridAdaptive::project(double tolerance, double dt, std::array<double, 2> acceleration) {
    lastPressureInterval = dt;
    if(dt>0.)lastAcceleration=acceleration;
    fillGhosts();
    const int nc = components;
    for (auto& entry : data) {
        auto& level = *entry;
        const auto domain = level.geometry.Domain();
        for (int d = 0; d < 2; ++d) {
            const int di = d == 0, dj = d == 1;
            const double impulse = dt * acceleration[d];
#pragma omp parallel for
            for (int patchIndex=0;patchIndex<level.velocity[d].size();++patchIndex) {
                const auto q = level.conserved.const_array(patchIndex);
                const auto u = level.velocity[d].array(patchIndex), b = level.beta[d].array(patchIndex);
                forCells(level.velocity[d].box(patchIndex), [=](int i, int j) noexcept {
                    double left = 0., right = 0.;
                    for (int n = 0; n < nc; ++n) { left += q(i - di, j - dj, n); right += q(i, j, n); }
                    b(i, j) = 2. / (left + right);
                    const int coordinate = d == 0 ? i : j;
                    const bool wall = coordinate == domain.smallEnd(d) || coordinate == domain.bigEnd(d) + 1;
                    u(i, j) = wall ? 0. : 0.5 * (q(i - di, j - dj, nc + d) / left + q(i, j, nc + d) / right) + impulse;
                });
            }
            level.faceSync[d]->apply(level.velocity[d]);
        }
    }
    // Match MacProjector's composite predictor before testing the actual divergence contract.
    averageFaces(&Level::velocity);
    const double predictedDivergence = maxDivergence();
    const bool needsProjection = predictedDivergence > tolerance;
    int iterations = 0;
    if (needsProjection) {
        for (auto& entry : data) for (int d = 0; d < 2; ++d) entry->predictedVelocity[d].copyValid(entry->velocity[d]);
        iterations = pressureWorkspace->project(data, tolerance, liquidComponents ? 2000 : 200, equalIngredientDensities);
        averageFaces(&Level::velocity);
    } else {
        for (auto& entry : data) entry->pressure.setVal(0.);
        if (dt * acceleration[0] == 0. && dt * acceleration[1] == 0.) return {0, predictedDivergence, true};
    }
    for (auto& entry : data) {
        auto& level = *entry;
        const auto domain = level.geometry.Domain();
#pragma omp parallel for
        for (int patchIndex=0;patchIndex<level.conserved.size();++patchIndex) {
            const auto q = level.conserved.array(patchIndex);
            const auto px = level.predictedVelocity[0].const_array(patchIndex), py = level.predictedVelocity[1].const_array(patchIndex);
            const auto ux = level.velocity[0].const_array(patchIndex), uy = level.velocity[1].const_array(patchIndex);
            const double ax = dt * acceleration[0], ay = dt * acceleration[1];
            forCells(level.conserved.box(patchIndex), [=](int i, int j) noexcept {
                double rho = 0.;
                for (int n = 0; n < nc; ++n) rho += q(i, j, n);
                // The cell velocity receives only the averaged MAC correction, as in incflo's cell projection.
                // Body-force faces vanish on closed walls; averaging force and pressure together preserves rest equilibrium.
                const double wx = 0.5 * ((i != domain.smallEnd(0)) + (i != domain.bigEnd(0)));
                const double wy = 0.5 * ((j != domain.smallEnd(1)) + (j != domain.bigEnd(1)));
                // Projected and predicted faces are both averaged down; their difference is the same composite correction.
                const double cx = needsProjection ? 0.5 * ((ux(i, j) - px(i, j)) + (ux(i + 1, j) - px(i + 1, j))) : 0.;
                const double cy = needsProjection ? 0.5 * ((uy(i, j) - py(i, j)) + (uy(i, j + 1) - py(i, j + 1))) : 0.;
                q(i, j, nc) += rho * (cx + wx * ax);
                q(i, j, nc + 1) += rho * (cy + wy * ay);
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

void FluidGridAdaptive::transport(double dt) {
    if (dt == 0.) return;
    if(liquidComponents) { transportPhases(dt); return; }
    fillGhosts();
    const int fields = components + 2;
    const int nc=components;
    for (auto& entry : data) {
        auto& level = *entry;
        const double halfDt=.5*dt/level.geometry.CellSize(0);
#pragma omp parallel for
        for (int patchIndex=0;patchIndex<level.slopes.size();++patchIndex) {
            const auto q=level.conserved.const_array(patchIndex); const auto slopes=level.slopes.array(patchIndex);
            forCells(level.slopes.box(patchIndex).grown(1),[=](int i,int j) noexcept {
                auto sample=[&](int dx,int dy,int n) { return q(i+dx,j+dy,n); };
                const auto limiter=linearLimiter(nc,sample);
                for (int n=0; n<nc; ++n) {
                    const auto s=linearSlope(n,sample,limiter);
                    slopes(i,j,n)=s.x; slopes(i,j,nc+n)=s.y;
                }
            });
        }
        for (int d = 0; d < 2; ++d) {
            const int di = d == 0, dj = d == 1;
#pragma omp parallel for
            for (int patchIndex=0;patchIndex<level.flux[d].size();++patchIndex) {
                const auto q = level.conserved.const_array(patchIndex), u = level.velocity[d].const_array(patchIndex);
                const auto slopes=level.slopes.const_array(patchIndex); const auto flux = level.flux[d].array(patchIndex);
                forCells(level.flux[d].box(patchIndex), [=](int i, int j) noexcept {
                    const double speed=u(i,j);
                    const bool left=speed>=0.;
                    const int x=i-(left?di:0), y=j-(left?dj:0);
                    double rho=0.; for (int n=0; n<nc; ++n) rho+=q(x,y,n);
                    // Trace the donor plane back half a step in both directions.
                    // The normal projected face speed keeps that point inside
                    // the donor; transverse tracing is bounded by the same box.
                    const double normal=(left?.5:-.5)-halfDt*speed;
                    const double transverse=std::clamp(-halfDt*q(x,y,nc+1-d)/rho,-.5,.5);
                    const double dx=di?normal:transverse, dy=dj?normal:transverse;
                    double massFlux=0.;
                    for (int n=0; n<nc; ++n) {
                        const double transfer=speed*(q(x,y,n)+dx*slopes(x,y,n)+dy*slopes(x,y,nc+n));
                        flux(i,j,n)=transfer; massFlux+=transfer;
                    }
                    // Preserve the existing donor-velocity momentum scheme,
                    // using exactly the mass transported by the ingredients.
                    for (int axis=0; axis<2; ++axis) flux(i,j,nc+axis)=massFlux*q(x,y,nc+axis)/rho;
                });
            }
            level.faceSync[d]->apply(level.flux[d]);
        }
    }
    // Same-timestep reflux: each coarse boundary uses the sum of the actual fine ingredient AND momentum fluxes.
    averageFaces(&Level::flux);
    for (auto& entry : data) {
        auto& level = *entry;
        const double scale = dt / level.geometry.CellSize(0);
#pragma omp parallel for
        for (int patchIndex=0;patchIndex<level.conserved.size();++patchIndex) {
            const auto q = level.conserved.array(patchIndex);
            const auto fx = level.flux[0].const_array(patchIndex), fy = level.flux[1].const_array(patchIndex);
            forCells(level.conserved.box(patchIndex), fields, [=](int i, int j, int n) noexcept {
                q(i, j, n) -= scale * (fx(i + 1, j, n) - fx(i, j, n) + fy(i, j + 1, n) - fy(i, j, n));
            });
        }
    }
    restrictState();
}

void FluidGridAdaptive::transportPhases(double dt) {
    const int nc=components,nl=liquidComponents,fields=nc+2;
    const auto* material=mixture.components.data();
    // Freeze the Weymouth-Yue split correction before either directional pass.
    // Its X/Y contributions cancel for a divergence-free advecting field.
    for(auto& entry:data) {
        auto& level=*entry;
#pragma omp parallel for
        for (int patchIndex=0;patchIndex<level.conserved.size();++patchIndex) {
            const auto q=level.conserved.const_array(patchIndex);const auto w=level.splitWeights.array(patchIndex);
            forCells(level.conserved.box(patchIndex),[=](int i,int j) noexcept {
                double liquid=0.,gas=0.,rho=0.;
                for(int n=0;n<nc;++n) {
                    (n<nl?liquid:gas)+=q(i,j,n)/material[n].density;rho+=q(i,j,n);
                }
                const bool selected=liquid>.5; const double volume=selected?liquid:gas;
                double mass=0.;
                for(int n=0;n<nc;++n) {
                    w(i,j,n)=((n<nl)==selected)?q(i,j,n)/volume:0.;mass+=w(i,j,n);
                }
                for(int d=0;d<2;++d)w(i,j,nc+d)=mass*q(i,j,nc+d)/rho;
            });
        }
    }
    for(int d=0;d<2;++d) {
        fillGhosts();
        const int di=d==0,dj=d==1;
        for(auto& entry:data) {
            auto& level=*entry;
            const double scale=dt/level.geometry.CellSize(0);
#pragma omp parallel for
            for (int patchIndex=0;patchIndex<level.phaseGeometry.size();++patchIndex) {
                const auto q=level.conserved.const_array(patchIndex);const auto geometry=level.phaseGeometry.array(patchIndex);
                forCells(level.phaseGeometry.box(patchIndex).grown(1),[=](int i,int j) noexcept {
                    double f=0.,gas=0.;
                    for(int n=0;n<nl;++n)f+=q(i,j,n)/material[n].density;
                    for(int n=nl;n<nc;++n)gas+=q(i,j,n)/material[n].density;
                    geometry(i,j)=f;geometry(i,j,1)=gas;
                    // Reconstruct the smaller phase directly. Subtracting a
                    // nearly full liquid flux can overwhelm a tiny gas sliver.
                    const bool liquid=f<=gas;const double minority=liquid?f:gas;
                    auto fraction=[&](int dx,int dy) {
                        double value=0.;for(int n=liquid?0:nl;n<(liquid?nl:nc);++n)value+=q(i+dx,j+dy,n)/material[n].density;
                        return value;
                    };
                    if(minority>0.) {
                        const auto normal=interface_geometry::normal(fraction);
                        geometry(i,j,2)=normal.x;geometry(i,j,3)=normal.y;
                        geometry(i,j,4)=interface_geometry::offset(normal,minority);
                    }
                });
            }
#pragma omp parallel for
            for (int patchIndex=0;patchIndex<level.flux[d].size();++patchIndex) {
                const auto q=level.conserved.const_array(patchIndex),u=level.velocity[d].const_array(patchIndex);
                const auto geometry=level.phaseGeometry.const_array(patchIndex);const auto flux=level.flux[d].array(patchIndex);
                forCells(level.flux[d].box(patchIndex),[=](int i,int j) noexcept {
                    const double speed=u(i,j);
                    if(speed==0.) { for(int n=0;n<fields;++n)flux(i,j,n)=0.;return; }
                    const bool positive=speed>0.;const int x=i-(positive?di:0),y=j-(positive?dj:0);
                    const double f=geometry(x,y),g=geometry(x,y,1),total=speed*scale;
                    const bool smallLiquid=f<=g;const double minority=smallLiquid?f:g;
                    double smallTransfer=0.;
                    if(minority>0.) {
                        using interface_geometry::Vec;
                        const double strip=std::abs(total),lo=positive?1.-strip:0.,hi=positive?1.:strip;
                        const Vec a{di?lo:0.,dj?lo:0.},b{di?hi:1.,dj?hi:1.};
                        const Vec normal{geometry(x,y,2),geometry(x,y,3)};
                        const double area=interface_geometry::area(normal,geometry(x,y,4),a,b);
                        // Same geometric roundoff bound as the uniform PLIC flux;
                        // neither phase may export more than its donor contains.
                        smallTransfer=std::copysign(std::clamp(area,std::max(0.,strip-(1.-minority)),std::min(strip,minority)),speed);
                    }
                    const double liquid=smallLiquid?smallTransfer:total-smallTransfer;
                    const double gas=smallLiquid?total-smallTransfer:smallTransfer;
                    double rho=0.,massFlux=0.;
                    for(int n=0;n<nc;++n) {
                        const double volume=n<nl?f:g;
                        const double transfer=volume>0. ? q(x,y,n)*((n<nl?liquid:gas)/volume)/scale : 0.;
                        flux(i,j,n)=transfer;massFlux+=transfer;rho+=q(x,y,n);
                    }
                    for(int axis=0;axis<2;++axis)flux(i,j,nc+axis)=massFlux*q(x,y,nc+axis)/rho;
                });
            }
            level.faceSync[d]->apply(level.flux[d]);
        }
        // A coarse face uses exactly the fine phase/ingredient/momentum flux.
        for(int lev=int(data.size())-1;lev>0;--lev)
            data[lev]->faceTransfer[d]->average(data[lev]->flux[d],data[lev-1]->flux[d]);
        for(auto& entry:data) {
            auto& level=*entry;const double scale=dt/level.geometry.CellSize(0);
#pragma omp parallel for
            for (int patchIndex=0;patchIndex<level.conserved.size();++patchIndex) {
                const auto q=level.conserved.array(patchIndex);const auto flux=level.flux[d].const_array(patchIndex);
                const auto u=level.velocity[d].const_array(patchIndex),w=level.splitWeights.const_array(patchIndex);
                forCells(level.conserved.box(patchIndex),fields,[=](int i,int j,int n) noexcept {
                    const double volume=scale*(u(i+di,j+dj)-u(i,j));
                    q(i,j,n)+=scale*(flux(i,j,n)-flux(i+di,j+dj,n))+w(i,j,n)*volume;
                });
            }
        }
        restrictState();
    }
}

double FluidGridAdaptive::stableTimeStep() const {
    double rate = 0.;
    if (data.size() == 1) {
        const auto& level = *data.front();
        // Bound what one donor can export, not unrelated global X/Y extrema.
        // The .45 margin protects a bounded plane up to twice its cell average.
#pragma omp parallel for reduction(max:rate)
        for (int patchIndex=0;patchIndex<level.conserved.size();++patchIndex) {
            const auto u = level.velocity[0].const_array(patchIndex), v = level.velocity[1].const_array(patchIndex);
            const auto box = level.conserved.box(patchIndex);
            for (int y = box.smallEnd(1); y <= box.bigEnd(1); ++y)
            for (int x = box.smallEnd(0); x <= box.bigEnd(0); ++x) {
                const double outflow = std::max(u(x+1,y),0.) + std::max(-u(x,y),0.) +
                                       std::max(v(x,y+1),0.) + std::max(-v(x,y),0.);
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

GasStepResult FluidGridAdaptive::step(double dt, double tolerance, double damping, std::array<double, 2> acceleration) {
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
        if(!liquidComponents)level->conserved.multiply(std::exp(-damping * dt),components,2);
        else {
            const int nl=liquidComponents,nc=components;
#pragma omp parallel for
            for (int patchIndex=0;patchIndex<level->conserved.size();++patchIndex) {
                const auto q=level->conserved.array(patchIndex);
                forCells(level->conserved.box(patchIndex),[=](int i,int j) noexcept {
                    for(int n=0;n<nl;++n)if(q(i,j,n)!=0.)return;
                    const double factor=std::exp(-damping*dt);
                    q(i,j,nc)*=factor;q(i,j,nc+1)*=factor;
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

double FluidGridAdaptive::maxDivergence() {
    double maximum = 0.;
    for (auto& entry : data) {
        auto& level = *entry;
        const double inverse=1./level.geometry.cell_size;
#pragma omp parallel for reduction(max:maximum)
        for(int p=0;p<level.divergence.size();++p) {
            const auto div=level.divergence.array(p);const auto u=level.velocity[0].const_array(p),v=level.velocity[1].const_array(p);
            const auto mask=level.active.const_array(p);
            forCells(level.divergence.box(p),[&](int x,int y) {
                div(x,y)=inverse*((u(x+1,y)-u(x,y))+(v(x,y+1)-v(x,y)));
                if(mask(x,y)) maximum=std::isfinite(div(x,y))?std::max(maximum,std::abs(div(x,y))):std::numeric_limits<double>::infinity();
            });
        }
    }
    return maximum;
}

GasTotals FluidGridAdaptive::totals() const {
    GasTotals result;
    result.ingredientMass.resize(components);
    result.minMassDensity = std::numeric_limits<double>::infinity();
    for (const auto& entry : data) {
        const auto& level = *entry;
        const double area = std::pow(level.geometry.CellSize(0), 2);
        for (int patchIndex=0;patchIndex<level.conserved.size();++patchIndex) {
            const auto q = level.conserved.const_array(patchIndex);
            const auto mask = level.active.const_array(patchIndex);
            for(int j=level.conserved.box(patchIndex).y;j<=level.conserved.box(patchIndex).bigEnd(1);++j) for(int i=level.conserved.box(patchIndex).x;i<=level.conserved.box(patchIndex).bigEnd(0);++i) {
                if (!mask(i, j)) continue;
                double rho = 0., volume = 0.;
                for (int n = 0; n < components; ++n) {
                    const double value = q(i, j, n);
                    result.finite &= std::isfinite(value);
                    result.ingredientMass[n] += value * area;
                    result.minMassDensity = std::min(result.minMassDensity, value);
                    rho += value;
                    volume += value / mixture.components[n].density;
                }
                const double mx = q(i, j, components), my = q(i, j, components + 1);
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
