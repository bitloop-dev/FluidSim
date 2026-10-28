#pragma once
#include "FluidGridCPU.h"
#include "FluidLinearReconstruction.h"
#include <cmath>
#include <utility>
#ifdef FLUIDSIM_HAS_AMR
#include "FluidGridAdaptive.h"
#endif

namespace fluid {
// Non-owning observation of committed cells. The only spatial cache is a patch
// index per 2x2 finest group; no masses, momentum or pressure are resampled.
class FluidCellView {
public:
    struct MaterialPlane {
        std::vector<double> stencil;
        std::vector<LinearSlope> slopes;
        int count=0;
        double mean(int n,int side=0) const { return stencil[side*count+n]; }
        double value(int n,double dx,double dy) const { return mean(n)+dx*slopes[n].x+dy*slopes[n].y; }
    };
    struct Patch {
        int x = 0, y = 0, width = 0, height = 0, scale = 1;
#ifdef FLUIDSIM_HAS_AMR
        AmrArray2D<const double> q, pressure, u, v;
        AmrArray2D<const int> active;
#endif
    };
    struct Cell {
        const FluidCellView* view = nullptr;
        const Patch* patch = nullptr;
        int x = 0, y = 0;
        explicit operator bool() const { return view != nullptr; }
        double size() const { return view->cellSize * patch->scale; }
        double cx() const { return (x + .5) * size(); }
        double cy() const { return (y + .5) * size(); }
        int index() const { return y * view->width + x; }
        bool liquid() const {
            if (!view->uniform) {
                for(int k=0;k<view->liquidComponents;++k)if(density(k)!=0.)return true;
                return false;
            }
            if (view->uniform->liquid_fraction[index()] != 0.) return true;
            for (const auto& mass : view->uniform->liquid_mass) if (mass[index()] != 0.) return true;
            return false;
        }
        double density(int k) const {
            if (view->uniform) return view->uniform->gas_mass[k][index()] / (size() * size());
#ifdef FLUIDSIM_HAS_AMR
            return patch->q(x,y,k);
#else
            return 0.;
#endif
        }
        double density() const { double rho=0.; for (int k=0; k<view->components(); ++k) rho+=density(k); return rho; }
        double fraction(int k) const { return density(k)/view->mixture->components[k].density; }
        double face(int axis, int offset) const {
            if (view->uniform) return axis == 0 ? view->uniform->u[view->uniform->uIndex(x+offset,y)] : view->uniform->v[view->uniform->vIndex(x,y+offset)];
#ifdef FLUIDSIM_HAS_AMR
            return axis == 0 ? patch->u(x+offset,y) : patch->v(x,y+offset);
#else
            return 0.;
#endif
        }
        double velocity(int axis) const {
            if (view->uniform) return .5*(face(axis,0)+face(axis,1));
#ifdef FLUIDSIM_HAS_AMR
            return patch->q(x,y,view->components()+axis)/density();
#else
            return 0.;
#endif
        }
        double momentum(int axis) const {
            if (view->uniform) return density()*velocity(axis);
#ifdef FLUIDSIM_HAS_AMR
            return patch->q(x,y,view->components()+axis);
#else
            return 0.;
#endif
        }
        double pressure() const {
            if (view->uniform) return view->uniform->pressure[index()];
#ifdef FLUIDSIM_HAS_AMR
            return view->pressureInterval > 0. ? patch->pressure(x,y)/view->pressureInterval : 0.;
#else
            return 0.;
#endif
        }
    };
    explicit FluidCellView(const FluidGridCPU& grid) : width(grid.width), height(grid.height), cellSize(grid.cell_size),
        mixture(&grid.gas_mixture), catalogue(&grid.elements), uniform(&grid), acceleration{0.,grid.gravity_y}, identity(&grid) {
        patches.push_back({0,0,width,height,1});
        owners.assign(size_t((width+1)/2)*((height+1)/2),0);
    }
#ifdef FLUIDSIM_HAS_AMR
    explicit FluidCellView(const FluidGridAdaptive& gas) : width(gas.finestWidth()), height(gas.finestHeight()), cellSize(gas.finestCellSize()),
        mixture(&gas.materialMixture()), catalogue(&gas.elements()), acceleration(gas.acceleration()), identity(&gas),
        pressureInterval(gas.pressureInterval()), liquidComponents(gas.liquidIngredientCount()) {
        owners.assign(size_t((width+1)/2)*((height+1)/2),-1);
        for (const auto& level : gas.levels()) {
            const int scale=int(std::lround(level->geometry.CellSize(0)/cellSize));
            for (int fab=0;fab<level->conserved.size();++fab) {
                const auto& b=level->conserved.box(fab);
                Patch p{b.smallEnd(0)*scale,b.smallEnd(1)*scale,b.length(0)*scale,b.length(1)*scale,scale};
                p.q=level->conserved.const_array(fab); p.pressure=level->pressure.const_array(fab);
                p.u=level->velocity[0].const_array(fab); p.v=level->velocity[1].const_array(fab); p.active=level->active.const_array(fab);
                const int id=int(patches.size()); patches.push_back(p);
                for (int y=p.y; y<p.y+p.height; y+=2) for (int x=p.x; x<p.x+p.width; x+=2)
                    if (p.active(x/scale,y/scale)) owners[(y/2)*((width+1)/2)+x/2]=id;
            }
        }
    }
#endif
#ifdef FLUIDSIM_HAS_AMR
    // Non-owning view over an explicit native-state observation. The caller
    // keeps the patch arrays alive; no physical state is created or advanced.
    FluidCellView(int w, int h, double dx, const FluidMixture& material, const ElementCatalogue& elements,
                  std::vector<Patch> nativePatches, std::array<double,2> force, double interval, int liquids, const void* id)
        : width(w), height(h), cellSize(dx), mixture(&material), catalogue(&elements), acceleration(force), identity(id),
          pressureInterval(interval), liquidComponents(liquids), patches(std::move(nativePatches))
    {
        owners.assign(size_t((width+1)/2)*((height+1)/2),-1);
        for (int index=0;index<int(patches.size());++index) {
            const auto& p=patches[index];
            for (int y=p.y;y<p.y+p.height;y+=2) for (int x=p.x;x<p.x+p.width;x+=2)
                if (p.active(x/p.scale,y/p.scale)) owners[(y/2)*((width+1)/2)+x/2]=index;
        }
    }
#endif
    Cell at(int x, int y) const {
        if (x<0 || y<0 || x>=width || y>=height) return {};
        const auto& patch=patches[owners[(y/2)*((width+1)/2)+x/2]];
        return {this,&patch,x/patch.scale,y/patch.scale};
    }
    int minimumScale(int x,int y,int scale) const {
        int stride=scale;
        for (int dy=0;dy<scale;dy+=2) for (int dx=0;dx<scale;dx+=2)
            stride=std::min(stride,at(x+dx,y+dy).patch->scale);
        return stride;
    }
    void reconstruct(MaterialPlane& plane,int x,int y,int scale,bool includeMomentum=false) const {
        plane.count=components()+(includeMomentum?2:0);
        plane.stencil.assign(5*plane.count,0.); plane.slopes.resize(plane.count);
        constexpr int offsets[5][2]={{0,0},{-1,0},{1,0},{0,-1},{0,1}};
        for (int side=0;side<5;++side) {
            const int gx=std::clamp(x+scale*offsets[side][0],0,width-scale), gy=std::clamp(y+scale*offsets[side][1],0,height-scale);
            const auto source=at(gx,gy);
            if (source.patch->scale>scale) {
                // A smaller stencil footprint inside a larger cell uses that
                // cell's conservative plane, matching coarse/fine ghost fill.
                MaterialPlane parent;
                const int parentScale=source.patch->scale;
                reconstruct(parent,source.x*parentScale,source.y*parentScale,parentScale,includeMomentum);
                for (int n=0;n<plane.count;++n)
                    plane.stencil[side*plane.count+n]=parent.value(n,(gx+.5*scale)/parentScale-source.x-.5,(gy+.5*scale)/parentScale-source.y-.5);
                continue;
            }
            // Restrict actual samples, including groups crossing fine patch edges.
            const int stride=minimumScale(gx,gy,scale), samples=(scale/stride)*(scale/stride);
            for (int dy=0;dy<scale;dy+=stride) for (int dx=0;dx<scale;dx+=stride) {
                const auto c=at(gx+dx,gy+dy);
                for (int n=0;n<components();++n) plane.stencil[side*plane.count+n]+=c.density(n)/samples;
                if (includeMomentum) for (int d=0;d<2;++d) {
                    const int requested=d?y+scale*offsets[side][1]:x+scale*offsets[side][0];
                    const bool reflected=requested<0 || requested>=(d?height:width);
                    plane.stencil[side*plane.count+components()+d]+=(reflected?-1.:1.)*c.momentum(d)/samples;
                }
            }
        }
        auto sample=[&](int dx,int dy,int n) { return plane.mean(n,dx<0?1:dx>0?2:dy<0?3:dy>0?4:0); };
        const auto limiter=linearLimiter(components(),sample);
        for (int n=0;n<components();++n) plane.slopes[n]=linearSlope(n,sample,limiter);
        if (includeMomentum) {
            const auto momentum=linearMomentumSlopes(components(),sample,limiter);
            for(int d=0;d<2;++d) plane.slopes[components()+d]=momentum[d];
        }
    }
    double materialSample(int x,int y,int ingredient) const {
        const auto c=at(x,y);
        if (c.patch->scale==1) return c.fraction(ingredient);
        MaterialPlane plane;
        const int scale=c.patch->scale;
        reconstruct(plane,c.x*scale,c.y*scale,scale);
        return plane.value(ingredient,(x+.5)/scale-c.x-.5,(y+.5)/scale-c.y-.5)/mixture->components[ingredient].density;
    }
    int components() const { return int(mixture->components.size()); }
    int width, height;
    double cellSize;
    const FluidMixture* mixture;
    const ElementCatalogue* catalogue;
    const FluidGridCPU* uniform=nullptr;
    std::array<double,2> acceleration{};
    const void* identity;
    double pressureInterval=0.;
    int liquidComponents=0;
    std::vector<Patch> patches;
    std::vector<int> owners;
};
}
