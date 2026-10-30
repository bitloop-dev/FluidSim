#include "FluidCoarseningPreview.h"
#include "FluidCellView.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#ifdef FLUIDSIM_HAS_AMR
#include "AmrField2D.h"
#endif

namespace fluid {
namespace {
using Cell = FluidCellView::Cell;
struct Assessment { FluidCoarseningPreview::Label label; bool quiet = false; };
void flatMaterialError(const FluidCellView& view, int x, int y, FluidCoarseningPreview::Error& error)
{
    // Uniform liquid previews retain their flat-mean representation contract.
    // Native gas uses the bounded plane below.
    const int offsets[5][2]={{0,0},{-2,0},{2,0},{0,-2},{0,2}};
    std::array<Cell,4> groups[5];
    for (int g=0;g<5;++g) {
        const int gx=x+offsets[g][0], gy=y+offsets[g][1];
        if (gx<0 || gy<0 || gx+1>=view.width || gy+1>=view.height) continue;
        for (int j=0;j<2;++j) for (int i=0;i<2;++i) groups[g][2*j+i]=view.at(gx+i,gy+j);
    }
    for (int k=-1;k<view.components();++k) {
        auto value=[&](const Cell& c) { return k<0?c.density():c.fraction(k); };
        double mean[5]{};
        for (int g=0;g<5;++g) if (groups[g][0])
            for (const auto& c:groups[g]) mean[g]+=.25*value(c);
        double detail=0., slope[2]{};
        for (const auto& c:groups[0]) detail=std::max(detail,std::abs(value(c)-mean[0]));
        for (int side=0;side<4;++side) if (groups[side+1][0]) {
            slope[side/2]=std::max(slope[side/2],std::abs(mean[side+1]-mean[0]));
            // Averages alone hide opposing fine samples on an adjoining face.
            constexpr int face[4][2]={{1,3},{0,2},{2,3},{0,1}};
            const auto& neighbour=groups[side+1];
            detail=std::max(detail,.5*std::abs(value(neighbour[face[side][0]])-value(neighbour[face[side][1]])));
        }
        // Resolved neighbour slopes predict variation at child centres. Apply
        // this before merging too, so a stationary step cannot merge/refine forever.
        detail=std::max(detail,.25*(slope[0]+slope[1]));
        if (k<0) error.density=detail/mean[0];
        else error.ingredient=std::max(error.ingredient,detail);
    }
}
void nativeRepresentationError(const FluidCellView& view,int x,int y,int scale,FluidCoarseningPreview::Error& error,FluidCellView::MaterialPlane& plane) {
    view.reconstruct(plane,x,y,scale,true);
    const int stride=view.minimumScale(x,y,scale);
    constexpr int offsets[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
    for (int k=-1;k<view.components();++k) {
        double mean=0.; LinearSlope slope;
        for (int n=0;n<view.components();++n) if (k<0 || k==n) {
            const double divisor=k<0?1.:view.mixture->components[n].density;
            mean+=plane.mean(n)/divisor; slope.x+=plane.slopes[n].x/divisor; slope.y+=plane.slopes[n].y/divisor;
        }
        auto value=[&](const Cell& c) { return k<0?c.density():c.fraction(k); };
        auto residual=[&](const Cell& c) {
            return value(c)-mean-(c.cx()/view.cellSize-x-.5*scale)*slope.x/scale-(c.cy()/view.cellSize-y-.5*scale)*slope.y/scale;
        };
        double detail=0.;
        if (stride<scale) for (int dy=0;dy<scale;dy+=stride) for (int dx=0;dx<scale;dx+=stride)
            detail=std::max(detail,std::abs(residual(view.at(x+dx,y+dy))));
        for (int side=0;side<4;++side) {
            const int gx=x+scale*offsets[side][0],gy=y+scale*offsets[side][1];
            if (gx<0 || gy<0 || gx+scale>view.width || gy+scale>view.height) continue;
            double neighbour=0.;
            for (int n=0;n<view.components();++n) if (k<0 || k==n) neighbour+=plane.mean(n,side+1)/(k<0?1.:view.mixture->components[n].density);
            detail=std::max(detail,.25*std::abs(neighbour-mean-offsets[side][0]*slope.x-offsets[side][1]*slope.y));
            // Preserve all resolved subface variation, including opposing samples
            // whose mean and endpoints alone would hide fine detail.
            double low=0.,high=0.;
            for (int j=0;j<scale;++j) {
                const auto c=view.at(gx+(side==0?scale-1:side==1?0:j),gy+(side==2?scale-1:side==3?0:j));
                const double r=residual(c);
                if (j==0) low=high=r; else { low=std::min(low,r); high=std::max(high,r); }
            }
            detail=std::max(detail,.5*(high-low));
        }
        if (k<0) error.density=detail/mean; else error.ingredient=std::max(error.ingredient,detail);
    }
    auto velocity=[&](int d,double dx,double dy) {
        double rho=0.; for(int k=0;k<view.components();++k) rho+=plane.value(k,dx,dy);
        return plane.value(view.components()+d,dx,dy)/rho;
    };
    for(int d=0;d<2;++d) {
        double rho=0.; LinearSlope densitySlope;
        for(int k=0;k<view.components();++k) { rho+=plane.mean(k); densitySlope.x+=plane.slopes[k].x; densitySlope.y+=plane.slopes[k].y; }
        const double mean=plane.mean(view.components()+d)/rho;
        const LinearSlope slope{(plane.slopes[view.components()+d].x-mean*densitySlope.x)/rho,
                                (plane.slopes[view.components()+d].y-mean*densitySlope.y)/rho};
        // Neighbour centres lie outside the bounded density plane. Extend the
        // velocity slope there, rather than divide by extrapolated density.
        auto extended=[&](double dx,double dy) { return mean+dx*slope.x+dy*slope.y; };
        auto residual=[&](const Cell& c) {
            return c.velocity(d)-extended((c.cx()/view.cellSize-x)/scale-.5,(c.cy()/view.cellSize-y)/scale-.5);
        };
        double detail=0.;
        for(int dy=0;dy<scale;dy+=stride) for(int dx=0;dx<scale;dx+=stride) {
            const auto c=view.at(x+dx,y+dy);
            detail=std::max(detail,std::abs(c.velocity(d)-velocity(d,(c.cx()/view.cellSize-x)/scale-.5,(c.cy()/view.cellSize-y)/scale-.5)));
            // MAC faces are independent of cell momentum. Check them too:
            // cancelling face fluctuations must not disappear in a good cell fit.
            for(int side=0;side<2;++side) {
                const double px=c.cx()/view.cellSize+(d==0?(side-.5)*c.patch->scale:0.);
                const double py=c.cy()/view.cellSize+(d==1?(side-.5)*c.patch->scale:0.);
                detail=std::max(detail,std::abs(c.face(d,side)-velocity(d,(px-x)/scale-.5,(py-y)/scale-.5)));
            }
        }
        for(int side=0;side<4;++side) {
            const int gx=x+scale*offsets[side][0],gy=y+scale*offsets[side][1];
            if(gx<0 || gy<0 || gx+scale>view.width || gy+scale>view.height) continue;
            double neighbourDensity=0.; for(int k=0;k<view.components();++k) neighbourDensity+=plane.mean(k,side+1);
            const double neighbour=plane.mean(view.components()+d,side+1)/neighbourDensity;
            detail=std::max(detail,.25*std::abs(neighbour-extended(offsets[side][0],offsets[side][1])));
            double low=0.,high=0.;
            for(int j=0;j<scale;++j) {
                const auto c=view.at(gx+(side==0?scale-1:side==1?0:j),gy+(side==2?scale-1:side==3?0:j));
                const double r=residual(c);
                if(j==0) low=high=r; else { low=std::min(low,r); high=std::max(high,r); }
                if(d==side/2) {
                    const double px=c.cx()/view.cellSize+(d==0?(side==0?.5:-.5)*c.patch->scale:0.);
                    const double py=c.cy()/view.cellSize+(d==1?(side==2?.5:-.5)*c.patch->scale:0.);
                    const double face=c.face(d,side%2==0?1:0)-velocity(d,(px-x)/scale-.5,(py-y)/scale-.5);
                    low=std::min(low,face); high=std::max(high,face);
                }
            }
            detail=std::max(detail,.5*(high-low));
        }
        error.velocity=std::max(error.velocity,detail);
    }
}
Assessment boundaryRefinement(const FluidCellView& grid, int fx, int fy, int scale, const FluidCoarseningPreview::Tolerances& tolerance, double quiet_ratio,FluidCellView::MaterialPlane& plane,FluidCoarseningPreview::Trace::Cell* observed)
{
    using P = FluidCoarseningPreview;
    const auto c = grid.at(fx,fy);
    if (c.liquid()) return {P::Refine};
    std::array<Cell,8> neighbours;
    // Average the two fine neighbours on a coarse face, retaining each sample
    // too: otherwise opposing fine variations could disappear at the boundary.
    for (int j = 0; j < 2; ++j)
    {
        neighbours[j] = grid.at(fx-1,fy+j*(scale/2));
        neighbours[2+j] = grid.at(fx+scale,fy+j*(scale/2));
        neighbours[4+j] = grid.at(fx+j*(scale/2),fy-1);
        neighbours[6+j] = grid.at(fx+j*(scale/2),fy+scale);
    }
    for (const auto& n : neighbours)
    {
        if (bool(n) && n.liquid()) return {P::Refine};
    }
    for (int dy : {-1,scale}) for (int dx : {-1,scale})
        if (const auto n = grid.at(fx+dx,fy+dy); bool(n) && n.liquid()) return {P::Refine};

    P::Error e;
    nativeRepresentationError(grid,fx,fy,scale,e,plane);

    double gradient[4]{};
    bool available[4]{};
    for (int side = 0; side < 4; ++side)
    {
        const auto& a = neighbours[2*side]; const auto& z = neighbours[2*side+1];
        if (!bool(a) || !bool(z)) continue;
        const double distance = side < 2 ? 0.5*(a.cx()+z.cx())-c.cx() : 0.5*(a.cy()+z.cy())-c.cy();
        gradient[side] = (0.5*(a.pressure()+z.pressure())-c.pressure())/distance;
        available[side] = true;
    }
    double slope[2]{};
    for (int axis = 0; axis < 2; ++axis)
    {
        const int a = 2*axis, z = a+1, count = int(available[a])+int(available[z]);
        if (count) slope[axis] = (gradient[a]+gradient[z])/count;
        if (count == 2) e.pressure_gradient = std::max(e.pressure_gradient,0.5*std::abs(gradient[a]-gradient[z]));
    }
    // Check each fine neighbour against the local plane, preserving affine
    // pressure across a 2:1 join while detecting cancelling sub-face patterns.
    for (const auto& n : neighbours) if (bool(n))
    {
        const double dx = n.cx()-c.cx(), dy = n.cy()-c.cy();
        const double residual = n.pressure()-c.pressure()-slope[0]*dx-slope[1]*dy;
        e.pressure_gradient = std::max(e.pressure_gradient,std::abs(residual)/(std::abs(dx)+std::abs(dy)));
    }
    const double score = e.score(tolerance);
    if (observed) { observed->error=e; observed->evaluated=true; }
    if (score > 1) return {P::Refine};
    return {P::Coarse, score <= quiet_ratio};
}
FluidCoarseningPreview::Error fineError(const FluidCellView& view,int x,int y,int scale,FluidCellView::MaterialPlane& plane) {
    FluidCoarseningPreview::Error e;
    double rho=0.,mean[2]{},pressureHalf[2][2]{};
    for (int dy=0;dy<scale;++dy) for (int dx=0;dx<scale;++dx) {
        const auto c=view.at(x+dx,y+dy);
        rho+=c.density(); for (int d=0;d<2;++d) mean[d]+=c.momentum(d);
        pressureHalf[0][dx>=scale/2]+=c.pressure(); pressureHalf[1][dy>=scale/2]+=c.pressure();
    }
    for (auto& v:mean) v/=rho;
    if(view.uniform) for (int dy=0;dy<scale;++dy) for (int dx=0;dx<scale;++dx) {
        const auto c=view.at(x+dx,y+dy);
        for (int d=0;d<2;++d) {
            e.velocity=std::max(e.velocity,std::abs(c.velocity(d)-mean[d]));
            for (int side=0;side<2;++side) e.velocity=std::max(e.velocity,std::abs(c.face(d,side)-mean[d]));
        }
    }
    if (view.uniform) flatMaterialError(view,x,y,e); else nativeRepresentationError(view,x,y,scale,e,plane);
    double gradient[2]{};
    for (int d=0;d<2;++d) gradient[d]=4.*(pressureHalf[d][1]-pressureHalf[d][0])/(scale*scale*scale*view.cellSize);
    for (int d=0;d<2;++d) for (int along=-1;along<scale;++along) for (int across=0;across<scale;++across) {
        const auto a=view.at(x+(d?across:along),y+(d?along:across));
        const auto b=view.at(x+(d?across:along)+!d,y+(d?along:across)+d);
        if (!a || !b) continue;
        const double distance=d?b.cy()-a.cy():b.cx()-a.cx();
        if (distance==0.) continue; // Both sample positions may belong to one actual coarse cell.
        const double tangent=d?b.cx()-a.cx():b.cy()-a.cy();
        e.pressure_gradient=std::max(e.pressure_gradient,std::abs((b.pressure()-a.pressure()-gradient[1-d]*tangent)/distance-gradient[d]));
    }
    return e;
}
}
double FluidCoarseningPreview::Error::score(const Tolerances& t) const
{
    // Positive tolerances are supplied by the scene controls.
    return std::max({velocity/t.velocity, pressure_gradient/t.pressure_gradient, density/t.density, ingredient/t.ingredient});
}

FluidCoarseningPreview::Error FluidCoarseningPreview::estimate(const FluidGrid& g,int x,int y) {
    FluidCellView::MaterialPlane plane; return fineError(FluidCellView(g),x,y,2,plane);
}
void FluidCoarseningPreview::reset() { assessments.clear(); observed_scales.clear(); statistics={}; counts.fill(0); labels.clear(); }
void FluidCoarseningPreview::update(const FluidGrid& grid,const Tolerances& t,const Stability& s,double time,double lookahead) {
    updateView(FluidCellView(grid),t,s,time,lookahead,2);
}
#ifdef FLUIDSIM_HAS_AMR
FluidCoarseningPreview::Error FluidCoarseningPreview::estimate(const FluidGridAdaptive& grid,int x,int y,int scale) {
    FluidCellView::MaterialPlane plane; return fineError(FluidCellView(grid),x,y,scale,plane);
}
void FluidCoarseningPreview::update(const FluidGridAdaptive& grid,const Tolerances& t,const Stability& s,double time,double lookahead,int maxCellScale,Trace* trace) {
    updateView(FluidCellView(grid),t,s,time,lookahead,maxCellScale,trace);
}
GasLayout FluidCoarseningPreview::proposeLayout(const FluidGridAdaptive& grid,Trace* trace) const {
    if (trace) trace->fallback_scale=0;
    auto mark=[&](std::vector<std::uint8_t>& tags,int tile,int x0,int y0,int x1,int y1) {
        const int nx=(grid.finestWidth()+tile-1)/tile;
        for (int y=std::max(0,y0)/tile;y<=std::min(grid.finestHeight()-1,y1)/tile;++y)
            for (int x=std::max(0,x0)/tile;x<=std::min(grid.finestWidth()-1,x1)/tile;++x) tags[y*nx+x]=1;
    };
    auto requests=[&](const AssessmentLevel& level,Trace::Level* observed,int tile) {
        const int nx=(grid.finestWidth()+tile-1)/tile,ny=(grid.finestHeight()+tile-1)/tile;
        if (observed) { observed->tile=tile; observed->tiles_x=nx; observed->tiles_y=ny; }
        std::vector<std::uint8_t> tags(size_t(nx)*ny,0);
        for (int y=0;y<level.height;++y) for (int x=0;x<level.width;++x) {
            const int i=y*level.width+x;
            const auto label=level.labels[i];
            const bool needed=level.history[i].coarse ? label==Refine || label==Approaching : label!=Candidate;
            if (observed) observed->cells[i].requested=needed;
            if (!needed) continue;
            const auto raw=level.history[i].raw;
            const int buffer=(raw==Detail || raw==LiquidBuffer || raw==Refine || raw==Approaching)?level.scale:0;
            // Quiet dwell retains its own cells without seeding fresh padding.
            mark(tags,tile,x*level.scale-buffer,y*level.scale-buffer,(x+1)*level.scale-1+buffer,(y+1)*level.scale-1+buffer);
        }
        if (observed) observed->padded=observed->nested=tags;
        return tags;
    };
    std::vector<Requests> masks;
    for (size_t lev=0;lev<assessments.size();++lev) {
        const auto& level=assessments[lev];
        masks.push_back({level.scale,requests(level,trace?&trace->levels[lev]:nullptr,std::max(16,level.scale))});
    }
    return clusterRequests(grid.finestWidth(),grid.finestHeight(),masks,trace);
}
GasLayout FluidCoarseningPreview::clusterRequests(int width,int height,const std::vector<Requests>& requests,Trace* trace) {
    if (trace) trace->fallback_scale=0;
    auto mark=[&](std::vector<std::uint8_t>& tags,int tile,int x0,int y0,int x1,int y1) {
        const int nx=(width+tile-1)/tile;
        for (int y=std::max(0,y0)/tile;y<=std::min(height-1,y1)/tile;++y)
            for (int x=std::max(0,x0)/tile;x<=std::min(width-1,x1)/tile;++x) tags[y*nx+x]=1;
    };
    auto cluster=[&](const std::vector<std::uint8_t>& tags,int tile) {
        const int nx=(width+tile-1)/tile,ny=(height+tile-1)/tile;
        std::vector<std::array<int,2>> points;
        for (int y=0;y<ny;++y) for (int x=0;x<nx;++x) if (tags[y*nx+x]) points.push_back({x,y});
        std::vector<GasPatch> patches;
        if (points.empty()) return patches;
        for (const auto& box:clusterAmrTags(std::move(points),.7)) {
            const int x=tile*box.smallEnd(0),y=tile*box.smallEnd(1);
            patches.push_back({x,y,std::min(width,tile*(box.bigEnd(0)+1))-x,
                                  std::min(height,tile*(box.bigEnd(1)+1))-y});
        }
        return patches;
    };
    auto nearlyFull=[&](const std::vector<GasPatch>& patches) {
        int area=0; for (const auto& p:patches) area+=p.width*p.height;
        return area>=3*width*height/4;
    };
    GasLayout layout{requests.back().scale,{}};
    // Parent padding is topological; retained history never becomes a new detail source.
    for (size_t lev=0;lev<requests.size();++lev) {
        const auto& request=requests[lev];
        const int tile=std::max(16,request.scale);
        auto tags=request.tiles;
        if (!layout.refinedPatches.empty()) for (const auto& p:layout.refinedPatches.back())
            mark(tags,tile,p.x-request.scale,p.y-request.scale,p.x+p.width+request.scale-1,p.y+p.height+request.scale-1);
        if (trace) trace->levels[lev].nested=tags;
        auto patches=cluster(tags,tile);
        if (nearlyFull(patches)) {
            layout.baseRatio=request.scale/2;
            if (trace) trace->fallback_scale=layout.baseRatio;
            break;
        }
        layout.refinedPatches.push_back(std::move(patches));
    }
    std::reverse(layout.refinedPatches.begin(),layout.refinedPatches.end());
    while (!layout.refinedPatches.empty() && layout.refinedPatches.back().empty()) layout.refinedPatches.pop_back();
    return layout;
}
#endif
void FluidCoarseningPreview::updateView(const FluidCellView& grid,const Tolerances& tolerance,const Stability& stability,double time,double lookahead,int maxCellScale,Trace* trace) {
    const int levelCount=std::countr_zero(static_cast<unsigned>(maxCellScale));
    if (trace) { *trace={}; trace->levels.resize(levelCount); }
    const bool fresh=assessments.size()!=levelCount || observed_grid!=grid.identity || width!=(grid.width+1)/2 || height!=(grid.height+1)/2 ||
        observed_cell_size!=grid.cellSize || observed_tolerances!=tolerance || observed_stability!=stability || observed_lookahead!=lookahead || time<last_time;
    width=(grid.width+1)/2; height=(grid.height+1)/2;
    if (fresh) assessments.assign(levelCount,{});
    observed_grid=grid.identity; observed_cell_size=grid.cellSize;
    observed_tolerances=tolerance; observed_stability=stability; observed_lookahead=lookahead; last_time=time;
    // Do not pack a group's footprint into a fixed-width integer: larger
    // groups would silently lose changes in their earlier samples.
    std::vector<std::uint8_t> scales(size_t(width)*height);
    for (int y=0;y<height;++y) for (int x=0;x<width;++x)
        scales[y*width+x]=std::uint8_t(grid.at(2*x,2*y).patch->scale);
    for (int lev=0;lev<levelCount;++lev) {
        auto& level=assessments[lev];
        const int scale=level.scale=2<<lev;
        level.width=(grid.width+scale-1)/scale; level.height=(grid.height+scale-1)/scale;
        const int count=level.width*level.height;
        auto* observed=trace?&trace->levels[lev]:nullptr;
        if (observed) { observed->scale=scale; observed->width=level.width; observed->height=level.height; observed->cells.resize(count); }
        level.labels.assign(count,Unassessed);
        if (fresh) level.history.assign(count,{});
        bl::Thread::forEachBatch(count,[&](int first,int last) {
            FluidCellView::MaterialPlane plane;
            for (int index=first;index<last;++index) {
                const int fx=scale*(index%level.width),fy=scale*(index/level.width);
                auto& h=level.history[index];
                if (!fresh) for (int dy=0;dy<scale && fy+dy<grid.height;dy+=2)
                    for (int dx=0;dx<scale && fx+dx<grid.width;dx+=2) {
                        const int i=((fy+dy)/2)*width+(fx+dx)/2;
                        if (scales[i]!=observed_scales[i]) h={};
                    }
                const int actual=grid.at(fx,fy).patch->scale;
                auto* sample=observed?&observed->cells[index]:nullptr;
                if (sample) sample->observed_scale=actual;
                h.quiet=false; h.coarse=actual>=scale;
                if (actual>scale) { level.labels[index]=Coarse; continue; }
                if (h.coarse) {
                    const auto a=boundaryRefinement(grid,fx,fy,scale,tolerance,stability.quiet_ratio,plane,sample);
                    level.labels[index]=a.label; h.quiet=a.quiet;
                } else if (fx+scale<=grid.width && fy+scale<=grid.height) {
                    bool liquid=false,complete=true;
                    for (int dy=-1;dy<=scale;++dy) for (int dx=-1;dx<=scale;++dx) {
                        const auto c=grid.at(fx+dx,fy+dy);
                        if (!c) complete &= !grid.uniform;
                        else liquid|=c.liquid();
                    }
                    if (liquid) level.labels[index]=LiquidBuffer;
                    else if (complete) {
                        const auto error=fineError(grid,fx,fy,scale,plane);
                        const double score=error.score(tolerance);
                        if (sample) { sample->error=error; sample->evaluated=true; }
                        level.labels[index]=score<=1?Candidate:Detail; h.quiet=score<=stability.quiet_ratio;
                    }
                }
            }
        },count>=8192?8:1);
        if (observed) for (int i=0;i<count;++i) observed->cells[i].source=level.labels[i];
        if (lookahead>0.) markApproaching(level,grid,tolerance,lookahead);
        stabilise(level,stability,time,fresh);
        if (observed) for (int i=0;i<count;++i) {
            auto& sample=observed->cells[i]; const auto& h=level.history[i];
            sample.raw=h.raw; sample.decision=h.decision; sample.quiet=h.quiet; sample.quiet_since=h.quiet_since;
        }
    }
    observed_scales=std::move(scales);
    // Project decisions only for display; physics and history remain per level.
    labels=assessments.front().labels; counts.fill(0); statistics={};
    for (int y=0;y<height;++y) for (int x=0;x<width;++x) {
        auto& label=labels[y*width+x];
        for (int lev=1;lev<levelCount && (label==Coarse || label==Candidate);++lev)
            label=assessments[lev].labels[(y>>lev)*assessments[lev].width+(x>>lev)];
    }
    for (auto label:labels) ++counts[label];
    for (const auto& level:assessments) {
        statistics.raw_changes+=level.statistics.raw_changes; statistics.decision_changes+=level.statistics.decision_changes;
        statistics.waiting+=level.statistics.waiting; statistics.held+=level.statistics.held;
    }
    ++revision;
}
void FluidCoarseningPreview::stabilise(AssessmentLevel& level,const Stability& stability,double time,bool fresh) {
    level.statistics={};
    auto refine=[](std::uint8_t label) { return label==Refine || label==Approaching; };
    auto action=[&](std::uint8_t label) { return label==Candidate?-1:refine(label)?1:0; };
    for (size_t i=0;i<level.labels.size();++i) {
        auto& h=level.history[i]; const auto raw=level.labels[i];
        if (!h.quiet || refine(raw)) h.quiet_since=-1;
        else if (h.quiet_since<0) h.quiet_since=time;
        const bool settled=h.quiet_since>=0 && time>=h.quiet_since+stability.quiet_seconds;
        auto decision=raw;
        if (raw==Candidate && h.decision!=Candidate && !settled) { decision=Detail; ++level.statistics.waiting; }
        else if ((raw==Coarse || raw==Unassessed) && refine(h.decision) && !settled) { decision=h.decision; ++level.statistics.held; }
        if (!fresh) {
            level.statistics.raw_changes+=action(raw)!=action(h.raw);
            level.statistics.decision_changes+=action(decision)!=action(h.decision);
        }
        h.raw=raw; h.decision=decision; level.labels[i]=decision;
    }
}
void FluidCoarseningPreview::markApproaching(AssessmentLevel& level,const FluidCellView& grid,const Tolerances& tolerance,double lookahead) {
    FluidCellView::MaterialPlane plane;
    const int width=level.width,height=level.height,scale=level.scale,stride=width+1;
    auto& swept=level.swept;
    swept.assign(size_t(stride)*(height+1),0);
    for (int y=0;y<height;++y) for (int x=0;x<width;++x) {
        const int fx=scale*x,fy=scale*y,index=y*width+x;
        const auto label=level.labels[index];
        bool source=label==Detail || label==LiquidBuffer || label==Refine;
        if (!source && label==Unassessed && !level.history[index].coarse && fx+scale<=grid.width && fy+scale<=grid.height)
            source=fineError(grid,fx,fy,scale,plane).score(tolerance)>1;
        if (!source) continue;
        double low[2]{},high[2]{};
        for (int dy=0;dy<scale;++dy) for (int dx=0;dx<scale;++dx) {
            const auto c=grid.at(fx+dx,fy+dy); if (!c) continue;
            for (int d=0;d<2;++d) for (int side=0;side<2;++side) {
                const double v=c.face(d,side),accelerated=v+grid.acceleration[d]*lookahead;
                low[d]=std::min({low[d],v,accelerated}); high[d]=std::max({high[d],v,accelerated});
            }
        }
        const double travel=lookahead/grid.cellSize;
        const int x0=int(std::floor(std::clamp((fx+low[0]*travel)/scale,0.,double(width))));
        const int x1=int(std::ceil(std::clamp((fx+scale+high[0]*travel)/scale,0.,double(width))));
        const int y0=int(std::floor(std::clamp((fy+low[1]*travel)/scale,0.,double(height))));
        const int y1=int(std::ceil(std::clamp((fy+scale+high[1]*travel)/scale,0.,double(height))));
        ++swept[y0*stride+x0]; --swept[y0*stride+x1]; --swept[y1*stride+x0]; ++swept[y1*stride+x1];
    }
    for (int y=0;y<height;++y) {
        int row=0;
        for (int x=0;x<width;++x) {
            row+=swept[y*stride+x];
            const int coverage=row+(y?swept[(y-1)*stride+x]:0); swept[y*stride+x]=coverage;
            const int index=y*width+x;
            if (coverage && level.history[index].coarse && level.labels[index]!=Refine)
                level.labels[index]=Approaching;
        }
    }
}
}
