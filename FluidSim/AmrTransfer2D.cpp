// Derived from AMReX (c) 2024 The Regents of the University of California.
// Redistribution terms: reference/pressure2d/AMReX-LICENSE; source map: reference/engine/.
#include "AmrTransfer2D.h"
#include "FluidLinearReconstruction.h"
#include <stdexcept>

namespace fluid {
namespace {
int owner(const AmrField2D<double>& field,int x,int y) {
    for(int p=0;p<field.size();++p)if(field.box(p).contains(x,y))return p;
    return -1;
}
}

AmrBoundary2D::AmrBoundary2D(const AmrField2D<double>& field,AmrGeometry2D geometry,const AmrField2D<double>* coarse)
    : ghosts(field.size()) {
    for(int p=0;p<field.size();++p) {
        const auto box=field.box(p);const auto q=field.const_array(p);
        forCells(box.grown(2),[&](int x,int y) {
            if(box.contains(x,y))return;
            const bool rx=x<0||x>=geometry.width, ry=y<0||y>=geometry.height;
            const int i=x<0?-x-1:x>=geometry.width?2*geometry.width-x-1:x;
            const int j=y<0?-y-1:y>=geometry.height?2*geometry.height-y-1:y;
            const int source=owner(field,i,j);
            if(source>=0) ghosts[p].push_back({q.index(x,y),field.const_array(source).index(i,j),source,0.,0.,false,rx,ry});
            else {
                const int parent=owner(*coarse,i/2,j/2);
                ghosts[p].push_back({q.index(x,y),coarse->const_array(parent).index(i/2,j/2),parent,(i%2? .25:-.25),(j%2? .25:-.25),true,rx,ry});
            }
        });
    }
}

void AmrBoundary2D::fill(AmrField2D<double>& field,const AmrField2D<double>* coarse,int ingredients,bool equalDensity) const {
    const int fields=field.nComp();
#pragma omp parallel for
    for(int p=0;p<field.size();++p) {
        const auto out=field.array(p);
        for(const auto& g:ghosts[p]) {
            const auto in=g.coarse?coarse->const_array(g.patch):field.const_array(g.patch);
            if(!g.coarse) for(int n=0;n<fields;++n)out.values[g.target+n*out.plane]=in.values[g.source+n*in.plane];
            else for(int first=0;first<fields;) {
                const int count=first==0?(equalDensity?ingredients:fields):2;
                auto sample=[&](int dx,int dy,int n){return in.values[g.source+dx+dy*in.stride+(first+n)*in.plane];};
                const auto limiter=linearLimiter(count,sample);
                for(int n=0;n<count;++n) {
                    const auto slope=linearSlope(n,sample,limiter);
                    out.values[g.target+(first+n)*out.plane]=sample(0,0,n)+g.dx*slope.x+g.dy*slope.y;
                }
                first+=count;
            }
            if(g.reflectX)out.values[g.target+ingredients*out.plane]*=-1.;
            if(g.reflectY)out.values[g.target+(ingredients+1)*out.plane]*=-1.;
        }
    }
}

AmrTransfer2D::AmrTransfer2D(const std::vector<AmrBox2D>& fine,const std::vector<AmrBox2D>& coarse,int ratio,int axis)
    : ratio(ratio),axis(axis),regions(coarse.size()) {
    for(int p=0;p<int(coarse.size());++p)for(int f=0;f<int(fine.size());++f) {
        auto cb=coarse[p],fb=fine[f].coarse(ratio);
        if(axis>=0){cb=cb.faces(axis);fb=fb.faces(axis);}
        const auto b=intersect(cb,fb);
        if(b.width&&b.height)regions[p].push_back({f,b});
    }
}

void AmrTransfer2D::average(const AmrField2D<double>& fine,AmrField2D<double>& coarse) const {
    const int nx=axis==0?1:ratio,ny=axis==1?1:ratio;
    const double weight=1./(nx*ny);
#pragma omp parallel for
    for(int p=0;p<coarse.size();++p) {
        const auto out=coarse.array(p);
        for(const auto& region:regions[p]) {
            const auto in=fine.const_array(region.fine);
            forCells(region.box,coarse.nComp(),[&](int x,int y,int n) {
                double sum=0.;
                for(int dy=0;dy<ny;++dy)for(int dx=0;dx<nx;++dx)sum+=in(x*ratio+dx,y*ratio+dy,n);
                out(x,y,n)=sum*weight;
            });
        }
    }
}

AmrFaceSync2D::AmrFaceSync2D(const std::vector<AmrBox2D>& boxes,int axis):regions(boxes.size()) {
    for(int p=0;p<int(boxes.size());++p)for(int f=p-1;f>=0;--f) {
        const auto b=intersect(boxes[p].faces(axis),boxes[f].faces(axis));
        if(b.width&&b.height)regions[p].push_back({f,b});
    }
}
void AmrFaceSync2D::apply(AmrField2D<double>& field) const {
    // Copy from the lowest-index owner; sources on a shared face are never destinations.
#pragma omp parallel for
    for(int p=0;p<field.size();++p) {
        const auto out=field.array(p);
        for(const auto& region:regions[p]) {
            const auto in=field.const_array(region.source);
            forCells(region.box,field.nComp(),[&](int x,int y,int n){out(x,y,n)=in(x,y,n);});
        }
    }
}

void copyAmrOverlap(AmrField2D<double>& target,const AmrField2D<double>& source) {
#pragma omp parallel for
    for(int p=0;p<target.size();++p)for(int f=0;f<source.size();++f) {
        const auto box=intersect(target.box(p),source.box(f));
        const auto out=target.array(p);const auto in=source.const_array(f);
        forCells(box,target.nComp(),[&](int x,int y,int n){out(x,y,n)=in(x,y,n);});
    }
}

void refineAmrField(AmrField2D<double>& fine,const AmrField2D<double>& coarse,int ingredients,bool equalDensity) {
#pragma omp parallel for
    for(int p=0;p<fine.size();++p)for(int f=0;f<coarse.size();++f) {
        const auto region=intersect(fine.box(p).coarse(2),coarse.box(f));
        const auto out=fine.array(p);const auto in=coarse.const_array(f);
        forCells(region,[&](int x,int y) {
            for(int first=0;first<fine.nComp();) {
                const int count=first==0?(equalDensity?ingredients:fine.nComp()):2;
                auto sample=[&](int dx,int dy,int n){return in(x+dx,y+dy,first+n);};
                const auto limiter=linearLimiter(count,sample);
                for(int n=0;n<count;++n) {
                    const auto slope=linearSlope(n,sample,limiter);
                    for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
                        out(x*2+dx,y*2+dy,first+n)=sample(0,0,n)+(dx?.25:-.25)*slope.x+(dy?.25:-.25)*slope.y;
                }
                first+=count;
            }
        });
    }
}

} // namespace fluid
