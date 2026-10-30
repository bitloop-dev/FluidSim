#pragma once
#include "AmrField2D.h"

namespace fluid {

// Cached address plans replace FillPatch/ParallelCopy's communication machinery.
// The order is coarse interpolation, same-level ownership, reflecting walls.
class AmrBoundary2D {
public:
    struct Ghost { int target,source,patch; double dx,dy; bool coarse,reflectX,reflectY; };
    AmrBoundary2D(const AmrField2D<double>& field,AmrGeometry2D geometry,const AmrField2D<double>* coarse);
    AmrBoundary2D(const AmrFieldLayout2D& field,AmrGeometry2D geometry,const AmrFieldLayout2D& coarse);
    const auto& recipes() const { return ghosts; }
    void fill(AmrField2D<double>& field,const AmrField2D<double>* coarse,int ingredients) const;
private:
    std::vector<std::vector<Ghost>> ghosts;
};

class AmrTransfer2D {
public:
    struct Region { int fine; AmrBox2D box; };
    AmrTransfer2D(const std::vector<AmrBox2D>& fine,const std::vector<AmrBox2D>& coarse,int ratio=2,int axis=-1);
    const auto& recipes() const { return regions; }
    void average(const AmrField2D<double>& fine,AmrField2D<double>& coarse) const;
private:
    int ratio,axis;
    std::vector<std::vector<Region>> regions;
};

class AmrFaceSync2D {
public:
    struct Region { int source; AmrBox2D box; };
    AmrFaceSync2D(const std::vector<AmrBox2D>& boxes,int axis);
    const auto& recipes() const { return regions; }
    void apply(AmrField2D<double>& field) const;
private:
    std::vector<std::vector<Region>> regions;
};

void copyAmrOverlap(AmrField2D<double>& target,const AmrField2D<double>& source);
void refineAmrField(AmrField2D<double>& fine,const AmrField2D<double>& coarse,int ingredients);

} // namespace fluid
