#pragma once
#include "AmrField2D.h"
#include <tuple>

namespace fluid {

struct GasPatch { int x,y,width,height; bool operator==(const GasPatch&) const = default; };
struct GasLayout {
    int baseRatio=1;
    // Successive 2:1 levels, aligned and properly nested in finest-cell coordinates.
    std::vector<std::vector<GasPatch>> refinedPatches;
    bool operator==(const GasLayout&) const = default;
};

inline void normalizeGasLayout(GasLayout& layout) {
    for (auto& patches:layout.refinedPatches)
        std::sort(patches.begin(),patches.end(),[](const GasPatch& a,const GasPatch& b) {
            return std::tie(a.y,a.x,a.height,a.width)<std::tie(b.y,b.x,b.height,b.width);
        });
}

struct AmrHierarchy2D {
    struct Level { AmrGeometry2D geometry; std::vector<AmrBox2D> boxes; };
    std::vector<Level> levels;
    long long activeCells=0, allocatedCells=0;
    AmrHierarchy2D(int width,int height,double cellSize,const GasLayout& layout,int maxGridSize) {
        for (int lev=0;lev<=int(layout.refinedPatches.size());++lev) {
            const int ratio=layout.baseRatio>>lev;
            Level level{{width/ratio,height/ratio,cellSize*ratio},{}};
            std::vector<AmrBox2D> boxes;
            if (!lev) boxes.push_back(level.geometry.Domain());
            else for (const auto& p:layout.refinedPatches[lev-1]) boxes.push_back({p.x/ratio,p.y/ratio,p.width/ratio,p.height/ratio});
            level.boxes=splitAmrBoxes(boxes,maxGridSize);
            for (const auto& b:level.boxes) {
                allocatedCells+=b.width*b.height;
                activeCells+=b.width*b.height-(lev?b.width*b.height/4:0);
            }
            levels.push_back(std::move(level));
        }
    }
};

} // namespace fluid
