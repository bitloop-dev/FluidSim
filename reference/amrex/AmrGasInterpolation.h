#pragma once
#include "FluidLinearReconstruction.h"
#include <AMReX_Interpolater.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuLaunch.H>

namespace fluid::reference {
// Use the transport plane for new fine averages and coarse/fine ghost values.
// AMReX still owns stencil gathering, physical BCs and overlap copies.
class GasLinearInterpolater final : public amrex::CellConservativeLinear {
public:
    void interp(const amrex::FArrayBox& coarse, int src, amrex::FArrayBox& fine, int dst, int count,
                const amrex::Box& region, const amrex::IntVect& ratio, const amrex::Geometry&, const amrex::Geometry&,
                const amrex::Vector<amrex::BCRec>&, int, int, amrex::RunOn) override {
        const auto q=coarse.const_array(); const auto out=fine.array();
        const auto lo=amrex::lbound(region), hi=amrex::ubound(region);
        amrex::ParallelFor(amrex::coarsen(region,ratio),[=](int i,int j,int) noexcept {
            auto sample=[&](int dx,int dy,int n) { return q(i+dx,j+dy,0,src+n); };
            const auto limiter=linearLimiter(count,sample);
            for (int n=0; n<count; ++n) {
                const auto s=linearSlope(n,sample,limiter);
                for (int y=std::max(j*ratio[1],lo.y); y<=std::min((j+1)*ratio[1]-1,hi.y); ++y)
                for (int x=std::max(i*ratio[0],lo.x); x<=std::min((i+1)*ratio[0]-1,hi.x); ++x)
                    out(x,y,0,dst+n)=q(i,j,0,src+n)+((x+.5)/ratio[0]-i-.5)*s.x+((y+.5)/ratio[1]-j-.5)*s.y;
            }
        });
    }
};
}
