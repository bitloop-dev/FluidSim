#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace fluid {
struct LinearSlope { double x = 0., y = 0.; };

// One limiter per mixture preserves linear identities (in particular summed
// ingredient volume). The plane stays within neighbouring cell extrema even
// at its corners. Sample offsets are measured in cells of this resolution.
template<class Sample>
LinearSlope linearLimiter(int count, const Sample& sample) {
    LinearSlope limiter{1.,1.};
    for (int n=0; n<count; ++n) {
        const double c=sample(0,0,n);
        auto factor=[&](double a,double b) {
            const double left=c-a, right=b-c, centered=.5*(b-a);
            if (centered==0.) return 1.;
            if (left*right<=0.) return 0.;
            return std::min(1.,2.*std::min(std::abs(left),std::abs(right))/std::abs(centered));
        };
        limiter.x=std::min(limiter.x,factor(sample(-1,0,n),sample(1,0,n)));
        limiter.y=std::min(limiter.y,factor(sample(0,-1,n),sample(0,1,n)));
    }
    double corner=1.;
    for (int n=0; n<count; ++n) {
        const double c=sample(0,0,n), l=sample(-1,0,n), r=sample(1,0,n), b=sample(0,-1,n), t=sample(0,1,n);
        const double excursion=.25*(std::abs(r-l)*limiter.x+std::abs(t-b)*limiter.y);
        if (excursion>0.) corner=std::min(corner,std::min(c-std::min({c,l,r,b,t}),std::max({c,l,r,b,t})-c)/excursion);
    }
    return {limiter.x*corner,limiter.y*corner};
}
template<class Sample>
LinearSlope linearSlope(int n, const Sample& sample, LinearSlope limiter) {
    return {.5*(sample(1,0,n)-sample(-1,0,n))*limiter.x,.5*(sample(0,1,n)-sample(0,-1,n))*limiter.y};
}
// Reconstruct momentum around the parent velocity: material slopes stay independent,
// while a zero-mean residual preserves momentum and constant velocity through regrids.
template<class Sample>
std::array<LinearSlope,2> linearMomentumSlopes(int ingredients,const Sample& sample,LinearSlope materialLimiter) {
    constexpr int offsets[5][2]={{0,0},{-1,0},{1,0},{0,-1},{0,1}};
    std::array<double,5> rho{};
    for(int n=0;n<ingredients;++n) for(int side=0;side<5;++side)
        rho[side]+=sample(offsets[side][0],offsets[side][1],n);
    const LinearSlope densitySlope{.5*(rho[2]-rho[1])*materialLimiter.x,.5*(rho[4]-rho[3])*materialLimiter.y};
    std::array<LinearSlope,2> result;
    for(int d=0;d<2;++d) {
        const double velocity=sample(0,0,ingredients+d)/rho[0];
        double low=velocity,high=velocity;
        std::array<double,5> residual{};
        for(int side=1;side<5;++side) {
            const double momentum=sample(offsets[side][0],offsets[side][1],ingredients+d);
            residual[side]=momentum-velocity*rho[side];
            low=std::min(low,momentum/rho[side]); high=std::max(high,momentum/rho[side]);
        }
        auto relative=[&](int x,int y,int) { return residual[x<0?1:x>0?2:y<0?3:y>0?4:0]; };
        const auto slope=linearSlope(0,relative,linearLimiter(1,relative));
        double factor=1.;
        // Corner inequalities bound velocity everywhere in the cell without flattening material.
        for(double y:{-.5,.5}) for(double x:{-.5,.5}) {
            const double density=rho[0]+x*densitySlope.x+y*densitySlope.y;
            const double excursion=x*slope.x+y*slope.y;
            if(excursion>0.)factor=std::min(factor,(high-velocity)*density/excursion);
            else if(excursion<0.)factor=std::min(factor,(low-velocity)*density/excursion);
        }
        result[d]={velocity*densitySlope.x+factor*slope.x,velocity*densitySlope.y+factor*slope.y};
    }
    return result;
}
}
