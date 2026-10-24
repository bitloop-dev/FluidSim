#pragma once
#include <algorithm>
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
}
