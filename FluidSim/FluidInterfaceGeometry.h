#pragma once
#include <algorithm>
#include <cmath>

namespace fluid::interface_geometry {
struct Vec { double x, y; };

// MYC normal from the same 3x3 phase stencil used by the uniform liquid engine.
template<class Sample> Vec normal(const Sample& f) {
    const double tl=f(-1,-1),t=f(0,-1),tr=f(1,-1),l=f(-1,0),r=f(1,0);
    const double bl=f(-1,1),b=f(0,1),br=f(1,1);
    const double left=tl+l+bl,right=tr+r+br,top=tl+t+tr,bottom=bl+b+br;
    Vec centered{.5*(left-right),.5*(top-bottom)};
    const Vec youngs{left+l-right-r,top+t-bottom-b};
    if(centered.x==0. && centered.y==0.)
        return youngs.x==0. && youngs.y==0. ? Vec{1.,0.} : youngs;
    if(std::abs(centered.x)<=std::abs(centered.y)) {
        centered.y=std::copysign(1.,centered.y);
        if(std::abs(youngs.x)>std::abs(centered.x*youngs.y))return youngs;
    } else {
        centered.x=std::copysign(1.,centered.x);
        if(std::abs(youngs.y)>std::abs(centered.y*youngs.x))return youngs;
    }
    return centered;
}

template<class Vector> double offset(Vector n,double fraction) {
    const double a=std::min(std::abs(n.x),std::abs(n.y)), b=std::max(std::abs(n.x),std::abs(n.y));
    const double triangle=.5*a/b;
    const double cut=fraction<triangle ? std::sqrt(2.*a*b*fraction) :
        fraction<=1.-triangle ? b*fraction+.5*a : a+b-std::sqrt(2.*a*b*(1.-fraction));
    return std::min(n.x,0.)+std::min(n.y,0.)+cut;
}

template<class Vector> double area(Vector n,double offset,Vector lo,Vector hi) {
    const double width=hi.x-lo.x,height=hi.y-lo.y,total=width*height;
    const double cut=offset-(n.x*(n.x>=0.?lo.x:hi.x)+n.y*(n.y>=0.?lo.y:hi.y));
    const double a=std::min(std::abs(n.x)*width,std::abs(n.y)*height);
    const double b=std::max(std::abs(n.x)*width,std::abs(n.y)*height);
    if(cut<=0.)return 0.;
    if(cut>=a+b)return total;
    if(a==0.)return total*cut/b;
    if(cut<a)return total*cut*cut/(2.*a*b);
    if(cut<b)return total*(cut-.5*a)/b;
    const double dry=a+b-cut;
    return total*(1.-dry*dry/(2.*a*b));
}
}
