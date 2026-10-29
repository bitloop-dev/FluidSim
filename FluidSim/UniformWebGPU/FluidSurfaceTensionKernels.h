#pragma once
#include "FluidWebGPUCompute.h"
#include "FluidWebGPUShaders.h"

namespace fluid::webgpu {

// Uniform 2D generalized height functions and balanced CSF. Algorithm references:
// Popinet (2009), JCP 228:5838-5866; basilisk.fr/src/{heights,curvature,iforce}.h.
// Local bounded columns replace tree/ghost exchanges. No physical fraction is modified.
struct SurfaceTensionKernels
{
    Kernel heights, curvature, complete, predict;

    void build(Compute& compute)
    {
        const std::string samples = R"wgsl(
fn mirror(i: i32, n: i32) -> i32 {
    let period=2*n; let q=((i%period)+period)%period;
    return select(q,period-1-q,q>=n);
}
// Even reflection of fractions imposes a neutral 90-degree contact angle.
fn fraction(p: vec2i) -> f32 { return liquid(vec2i(mirror(p.x,gridSize().x),mirror(p.y,gridSize().y))).x; }
fn interfacial(p: vec2i) -> bool {
    let f=fraction(p);
    if (f>0.0 && f<1.0) { return true; }
    let a=vec4f(fraction(p-vec2i(1,0)),fraction(p+vec2i(1,0)),fraction(p-vec2i(0,1)),fraction(p+vec2i(0,1)));
    return (f>=1.0 && any(a<=vec4f(0))) || (f<=0.0 && any(a>=vec4f(1)));
}
fn outward(p: vec2i) -> vec2f {
    var s: array<f32,9>;
    for(var y=0;y<3;y++) { for(var x=0;x<3;x++) { s[y*3+x]=fraction(p+vec2i(x-1,y-1)); } }
    return normalize(plicNormal(s));
}
)wgsl";
        const std::string fitting = R"wgsl(
struct Fit { s0:f32, s1:f32, s2:f32, s3:f32, s4:f32, y0:f32, y1:f32, y2:f32 }
fn addPoint(f: ptr<function,Fit>, p: vec2f, n: vec2f, weight:f32) {
    let x=dot(p,vec2f(-n.y,n.x)); let y=dot(p,n); let x2=x*x;
    (*f).s0+=weight; (*f).s1+=weight*x; (*f).s2+=weight*x2;
    (*f).s3+=weight*x2*x; (*f).s4+=weight*x2*x2;
    (*f).y0+=weight*y; (*f).y1+=weight*x*y; (*f).y2+=weight*x2*y;
}
fn fitCurvature(f:Fit) -> vec2f {
    if(f.s0==0.0) { return vec2f(0); }
    let a=f.s2-f.s1*f.s1/f.s0; let b=f.s3-f.s1*f.s2/f.s0; let c=f.s4-f.s2*f.s2/f.s0;
    let d=f.y1-f.s1*f.y0/f.s0; let e=f.y2-f.s2*f.y0/f.s0; let determinant=a*c-b*b;
    // Thin fragments can supply coincident/collinear fit coordinates; reject that fit.
    if(determinant<=1e-5*max(a*c,1e-12)) { return vec2f(0); }
    let quadratic=(a*e-b*d)/determinant; let slope=(c*d-b*e)/determinant;
    let curvature=-2.0*quadratic/pow(1.0+slope*slope,1.5);
    // The local fit cannot resolve radii below half a cell (Popinet's fit limit).
    return vec2f(clamp(curvature,-2.0,2.0)/cellSize(),1);
}
)wgsl";
        compute.build(heights,"fluid interface heights",1,{"liquid"},R"wgsl(
fn column(p:vec2i,axis:vec2i) -> vec2f {
    var start=-4; var base=fraction(p-4*axis); var have=(base<=0.0||base>=1.0);
    base=select(0.0,1.0,base>=1.0);
    var sum=0.0; var result=vec2f(0); var nearest=10.0;
    for(var k=-3;k<=4;k++) {
        let f=fraction(p+k*axis);
        if(f<=0.0||f>=1.0) {
            let pure=select(0.0,1.0,f>=1.0);
            if(have && pure!=base) {
                let h=f32(start)+0.5+select(f32(k-start-1)-sum,sum,base==1.0);
                if(abs(h)<nearest) { nearest=abs(h); result=vec2f(h,select(-1.0,1.0,base==1.0)); }
            }
            start=k; base=pure; have=true; sum=0.0;
        } else { sum+=f; }
    }
    return result;
}
fn execute() {
    let p=cell(); var lo=1.0; var hi=0.0;
    for(var y=-1;y<=1;y++) { for(var x=-1;x<=1;x++) { let f=fraction(p+vec2i(x,y)); lo=min(lo,f); hi=max(hi,f); } }
    if(lo==hi) { write0(p,vec4f(0)); return; }
    let x=column(p,vec2i(1,0)); let y=column(p,vec2i(0,1));
    write0(p,vec4f(x.x,y.x,x.y,y.y));
}
)wgsl",std::string(geometry)+samples);

        const std::string heightSamples=R"wgsl(
fn heightAt(p:vec2i) -> vec4f {
    let q=vec2i(mirror(p.x,gridSize().x),mirror(p.y,gridSize().y));
    var h=height_field(q);
    if(p.x<0||p.x>=gridSize().x) { h.x=-h.x; h.z=-h.z; }
    if(p.y<0||p.y>=gridSize().y) { h.y=-h.y; h.w=-h.w; }
    return h;
}
fn heightCurvature(p:vec2i,axis:vec2i) -> vec2f {
    let tangent=vec2i(axis.y,axis.x); let xAxis=axis.x==1;
    let a=heightAt(p-tangent); let b=heightAt(p); let c=heightAt(p+tangent);
    let orientation=select(b.w,b.z,xAxis);
    if(orientation==0.0 || select(a.w,a.z,xAxis)!=orientation || select(c.w,c.z,xAxis)!=orientation) { return vec2f(0); }
    let left=select(a.y,a.x,xAxis); let mid=select(b.y,b.x,xAxis); let right=select(c.y,c.x,xAxis);
    let slope=0.5*(right-left); let second=(right-mid)-(mid-left);
    let k=-orientation*second/pow(1.0+slope*slope,1.5);
    return vec2f(clamp(k,-1.0,1.0)/cellSize(),1);
}
)wgsl";
        compute.build(curvature,"fluid height curvature",1,{"liquid","height_field"},R"wgsl(
fn execute() {
    let p=cell(); if(!interfacial(p)) { write0(p,vec4f(0)); return; }
    let n=outward(p); let first=select(vec2i(0,1),vec2i(1,0),abs(n.x)>abs(n.y));
    var k=heightCurvature(p,first);
    if(k.y==0.0) { k=heightCurvature(p,vec2i(first.y,first.x)); }
    if(k.y==0.0) {
        var fit:Fit;
        for(var j=-1;j<=1;j++) {
            let hx=heightAt(p+vec2i(0,j)); let hy=heightAt(p+vec2i(j,0));
            if(hx.z*n.x>0.0) { addPoint(&fit,vec2f(hx.x,f32(j)),n,1.0); }
            if(hy.w*n.y>0.0) { addPoint(&fit,vec2f(f32(j),hy.y),n,1.0); }
        }
        k=fitCurvature(fit);
    }
    write0(p,vec4f(k,0,0));
}
)wgsl",std::string(geometry)+samples+fitting+heightSamples);

        compute.build(complete,"fluid complete curvature",1,{"liquid","raw_curvature"},R"wgsl(
fn segment(p:vec2i,n:vec2f) -> vec3f {
    let offset=plicOffset(n,fraction(p)); let t=vec2f(-n.y,n.x);
    var low=10.0; var high=-10.0;
    for(var i=0;i<=1;i++) {
        if(abs(n.x)>1e-7) {
            let x=(offset-n.y*f32(i))/n.x;
            if(x>=0.0&&x<=1.0) { let s=dot(vec2f(x,f32(i)),t); low=min(low,s); high=max(high,s); }
        }
        if(abs(n.y)>1e-7) {
            let y=(offset-n.x*f32(i))/n.y;
            if(y>=0.0&&y<=1.0) { let s=dot(vec2f(f32(i),y),t); low=min(low,s); high=max(high,s); }
        }
    }
    if(high<=low) { return vec3f(0); }
    return vec3f(n*offset+t*(0.5*(low+high))-vec2f(0.5),high-low);
}
fn execute() {
    let p=cell(); var k=raw_curvature(p).xy;
    if(k.y!=0.0 || !interfacial(p)) { write0(p,vec4f(k,0,0)); return; }
    let n=outward(p); var sum=0.0; var count=0.0;
    for(var y=-1;y<=1;y++) { for(var x=-1;x<=1;x++) {
        let q=p+vec2i(x,y);
        let kq=raw_curvature(vec2i(mirror(q.x,gridSize().x),mirror(q.y,gridSize().y))).xy;
        if(kq.y!=0.0) { sum+=kq.x; count+=1.0; }
    } }
    if(count>0.0) { write0(p,vec4f(sum/count,1,0,0)); return; }
    var fit:Fit;
    for(var y=-1;y<=1;y++) { for(var x=-1;x<=1;x++) {
        let q=p+vec2i(x,y); let f=fraction(q);
        if(f>0.0&&f<1.0) {
            let nq=outward(q);
            if(dot(n,nq)>0.0) { let s=segment(q,nq); if(s.z>0.0) { addPoint(&fit,s.xy+vec2f(f32(x),f32(y)),n,s.z); } }
        }
    } }
    k=fitCurvature(fit);
    write0(p,vec4f(k,0,0));
}
)wgsl",std::string(geometry)+samples+fitting);

        compute.build(predict,"fluid balanced capillary predictor",1,{"velocity","pressure","weights","liquid","curvature_field"},R"wgsl(
fn execute() {
    let p=cell(); let horizontal=mode()==0; let axis=select(vec2i(0,1),vec2i(1,0),horizontal);
    let coordinate=select(p.y,p.x,horizontal); let extent=select(gridSize().y,gridSize().x,horizontal);
    if(coordinate==0||coordinate==extent) { write0(p,vec4f(0)); return; }
    let left=curvature_field(p-axis).xy; let right=curvature_field(p).xy;
    var k=0.0;
    if(left.y+right.y>0.0) { k=(left.x*left.y+right.x*right.y)/(left.y+right.y); }
    let capillary=params.value.w*k*(liquid(p).x-liquid(p-axis).x);
    let pressure_jump=pressure(p).x-pressure(p-axis).x;
    let a=weights(p); let inverse_density=select(a.z,a.x,horizontal);
    write0(p,vec4f(velocity(p).x+timeStep()*inverse_density*(capillary-pressure_jump)/cellSize()));
}
)wgsl");
    }
};
}
