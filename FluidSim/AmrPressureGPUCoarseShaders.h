#pragma once
#include <string_view>

namespace fluid::webgpu::amrpressure
{
inline constexpr int coarseCapacity = 2048;
inline constexpr unsigned coarseThreads = 128;

inline constexpr std::string_view coarsePack = R"wgsl(
fn execute(){
    let i=u32(point.x);if(i>=u32(params.grid.x)||!valid(i)){return;}
    let s=cellStride(i);let a=6u*(u32(params.grid.y)+i);
    output0Data[a]=bxData[i];output0Data[a+1u]=bxData[i+1u];
    output0Data[a+2u]=byData[i];output0Data[a+3u]=byData[i+s];
    output0Data[a+4u]=diagonalData[2u*i];output0Data[a+5u]=diagonalData[2u*i+1u];
}
)wgsl";

// Two planes fit in 16 KiB, including every patch halo in the coupled tail.
inline constexpr std::string_view coarseCycle = R"wgsl(
var<workgroup> solution: array<f32,2048>;
var<workgroup> forcing: array<f32,2048>;
fn size(l:u32)->u32{return u32(params.vectors[l].x);}
fn index(l:u32,i:u32)->u32{return u32(params.vectors[l].y)+i;}
fn word(l:u32,i:u32,k:u32)->u32{return bitcast<u32>(recipesData[12u*index(l,i)+k]);}
fn integer(l:u32,i:u32,k:u32)->i32{return bitcast<i32>(recipesData[12u*index(l,i)+k]);}
fn weight(l:u32,i:u32,k:u32)->f32{return recipesData[12u*index(l,i)+k];}
fn valid(l:u32,i:u32)->bool{return word(l,i,0u)!=0u;}
fn stencil(l:u32,i:u32)->vec4f{
    let a=6u*index(l,i);return vec4f(coefficientsData[a],coefficientsData[a+1u],coefficientsData[a+2u],coefficientsData[a+3u]);
}
fn fillBoundary(l:u32,lane:u32){
    for(var i=lane;i<size(l);i+=128u){
        let source=integer(l,i,2u);if(source == -2){continue;}
        var value=0.;
        if(source>=0){value=solution[index(l,u32(source))];}
        else{
            value=weight(l,i,5u)*solution[index(l,word(l,i,3u))];
            if(weight(l,i,6u)!=0.){value+=weight(l,i,6u)*solution[index(l,word(l,i,4u))];}
        }
        solution[index(l,i)]=value;
    }
    workgroupBarrier();
}
fn smoothLevel(l:u32,lane:u32,sweeps:u32){
    for(var sweep=0u;sweep<sweeps;sweep++){
        for(var color=0u;color<2u;color++){
            fillBoundary(l,lane);
            for(var i=lane;i<size(l);i+=128u){
                if(!valid(l,i)||word(l,i,1u)!=color){continue;}
                let s=word(l,i,0u);let a=stencil(l,i);let g=index(l,i);let f=params.vectors[l].z;
                let neighbors=f*(a.x*solution[g-1u]+a.y*solution[g+1u])+f*(a.z*solution[g-s]+a.w*solution[g+s]);
                let delta=coefficientsData[6u*g+4u];let diagonal=coefficientsData[6u*g+5u];
                solution[g]=(forcing[g]+neighbors-solution[g]*delta)/(diagonal-delta);
            }
            workgroupBarrier();
        }
    }
}
fn residual(l:u32,i:u32)->f32{
    let s=word(l,i,0u);let a=stencil(l,i);let g=index(l,i);let p=solution[g];let f=params.vectors[l].z;
    let value=-f*(a.y*(solution[g+1u]-p)-a.x*(p-solution[g-1u]))-f*(a.w*(solution[g+s]-p)-a.z*(p-solution[g-s]));
    return -value+forcing[g];
}
fn execute(){
    let lane=u32(point.x);let last=u32(params.grid.x)-1u;
    for(var i=lane;i<u32(params.grid.y);i+=128u){
        solution[i]=0.;forcing[i]=0.;if(i<size(0u)){forcing[i]=rhsData[i];}
    }
    workgroupBarrier();
    for(var l=0u;l<last;l++){
        smoothLevel(l,lane,2u);fillBoundary(l,lane);
        // Each fine residual belongs to one coarse average, so no scratch plane is needed.
        for(var i=lane;i<size(l+1u);i+=128u){
            if(!valid(l+1u,i)){continue;}
            let a=residual(l,word(l+1u,i,8u));let b=residual(l,word(l+1u,i,9u));
            let c=residual(l,word(l+1u,i,10u));let d=residual(l,word(l+1u,i,11u));
            forcing[index(l+1u,i)]=.25*(a+b+c+d);
        }
        workgroupBarrier();
    }
    smoothLevel(last,lane,8u);
    for(var l=i32(last)-1;l>=0;l--){
        let level=u32(l);
        for(var i=lane;i<size(level);i+=128u){
            if(valid(level,i)){solution[index(level,i)]+=solution[index(level+1u,word(level,i,7u))];}
        }
        workgroupBarrier();smoothLevel(level,lane,2u);
    }
    for(var i=lane;i<size(0u);i+=128u){output0Data[i]=solution[i];}
}
)wgsl";
} // namespace fluid::webgpu::amrpressure
