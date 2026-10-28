#pragma once
#include <string_view>

namespace fluid::webgpu {

// Three 1360-float planes fit in 16 KiB; the packed tail may also have truncated heights.
inline constexpr std::string_view coarse_cycle_shader=R"wgsl(
var<workgroup> solution: array<f32,1360>;
var<workgroup> scratch: array<f32,1360>;
var<workgroup> forcing: array<f32,1360>;

fn shape(l: i32) -> vec2i { return vec2i(params.vectors[u32(l)].xy); }
fn coordinate(i: i32,size: vec2i) -> vec2i { return vec2i(i&(size.x-1),i>>firstTrailingBit(u32(size.x))); }
fn index(l: i32,p: vec2i) -> i32 {
    return i32(params.vectors[u32(l)].w)+p.y*shape(l).x+p.x;
}
fn sample(l: i32,p: vec2i) -> f32 { return solution[index(l,clamp(p,vec2i(0),shape(l)-1))]; }
fn coefficients(l: i32,p: vec2i) -> vec4f {
    let first=4*index(l,clamp(p,vec2i(0),shape(l)-1));
    return vec4f(packedData[first],packedData[first+1],packedData[first+2],packedData[first+3]);
}
fn mapAt(offset: i32,width: i32,p: vec2i) -> vec4f {
    let first=offset+4*(p.y*width+p.x);
    return vec4f(mapsData[first],mapsData[first+1],mapsData[first+2],mapsData[first+3]);
}
fn firstColor(l: i32,point: vec2i,fromZero: bool) -> f32 {
    let p=clamp(point,vec2i(0),shape(l)-1); let a=coefficients(l,p);
    let diagonal=(a.x+a.y)+(a.z+a.w); var value=forcing[index(l,p)];
    if (!fromZero) {
        value+=(a.x*sample(l,p-vec2i(1,0))+a.y*sample(l,p+vec2i(1,0)))+
        (a.z*sample(l,p-vec2i(0,1))+a.w*sample(l,p+vec2i(0,1)));
    }
    if (diagonal>0.0) { return value/diagonal; } return 0.0;
}
// Reuse the first colour through shared memory instead of recomputing it for every neighbour.
fn smoothLevel(l: i32,lane: i32,reverse: bool,fromZero: bool) {
    let size=shape(l);
    for (var phase=0; phase<2; phase++) {
        let color=select(0,1,reverse)^phase;
        for (var batch=0; batch<size.x*size.y; batch+=256) {
            let i=batch+lane;
            if (i<size.x*size.y) {
                let p=coordinate(i,size);
                if ((p.x+p.y)%2==color) { solution[index(l,p)]=firstColor(l,p,fromZero && phase==0); }
            }
        }
        workgroupBarrier();
    }
}
fn solveBottom(l: i32) {
    // Match the generic solve: DXC can miscompile an early return inside the factor loop.
    let size=shape(l); let count=size.x*size.y; let n=count-1;
    let offset=index(l,vec2i(0)); var values: array<f32,15>; var mean=0.0; var valid=true;
    for (var i=0; i<count; i++) { mean+=forcing[offset+i]; } mean/=f32(count);
    for (var row=0; row<n; row++) {
        let diagonal=factor(vec2i(row,row)).x;
        valid = valid && diagonal>0.0 && finite(diagonal);
        var value=forcing[offset+row+1]-mean;
        for (var col=0; col<row; col++) { value-=factor(vec2i(col,row)).x*values[col]; }
        values[row]=value/diagonal;
    }
    for (var row=n-1; row>=0; row--) {
        var value=values[row];
        for (var col=row+1; col<n; col++) { value-=factor(vec2i(row,col)).x*values[col]; }
        values[row]=value/factor(vec2i(row,row)).x;
    }
    mean=0.0; for (var i=0; i<n; i++) { mean+=values[i]; } mean/=f32(count);
    solution[offset]=select(0.0,-mean,valid);
    for (var i=1; i<count; i++) { solution[offset+i]=select(0.0,values[i-1]-mean,valid); }
}
fn execute() {
    let lane=cell().y*16+cell().x; let size=shape(0); let last=params.grid.w-1;
    for (var batch=0; batch<size.x*size.y; batch+=256) {
        let i=batch+lane;
        if (i<size.x*size.y) { forcing[i]=rhsAt(coordinate(i,size)); }
    }
    workgroupBarrier();
    for (var l=0; l<last; l++) {
        for (var sweep=0; sweep<5; sweep++) { smoothLevel(l,lane,false,sweep==0); }
        let fine=shape(l); let coarse=shape(l+1); let transfer=params.vectors[u32(l+4)];
        for (var batch=0; batch<fine.x*fine.y; batch+=256) {
            let i=batch+lane;
            if (i<fine.x*fine.y) {
                let p=coordinate(i,fine); let a=coefficients(l,p); let c=sample(l,p);
                let product=(a.x*(c-sample(l,p-vec2i(1,0)))+a.y*(c-sample(l,p+vec2i(1,0))))+
                (a.z*(c-sample(l,p-vec2i(0,1)))+a.w*(c-sample(l,p+vec2i(0,1))));
                scratch[index(l,p)]=forcing[index(l,p)]-product;
            }
        }
        workgroupBarrier();
        for (var batch=0; batch<coarse.x*coarse.y; batch+=256) {
            let i=batch+lane;
            if (i<coarse.x*coarse.y) {
                let p=coordinate(i,coarse); let base=p*2-1;
                let wx=mapAt(i32(transfer.y),i32(transfer.w),vec2i(p.x,0));
                let wy=mapAt(i32(transfer.y),i32(transfer.w),vec2i(p.y,1));
                var value=0.0;
                for (var y=0; y<4; y++) {
                    var horizontal=0.0;
                    for (var x=0; x<4; x++) {
                        let q=base+vec2i(x,y); if (any(q<vec2i(0)) || any(q>=fine)) { continue; }
                        horizontal+=wx[x]*scratch[index(l,q)];
                    }
                    value+=wy[y]*horizontal;
                }
                forcing[index(l+1,p)]=value;
            }
        }
        workgroupBarrier();
    }
    if (lane==0) { solveBottom(last); }
    workgroupBarrier();
    for (var l=last-1; l>=0; l--) {
        let fine=shape(l); let transfer=params.vectors[u32(l+4)];
        for (var batch=0; batch<fine.x*fine.y; batch+=256) {
            let i=batch+lane;
            if (i<fine.x*fine.y) {
                let p=coordinate(i,fine);
                let px=mapAt(i32(transfer.x),i32(transfer.z),vec2i(p.x,0));
                let py=mapAt(i32(transfer.x),i32(transfer.z),vec2i(p.y,1));
                let a=(1.0-px.z)*sample(l+1,vec2i(i32(px.x),i32(py.x)))+px.z*sample(l+1,vec2i(i32(px.y),i32(py.x)));
                let b=(1.0-px.z)*sample(l+1,vec2i(i32(px.x),i32(py.y)))+px.z*sample(l+1,vec2i(i32(px.y),i32(py.y)));
                scratch[index(l,p)]=sample(l,p)+(1.0-py.z)*a+py.z*b;
            }
        }
        workgroupBarrier();
        for (var batch=0; batch<fine.x*fine.y; batch+=256) {
            let i=batch+lane;
            if (i<fine.x*fine.y) {
                let p=coordinate(i,fine); solution[index(l,p)]=scratch[index(l,p)];
            }
        }
        workgroupBarrier();
        for (var sweep=0; sweep<5; sweep++) { smoothLevel(l,lane,true,false); }
    }
    for (var batch=0; batch<size.x*size.y; batch+=256) {
        let i=batch+lane;
        if (i<size.x*size.y) { write0(coordinate(i,size),vec4f(solution[i])); }
    }
}
)wgsl";

} // namespace fluid::webgpu
