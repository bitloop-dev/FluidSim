#pragma once
#include "FluidWebGPUCompute.h"

namespace fluid::webgpu {

// Pressure statistics only: sum/max residual, dot product, or correction sum/dot.
// A bounded first pass and one finishing workgroup replace the 4x4 reduction chain.
class UniformPressureReduction
{
public:
    UniformPressureReduction(Compute& compute, int cells)
        : compute(compute), partials(compute.field(groupCount(cells), 1, 4, "uniform pressure partials"))
    {
        constexpr std::string_view shared = R"wgsl(
var<workgroup> partial:array<vec4f,128>;
fn combine(a:vec4f,b:vec4f)->vec4f {
    return vec4f(a.x+b.x,max(a.y,b.y),a.z+b.z,max(a.w,b.w));
}
fn checked(value:vec4f)->vec4f {
    if(!finite(value.x)||!finite(value.y)||!finite(value.z)){return vec4f(0,0,0,1);}
    return value;
}
fn groupReduction(lane:u32,value:vec4f)->vec4f {
    partial[lane]=checked(value);workgroupBarrier();
    for(var stride=64u;stride>0u;stride/=2u){
        if(lane<stride){partial[lane]=combine(partial[lane],partial[lane+stride]);}
        workgroupBarrier();
    }
    return checked(partial[0]);
}
)wgsl";
        compute.build(first, "uniform pressure reduction partials", 1, {"source", "other"}, R"wgsl(
fn execute(){
    let lane=u32(cell().x)%128u;let group=u32(cell().x)/128u;
    let count=u32(params.grid.x);let groups=u32(params.grid.y);let width=u32(sourceSize().x);
    var total=vec4f(0);
    for(var base=group*512u;base<count;base+=groups*512u){
        for(var offset=0u;offset<512u;offset+=128u){
            let i=base+offset+lane;if(i>=count){continue;}
            let p=vec2i(i32(i%width),i32(i/width));var value=source(p);
            if(mode()==0){value=vec4f(value.x,abs(value.x),0,0);}
            if(mode()==1){value=vec4f(value.x*other(p).x,0,0,0);}
            if(mode()==2){value=vec4f(value.x,0,value.x*other(p).x,0);}
            total=combine(total,checked(value));
        }
    }
    let result=groupReduction(lane,total);
    if(lane==0u){write0(vec2i(i32(group),0),result);}
})wgsl", shared, true, {threads, 1});
        compute.build(finish, "uniform pressure reduction finish", 1, {"source"}, R"wgsl(
fn execute(){
    let lane=u32(cell().x);var total=vec4f(0);
    for(var i=lane;i<u32(params.grid.x);i+=128u){total=combine(total,checked(source(vec2i(i32(i),0))));}
    let result=groupReduction(lane,total);
    if(lane==0u){write0(vec2i(0),result);}
})wgsl", shared, true, {threads, 1});
    }

    void reduce(const Field& input, Field& result, int mode, const Field* other = nullptr)
    {
        const int cells = input.width * input.height;
        const int groups = groupCount(cells);
        Parameters parameters;
        parameters.grid = {cells, groups, mode, 0};
        compute.run(first, {&partials}, {{"source", &input}, {"other", other ? other : &input}},
                    parameters, {unsigned(groups) * threads, 1, 1});
        parameters.grid[0] = groups;
        compute.run(finish, {&result}, {{"source", &partials}}, parameters, {threads, 1, 1});
    }

private:
    static constexpr unsigned threads = 128;
    static int groupCount(int cells) { return std::min((cells + 511) / 512, 1024); }
    Compute& compute;
    Field partials;
    Kernel first, finish;
};

} // namespace fluid::webgpu
