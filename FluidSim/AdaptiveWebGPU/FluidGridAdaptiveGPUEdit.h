#pragma once
#include <string_view>
namespace fluid::amr_gpu {
// Explicit material source/sink only. The ordinary transport/projection is unchanged.
inline constexpr std::string_view replaceMaterial=R"wgsl(
fn execute() {
    let i=cell().y*i32(params.extent.x)+cell().x;
    if(i>=params.grid.x){return;}
    let base=bitcast<i32>(editsData[4*i]);
    let plane=bitcast<i32>(editsData[4*i+1]);
    let component=bitcast<i32>(editsData[4*i+2]);
    let density=editsData[4*i+3];
    var previous=0.0;
    for(var n=0;n<params.grid.z;n++){previous+=output0Data[base+n*plane];}
    // Replacement material inherits the current cell velocity.
    for(var d=0;d<2;d++){output0Data[base+(params.grid.z+d)*plane]*=density/previous;}
    for(var n=0;n<params.grid.z;n++){output0Data[base+n*plane]=select(0.0,density,n==component);}
}
)wgsl";
}
