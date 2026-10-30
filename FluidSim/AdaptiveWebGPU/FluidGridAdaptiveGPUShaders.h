#pragma once
#include <string_view>

namespace fluid::amr_gpu {

// Patch record: box[0:4], q[4:7], slopes[7:10], weights[10:12], faces[12:16],
// flux offsets[16:18], pressure[18:20], geometry offset[20], cell-mask offset[21].
inline constexpr std::string_view addressing=R"wgsl(
fn patchValue(k:i32)->i32 { return bitcast<i32>(metadataData[i32(slice())*24+k]); }
fn valid(p:vec2i)->bool { return all(p>=vec2i(0)) && all(p<vec2i(patchValue(2),patchValue(3))); }
fn qi(p:vec2i,n:i32)->i32 { return patchValue(4)+(p.y+2)*patchValue(5)+p.x+2+n*patchValue(6); }
fn si(p:vec2i,n:i32)->i32 { return patchValue(7)+(p.y+1)*patchValue(8)+p.x+1+n*patchValue(9); }
fn gi(p:vec2i,n:i32)->i32 { return patchValue(20)+(p.y+1)*patchValue(8)+p.x+1+n*patchValue(9); }
fn wi(p:vec2i,n:i32)->i32 { return patchValue(10)+p.y*patchValue(2)+p.x+n*patchValue(11); }
fn ci(p:vec2i)->i32 { return patchValue(21)+p.y*patchValue(2)+p.x; }
fn fi(p:vec2i,d:i32,n:i32,flux:bool)->i32 {
    let base=select(patchValue(12+2*d),patchValue(16+d),flux);
    return base+p.y*(patchValue(2)+select(0,1,d==0))+p.x+n*patchValue(13+2*d);
}
fn pi(p:vec2i)->i32 { return patchValue(18)+(p.y+1)*patchValue(19)+p.x+1; }
fn nc()->i32 { return params.grid.z; }
fn nl()->i32 { return params.grid.w; }
fn axis()->i32 { return i32(params.level.x); }
fn dir(d:i32)->vec2i { return select(vec2i(0,1),vec2i(1,0),d==0); }
fn faceValid(p:vec2i,d:i32)->bool { return all(p<vec2i(patchValue(2),patchValue(3))+dir(d)); }
)wgsl";

inline constexpr std::string_view qLimiter=R"wgsl(
fn q(p:vec2i,n:i32)->f32 { return stateData[qi(p,n)]; }
fn stencil(p:vec2i)->vec3i { return vec3i(qi(p,0),patchValue(6),patchValue(5)); }
fn reconstructionSample(s:vec3i,n:i32,side:i32)->f32 {
    let offsets=array<i32,5>(0,-1,1,-s.z,s.z);
    return stateData[s.x+n*s.y+offsets[side]];
}
)wgsl";

inline constexpr std::string_view slopes=R"wgsl(
fn execute() {
    let p=cell()-1;
    if(any(p>=vec2i(patchValue(2),patchValue(3))+1)){return;}
    let lim=materialLimiter(stencil(p),nc());
    for(var n=0;n<nc();n++) {
        let slope=0.5*vec2f(q(p+vec2i(1,0),n)-q(p-vec2i(1,0),n),q(p+vec2i(0,1),n)-q(p-vec2i(0,1),n))*lim;
        output0Data[si(p,n)]=slope.x;output0Data[si(p,nc()+n)]=slope.y;
    }
})wgsl";

inline constexpr std::string_view gasFlux=R"wgsl(
fn execute() {
    let p=cell();let d=axis();if(!faceValid(p,d)){return;}
    let speed=velocityData[fi(p,d,0,false)];
    let donor=p-select(vec2i(0),dir(d),speed>=0.0);
    var rho=0.0;for(var n=0;n<nc();n++){rho+=stateData[qi(donor,n)];}
    let halfDt=0.5*timeStep()/cellSize();
    let normal=select(-0.5,0.5,speed>=0.0)-halfDt*speed;
    let transverse=clamp(-halfDt*stateData[qi(donor,nc()+1-d)]/rho,-0.5,0.5);
    let delta=select(vec2f(transverse,normal),vec2f(normal,transverse),d==0);
    var mass=0.0;
    for(var n=0;n<nc();n++) {
        let transfer=speed*(stateData[qi(donor,n)]+delta.x*slopesData[si(donor,n)]+delta.y*slopesData[si(donor,nc()+n)]);
        output0Data[fi(p,d,n,true)]=transfer;mass+=transfer;
    }
    for(var a=0;a<2;a++){output0Data[fi(p,d,nc()+a,true)]=mass*stateData[qi(donor,nc()+a)]/rho;}
})wgsl";

inline constexpr std::string_view color=R"wgsl(
fn execute(){let p=cell();if(!valid(p)||maskData[ci(p)]==0.0){return;}
    let ratio=i32(params.level.y);var lim=vec2f(0);if(ratio>1){lim=materialLimiter(stencil(p),nc());}
    for(var y=0;y<ratio;y++){for(var x=0;x<ratio;x++){
        let delta=(vec2f(f32(x),f32(y))+0.5)/f32(ratio)-0.5;var rgb=vec3f(0);var liquid=0.0;
        for(var n=0;n<nc();n++){var mass=q(p,n);if(ratio>1){mass+=dot(delta,0.5*vec2f(q(p+vec2i(1,0),n)-q(p-vec2i(1,0),n),q(p+vec2i(0,1),n)-q(p-vec2i(0,1),n))*lim);}
            let k=n*4;rgb+=mass/paletteData[k+3]*vec3f(paletteData[k],paletteData[k+1],paletteData[k+2]);
            if(n<nl()){liquid+=mass/paletteData[k+3];}}
        write0((p+vec2i(patchValue(0),patchValue(1)))*ratio+vec2i(x,y),vec4f(rgb,select(1.0,liquid,params.level.z!=0.0)));
    }}
})wgsl";

inline constexpr std::string_view gasUpdate=R"wgsl(
fn execute() {
    let p=cell();if(!valid(p)){return;}let scale=timeStep()/cellSize();
    for(var n=0;n<nc()+2;n++) {
        output0Data[qi(p,n)]-=scale*(fxData[fi(p+vec2i(1,0),0,n,true)]-fxData[fi(p,0,n,true)]+fyData[fi(p+vec2i(0,1),1,n,true)]-fyData[fi(p,1,n,true)]);
    }
})wgsl";

inline constexpr std::string_view splitWeights=R"wgsl(
fn execute() {
    let p=cell();if(!valid(p)){return;}
    var liquid=0.0;var gas=0.0;var rho=0.0;
    for(var n=0;n<nc();n++) {let mass=stateData[qi(p,n)];rho+=mass;if(n<nl()){liquid+=mass/densityData[n];}else{gas+=mass/densityData[n];}}
    let selected=liquid>0.5;let volume=select(gas,liquid,selected);var mass=0.0;
    for(var n=0;n<nc();n++){var w=0.0;if((n<nl())==selected){w=stateData[qi(p,n)]/volume;}output0Data[wi(p,n)]=w;mass+=w;}
    for(var d=0;d<2;d++){output0Data[wi(p,nc()+d)]=mass*stateData[qi(p,nc()+d)]/rho;}
})wgsl";

inline constexpr std::string_view phaseGeometry=R"wgsl(
fn fraction(p:vec2i,liquid:bool)->f32 {
    var result=0.0;for(var n=select(nl(),0,liquid);n<select(nc(),nl(),liquid);n++){result+=stateData[qi(p,n)]/densityData[n];}return result;
}
fn execute() {
    let p=cell()-1;if(any(p>=vec2i(patchValue(2),patchValue(3))+1)){return;}
    let f=fraction(p,true);let gas=fraction(p,false);let liquid=f<=gas;let minority=select(gas,f,liquid);
    output0Data[gi(p,0)]=f;output0Data[gi(p,1)]=gas;
    if(minority>0.0) {
        var s:array<f32,9>;for(var y=0;y<3;y++){for(var x=0;x<3;x++){s[y*3+x]=fraction(p+vec2i(x-1,y-1),liquid);}}
        let normal=plicNormal(s);output0Data[gi(p,2)]=normal.x;output0Data[gi(p,3)]=normal.y;output0Data[gi(p,4)]=plicOffset(normal,minority);
    }
})wgsl";

inline constexpr std::string_view phaseFlux=R"wgsl(
fn execute() {
    let p=cell();let d=axis();if(!faceValid(p,d)){return;}
    let speed=velocityData[fi(p,d,0,false)];
    if(speed==0.0){for(var n=0;n<nc()+2;n++){output0Data[fi(p,d,n,true)]=0.0;}return;}
    let positive=speed>0.0;let donor=p-select(vec2i(0),dir(d),positive);
    let f=geometryData[gi(donor,0)];let gas=geometryData[gi(donor,1)];let scale=timeStep()/cellSize();let total=speed*scale;
    let smallLiquid=f<=gas;let minority=select(gas,f,smallLiquid);var smallTransfer=0.0;
    if(minority>0.0) {
        let strip=abs(total);let lo=select(0.0,1.0-strip,positive);let hi=select(strip,1.0,positive);
        let a=select(vec2f(0,lo),vec2f(lo,0),d==0);let b=select(vec2f(1,hi),vec2f(hi,1),d==0);
        let normal=vec2f(geometryData[gi(donor,2)],geometryData[gi(donor,3)]);
        let area=plicArea(normal,geometryData[gi(donor,4)],a,b);
        smallTransfer=sign(speed)*clamp(area,max(0.0,strip-(1.0-minority)),min(strip,minority));
    }
    let liquid=select(total-smallTransfer,smallTransfer,smallLiquid);let gasTransfer=select(smallTransfer,total-smallTransfer,smallLiquid);
    var rho=0.0;var mass=0.0;
    for(var n=0;n<nc();n++) {
        let volume=select(gas,f,n<nl());var transfer=0.0;
        if(volume>0.0){transfer=stateData[qi(donor,n)]*(select(gasTransfer,liquid,n<nl())/volume)/scale;}
        output0Data[fi(p,d,n,true)]=transfer;mass+=transfer;rho+=stateData[qi(donor,n)];
    }
    for(var a=0;a<2;a++){output0Data[fi(p,d,nc()+a,true)]=mass*stateData[qi(donor,nc()+a)]/rho;}
})wgsl";

inline constexpr std::string_view phaseUpdate=R"wgsl(
fn execute() {
    let p=cell();if(!valid(p)){return;}let d=axis();let delta=dir(d);let scale=timeStep()/cellSize();
    let volume=scale*(velocityData[fi(p+delta,d,0,false)]-velocityData[fi(p,d,0,false)]);
    for(var n=0;n<nc()+2;n++){
        output0Data[qi(p,n)]+=scale*(fluxData[fi(p,d,n,true)]-fluxData[fi(p+delta,d,n,true)])+weightsData[wi(p,n)]*volume;
    }
})wgsl";

inline constexpr std::string_view predictor=R"wgsl(
fn execute() {
    let p=cell();let d=axis();if(!faceValid(p,d)){return;}let delta=dir(d);var left=0.0;var right=0.0;
    for(var n=0;n<nc();n++){left+=stateData[qi(p-delta,n)];right+=stateData[qi(p,n)];}
    output1Data[fi(p,d,0,false)]=2.0/(left+right);
    let global=p+vec2i(patchValue(0),patchValue(1));let wall=global[d]==0 || global[d]==params.grid[d];
    var speed=0.0;if(!wall){speed=0.5*(stateData[qi(p-delta,nc()+d)]/left+stateData[qi(p,nc()+d)]/right)+params.vectors[0][d];}
    output0Data[fi(p,d,0,false)]=speed;
})wgsl";

inline constexpr std::string_view divergence=R"wgsl(
fn execute() {
    let p=cell();if(!valid(p)){return;}
    let div=(uxData[fi(p+vec2i(1,0),0,0,false)]-uxData[fi(p,0,0,false)]+uyData[fi(p+vec2i(0,1),1,0,false)]-uyData[fi(p,1,0,false)])/cellSize();
    output0Data[pi(p)]=-div;
    output1Data[ci(p)]=select(0.0,select(3.402823466e+38,abs(div),finite(div)),maskData[ci(p)]!=0.0);
})wgsl";

inline constexpr std::string_view packBeta=R"wgsl(
fn execute(){let p=cell();let d=axis();if(faceValid(p,d)){output0Data[pi(p)]=betaData[fi(p,d,0,false)];}}
)wgsl";
inline constexpr std::string_view correctFaces=R"wgsl(
fn execute(){let p=cell();let d=axis();if(faceValid(p,d)){output0Data[fi(p,d,0,false)]=predictedData[fi(p,d,0,false)]+fluxData[pi(p)];}}
)wgsl";

inline constexpr std::string_view correctMomentum=R"wgsl(
fn execute(){
    let p=cell();if(!valid(p)){return;}var rho=0.0;for(var n=0;n<nc();n++){rho+=output0Data[qi(p,n)];}
    let global=p+vec2i(patchValue(0),patchValue(1));
    let wallWeight=0.5*(select(vec2f(0),vec2f(1),global!=vec2i(0))+select(vec2f(0),vec2f(1),global!=params.grid.xy-1));
    var correction=vec2f(0);
    if(params.value.w!=0.0){
        correction.x=0.5*((uxData[fi(p,0,0,false)]-pxData[fi(p,0,0,false)])+(uxData[fi(p+vec2i(1,0),0,0,false)]-pxData[fi(p+vec2i(1,0),0,0,false)]));
        correction.y=0.5*((uyData[fi(p,1,0,false)]-pyData[fi(p,1,0,false)])+(uyData[fi(p+vec2i(0,1),1,0,false)]-pyData[fi(p+vec2i(0,1),1,0,false)]));
    }
    correction+=wallWeight*params.vectors[0].xy;
    output0Data[qi(p,nc())]+=rho*correction.x;output0Data[qi(p,nc()+1)]+=rho*correction.y;
})wgsl";

inline constexpr std::string_view damping=R"wgsl(
fn execute(){let p=cell();if(!valid(p)){return;}for(var n=0;n<nl();n++){if(output0Data[qi(p,n)]!=0.0){return;}}
    output0Data[qi(p,nc())]*=params.value.w;output0Data[qi(p,nc()+1)]*=params.value.w;
})wgsl";
inline constexpr std::string_view donorCfl=R"wgsl(
fn execute(){let p=cell();if(!valid(p)){return;}
    output0Data[ci(p)]=(max(uxData[fi(p+vec2i(1,0),0,0,false)],0.0)+max(-uxData[fi(p,0,0,false)],0.0)+max(uyData[fi(p+vec2i(0,1),1,0,false)],0.0)+max(-uyData[fi(p,1,0,false)],0.0))/cellSize();
})wgsl";

inline constexpr std::string_view reduceMax=R"wgsl(
fn execute(){let index=u32(cell().y)*params.extent.x+u32(cell().x);let count=u32(params.grid.x);var value=0.0;
    for(var i=index*256u;i<min(count,(index+1u)*256u);i++){let x=sourceData[i];value=max(value,select(3.402823466e+38,abs(x),finite(x)));}output0Data[index]=value;
})wgsl";
inline constexpr std::string_view zero=R"wgsl(
fn execute(){let i=u32(cell().y)*params.extent.x+u32(cell().x);let info=params.fields[0];
    for(var c=0u;c<info.z;c++){output0Data[c*info.x*info.y+i]=0.0;}}
)wgsl";
inline constexpr std::string_view copy=R"wgsl(
fn execute(){let i=u32(cell().y)*params.extent.x+u32(cell().x);let info=params.fields[0];
    for(var c=0u;c<info.z;c++){let at=c*info.x*info.y+i;output0Data[at]=sourceData[at];}}
)wgsl";

// Recipes encode integer addresses by bit representation, never float conversion.
inline constexpr std::string_view ghost=R"wgsl(
fn ri(i:i32,k:i32)->i32{return bitcast<i32>(recipesData[i*8+k]);}
fn reconstructionSample(s:vec3i,n:i32,side:i32)->f32 {
    let offsets=array<i32,5>(0,-1,1,-s.z,s.z);
    return parentData[s.x+n*s.y+offsets[side]];
}
fn execute(){
    let i=cell().y*i32(params.extent.x)+cell().x;if(i>=params.grid.x){return;}
    let destination=ri(i,0);let source=ri(i,1);let plane=ri(i,2);let stride=ri(i,3);let flags=ri(i,4);let destinationPlane=ri(i,7);let coarse=(flags&1)!=0;
    let count=params.grid.z+2;
    if(!coarse){for(var n=0;n<count;n++){output0Data[destination+n*destinationPlane]=output0Data[source+n*plane];}}
    else {
        let s=vec3i(source,plane,stride);let limited=materialLimiter(s,params.grid.z);
        let momentum=momentumSlopes(s,params.grid.z,limited);let delta=vec2f(recipesData[i*8+5],recipesData[i*8+6]);
        for(var n=0;n<count;n++) {
            var slope:vec2f;
            if(n<params.grid.z){slope=materialSlope(s,n,limited);}else{slope=momentum[n-params.grid.z];}
            output0Data[destination+n*destinationPlane]=parentData[source+n*plane]+dot(delta,slope);
        }
    }
    if((flags&2)!=0){output0Data[destination+params.grid.z*destinationPlane]*=-1.0;}
    if((flags&4)!=0){output0Data[destination+(params.grid.z+1)*destinationPlane]*=-1.0;}
})wgsl";

inline constexpr std::string_view average=R"wgsl(
fn r(i:i32,k:i32)->i32{return bitcast<i32>(recipesData[i*8+k]);}
fn execute(){let i=cell().y*i32(params.extent.x)+cell().x;if(i>=params.grid.x){return;}
    // Direct deep coarsening can combine 1024 samples; retain the low-order float sum.
    for(var n=0;n<params.grid.z;n++){var sum=0.0;var correction=0.0;for(var y=0;y<r(i,6);y++){for(var x=0;x<r(i,5);x++){
        let value=sourceData[r(i,1)+n*r(i,3)+y*r(i,4)+x]-correction;let next=sum+value;correction=(next-sum)-value;sum=next;}}
        output0Data[r(i,0)+n*r(i,2)]=sum/f32(r(i,5)*r(i,6));}
})wgsl";
inline constexpr std::string_view sync=R"wgsl(
fn r(i:i32,k:i32)->i32{return bitcast<i32>(recipesData[i*4+k]);}
fn execute(){let i=cell().y*i32(params.extent.x)+cell().x;if(i>=params.grid.x){return;}
    for(var n=0;n<params.grid.z;n++){output0Data[r(i,0)+n*r(i,2)]=output0Data[r(i,1)+n*r(i,3)];}}
)wgsl";
inline constexpr std::string_view overlap=R"wgsl(
fn r(i:i32,k:i32)->i32{return bitcast<i32>(recipesData[i*4+k]);}
fn execute(){let i=cell().y*i32(params.extent.x)+cell().x;if(i>=params.grid.x){return;}
    for(var n=0;n<params.grid.z;n++){output0Data[r(i,0)+n*r(i,2)]=sourceData[r(i,1)+n*r(i,3)];}}
)wgsl";

inline constexpr std::string_view refine=R"wgsl(
fn r(i:i32,k:i32)->i32{return bitcast<i32>(recipesData[i*8+k]);}
fn reconstructionSample(s:vec3i,n:i32,side:i32)->f32 {
    let offsets=array<i32,5>(0,-1,1,-s.z,s.z);
    return sourceData[s.x+n*s.y+offsets[side]];
}
fn execute(){let i=cell().y*i32(params.extent.x)+cell().x;if(i>=params.grid.x){return;}
    let destination=r(i,0);let source=r(i,1);let tp=r(i,2);let sp=r(i,3);let stride=r(i,4);let count=params.grid.z+2;
    let s=vec3i(source,sp,stride);let limited=materialLimiter(s,params.grid.z);
    let momentum=momentumSlopes(s,params.grid.z,limited);let delta=vec2f(recipesData[i*8+5],recipesData[i*8+6]);
    for(var n=0;n<count;n++) {
        var slope:vec2f;
        if(n<params.grid.z){slope=materialSlope(s,n,limited);}else{slope=momentum[n-params.grid.z];}
        output0Data[destination+n*tp]=sourceData[source+n*sp]+dot(delta,slope);
    }
})wgsl";

} // namespace fluid::amr_gpu
