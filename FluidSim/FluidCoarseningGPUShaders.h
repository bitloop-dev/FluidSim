#pragma once
#include <string_view>

namespace fluid::webgpu::coarsening {
inline constexpr std::string_view nativeView=R"wgsl(
const Coarse=0u; const Unassessed=1u; const LiquidBuffer=2u; const Detail=3u; const Candidate=4u; const Refine=5u; const Approaching=6u;
const sides=array<vec2i,5>(vec2i(0),vec2i(-1,0),vec2i(1,0),vec2i(0,-1),vec2i(0,1));
fn word(i:u32)->u32 { return bitcast<u32>(metadataData[i]); }
fn record(l:u32,n:u32)->u32 { return word(params.padding[0].w+8u*l+n); }
fn patchWord(p:u32,n:u32)->u32 { return word(params.padding[0].y+20u*p+n); }
fn patchInt(p:u32,n:u32)->i32 { return bitcast<i32>(patchWord(p,n)); }
fn ingredients()->u32 { return u32(params.grid.z); }
fn lanes()->u32 { return ingredients()+2u; }
fn intrinsic(k:u32)->f32 { return metadataData[params.padding[0].z+k]; }
struct Cell { owner:u32, native:vec2i, scale:i32, valid:bool }
fn at(p:vec2i)->Cell {
    if(any(p<vec2i(0)) || any(p>=gridSize())) { return Cell(0u,vec2i(0),1,false); }
    let owner=word(params.padding[0].x+u32(p.y/2)*params.padding[3].y+u32(p.x/2));
    let scale=patchInt(owner,0u);
    return Cell(owner,p/scale,scale,true);
}
fn center(c:Cell)->vec2f { return (vec2f(c.native)+vec2f(.5))*f32(c.scale); }
fn q(c:Cell,n:u32)->f32 {
    let p=c.owner;
    let i=(c.native.y-patchInt(p,3u))*patchInt(p,4u)+c.native.x-patchInt(p,2u);
    return stateData[patchWord(p,1u)+u32(i)+n*patchWord(p,5u)];
}
fn density(c:Cell)->f32 { var rho=0.; for(var k=0u;k<ingredients();k++) { rho+=q(c,k); } return rho; }
fn liquid(c:Cell)->bool { if(!c.valid) { return false; } for(var k=0u;k<u32(params.grid.w);k++) { if(q(c,k)!=0.) { return true; } } return false; }
fn face(c:Cell,d:u32,side:u32)->f32 {
    let p=c.owner; let base=10u+4u*d;
    var xy=c.native; xy[d]+=i32(side);
    let i=(xy.y-patchInt(p,base+2u))*patchInt(p,base+3u)+xy.x-patchInt(p,base+1u);
    return stateData[patchWord(p,base)+u32(i)];
}
fn pressureParts(c:Cell)->vec2f {
    let p=c.owner; let i=(c.native.y-patchInt(p,8u))*patchInt(p,9u)+c.native.x-patchInt(p,7u);
    let index=patchWord(p,6u)+u32(i);var value=vec2f(stateData[index],0.);let low=patchWord(p,18u);
    if(low!=0u) { value.y=stateData[index+low]; } return value;
}
fn pressureDelta(a:Cell,b:Cell)->f32 {
    if(params.value.y==0.) { return 0.; }
    // Cancel the common high pressure before adding the retained low bits.
    let difference=pressureParts(a)-pressureParts(b);return (difference.x+difference.y)*params.value.y;
}
fn minimumScale(origin:vec2i,scale:i32)->i32 {
    var stride=scale;
    for(var y=0;y<scale;y+=2) { for(var x=0;x<scale;x+=2) { stride=min(stride,at(origin+vec2i(x,y)).scale); } }
    return stride;
}
fn groupIndex(l:u32,p:vec2i)->u32 { return record(l,3u)+u32(p.y)*record(l,1u)+u32(p.x); }
fn planeIndex(g:u32,n:u32,k:u32)->u32 { return (g*lanes()+n)*7u+k; }
fn levelFor(scale:i32)->u32 { return firstLeadingBit(u32(scale))-1u; }
fn componentValue(c:Cell,k:i32)->f32 { if(k<0) { return density(c); } return q(c,u32(k))/intrinsic(u32(k)); }
fn errorScore(e:vec4f)->f32 { let v=e/params.level; return max(max(v.x,v.y),max(v.z,v.w)); }
)wgsl";

inline constexpr std::string_view planeBuilder=R"wgsl(
fn readPlane(g:u32,n:u32,k:u32)->f32 { return output0Data[planeIndex(g,n,k)]; }
fn parentValue(c:Cell,position:vec2f,n:u32)->f32 {
    let lev=levelFor(c.scale); let g=groupIndex(lev,c.native);
    let delta=position/f32(c.scale)-vec2f(c.native)-vec2f(.5);
    return readPlane(g,n,0u)+delta.x*readPlane(g,n,5u)+delta.y*readPlane(g,n,6u);
}
fn reconstructionSample(s:vec3i,n:i32,side:i32)->f32 { return readPlane(u32(s.x),u32(n),u32(side)); }
fn execute() {
    let lev=params.padding[1].x; let scale=i32(params.padding[1].y); let origin=cell()*scale;
    if(any(origin+vec2i(scale)>gridSize())) { return; }
    let g=groupIndex(lev,cell());
    for(var side=0u;side<5u;side++) {
        let requested=origin+scale*sides[side]; let start=clamp(requested,vec2i(0),gridSize()-vec2i(scale)); let source=at(start);
        if(source.scale>scale) {
            for(var n=0u;n<lanes();n++) { output0Data[planeIndex(g,n,side)]=parentValue(source,vec2f(start)+vec2f(.5*f32(scale)),n); }
        } else {
            let stride=minimumScale(start,scale); let samples=f32((scale/stride)*(scale/stride));
            for(var n=0u;n<lanes();n++) {
                // Match native restriction, including deep groups of 1024 samples.
                var mean=0.; var correction=0.;
                for(var y=0;y<scale;y+=stride) { for(var x=0;x<scale;x+=stride) {
                    let value=q(at(start+vec2i(x,y)),n)-correction; let next=mean+value; correction=(next-mean)-value; mean=next;
                } }
                mean/=samples;
                if(n>=ingredients()) { let d=n-ingredients(); if(requested[d]<0 || requested[d]>=gridSize()[d]) { mean=-mean; } }
                output0Data[planeIndex(g,n,side)]=mean;
            }
        }
    }
    let stencil=vec3i(i32(g),0,0);let material=materialLimiter(stencil,i32(ingredients()));
    let momentum=momentumSlopes(stencil,i32(ingredients()),material);
    for(var n=0u;n<lanes();n++) {
        var slope:vec2f;
        if(n<ingredients()){slope=materialSlope(stencil,i32(n),material);}else{slope=momentum[n-ingredients()];}
        output0Data[planeIndex(g,n,5u)]=slope.x;output0Data[planeIndex(g,n,6u)]=slope.y;
    }
}
)wgsl";

// Separate material, velocity and pressure tests to keep native shader compilation bounded.
inline constexpr std::string_view materialAssessment=R"wgsl(
fn readPlane(g:u32,n:u32,k:u32)->f32 { return planesData[planeIndex(g,n,k)]; }
fn representation(origin:vec2i,scale:i32,g:u32)->vec4f {
    var error=vec4f(0); let stride=minimumScale(origin,scale);
    for(var k=-1;k<i32(ingredients());k++) {
        var mean=0.; var slope=vec2f(0);
        for(var n=0u;n<ingredients();n++) { if(k<0 || k==i32(n)) {
            let divisor=select(intrinsic(n),1.,k<0);
            mean+=readPlane(g,n,0u)/divisor; slope+=vec2f(readPlane(g,n,5u),readPlane(g,n,6u))/divisor;
        } }
        var detail=0.;
        if(stride<scale) { for(var y=0;y<scale;y+=stride) { for(var x=0;x<scale;x+=stride) {
            let c=at(origin+vec2i(x,y)); let delta=(center(c)-vec2f(origin))/f32(scale)-vec2f(.5);
            detail=max(detail,abs(componentValue(c,k)-mean-dot(delta,slope)));
        } } }
        for(var side=0u;side<4u;side++) {
            let start=origin+scale*sides[side+1u];
            if(any(start<vec2i(0)) || any(start+vec2i(scale)>gridSize())) { continue; }
            var neighbour=0.;
            for(var n=0u;n<ingredients();n++) { if(k<0 || k==i32(n)) { neighbour+=readPlane(g,n,side+1u)/select(intrinsic(n),1.,k<0); } }
            detail=max(detail,.25*abs(neighbour-mean-dot(vec2f(sides[side+1u]),slope)));
            var low=0.; var high=0.;
            for(var j=0;j<scale;j++) {
                let p=start+vec2i(select(select(j,0,side==1u),scale-1,side==0u),select(select(j,0,side==3u),scale-1,side==2u));
                let c=at(p); let delta=(center(c)-vec2f(origin))/f32(scale)-vec2f(.5); let residual=componentValue(c,k)-mean-dot(delta,slope);
                if(j==0) { low=residual; high=residual; } else { low=min(low,residual); high=max(high,residual); }
            }
            detail=max(detail,.5*(high-low));
        }
        if(k<0) { error.z=detail/mean; } else { error.w=max(error.w,detail); }
    }
    return error;
}
fn coarseLiquid(origin:vec2i,scale:i32)->bool {
    if(liquid(at(origin))) { return true; }
    for(var j=0;j<2;j++) {
        if(liquid(at(origin+vec2i(-1,j*(scale/2)))) || liquid(at(origin+vec2i(scale,j*(scale/2)))) ||
           liquid(at(origin+vec2i(j*(scale/2),-1))) || liquid(at(origin+vec2i(j*(scale/2),scale)))) { return true; }
    }
    for(var y=-1;y<=scale;y+=scale+1) { for(var x=-1;x<=scale;x+=scale+1) { if(liquid(at(origin+vec2i(x,y)))) { return true; } } }
    return false;
}
fn execute() {
    let lev=params.padding[1].x; let scale=i32(params.padding[1].y); let origin=cell()*scale; let g=groupIndex(lev,cell()); let base=24u*g;
    let actual=at(origin).scale; var label=Unassessed; var evaluated=false; var error=vec4f(0);
    if(actual>scale) { label=Coarse; }
    else if(actual==scale) {
        if(coarseLiquid(origin,scale)) { label=Refine; } else { evaluated=true; }
    } else if(all(origin+vec2i(scale)<=gridSize())) {
        var wet=false;
        for(var y=-1;y<=scale;y++) { for(var x=-1;x<=scale;x++) { wet=wet || liquid(at(origin+vec2i(x,y))); } }
        if(wet) { label=LiquidBuffer; } else { evaluated=true; }
    }
    if(evaluated) { error=representation(origin,scale,g); }
    for(var i=0u;i<4u;i++) { output0Data[base+i]=error[i]; }
    output0Data[base+4u]=f32(label); output0Data[base+6u]=f32(actual); output0Data[base+7u]=select(0.,1.,evaluated);
}
)wgsl";

inline constexpr std::string_view velocityAssessment=R"wgsl(
fn readPlane(g:u32,n:u32,k:u32)->f32 { return planesData[planeIndex(g,n,k)]; }
fn planeVelocity(momentum:vec3f,density:vec3f,delta:vec2f)->f32 {
    return (momentum.x+dot(delta,momentum.yz))/(density.x+dot(delta,density.yz));
}
fn velocityError(origin:vec2i,scale:i32,g:u32)->f32 {
    let stride=minimumScale(origin,scale); var error=0.;
    var rho=0.; var densitySlope=vec2f(0);
    for(var k=0u;k<ingredients();k++) { rho+=readPlane(g,k,0u); densitySlope+=vec2f(readPlane(g,k,5u),readPlane(g,k,6u)); }
    // Sum the density plane once instead of expanding its ingredient loop at every velocity sample.
    let densityPlane=vec3f(rho,densitySlope);
    for(var d=0u;d<2u;d++) {
        let momentumPlane=vec3f(readPlane(g,ingredients()+d,0u),readPlane(g,ingredients()+d,5u),readPlane(g,ingredients()+d,6u));
        let mean=momentumPlane.x/rho;
        let slope=(momentumPlane.yz-mean*densitySlope)/rho;
        var detail=0.;
        for(var y=0;y<scale;y+=stride) { for(var x=0;x<scale;x+=stride) {
            let c=at(origin+vec2i(x,y)); let delta=(center(c)-vec2f(origin))/f32(scale)-vec2f(.5);
            detail=max(detail,abs(q(c,ingredients()+d)/density(c)-planeVelocity(momentumPlane,densityPlane,delta)));
            for(var side=0u;side<2u;side++) {
                var faceDelta=delta; faceDelta[d]+=(f32(side)-.5)*f32(c.scale)/f32(scale);
                detail=max(detail,abs(face(c,d,side)-planeVelocity(momentumPlane,densityPlane,faceDelta)));
            }
        } }
        for(var side=0u;side<4u;side++) {
            let start=origin+scale*sides[side+1u];
            if(any(start<vec2i(0)) || any(start+vec2i(scale)>gridSize())) { continue; }
            var neighbourDensity=0.; for(var k=0u;k<ingredients();k++) { neighbourDensity+=readPlane(g,k,side+1u); }
            let neighbour=readPlane(g,ingredients()+d,side+1u)/neighbourDensity;
            detail=max(detail,.25*abs(neighbour-mean-dot(vec2f(sides[side+1u]),slope)));
            var low=0.; var high=0.;
            for(var j=0;j<scale;j++) {
                let p=start+vec2i(select(select(j,0,side==1u),scale-1,side==0u),select(select(j,0,side==3u),scale-1,side==2u));
                let c=at(p); let delta=(center(c)-vec2f(origin))/f32(scale)-vec2f(.5);
                let residual=q(c,ingredients()+d)/density(c)-mean-dot(delta,slope);
                if(j==0) { low=residual; high=residual; } else { low=min(low,residual); high=max(high,residual); }
                if(d==side/2u) {
                    var faceDelta=delta; faceDelta[d]+=select(-.5,.5,side%2u==0u)*f32(c.scale)/f32(scale);
                    let residualFace=face(c,d,select(0u,1u,side%2u==0u))-planeVelocity(momentumPlane,densityPlane,faceDelta);
                    low=min(low,residualFace); high=max(high,residualFace);
                }
            }
            detail=max(detail,.5*(high-low));
        }
        error=max(error,detail);
    }
    return error;
}
fn execute() {
    let lev=params.padding[1].x; let scale=i32(params.padding[1].y); let g=groupIndex(lev,cell()); let base=24u*g;
    if(output0Data[base+7u]!=0.) { output0Data[base]=velocityError(cell()*scale,scale,g); }
}
)wgsl";

inline constexpr std::string_view assessment=R"wgsl(
fn finePressure(origin:vec2i,scale:i32)->f32 {
    var halves=array<vec2f,2>(vec2f(0),vec2f(0));
    var corrections=array<vec2f,2>(vec2f(0),vec2f(0));let reference=at(origin);
    for(var y=0;y<scale;y++) { for(var x=0;x<scale;x++) {
        // A common gauge cancels from both half means. Sum only differences so
        // a large hydrostatic pressure cannot erase small resolved curvature.
        let p=pressureDelta(at(origin+vec2i(x,y)),reference);let side=vec2u(select(0u,1u,x>=scale/2),select(0u,1u,y>=scale/2));
        for(var d=0u;d<2u;d++) { let value=p-corrections[d][side[d]];let next=halves[d][side[d]]+value;
            corrections[d][side[d]]=(next-halves[d][side[d]])-value;halves[d][side[d]]=next; }
    } }
    let gradient=4.*vec2f(halves[0].y-halves[0].x,halves[1].y-halves[1].x)/(f32(scale*scale*scale)*cellSize());
    var error=0.;
    for(var d=0u;d<2u;d++) { for(var along=-1;along<scale;along++) { for(var across=0;across<scale;across++) {
        let p=origin+select(vec2i(along,across),vec2i(across,along),d!=0u); var step=vec2i(0); step[d]=1;
        let a=at(p); let b=at(p+step); if(!a.valid || !b.valid) { continue; }
        let delta=(center(b)-center(a))*cellSize(); if(delta[d]==0.) { continue; }
        error=max(error,abs((pressureDelta(b,a)-gradient[1u-d]*delta[1u-d])/delta[d]-gradient[d]));
    } } }
    return error;
}
fn pressureNeighbour(origin:vec2i,scale:i32,side:u32,j:u32)->Cell {
    let across=select(-1,scale,side%2u!=0u); let along=i32(j)*(scale/2);
    return at(origin+select(vec2i(across,along),vec2i(along,across),side>=2u));
}
fn coarsePressure(origin:vec2i,scale:i32)->f32 {
    let c=at(origin);
    var gradient=vec4f(0); var available=vec4u(0); var error=0.;
    for(var side=0u;side<4u;side++) {
        let a=pressureNeighbour(origin,scale,side,0u); let b=pressureNeighbour(origin,scale,side,1u); if(!a.valid || !b.valid) { continue; }
        let distance=(.5*(center(a)+center(b))-center(c))*cellSize();
        gradient[side]=.5*(pressureDelta(a,c)+pressureDelta(b,c))/distance[side/2u]; available[side]=1u;
    }
    var slope=vec2f(0);
    for(var d=0u;d<2u;d++) { let count=available[2u*d]+available[2u*d+1u];
        if(count>0u) { slope[d]=(gradient[2u*d]+gradient[2u*d+1u])/f32(count); }
        if(count==2u) { error=max(error,.5*abs(gradient[2u*d]-gradient[2u*d+1u])); }
    }
    for(var j=0u;j<8u;j++) { let n=pressureNeighbour(origin,scale,j/2u,j%2u); if(n.valid) {
        let delta=(center(n)-center(c))*cellSize();
        error=max(error,abs(pressureDelta(n,c)-dot(slope,delta))/(abs(delta.x)+abs(delta.y)));
    } }
    return error;
}
fn execute() {
    let lev=params.padding[1].x; let scale=i32(params.padding[1].y); let origin=cell()*scale; let g=groupIndex(lev,cell()); let base=24u*g;
    let actual=i32(output0Data[base+6u]); var label=u32(output0Data[base+4u]); let evaluated=output0Data[base+7u]!=0.;
    var error=vec4f(output0Data[base],output0Data[base+1u],output0Data[base+2u],output0Data[base+3u]); var quiet=false;
    if(evaluated) {
        if(actual==scale) { error.y=coarsePressure(origin,scale); } else { error.y=finePressure(origin,scale); }
        let score=errorScore(error); quiet=score<=params.vectors[0].x;
        if(actual==scale) { label=select(Coarse,Refine,score>1.); } else { label=select(Detail,Candidate,score<=1.); }
    }
    for(var i=0u;i<4u;i++) { output0Data[base+i]=error[i]; }
    output0Data[base+4u]=f32(label); output0Data[base+5u]=select(0.,1.,quiet); output0Data[base+6u]=f32(actual); output0Data[base+7u]=select(0.,1.,evaluated);
    var rectangle=vec4i(0);
    if(params.value.w>0. && (label==Detail || label==LiquidBuffer || label==Refine)) {
        var low=vec2f(0); var high=vec2f(0);
        for(var y=0;y<scale;y++) { for(var x=0;x<scale;x++) { let c=at(origin+vec2i(x,y)); if(!c.valid) { continue; }
            for(var d=0u;d<2u;d++) { for(var side=0u;side<2u;side++) {
                let v=face(c,d,side); let accelerated=v+params.vectors[0][2u+d]*params.value.w;
                low[d]=min(low[d],min(v,accelerated)); high[d]=max(high[d],max(v,accelerated));
            } }
        } }
        let travel=params.value.w/cellSize(); let shape=vec2f(params.padding[1].zw);
        rectangle=vec4i(vec2i(floor(clamp((vec2f(origin)+low*travel)/f32(scale),vec2f(0),shape))),
                       vec2i(ceil(clamp((vec2f(origin)+vec2f(f32(scale))+high*travel)/f32(scale),vec2f(0),shape))));
    }
    for(var i=0u;i<4u;i++) { output0Data[base+8u+i]=bitcast<f32>(rectangle[i]); }
    output0Data[base+12u]=f32(label);
    var changed=params.padding[2].x!=0u;
    for(var y=0;y<scale && origin.y+y<gridSize().y;y+=2) { for(var x=0;x<scale && origin.x+x<gridSize().x;x+=2) {
        let p=(origin+vec2i(x,y))/2; let i=u32(p.y)*params.padding[3].y+u32(p.x);
        changed=changed || previousData[i]!=f32(at(2*p).scale);
    } }
    output0Data[base+20u]=select(0.,1.,changed);
}
)wgsl";

inline constexpr std::string_view rowPrefix=R"wgsl(
fn execute() { let y=u32(cell().x); let stride=params.padding[1].z+1u; var sum=0;
    for(var x=0u;x<params.padding[1].z;x++) { sum+=bitcast<i32>(differencesData[y*stride+x]); output0Data[y*params.padding[1].z+x]=f32(sum); }
}
)wgsl";
inline constexpr std::string_view columnPrefix=R"wgsl(
fn execute() { let x=u32(cell().x); let width=params.padding[1].z; var sum=0.;
    for(var y=0u;y<params.padding[1].w;y++) { sum+=rowsData[y*width+x]; let g=groupIndex(params.padding[1].x,vec2i(i32(x),i32(y))); let base=24u*g;
        if(sum!=0. && output0Data[base+6u]>=f32(params.padding[1].y) && u32(output0Data[base+12u])!=Refine) { output0Data[base+12u]=f32(Approaching); }
    }
}
)wgsl";
inline constexpr std::string_view stabilise=R"wgsl(
fn refines(label:u32)->bool { return label==Refine || label==Approaching; }
fn action(label:u32)->i32 { return select(select(0,1,refines(label)),-1,label==Candidate); }
fn execute() {
    let g=groupIndex(params.padding[1].x,cell()); let b=24u*g; let h=3u*g; let reset=output0Data[b+20u]!=0.;
    let raw=u32(output0Data[b+12u]); let oldDecision=select(u32(output1Data[h+1u]),Unassessed,reset); let oldRaw=select(u32(output1Data[h+2u]),Unassessed,reset);
    var since=select(output1Data[h],-1.,reset);
    if(output0Data[b+5u]==0. || refines(raw)) { since=-1.; } else if(since<0.) { since=params.value.z; }
    let settled=since>=0. && params.value.z>=since+params.vectors[0].y;
    var decision=raw; var waiting=0.; var held=0.;
    if(raw==Candidate && oldDecision!=Candidate && !settled) { decision=Detail; waiting=1.; }
    else if((raw==Coarse || raw==Unassessed) && refines(oldDecision) && !settled) { decision=oldDecision; held=1.; }
    let coarse=output0Data[b+6u]>=f32(params.padding[1].y); let needed=select(decision!=Candidate,refines(decision),coarse);
    output0Data[b+13u]=f32(decision); output0Data[b+14u]=select(0.,1.,needed); output0Data[b+15u]=since;
    output0Data[b+16u]=select(0.,1.,params.padding[2].x==0u && action(raw)!=action(oldRaw));
    output0Data[b+17u]=select(0.,1.,params.padding[2].x==0u && action(decision)!=action(oldDecision)); output0Data[b+18u]=waiting; output0Data[b+19u]=held;
    output1Data[h]=since; output1Data[h+1u]=f32(decision); output1Data[h+2u]=f32(raw);
}
)wgsl";
inline constexpr std::string_view tileRequests=R"wgsl(
fn execute() {
    let scale=i32(params.padding[1].y); let tile=max(16,scale); let lo=cell()*tile; let hi=min(lo+vec2i(tile),gridSize());
    let first=max(vec2i(0),lo/scale-vec2i(1)); let last=min(vec2i(params.padding[1].zw),(hi+vec2i(scale-1))/scale+vec2i(1));
    var needed=false;
    for(var y=first.y;y<last.y;y++) { for(var x=first.x;x<last.x;x++) {
        let b=24u*groupIndex(params.padding[1].x,vec2i(x,y)); if(observationsData[b+14u]==0.) { continue; }
        let raw=u32(observationsData[b+12u]); let buffer=select(0,scale,raw==Detail || raw==LiquidBuffer || raw==Refine || raw==Approaching);
        let a=vec2i(x,y)*scale-vec2i(buffer); let z=(vec2i(x,y)+vec2i(1))*scale+vec2i(buffer);
        needed=needed || (all(a<hi) && all(z>lo));
    } }
    let tileWidth=(gridSize().x+tile-1)/tile;
    output0Data[record(params.padding[1].x,6u)+u32(cell().y*tileWidth+cell().x)]=select(0.,1.,needed);
}
)wgsl";
inline constexpr std::string_view displayAndScales=R"wgsl(
fn execute() {
    let p=cell(); let i=u32(p.y)*params.padding[3].y+u32(p.x); var label=u32(observationsData[24u*groupIndex(0u,p)+13u]);
    for(var lev=1u;lev<params.padding[2].y && (label==Coarse || label==Candidate);lev++) { label=u32(observationsData[24u*groupIndex(lev,p>>vec2u(lev))+13u]); }
    output0Data[i]=f32(label); output1Data[i]=f32(at(2*p).scale);
}
)wgsl";
inline constexpr std::string_view summary=R"wgsl(
fn execute() {
    let block=u32(cell().x); var totals:array<f32,11>;
    for(var i=block*256u;i<min((block+1u)*256u,params.padding[3].x);i++) { totals[u32(labelsData[i])]+=1.; }
    for(var i=block*256u;i<min((block+1u)*256u,params.padding[2].w);i++) { for(var j=0u;j<4u;j++) { totals[7u+j]+=observationsData[24u*i+16u+j]; } }
    for(var j=0u;j<11u;j++) { output0Data[11u*block+j]=totals[j]; }
}
)wgsl";
inline constexpr std::string_view reduceSummary=R"wgsl(
fn execute() { let metric=u32(cell().x); var total=0.; for(var b=0u;b<params.padding[3].w;b++) { total+=partialsData[11u*b+metric]; } output0Data[metric]=total; }
)wgsl";
} // namespace fluid::webgpu::coarsening
