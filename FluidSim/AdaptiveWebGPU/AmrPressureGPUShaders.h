#pragma once
#include <string_view>

namespace fluid::webgpu::amrpressure
{
// Only accumulated pressure uses two floats; bulk Krylov and multigrid work
// remains single precision. Error-free sums preserve corrections below its ULP.
inline constexpr std::string_view compensated = R"wgsl(
fn pairAdd(a:vec2f,b:vec2f)->vec2f{
    let s=a.x+b.x;let v=s-a.x;
    let t=((b.x-v)+(a.x-(s-v)))+a.y+b.y;
    let high=s+t;return vec2f(high,t-(high-s));
}
fn pairScale(a:vec2f,b:f32)->vec2f{
    let product=a.x*b;let error=fma(a.x,b,-product)+a.y*b;
    let high=product+error;return vec2f(high,error-(high-product));
}
fn pairValue(a:vec2f)->f32{return a.x+a.y;}
)wgsl";
inline constexpr std::string_view index = R"wgsl(
fn index() -> u32 { return u32(point.y) * params.fields[0].x + u32(point.x); }
)wgsl";
inline constexpr std::string_view metadata = R"wgsl(
fn index() -> u32 { return u32(point.y) * params.fields[0].x + u32(point.x); }
fn readWord(i:u32)->u32{return bitcast<u32>(geometryData[i]);}
fn flags(i:u32)->u32{return readWord(i);}
fn valid(i:u32)->bool{return (flags(i)&1u)!=0u;}
fn cellStride(i:u32)->u32{return readWord(u32(params.grid.x)+i);}
fn ghostAddress(entry:u32)->u32{return bitcast<u32>(params.vectors[11].x)+14u*entry;}
fn patchCell(color:bool)->vec3u{
    let thread=u32(point.y)*params.extent.x+u32(point.x);let job=thread/128u;
    let work=params.vectors[10];let count=u32(select(work.y,work.w,color));
    if(job>=count){return vec3u(0);}
    let entry=select(bitcast<u32>(work.x),bitcast<u32>(work.z),color)+2u*job;
    let box=readWord(entry);let cell=readWord(entry+1u)+thread%128u;
    let width=readWord(box+2u);let height=readWord(box+3u);let stride=readWord(box+1u);
    let rowWidth=select(width,(width+1u)/2u,color);let y=cell/rowWidth;var x=cell%rowWidth;
    if(color){x=2u*x+((readWord(box+4u)+y+u32(mode()))&1u);}
    if(x>=width||y>=height){return vec3u(0);}
    return vec3u(readWord(box)+y*stride+x,stride,1u);
}
)wgsl";
inline constexpr std::string_view vector = R"wgsl(
fn execute() {
    let i=index(); if(i>=u32(params.grid.x)){return;}
    let operation=mode();
    if(operation==0){storePair(i,vec2f(0));return;}
    if(operation==1){storePair(i,sourcePair(i));return;}
    if(!valid(i)){return;}
    if(params.level.w>0. && scalarData[3]==0.){return;}
    var factor=params.value.x;
    if(params.grid.w>=0){factor=scalarData[u32(params.grid.w)];}
    if(params.fields[0].z==2u){
        if(operation==2){storePair(i,pairAdd(output0Pair(i),pairScale(sourcePair(i),factor)));}
        if(operation==3){storePair(i,pairScale(output0Pair(i),factor));}
        if(operation==4){storePair(i,pairAdd(output0Pair(i),vec2f(-factor,0.)));}
        return;
    }
    if(operation==2){output0Data[i]+=factor*sourceData[i];}
    if(operation==3){output0Data[i]*=factor;}
    if(operation==4){output0Data[i]-=factor;}
}
)wgsl";
inline constexpr std::string_view boundary = R"wgsl(
fn execute() {
    let entry=u32(point.y)*params.extent.x+u32(point.x);if(entry>=u32(params.vectors[11].y)){return;}
    let a=ghostAddress(entry);let i=readWord(a);let source=bitcast<i32>(geometryData[a+1u]);
    let first=readWord(a+2u);let second=readWord(a+3u);let count=readWord(a+4u);
    let firstWeight=geometryData[a+5u];let secondWeight=geometryData[a+6u];let boundaryWeight=geometryData[a+7u];
    if(params.fields[0].z==2u){
        if(source>=0){storePair(i,output0Pair(u32(source)));return;}
        // Relative differences preserve the constant nullspace with rounded interpolation weights.
        let anchor=output0Pair(first);var value=anchor;
        if(secondWeight!=0.){value=pairAdd(value,pairScale(pairAdd(output0Pair(second),-anchor),secondWeight));}
        if(boundaryWeight!=0.){
            var b=vec2f(0);
            if(params.grid.z!=0){
                let coarseAnchor=coarsePair(readWord(a+8u));b=coarseAnchor;
                for(var k=1u;k<count;k++){b=pairAdd(b,pairScale(pairAdd(coarsePair(readWord(a+8u+k)),-coarseAnchor),geometryData[a+11u+k]));}
            }
            value=pairAdd(value,pairScale(pairAdd(b,-anchor),boundaryWeight));
        }
        storePair(i,value);return;
    }
    if(source>=0){output0Data[i]=output0Data[u32(source)];return;}
    var value=firstWeight*output0Data[first];
    if(secondWeight!=0.){value+=secondWeight*output0Data[second];}
    if(params.grid.z!=0){
        var b=0.;for(var k=0u;k<count;k++){b+=geometryData[a+11u+k]*coarseData[readWord(a+8u+k)];}
        value+=boundaryWeight*b;
    }
    output0Data[i]=value;
}
)wgsl";
inline constexpr std::string_view diagonal = R"wgsl(
fn execute() {
    let i=index(); if(i>=u32(params.grid.x)||!valid(i)){return;}
    let s=cellStride(i); let f=params.value.x;
    output0Data[2u*i+1u]=f*(bxData[i]+bxData[i+1u]+byData[i]+byData[i+s]);
    var delta=0.;let start=readWord(2u*u32(params.grid.x)+i);var count=0u;if(start!=0u){count=readWord(start);}
    for(var j=0u;j<count;j++){
        let a=start+1u+3u*j;let axis=bitcast<u32>(geometryData[a]);let face=bitcast<u32>(geometryData[a+1u]);
        var beta=bxData[face];if(axis!=0u){beta=byData[face];}
        delta+=f*beta*geometryData[a+2u];
    }
    output0Data[2u*i]=delta;
}
)wgsl";
inline constexpr std::string_view apply = R"wgsl(
fn operatorValue(i:u32,s:u32)->f32 {
    let p=pressureData[i];let f=params.value.x;
    if(params.fields[1].z==2u){
        let center=pressurePair(i);
        let x=pairAdd(pairScale(pairAdd(pressurePair(i+1u),-center),bxData[i+1u]),-pairScale(pairAdd(center,-pressurePair(i-1u)),bxData[i]));
        let y=pairAdd(pairScale(pairAdd(pressurePair(i+s),-center),byData[i+s]),-pairScale(pairAdd(center,-pressurePair(i-s)),byData[i]));
        return pairValue(pairScale(pairAdd(x,y),-f));
    }
    return -f*(bxData[i+1u]*(pressureData[i+1u]-p)-bxData[i]*(p-pressureData[i-1u]))
           -f*(byData[i+s]*(pressureData[i+s]-p)-byData[i]*(p-pressureData[i-s]));
}
fn execute() {
    let cell=patchCell(false);if(cell.z==0u){return;}
    let i=cell.x;let value=operatorValue(i,cell.y);
    if(mode()==0){output0Data[i]=value;}
    else if(mode()==1){output0Data[i]=-value+rhsData[i];}
    else{output0Data[i]+=-value;}
}
)wgsl";
inline constexpr std::string_view smooth = R"wgsl(
fn execute() {
    let cell=patchCell(true);if(cell.z==0u){return;}
    let i=cell.x;let s=cell.y;let f=params.value.x;
    let neighbors=f*(bxData[i]*output0Data[i-1u]+bxData[i+1u]*output0Data[i+1u])
                 +f*(byData[i]*output0Data[i-s]+byData[i+s]*output0Data[i+s]);
    output0Data[i]=(rhsData[i]+neighbors-output0Data[i]*diagonalData[2u*i])/(diagonalData[2u*i+1u]-diagonalData[2u*i]);
}
)wgsl";
inline constexpr std::string_view transfer = R"wgsl(
fn execute() {
    let entry=u32(point.y)*params.extent.x+u32(point.x);if(entry>=u32(params.grid.w)){return;}
    let r=u32(params.grid.y)+entry*select(3u,5u,mode()==0);
    let i=bitcast<u32>(recipeData[r]);let a=bitcast<u32>(recipeData[r+1u]);let b=bitcast<u32>(recipeData[r+2u]);
    var value=.5*(fineData[a]+fineData[b]);
    if(mode()==0){value=.25*(fineData[a]+fineData[b]+fineData[bitcast<u32>(recipeData[r+3u])]+fineData[bitcast<u32>(recipeData[r+4u])]);}
    output0Data[i]=value;
}
)wgsl";
inline constexpr std::string_view interpolate = R"wgsl(
fn execute() {
    let cell=patchCell(false);if(cell.z==0u){return;}let i=cell.x;
    let a=readWord(3u*u32(params.grid.x)+i);
    if(mode()==0){output0Data[i]+=coarseData[readWord(a)];return;}
    var value=0.;let weights=array<f32,4>(.5625,.1875,.1875,.0625);
    for(var k=0u;k<4u;k++){let c=bitcast<i32>(geometryData[a+k]);if(c>=0){value+=weights[k]*coarseData[u32(c)];}}
    output0Data[i]=value;output1Data[i]+=value;
    if(params.grid.w!=0){output2Data[i]+=value;}
}
)wgsl";
inline constexpr std::string_view flux = R"wgsl(
fn execute() {
    let i=index();if(i>=u32(params.grid.x)){return;}
    let faces=flags(i)>>3u;let f=params.value.x;
    if(params.fields[2].z==2u){
        if((faces&1u)!=0u){output0Data[i]=pairValue(pairScale(pairAdd(pressurePair(i),-pressurePair(i-1u)),-f*bxData[i]));}
        if((faces&2u)!=0u){let s=cellStride(i);output1Data[i]=pairValue(pairScale(pairAdd(pressurePair(i),-pressurePair(i-s)),-f*byData[i]));}
        return;
    }
    if((faces&1u)!=0u){output0Data[i]=-f*bxData[i]*(pressureData[i]-pressureData[i-1u]);}
    if((faces&2u)!=0u){let s=cellStride(i);output1Data[i]=-f*byData[i]*(pressureData[i]-pressureData[i-s]);}
}
)wgsl";
inline constexpr std::string_view reflux = R"wgsl(
fn execute() {
    let entry=u32(point.y)*params.extent.x+u32(point.x);if(entry>=u32(params.grid.w)){return;}
    let r=u32(params.grid.y)+3u*entry;let i=bitcast<u32>(recipeData[r]);
    let first=bitcast<u32>(recipeData[r+1u]);let count=bitcast<u32>(recipeData[r+2u]);
    var value=output0Data[i];
    for(var k=0u;k<count;k++){
        let a=first+8u*k;
        let cs=bitcast<i32>(recipeData[a+1u]);let cf=bitcast<u32>(recipeData[a+2u]);
        let fi=bitcast<u32>(recipeData[a+3u]);let fs=bitcast<i32>(recipeData[a+4u]);let ff=bitcast<u32>(recipeData[a+5u]);
        let axis=bitcast<u32>(recipeData[a+6u]);let sign=f32(bitcast<i32>(recipeData[a+7u]));
        var fb=fbxData[ff];var cb=cbxData[cf];if(axis!=0u){fb=fbyData[ff];cb=cbyData[cf];}
        var fc=-sign*fb*(fineData[fi]-fineData[u32(i32(fi)-fs)])/params.value.y;
        var cc=-sign*cb*(coarseData[u32(i32(i)+cs)]-coarseData[i])/params.value.x;
        if(params.fields[1].z==2u){
            fc=pairValue(pairScale(pairAdd(finePair(fi),-finePair(u32(i32(fi)-fs))),-sign*fb/params.value.y));
            cc=pairValue(pairScale(pairAdd(coarsePair(u32(i32(i)+cs)),-coarsePair(i)),-sign*cb/params.value.x));
        }
        value+=params.value.z*.5*sign*(fc-cc)/params.value.x;
    }
    output0Data[i]=value;
}
)wgsl";
inline constexpr std::string_view reductionLayout = R"wgsl(
const reductionWidth=128u;
const reductionBlock=4u*reductionWidth;
const reductionGroups=1024u;
var<workgroup> reductionValues: array<vec2f,128>;
fn reductionLane()->u32{return u32(point.x)%reductionWidth;}
fn reductionGroup()->u32{return u32(point.y)*(params.extent.x/reductionWidth)+u32(point.x)/reductionWidth;}
fn combineReduction(a:vec2f,b:vec2f,maximum:bool)->vec2f{
    var value=a.x+b.x;if(maximum){value=max(a.x,b.x);}
    return vec2f(value,max(a.y,b.y));
}
)wgsl";
inline constexpr std::string_view reductionShared = R"wgsl(
fn reduceWorkgroup(value:vec2f,maximum:bool)->vec2f{
    let lane=reductionLane();reductionValues[lane]=value;
    workgroupBarrier();
    for(var stride=reductionWidth/2u;stride>0u;stride/=2u){
        if(lane<stride){reductionValues[lane]=combineReduction(reductionValues[lane],reductionValues[lane+stride],maximum);}
        workgroupBarrier();
    }
    return reductionValues[0];
}
)wgsl";
inline constexpr std::string_view reductionSubgroups = R"wgsl(
fn reduceWorkgroup(value:vec2f,maximum:bool)->vec2f{
    let lane=reductionLane();var total=subgroupAdd(value.x);
    if(maximum){total=subgroupMax(value.x);}
    let failed=subgroupMax(value.y);
    reductionValues[lane]=select(vec2f(0),vec2f(total,failed),subgroupElect());
    workgroupBarrier();
    // Subgroups need not follow local-index order; fold the elected lanes explicitly.
    let lanes=subgroupAdd(1u);var partial=vec2f(0);
    for(var i=subgroupLane;i<reductionWidth;i+=lanes){partial=combineReduction(partial,reductionValues[i],maximum);}
    total=subgroupAdd(partial.x);if(maximum){total=subgroupMax(partial.x);}
    let result=vec2f(total,subgroupMax(partial.y));
    workgroupBarrier();return result;
}
)wgsl";
inline constexpr std::string_view reduction = R"wgsl(
fn execute() {
    let group=reductionGroup();let maximum=mode()==2||mode()==3;
    var value=vec2f(0);
    // Adjacent lanes read adjacent cells; every lane reaches the reduction barriers.
    let groups=min((u32(params.grid.x)+reductionBlock-1u)/reductionBlock,reductionGroups);
    for(var base=group*reductionBlock;base<u32(params.grid.x);base+=groups*reductionBlock){
        for(var offset=0u;offset<reductionBlock;offset+=reductionWidth){
            let i=base+offset+reductionLane();if(i>=u32(params.grid.x)){continue;}
            if(!valid(i)){continue;}
            if((mode()==0||mode()==2)&&(flags(i)&4u)==0u){continue;}
            var term=firstData[i];if(mode()==0||mode()==4){term*=secondData[i];}
            if(maximum){term=abs(term);}
            value=combineReduction(value,vec2f(term,select(1.,0.,finite(term))),maximum);
        }
    }
    value=reduceWorkgroup(value,maximum);
    if(reductionLane()==0u&&group*reductionBlock<u32(params.grid.x)){
        output0Data[2u*group]=value.x;output0Data[2u*group+1u]=value.y;
    }
}
)wgsl";
inline constexpr std::string_view reduceFinish = R"wgsl(
fn execute() {
    let maximum=params.grid.z==1;
    var value=vec2f(0);
    for(var i=reductionLane();i<u32(params.grid.x);i+=reductionWidth){
        value=combineReduction(value,vec2f(partialData[2u*i],partialData[2u*i+1u]),maximum);
    }
    value=reduceWorkgroup(value,maximum);
    if(reductionLane()!=0u){return;}
    let destination=u32(params.grid.y);
    if(params.grid.w==0){output0Data[destination]=value.x;}
    else if(maximum){output0Data[destination]=max(output0Data[destination],value.x);}
    else{output0Data[destination]+=value.x;}
    if(value.y!=0.||!finite(value.x)){output0Data[6]=1.;}
}
)wgsl";
inline constexpr std::string_view productsLayout = R"wgsl(
var<workgroup> productSums: array<vec4f,128>;
var<workgroup> productFailures: array<f32,128>;
fn storeProducts(group:u32,value:vec4f){
    for(var j=0u;j<4u;j++){output0Data[5u*group+j]=value[j];}
    output0Data[5u*group+4u]=productFailures[0];
}
)wgsl";
inline constexpr std::string_view productsShared = R"wgsl(
fn reduceProducts(value:vec4f,failed:f32)->vec4f{
    let lane=reductionLane();productSums[lane]=value;productFailures[lane]=failed;
    workgroupBarrier();
    for(var stride=reductionWidth/2u;stride>0u;stride/=2u){
        if(lane<stride){productSums[lane]+=productSums[lane+stride];productFailures[lane]=max(productFailures[lane],productFailures[lane+stride]);}
        workgroupBarrier();
    }
    return productSums[0];
}
)wgsl";
inline constexpr std::string_view productsSubgroups = R"wgsl(
fn reduceProducts(value:vec4f,failed:f32)->vec4f{
    let lane=reductionLane();let total=subgroupAdd(value);let failure=subgroupMax(failed);let elected=subgroupElect();
    productSums[lane]=select(vec4f(0),total,elected);productFailures[lane]=select(0.,failure,elected);
    workgroupBarrier();
    let lanes=subgroupAdd(1u);var partial=vec4f(0);var bad=0.;
    for(var i=subgroupLane;i<reductionWidth;i+=lanes){partial+=productSums[i];bad=max(bad,productFailures[i]);}
    let result=subgroupAdd(partial);let resultFailure=subgroupMax(bad);
    workgroupBarrier();if(lane==0u){productFailures[0]=resultFailure;}return result;
}
)wgsl";
inline constexpr std::string_view products = R"wgsl(
fn execute(){
    let group=reductionGroup();var value=vec4f(0);var failed=0.;
    let groups=min((u32(params.grid.x)+reductionBlock-1u)/reductionBlock,reductionGroups);
    for(var base=group*reductionBlock;base<u32(params.grid.x);base+=groups*reductionBlock){
        for(var offset=0u;offset<reductionBlock;offset+=reductionWidth){
            let i=base+offset+reductionLane();
            if(i>=u32(params.grid.x)||!valid(i)||(flags(i)&4u)==0u){continue;}
            let source=sourceData[i];
            for(var j=0u;j<u32(params.grid.y);j++){
                let term=source*basisData[j*u32(params.grid.x)+i];value[j]+=term;
                if(!finite(term)){failed=1.;}
            }
        }
    }
    value=reduceProducts(value,failed);
    if(reductionLane()==0u&&group*reductionBlock<u32(params.grid.x)){storeProducts(group,value);}
}
)wgsl";
inline constexpr std::string_view productsFinish = R"wgsl(
fn execute(){
    var value=vec4f(0);var failed=0.;
    for(var i=reductionLane();i<u32(params.grid.x);i+=reductionWidth){
        for(var j=0u;j<4u;j++){value[j]+=partialData[5u*i+j];}
        failed=max(failed,partialData[5u*i+4u]);
    }
    value=reduceProducts(value,failed);
    if(reductionLane()!=0u){return;}
    for(var j=0u;j<u32(params.grid.z);j++){
        let destination=1100u+u32(params.grid.y)+j;
        if(params.grid.w==0){output0Data[destination]=value[j];}else{output0Data[destination]+=value[j];}
        if(!finite(value[j])){output0Data[6]=1.;}
    }
    if(productFailures[0]!=0.){output0Data[6]=1.;}
}
)wgsl";
inline constexpr std::string_view normalizeBasis = R"wgsl(
fn execute(){
    let cell=patchCell(false);if(cell.z==0u){return;}let i=cell.x;
    let value=output0Data[i]*scalarData[2];
    output0Data[i]=value;output1Data[u32(params.grid.y)*u32(params.grid.x)+i]=value;
}
)wgsl";
inline constexpr std::string_view basisUpdate = R"wgsl(
fn execute(){
    let cell=patchCell(false);if(cell.z==0u){return;}let i=cell.x;
    var value=output0Data[i];if(mode()!=0&&params.grid.w==0){value=0.;}
    for(var j=0u;j<u32(params.grid.y);j++){
        let basis=u32(params.grid.w)+j;var factor=-scalarData[1100u+basis];
        if(mode()!=0){if(basis>=u32(scalarData[4])){break;}factor=scalarData[1000u+basis];}
        value+=factor*basisData[j*u32(params.grid.x)+i];
    }
    output0Data[i]=value;
}
)wgsl";
// Hessenberg, Givens rotations, stopping flags and coefficients never leave the device.
inline constexpr std::string_view scalar = R"wgsl(
fn execute() {
    if(index()!=0u){return;}
    let k=u32(params.grid.x);let operation=mode();
    if(operation==0){output0Data[k]=params.value.x;return;}
    if(operation==1){output0Data[k]=sqrt(output0Data[k]);return;}
    if(operation==2){output0Data[k]/=params.value.x;return;}
    if(operation==9){for(var i=0u;i<1152u;i++){output0Data[i]=0.;}return;}
    if(operation==3){
        for(var i=64u;i<1152u;i++){output0Data[i]=0.;}
        output0Data[4]=0.;output0Data[7]=output0Data[0];output0Data[1000]=output0Data[0];
        output0Data[3]=select(0.,1.,output0Data[0]>=params.value.z&&output0Data[6]==0.);
        output0Data[2]=0.;if(output0Data[0]>0.){output0Data[2]=1./output0Data[0];}return;
    }
    if(operation==4){
        if(output0Data[3]==0.){return;}
        for(var j=0u;j<=k;j++){output0Data[64u+30u*j+k]+=output0Data[1100u+j];}return;
    }
    if(operation==5){output0Data[2]=0.;if(output0Data[0]>0.){output0Data[2]=1./output0Data[0];}return;}
    if(operation==6){
        if(output0Data[3]==0.){return;}
        let length=output0Data[0];output0Data[64u+30u*(k+1u)+k]=length;
        for(var j=0u;j<k;j++){
            let a=64u+30u*j+k;let b=a+30u;let old=output0Data[a];
            output0Data[a]=output0Data[1032u+j]*old+output0Data[1064u+j]*output0Data[b];
            output0Data[b]=output0Data[1032u+j]*output0Data[b]-output0Data[1064u+j]*old;
        }
        let diagonal=64u+31u*k;
        if(length>0.){
            let x=output0Data[diagonal];let y=output0Data[diagonal+30u];let rotation=sqrt(x*x+y*y);
            if(rotation==0.||!finite(rotation)){output0Data[6]=1.;output0Data[3]=0.;return;}
            let c=x/rotation;let s=y/rotation;output0Data[1032u+k]=c;output0Data[1064u+k]=s;
            output0Data[1001u+k]=-s*output0Data[1000u+k];output0Data[1000u+k]*=c;
            output0Data[diagonal]=c*x+s*y;output0Data[7]=abs(output0Data[1001u+k]);
        }else{output0Data[7]=0.;}
        output0Data[4]+=1.;output0Data[5]+=1.;
        if(!finite(output0Data[7])){output0Data[6]=1.;}
        output0Data[3]=select(0.,1.,length>0.&&output0Data[7]>=params.value.z&&output0Data[6]==0.);return;
    }
    if(operation==7){
        let count=u32(output0Data[4]);if(k>=count){return;}
        var value=output0Data[1000u+k];
        for(var j=k+1u;j<count;j++){value-=output0Data[64u+30u*k+j]*output0Data[1000u+j];}
        let diagonal=output0Data[64u+31u*k];
        if(diagonal==0.||!finite(value)){output0Data[6]=1.;return;}
        output0Data[1000u+k]=value/diagonal;return;
    }
    if(operation==10){
        output0Data[7]=output0Data[0];
        output0Data[3]=select(0.,1.,output0Data[0]>params.value.z&&output0Data[5]<params.value.w&&output0Data[6]==0.);return;
    }
    if(operation==11&&output0Data[3]!=0.){output0Data[5]+=1.;}
}
)wgsl";

// The ordinarily 2x2 bottom solve is serial on the GPU, retaining the normalized
// BiCGStab arithmetic and useful-progress/eight-sweep fallback without host waits.
inline constexpr std::string_view bottom = R"wgsl(
fn workspace(c:u32,i:u32)->u32{return c*u32(params.grid.x)+i;}
fn fillWork(c:u32){
    for(var entry=0u;entry<u32(params.vectors[11].y);entry++){
        let a=ghostAddress(entry);let i=readWord(a);let source=bitcast<i32>(geometryData[a+1u]);
        var value=0.;
        if(source>=0){value=output1Data[workspace(c,u32(source))];}
        else{
            value=geometryData[a+5u]*output1Data[workspace(c,readWord(a+2u))];
            if(geometryData[a+6u]!=0.){value+=geometryData[a+6u]*output1Data[workspace(c,readWord(a+3u))];}
        }
        output1Data[workspace(c,i)]=value;
    }
}
fn applyWork(destination:u32,source:u32){
    fillWork(source);
    for(var i=0u;i<u32(params.grid.x);i++){
        if(!valid(i)){continue;}let s=cellStride(i);let f=params.value.x;let p=output1Data[workspace(source,i)];
        let value=-f*(bxData[i+1u]*(output1Data[workspace(source,i+1u)]-p)-bxData[i]*(p-output1Data[workspace(source,i-1u)]))
                   -f*(byData[i+s]*(output1Data[workspace(source,i+s)]-p)-byData[i]*(p-output1Data[workspace(source,i-s)]));
        output1Data[workspace(destination,i)]=value/diagonalData[2u*i+1u];
    }
}
fn product(a:u32,b:u32)->f32{
    var value=0.;for(var i=0u;i<u32(params.grid.x);i++){if(valid(i)){value+=output1Data[workspace(a,i)]*output1Data[workspace(b,i)];}}return value;
}
fn maximum(c:u32)->f32{
    var value=0.;for(var i=0u;i<u32(params.grid.x);i++){if(valid(i)){value=max(value,abs(output1Data[workspace(c,i)]));}}return value;
}
fn execute(){
    if(index()!=0u){return;}
    let n=u32(params.grid.x);var sum=0.;
    for(var i=0u;i<n;i++){
        output0Data[i]=0.;for(var c=0u;c<5u;c++){output1Data[workspace(c,i)]=0.;}
        if(valid(i)){sum+=rhsData[i];}
    }
    let mean=sum/f32(params.grid.y);
    for(var i=0u;i<n;i++){if(valid(i)){let r=(rhsData[i]-mean)/diagonalData[2u*i+1u];output1Data[workspace(0u,i)]=r;output1Data[workspace(1u,i)]=r;}}
    let initial=maximum(0u);let threshold=1.e-4*initial;if(initial==0.){return;}
    var previous=0.;var alpha=0.;var omega=0.;var magnitude=initial;var breakdown=false;var converged=false;
    for(var iteration=0u;iteration<200u;iteration++){
        let rho=product(1u,0u);if(rho==0.||!finite(rho)){breakdown=true;break;}
        var beta=0.;if(iteration!=0u){if(previous==0.||omega==0.){breakdown=true;break;}beta=(rho/previous)*(alpha/omega);}
        if(!finite(beta)){breakdown=true;break;}
        for(var i=0u;i<n;i++){if(valid(i)){
            let r=output1Data[workspace(0u,i)];var p=r;
            if(iteration!=0u){p+=beta*(output1Data[workspace(2u,i)]-omega*output1Data[workspace(3u,i)]);}
            output1Data[workspace(2u,i)]=p;
        }}
        applyWork(3u,2u);let denominator=product(1u,3u);if(denominator==0.||!finite(denominator)){breakdown=true;break;}alpha=rho/denominator;
        if(!finite(alpha)){breakdown=true;break;}
        for(var i=0u;i<n;i++){if(valid(i)){output0Data[i]+=alpha*output1Data[workspace(2u,i)];output1Data[workspace(0u,i)]-=alpha*output1Data[workspace(3u,i)];}}
        magnitude=maximum(0u);if(magnitude<threshold){converged=true;break;}
        applyWork(4u,0u);let tt=product(4u,4u);if(tt==0.||!finite(tt)){breakdown=true;break;}omega=product(4u,0u)/tt;
        if(!finite(omega)){breakdown=true;break;}
        for(var i=0u;i<n;i++){if(valid(i)){output0Data[i]+=omega*output1Data[workspace(0u,i)];output1Data[workspace(0u,i)]-=omega*output1Data[workspace(4u,i)];}}
        magnitude=maximum(0u);if(magnitude<threshold){converged=true;break;}
        if(omega==0.||!finite(magnitude)){breakdown=true;break;}previous=rho;
    }
    if(converged){return;}
    if(breakdown||!(magnitude<initial)){for(var i=0u;i<n;i++){output0Data[i]=0.;}}
    for(var sweep=0u;sweep<8u;sweep++){for(var color=0u;color<2u;color++){
        for(var entry=0u;entry<u32(params.vectors[11].y);entry++){
            let a=ghostAddress(entry);let i=readWord(a);let source=bitcast<i32>(geometryData[a+1u]);
            var value=0.;if(source>=0){value=output0Data[u32(source)];}
            else{value=geometryData[a+5u]*output0Data[readWord(a+2u)];if(geometryData[a+6u]!=0.){value+=geometryData[a+6u]*output0Data[readWord(a+3u)];}}
            output0Data[i]=value;
        }
        for(var i=0u;i<n;i++){
            if(!valid(i)||((flags(i)>>1u)&1u)!=color){continue;}let s=cellStride(i);let f=params.value.x;
            let neighbors=f*(bxData[i]*output0Data[i-1u]+bxData[i+1u]*output0Data[i+1u])+f*(byData[i]*output0Data[i-s]+byData[i+s]*output0Data[i+s]);
            output0Data[i]=(rhsData[i]+neighbors-output0Data[i]*diagonalData[2u*i])/(diagonalData[2u*i+1u]-diagonalData[2u*i]);
        }
    }}
}
)wgsl";

} // namespace fluid::webgpu::amrpressure

