#include "PressureMultigridGPU.h"
#include "PressureMultigridGPUShaders.h"
#include <cmath>

namespace fluid::webgpu {

PressureMultigridGPU::Level::Level(Compute& compute, int w, int h, int s, bool finest) :
    width(w), height(h), spacing(s), coefficients(finest ? Field{} : compute.field(w,h,4)),
    rhs(finest ? Field{} : compute.field(w,h)), solution(compute.field(w,h)), next_solution(compute.field(w,h)), residual(compute.field(w,h)) {}

PressureMultigridGPU::PressureMultigridGPU(Compute& compute, int width, int height) : compute(compute)
{
    // D^-1 A has eigenvalues in [0,2]; reverse the fixed Chebyshev weights after the coarse correction.
    for (int i=0; i<smoothing_sweeps; ++i)
        smoothing_weights[i]=float(1.0/(1.1+0.9*std::cos(3.14159265358979323846*(i+0.5)/smoothing_sweeps)));
    int w=width, h=height, spacing=1;
    for (;;)
    {
        levels.emplace_back(compute,w,h,spacing,levels.empty());
        if (w*h<=16) break;
        w=(w+1)/2; h=(h+1)/2; spacing*=2;
    }
    int tile_cells=0;
    for (int i=0; i+1<int(levels.size()); ++i)
    {
        int count=0;
        for (int l=i; l<int(levels.size()); ++l) count+=levels[l].width*levels[l].height;
        if (levels[i].width*levels[i].height<=1024 && levels.size()-i<=4 && count<=1360 &&
            (levels[i].width&(levels[i].width-1))==0 && (levels[i].height&(levels[i].height-1))==0)
        {
            tile_level=i; tile_cells=count;
            int offset=0;
            for (int l=i; l<int(levels.size()); ++l) { levels[l].tile_offset=offset; offset+=levels[l].width*levels[l].height; }
            break;
        }
    }
    std::vector<float> packed_maps;
    for (size_t i=0; i+1<levels.size(); ++i)
        buildTransfers(levels[i],levels[i+1],tile_level>=0 && int(i)>=tile_level ? &packed_maps : nullptr);
    const int unknowns=std::max(1,w*h-1);
    coarse_factor=compute.field(unknowns,unknowns);

    const std::string rhs_helpers=R"wgsl(
fn rhsAt(p: vec2i) -> f32 {
    let divisor=params.level.y;
    if (divisor>0.0) {
        // Center the numerator before division so a constant RHS cancels exactly on odd grids.
        return fma(rhs(p).x,divisor,-rhs_stats(vec2i(0)).x)/divisor;
    }
    return rhs(p).x;
}
)wgsl";
    const std::string product_helper=R"wgsl(
fn product(p: vec2i) -> f32 {
    let a=weights(p); let c=source(p).x;
    return (a.x*(c-source(p-vec2i(1,0)).x)+a.y*(c-source(p+vec2i(1,0)).x))+
           (a.z*(c-source(p-vec2i(0,1)).x)+a.w*(c-source(p+vec2i(0,1)).x));
}
)wgsl";
    compute.build(coarsen,"fluid MG coarse conductances",1,{"source"},R"wgsl(
fn center(i: i32, step: f32, extent: f32) -> f32 { return 0.5*(f32(i)*step+min(f32(i+1)*step,extent)); }
fn face(p: vec2i, axis: u32) -> f32 {
    let fine_size=sourceSize(); let coordinate=p[axis]; let extent=gridSize()[axis];
    if (coordinate==0||coordinate==extent) { return 0.0; }
    let edge=2*coordinate; let spacing=params.level.x; let step=spacing*0.5; let domain=params.level.zw[axis];
    let before=(center(edge-1,step,domain)-center(coordinate-1,spacing,domain))/(center(edge-1,step,domain)-center(edge-2,step,domain));
    var after=0.0;
    if (edge+1<fine_size[axis]) { after=(center(coordinate,spacing,domain)-center(edge,step,domain))/(center(edge+1,step,domain)-center(edge,step,domain)); }
    let perpendicular=1u-axis; var value=0.0;
    for (var j=0; j<2; j++) {
        var q=vec2i(0); q[axis]=edge-1; q[perpendicular]=2*p[perpendicular]+j;
        if (q[perpendicular]>=fine_size[perpendicular]) { continue; }
        let direction=select(vec2i(0,1),vec2i(1,0),axis==0u); let channel=select(3u,1u,axis==0u);
        var resistance=before/source(q-direction)[channel]+1.0/source(q)[channel];
        if (after!=0.0) { resistance+=after/source(q+direction)[channel]; }
        value+=1.0/resistance;
    }
    return value;
}
fn execute() { let p=cell(); write0(p,vec4f(face(p,0u),face(p+vec2i(1,0),0u),face(p,1u),face(p+vec2i(0,1),1u))); }
)wgsl");
    compute.build(smooth_pair,"fluid MG symmetric red black",1,{"source","rhs","weights","rhs_stats"},R"wgsl(
fn firstColor(point: vec2i) -> f32 {
    let p=clamp(point,vec2i(0),gridSize()-1); let a=weights(p); let diagonal=(a.x+a.y)+(a.z+a.w); var value=rhsAt(p);
    if (mode()<2) { value+=(a.x*source(p-vec2i(1,0)).x+a.y*source(p+vec2i(1,0)).x)+(a.z*source(p-vec2i(0,1)).x+a.w*source(p+vec2i(0,1)).x); }
    if (diagonal>0.0) { return value/diagonal; } return 0.0;
}
fn execute() {
    let p=cell(); if ((p.x+p.y)%2==mode()%2) { write0(p,vec4f(firstColor(p))); return; }
    let a=weights(p); let diagonal=(a.x+a.y)+(a.z+a.w);
    let value=rhsAt(p)+(a.x*firstColor(p-vec2i(1,0))+a.y*firstColor(p+vec2i(1,0)))+(a.z*firstColor(p-vec2i(0,1))+a.w*firstColor(p+vec2i(0,1)));
    var result=0.0; if (diagonal>0.0) { result=value/diagonal; } write0(p,vec4f(result));
}
)wgsl",rhs_helpers);
    compute.build(polynomial_smooth,"fluid MG polynomial smoother",1,{"source","rhs","weights","rhs_stats"},R"wgsl(
fn execute() {
    let p=cell(); let a=weights(p); let diagonal=(a.x+a.y)+(a.z+a.w); var value=rhsAt(p); var result=0.0;
    if (mode()==0) { value-=product(p); result=source(p).x; }
    if (diagonal>0.0) { result+=params.value.w*value/diagonal; } write0(p,vec4f(result));
}
)wgsl",rhs_helpers+product_helper);
    compute.build(residual,"fluid MG residual",1,{"source","rhs","weights","rhs_stats"},R"wgsl(
fn execute() {
    write0(cell(),vec4f(rhsAt(cell())-product(cell())));
}
)wgsl",rhs_helpers+product_helper);
    compute.build(restrict_pass,"fluid MG transpose restriction",1,{"source","maps"},R"wgsl(
fn execute() {
    let p=cell(); let base=p*2-1; let wx=maps(vec2i(p.x,0)); let wy=maps(vec2i(p.y,1)); var value=0.0;
    for (var y=0; y<4; y++) {
        var horizontal=0.0;
        for (var x=0; x<4; x++) {
            let q=base+vec2i(x,y); if (any(q<vec2i(0))||any(q>=sourceSize())) { continue; }
            horizontal+=wx[x]*source(q).x;
        }
        value+=wy[y]*horizontal;
    }
    write0(p,vec4f(value));
}
)wgsl");
    compute.build(prolong,"fluid MG prolongation",1,{"source","coarse","maps"},R"wgsl(
fn execute() {
    let p=cell(); let px=maps(vec2i(p.x,0)); let py=maps(vec2i(p.y,1));
    let a=(1.0-px.z)*coarse(vec2i(i32(px.x),i32(py.x))).x+px.z*coarse(vec2i(i32(px.y),i32(py.x))).x;
    let b=(1.0-px.z)*coarse(vec2i(i32(px.x),i32(py.y))).x+px.z*coarse(vec2i(i32(px.y),i32(py.y))).x;
    write0(p,vec4f(source(p).x+(1.0-py.z)*a+py.z*b));
}
)wgsl");
    compute.build(factor,"fluid MG tiny Cholesky",1,{"weights"},R"wgsl(
fn execute() {
    if (any(cell()!=vec2i(0))) { return; }
    let n=gridSize().x*gridSize().y-1;
    if (n==0) { write0(vec2i(0),vec4f(1)); return; }
    var lower: array<f32,225>;
    for (var row=0; row<n; row++) {
        let index=row+1; let p=vec2i(index%gridSize().x,index/gridSize().x); let a=weights(p);
        for (var col=0; col<=row; col++) {
            var value=0.0;
            if (col==row) { value=(a.x+a.y)+(a.z+a.w); }
            else if (col==row-1 && p.x>0) { value=-a.x; }
            else if (col==row-gridSize().x && p.y>0) { value=-a.z; }
            for (var k=0; k<col; k++) { value-=lower[row*15+k]*lower[col*15+k]; }
            if (row==col) {
                if (value<=0.0||!finite(value)) { write0(vec2i(row,row),vec4f(-1)); return; }
                value=sqrt(value);
            } else { value/=lower[col*15+col]; }
            lower[row*15+col]=value; write0(vec2i(col,row),vec4f(value));
        }
    }
}
)wgsl");
    compute.build(coarse_solve,"fluid MG tiny solve",1,{"rhs","factor","rhs_stats"},R"wgsl(
fn execute() {
    // Keep validity outside nested-loop exits: DXC can miscompile an early return here.
    let count=gridSize().x*gridSize().y; let n=count-1; var values: array<f32,15>; var mean=0.0; var valid=true;
    for (var i=0; i<count; i++) { mean+=rhsAt(vec2i(i%gridSize().x,i/gridSize().x)); } mean/=f32(count);
    for (var row=0; row<n; row++) {
        let diagonal=factor(vec2i(row,row)).x;
        valid = valid && diagonal>0.0 && finite(diagonal);
        var value=rhsAt(vec2i((row+1)%gridSize().x,(row+1)/gridSize().x))-mean;
        for (var col=0; col<row; col++) { value-=factor(vec2i(col,row)).x*values[col]; } values[row]=value/diagonal;
    }
    for (var row=n-1; row>=0; row--) {
        var value=values[row];
        for (var col=row+1; col<n; col++) { value-=factor(vec2i(row,col)).x*values[col]; }
        values[row]=value/factor(vec2i(row,row)).x;
    }
    mean=0.0; for (var i=0; i<n; i++) { mean+=values[i]; } mean/=f32(count);
    let index=cell().y*gridSize().x+cell().x; var value=0.0; if (index!=0) { value=values[index-1]; }
    write0(cell(),vec4f(select(0.0,value-mean,valid)));
}
)wgsl",rhs_helpers);
    if (tile_level>=0)
    {
        tile_coefficients=compute.field(tile_cells,1,4);
        tile_maps=compute.field(int(packed_maps.size()),1);
        compute.write(tile_maps,packed_maps);
        compute.build(pack_coefficients,"fluid MG packed coarse coefficients",1,{"source"},R"wgsl(
fn execute() {
    let p=cell(); let first=4*(mode()+p.y*gridSize().x+p.x); let a=source(p);
    output0Data[first]=a.x; output0Data[first+1]=a.y; output0Data[first+2]=a.z; output0Data[first+3]=a.w;
}
)wgsl");
        compute.build(tile_cycle,"fluid MG shared coarse cycle",1,{"rhs","rhs_stats","packed","maps","factor"},coarse_cycle_shader,rhs_helpers,true,{16,16});
    }
}

void PressureMultigridGPU::buildTransfers(Level& fine, const Level& coarse, std::vector<float>* packed_maps)
{
    const int fine_extent=std::max(fine.width,fine.height), coarse_extent=std::max(coarse.width,coarse.height);
    fine.prolong_map=compute.field(fine_extent,2,4); fine.restrict_map=compute.field(coarse_extent,2,4);
    std::vector<float> interpolation(size_t(fine_extent)*8), restriction(size_t(coarse_extent)*8);
    for (int axis=0; axis<2; ++axis)
    {
        const int size=axis==0?fine.width:fine.height, parent_size=axis==0?coarse.width:coarse.height;
        const int extent=axis==0?levels.front().width:levels.front().height;
        auto center=[&](int i,int spacing) { return 0.5*(i*spacing+std::min((i+1)*spacing,extent)); };
        for (int f=0; f<size; ++f)
        {
            const int parent=f/2;
            const int first=std::max(0,parent-int(center(f,fine.spacing)<center(parent,coarse.spacing)));
            const int second=std::min(first+1,parent_size-1);
            const float weight=first==second?0.0f:float(std::clamp((center(f,fine.spacing)-center(first,coarse.spacing))/(center(second,coarse.spacing)-center(first,coarse.spacing)),0.0,1.0));
            const int i=(axis*fine_extent+f)*4;
            interpolation[i]=float(first); interpolation[i+1]=float(second); interpolation[i+2]=weight;
            restriction[(axis*coarse_extent+first)*4+f-(2*first-1)]+=1.0f-weight;
            if (weight!=0.0f) restriction[(axis*coarse_extent+second)*4+f-(2*second-1)]+=weight;
        }
    }
    compute.write(fine.prolong_map,interpolation); compute.write(fine.restrict_map,restriction);
    if (packed_maps)
    {
        fine.prolong_offset=int(packed_maps->size()); packed_maps->insert(packed_maps->end(),interpolation.begin(),interpolation.end());
        fine.restrict_offset=int(packed_maps->size()); packed_maps->insert(packed_maps->end(),restriction.begin(),restriction.end());
    }
}

void PressureMultigridGPU::run(Kernel& kernel, int index, std::initializer_list<const Field*> outputs,
                             std::initializer_list<Compute::Input> inputs, int mode, float relaxation)
{
    Parameters p;
    p.grid={levels[index].width,levels[index].height,mode,0};
    p.value[3]=relaxation;
    p.level={float(levels[index].spacing),index==0?float(levels.front().width*levels.front().height):0.0f,float(levels.front().width),float(levels.front().height)};
    compute.run(kernel,outputs,inputs,p);
}

const Field& PressureMultigridGPU::coefficients(int level) const { return level==0?*fine_coefficients:levels[level].coefficients; }

void PressureMultigridGPU::rebuild(const Field& fine)
{
    fine_coefficients=&fine;
    const int last=int(levels.size())-1;
    for (int i=1; i<=last; ++i) run(coarsen,i,{&levels[i].coefficients},{{"source",&coefficients(i-1)}});
    run(factor,last,{&coarse_factor},{{"weights",&coefficients(last)}});
    for (int i=tile_level; i>=0 && i<=last; ++i)
    {
        const auto& level=levels[i]; Parameters p;
        p.grid={level.width,level.height,level.tile_offset,0};
        compute.run(pack_coefficients,{&tile_coefficients},{{"source",&coefficients(i)}},p,{unsigned(level.width),unsigned(level.height),1});
    }
}

void PressureMultigridGPU::smoothLevel(int index, const Field& rhs, bool reverse, bool from_zero, int sweep)
{
    auto& level=levels[index];
    const bool polynomial=level.width*level.height>65536;
    const int mode=polynomial?int(from_zero):int(reverse)+(from_zero?2:0);
    const float weight=polynomial?smoothing_weights[reverse?smoothing_sweeps-1-sweep:sweep]:0.0f;
    run(polynomial?polynomial_smooth:smooth_pair,index,{&level.next_solution},
        {{"source",&level.solution},{"rhs",&rhs},{"weights",&coefficients(index)},{"rhs_stats",residual_stats}},mode,weight);
    level.solution.swap(level.next_solution);
}

void PressureMultigridGPU::cycle(int index, const Field& rhs)
{
    auto& fine=levels[index];
    if (index==tile_level)
    {
        // Execute the same coarse V-cycle locally, preserving every level and transfer weight.
        Parameters p;
        p.grid={fine.width,fine.height,0,int(levels.size())-index};
        p.level={float(fine.spacing),index==0?float(fine.width*fine.height):0.f,float(levels.front().width),float(levels.front().height)};
        for (int i=index; i<int(levels.size()); ++i)
        {
            const auto& l=levels[i];
            p.vectors[i-index]={float(l.width),float(l.height),float(l.spacing),float(l.tile_offset)};
            p.vectors[i-index+4]={float(l.prolong_offset),float(l.restrict_offset),float(std::max(l.width,l.height)),
                i+1<int(levels.size())?float(std::max(levels[i+1].width,levels[i+1].height)):0.f};
        }
        compute.run(tile_cycle,{&fine.solution},{{"rhs",&rhs},{"rhs_stats",residual_stats},{"packed",&tile_coefficients},
            {"maps",&tile_maps},{"factor",&coarse_factor}},p,{16,16,1});
        return;
    }
    if (index+1==int(levels.size()))
    {
        run(coarse_solve,index,{&fine.solution},{{"rhs",&rhs},{"factor",&coarse_factor},{"rhs_stats",residual_stats}});
        return;
    }
    auto& coarse=levels[index+1];
    smoothLevel(index,rhs,false,true,0);
    for (int sweep=1; sweep<smoothing_sweeps; ++sweep) smoothLevel(index,rhs,false,false,sweep);
    run(residual,index,{&fine.residual},{{"source",&fine.solution},{"rhs",&rhs},{"weights",&coefficients(index)},{"rhs_stats",residual_stats}});
    run(restrict_pass,index,{&coarse.rhs},{{"source",&fine.residual},{"maps",&fine.restrict_map}});
    cycle(index+1,coarse.rhs);
    run(prolong,index,{&fine.next_solution},{{"source",&fine.solution},{"coarse",&coarse.solution},{"maps",&fine.prolong_map}});
    fine.solution.swap(fine.next_solution);
    for (int sweep=0; sweep<smoothing_sweeps; ++sweep) smoothLevel(index,rhs,true,false,sweep);
}

const Field& PressureMultigridGPU::apply(const Field& residual, const Field& stats)
{
    residual_stats=&stats; cycle(0,residual); return levels.front().solution;
}

} // namespace fluid::webgpu
