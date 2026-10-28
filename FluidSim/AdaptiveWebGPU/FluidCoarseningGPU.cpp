#include "FluidCoarseningGPU.h"
#include "FluidGridAdaptiveGPU.h"
#include "FluidCoarseningGPUShaders.h"
#include "../UniformWebGPU/FluidWebGPUShaders.h"
#include "FluidLinearReconstructionShaders.h"
#include <bit>
#include <cmath>

namespace fluid {
namespace {
using namespace webgpu;
float packed(unsigned value) { return std::bit_cast<float>(value); }
float packed(int value) { return std::bit_cast<float>(value); }
std::string joined(std::string_view a,std::string_view b) { return std::string(a)+std::string(b); }
constexpr std::string_view sweepShader=R"wgsl(
struct Config { width:u32,height:u32,offset:u32,count:u32 }
@group(0) @binding(0) var<uniform> config:Config;
@group(0) @binding(1) var<storage,read> observations:array<f32>;
@group(0) @binding(2) var<storage,read_write> differences:array<atomic<i32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3u) {
    if(id.x>=config.count) { return; }
    let b=24u*(config.offset+id.x);
    let x0=bitcast<u32>(observations[b+8u]); let y0=bitcast<u32>(observations[b+9u]);
    let x1=bitcast<u32>(observations[b+10u]); let y1=bitcast<u32>(observations[b+11u]);
    if(x0==x1 || y0==y1) { return; }
    let stride=config.width+1u;
    atomicAdd(&differences[y0*stride+x0],1); atomicAdd(&differences[y0*stride+x1],-1);
    atomicAdd(&differences[y1*stride+x0],-1); atomicAdd(&differences[y1*stride+x1],1);
}
)wgsl";
}

struct FluidCoarseningGPU::Impl {
    struct Assessment {
        int scale=2,width=0,height=0,groupOffset=0,tileOffset=0,tileCount=0;
        Field differences,rows;
        wgpu::Buffer uniform;
        wgpu::BindGroup sweepBinding;
    };
    Compute& commands;
    wgpu::Device device;
    Kernel buildPlane,assessMaterial,assessVelocity,assess,rowPrefix,columnPrefix,stabilise,makeTiles,display,summary,reduce;
    wgpu::ComputePipeline sweep;
    wgpu::BindGroupLayout sweepLayout;
    Field metadata,snapshot,planes,observations,history,previous,tiles,displayLabels,partials,totals;
    std::vector<Assessment> assessments;
    std::vector<std::array<size_t,4>> stateOffsets;
    std::vector<FluidCoarseningPreview::Requests> requests;
    Parameters parameters;
    const FluidGridAdaptiveGPU* observedGrid=nullptr;
    Tolerances tolerance;
    Stability stability;
    GasLayout observedLayout;
    int width=0,height=0,components=0,levelCount=0,planeCount=0,groupCount=0;
    double cellSize=0.,lookahead=0.,lastTime=0.;

    explicit Impl(Compute& commands):commands(commands),device(commands.gpu.device()) {
        namespace shader=webgpu::coarsening;
        const auto native=std::string(shader::nativeView);
        commands.build(buildPlane,"AMR bounded selection planes",1,{"metadata","state"},shader::planeBuilder,joined(native,webgpu::geometry)+std::string(webgpu::linearReconstruction));
        commands.build(assessMaterial,"AMR native material detail",1,{"metadata","state","planes"},shader::materialAssessment,native);
        commands.build(assessVelocity,"AMR native velocity detail",1,{"metadata","state","planes"},shader::velocityAssessment,native);
        commands.build(assess,"AMR native pressure and decisions",1,{"metadata","state","previous"},shader::assessment,native);
        commands.build(rowPrefix,"AMR swept request row scan",1,{"differences"},shader::rowPrefix);
        commands.build(columnPrefix,"AMR swept request column scan",1,{"metadata","state","rows"},shader::columnPrefix,native);
        commands.build(stabilise,"AMR quiet decision history",2,{"metadata","state"},shader::stabilise,native);
        commands.build(makeTiles,"AMR padded request tiles",1,{"metadata","state","observations"},shader::tileRequests,native);
        commands.build(display,"AMR selection display labels",2,{"metadata","state","observations"},shader::displayAndScales,native);
        commands.build(summary,"AMR selection summary",1,{"labels","observations"},shader::summary);
        commands.build(reduce,"AMR selection summary reduction",1,{"partials"},shader::reduceSummary);
        std::array<wgpu::BindGroupLayoutEntry,3> entries{};
        for(unsigned i=0;i<entries.size();++i) { entries[i].binding=i;entries[i].visibility=wgpu::ShaderStage::Compute; }
        entries[0].buffer.type=wgpu::BufferBindingType::Uniform;entries[0].buffer.minBindingSize=16;
        entries[1].buffer.type=wgpu::BufferBindingType::ReadOnlyStorage;entries[2].buffer.type=wgpu::BufferBindingType::Storage;
        wgpu::BindGroupLayoutDescriptor group;group.entryCount=entries.size();group.entries=entries.data();
        sweepLayout=commands.gpu.device().CreateBindGroupLayout(&group);
        wgpu::PipelineLayoutDescriptor layoutDescriptor;layoutDescriptor.bindGroupLayoutCount=1;layoutDescriptor.bindGroupLayouts=&sweepLayout;
        auto layout=commands.gpu.device().CreatePipelineLayout(&layoutDescriptor);
        wgpu::ComputePipelineDescriptor descriptor;descriptor.label="AMR swept rectangle accumulation";descriptor.layout=layout;
        descriptor.compute.module=commands.gpu.shader(sweepShader,"AMR swept rectangle accumulation");descriptor.compute.entryPoint="main";
        sweep=commands.gpu.device().CreateComputePipeline(&descriptor);
    }
    void invalidateBindings() {
        for(auto* kernel:{&buildPlane,&assessMaterial,&assessVelocity,&assess,&rowPrefix,&columnPrefix,&stabilise,&makeTiles,&display,&summary,&reduce}) kernel->bindings.clear();
    }
    Parameters settings(int level) const {
        auto p=parameters; const auto& a=assessments[level];p.padding[1]={unsigned(level),unsigned(a.scale),unsigned(a.width),unsigned(a.height)};return p;
    }
    void allocate(int w,int h,int n,int count,int allPlanes,bool keepHistory) {
        invalidateBindings();width=w;height=h;components=n;levelCount=count;planeCount=allPlanes;
        assessments.clear();assessments.resize(planeCount);
        int planeGroups=0,tileCount=0;
        for(int lev=0;lev<planeCount;++lev) {
            auto& a=assessments[lev];a.scale=2<<lev;a.width=(w+a.scale-1)/a.scale;a.height=(h+a.scale-1)/a.scale;a.groupOffset=planeGroups;
            planeGroups+=a.width*a.height;
            if(lev>=levelCount) continue;
            groupCount=planeGroups;
            const int tile=std::max(16,a.scale);a.tileOffset=tileCount;a.tileCount=((w+tile-1)/tile)*((h+tile-1)/tile);tileCount+=a.tileCount;
            a.differences=commands.field((a.width+1)*(a.height+1),1,1,"AMR swept corners");a.rows=commands.field(a.width*a.height,1,1,"AMR swept rows");
        }
        const int fineGroups=((w+1)/2)*((h+1)/2),blocks=(std::max(fineGroups,groupCount)+255)/256;
        planes=commands.field(planeGroups*(n+2)*7,1,1,"AMR derived assessment planes");
        observations=commands.field(groupCount*24,1,1,"AMR selection diagnostics");
        if(!keepHistory) { history=commands.field(groupCount*3,1,1,"AMR quiet history");previous=commands.field(fineGroups,1,1,"AMR previous exact footprint scales"); }
        displayLabels=commands.field(fineGroups,1,1,"AMR preview labels");
        tiles=commands.field(tileCount,1,1,"AMR compact request tiles");partials=commands.field(blocks*11,1,1,"AMR selection partial counts");totals=commands.field(11,1,1,"AMR selection counts");
        parameters.padding[2]={0,unsigned(levelCount),unsigned(planeCount),unsigned(groupCount)};
        parameters.padding[3]={unsigned(fineGroups),unsigned((w+1)/2),unsigned((h+1)/2),unsigned(blocks)};
        for(int lev=0;lev<levelCount;++lev) {
            auto& a=assessments[lev]; const std::array<unsigned,4> config{unsigned(a.width),unsigned(a.height),unsigned(a.groupOffset),unsigned(a.width*a.height)};
            a.uniform=commands.gpu.buffer(sizeof(config),wgpu::BufferUsage::Uniform|wgpu::BufferUsage::CopyDst,"AMR sweep shape");
            commands.gpu.queue().WriteBuffer(a.uniform,0,config.data(),sizeof(config));
            std::array<wgpu::BindGroupEntry,3> entries{};
            entries[0].binding=0;entries[0].buffer=a.uniform;entries[0].size=sizeof(config);
            entries[1].binding=1;entries[1].buffer=observations.buffer;entries[1].size=observations.bytes();
            entries[2].binding=2;entries[2].buffer=a.differences.buffer;entries[2].size=a.differences.bytes();
            wgpu::BindGroupDescriptor binding;binding.layout=sweepLayout;binding.entryCount=entries.size();binding.entries=entries.data();
            a.sweepBinding=commands.gpu.device().CreateBindGroup(&binding);
        }
    }
    void geometry(const FluidGridAdaptiveGPU& grid) {
        invalidateBindings();
        const int ownersWidth=(width+1)/2,ownersHeight=(height+1)/2;
        std::vector<float> data(size_t(ownersWidth)*ownersHeight,packed(-1));
        const unsigned patchOffset=unsigned(data.size());
        stateOffsets.resize(grid.levels().size());
        size_t stateSize=0;int patchId=0;
        auto descriptor=[&](const AmrFieldLayout2D::Patch& p,size_t base) {
            return std::array<float,4>{packed(unsigned(base+p.offset)),packed(p.box.x-p.ghosts),packed(p.box.y-p.ghosts),packed(p.stride)};
        };
        for(size_t lev=0;lev<grid.levels().size();++lev) {
            const auto& level=*grid.levels()[lev];const int scale=int(std::lround(level.geometry.cell_size/grid.finestCellSize()));
            auto& offsets=stateOffsets[lev];
            offsets[0]=stateSize;stateSize+=level.conserved.bytes()/sizeof(float);offsets[1]=stateSize;stateSize+=level.pressure.bytes()/sizeof(float);
            offsets[2]=stateSize;stateSize+=level.velocity[0].bytes()/sizeof(float);offsets[3]=stateSize;stateSize+=level.velocity[1].bytes()/sizeof(float);
            for(size_t p=0;p<level.qLayout.patches.size();++p,++patchId) {
                const auto& q=level.qLayout.patches[p];std::array<float,20> record{};record[0]=packed(scale);
                const auto qd=descriptor(q,offsets[0]);std::copy(qd.begin(),qd.end(),record.begin()+1);record[5]=packed(q.plane);
                const auto pd=descriptor(level.pressureLayout.patches[p],offsets[1]);std::copy(pd.begin(),pd.end(),record.begin()+6);
                record[18]=packed(level.pressure.channels>1?level.pressure.width*level.pressure.height:0);
                for(int d=0;d<2;++d) { const auto face=descriptor(level.faceLayout[d].patches[p],offsets[2+d]);std::copy(face.begin(),face.end(),record.begin()+10+4*d); }
                data.insert(data.end(),record.begin(),record.end());
                const auto& b=q.box;
                for(int y=b.y*scale;y<(b.y+b.height)*scale;y+=2) for(int x=b.x*scale;x<(b.x+b.width)*scale;x+=2)
                    data[size_t(y/2)*ownersWidth+x/2]=packed(patchId);
            }
        }
        const unsigned recordsOffset=unsigned(data.size());
        for(int lev=0;lev<planeCount;++lev) {
            const auto& a=assessments[lev];const std::array<unsigned,8> record{unsigned(a.scale),unsigned(a.width),unsigned(a.height),unsigned(a.groupOffset),0,0,unsigned(a.tileOffset),0};
            for(auto value:record) data.push_back(packed(value));
        }
        const unsigned densityOffset=unsigned(data.size());for(const auto& component:grid.materialMixture().components) data.push_back(float(component.density));
        metadata=commands.field(int(data.size()),1,1,"AMR selection geometry");commands.write(metadata,data);
        snapshot=commands.field(int(stateSize),1,1,"AMR native selection snapshot");
        parameters.padding[0]={0,patchOffset,densityOffset,recordsOffset};observedLayout=grid.layout();
    }
    void copyNative(const FluidGridAdaptiveGPU& grid) {
        commands.flush();auto encoder=commands.gpu.device().CreateCommandEncoder();
        for(size_t lev=0;lev<grid.levels().size();++lev) {
            const auto& level=*grid.levels()[lev];const std::array<const Field*,4> source{&level.conserved,&level.pressure,&level.velocity[0],&level.velocity[1]};
            for(int f=0;f<4;++f) encoder.CopyBufferToBuffer(source[f]->buffer,0,snapshot.buffer,stateOffsets[lev][f]*sizeof(float),source[f]->bytes());
        }
        auto batch=encoder.Finish();commands.gpu.queue().Submit(1,&batch);
    }
    void markSwept() {
        commands.flush();auto encoder=commands.gpu.device().CreateCommandEncoder();
        for(int lev=0;lev<levelCount;++lev) encoder.ClearBuffer(assessments[lev].differences.buffer);
        auto pass=encoder.BeginComputePass();pass.SetPipeline(sweep);
        for(int lev=0;lev<levelCount;++lev) { const auto& a=assessments[lev];pass.SetBindGroup(0,a.sweepBinding);pass.DispatchWorkgroups((a.width*a.height+63)/64); }
        pass.End();auto batch=encoder.Finish();commands.gpu.queue().Submit(1,&batch);
        for(int lev=0;lev<levelCount;++lev) {
            const auto& a=assessments[lev];const auto p=settings(lev);
            commands.run(rowPrefix,{&a.rows},{{"differences",&a.differences}},p,{unsigned(a.height),1,1});
            commands.run(columnPrefix,{&observations},{{"metadata",&metadata},{"state",&snapshot},{"rows",&a.rows}},p,{unsigned(a.width),1,1});
        }
    }
};

FluidCoarseningGPU::FluidCoarseningGPU()=default;
FluidCoarseningGPU::~FluidCoarseningGPU()=default;
void FluidCoarseningGPU::reset() { if(impl) { impl->observedGrid=nullptr;impl->requests.clear(); } labels.clear();counts.fill(0);statistics={}; }

void FluidCoarseningGPU::update(const FluidGridAdaptiveGPU& grid,const Tolerances& tolerance,const Stability& stability,double time,double lookahead,int maxCellScale,Trace* trace) {
    auto& commands=grid.commands();if(!impl || &impl->commands!=&commands || impl->device.Get()!=commands.gpu.device().Get()) impl=std::make_unique<Impl>(commands);
    auto& p=*impl;
    const int levels=int(std::countr_zero(unsigned(maxCellScale))),planes=std::max(levels,int(std::countr_zero(unsigned(grid.layout().baseRatio))));
    const bool resized=p.width!=grid.finestWidth() || p.height!=grid.finestHeight() || p.components!=grid.ingredientCount() || p.levelCount!=levels;
    const bool storageChanged=resized || p.planeCount!=planes;
    const bool fresh=resized || p.observedGrid!=&grid || p.cellSize!=grid.finestCellSize() || p.tolerance!=tolerance || p.stability!=stability || p.lookahead!=lookahead || time<p.lastTime;
    const bool geometry=storageChanged || p.observedGrid!=&grid || p.observedLayout!=grid.layout();
    if(storageChanged) p.allocate(grid.finestWidth(),grid.finestHeight(),grid.ingredientCount(),levels,planes,!resized);
    if(geometry) p.geometry(grid);
    p.observedGrid=&grid;p.cellSize=grid.finestCellSize();p.tolerance=tolerance;p.stability=stability;p.lookahead=lookahead;p.lastTime=time;
    p.parameters.grid={grid.finestWidth(),grid.finestHeight(),grid.ingredientCount(),grid.liquidIngredientCount()};
    p.parameters.value={float(grid.finestCellSize()),grid.pressureInterval()>0.?float(1./grid.pressureInterval()):0.f,float(time),float(lookahead)};
    p.parameters.level={float(tolerance.velocity),float(tolerance.pressure_gradient),float(tolerance.density),float(tolerance.ingredient)};
    const auto acceleration=grid.acceleration();p.parameters.vectors[0]={float(stability.quiet_ratio),float(stability.quiet_seconds),float(acceleration[0]),float(acceleration[1])};
    p.parameters.padding[2][0]=fresh;
    p.copyNative(grid);
    // Higher native planes resolve every recursive coarse source before a
    // smaller footprint uses it; no fine physical state is manufactured.
    for(int lev=p.planeCount-1;lev>=0;--lev) {
        const auto& a=p.assessments[lev];commands.run(p.buildPlane,{&p.planes},{{"metadata",&p.metadata},{"state",&p.snapshot}},p.settings(lev),{unsigned(a.width),unsigned(a.height),1});
    }
    for(int lev=0;lev<levels;++lev) {
        const auto& a=p.assessments[lev];const auto settings=p.settings(lev);const std::array<unsigned,3> extent={unsigned(a.width),unsigned(a.height),1};
        commands.run(p.assessMaterial,{&p.observations},{{"metadata",&p.metadata},{"state",&p.snapshot},{"planes",&p.planes}},settings,extent);
        commands.run(p.assessVelocity,{&p.observations},{{"metadata",&p.metadata},{"state",&p.snapshot},{"planes",&p.planes}},settings,extent);
        commands.run(p.assess,{&p.observations},{{"metadata",&p.metadata},{"state",&p.snapshot},{"previous",&p.previous}},settings,extent);
    }
    if(lookahead>0.) p.markSwept();
    for(int lev=0;lev<levels;++lev) {
        const auto& a=p.assessments[lev];const auto settings=p.settings(lev);const int tile=std::max(16,a.scale);
        commands.run(p.stabilise,{&p.observations,&p.history},{{"metadata",&p.metadata},{"state",&p.snapshot}},settings,{unsigned(a.width),unsigned(a.height),1});
        commands.run(p.makeTiles,{&p.tiles},{{"metadata",&p.metadata},{"state",&p.snapshot},{"observations",&p.observations}},settings,{unsigned((p.width+tile-1)/tile),unsigned((p.height+tile-1)/tile),1});
    }
    width=(p.width+1)/2;height=(p.height+1)/2;
    commands.run(p.display,{&p.displayLabels,&p.previous},{{"metadata",&p.metadata},{"state",&p.snapshot},{"observations",&p.observations}},p.parameters,{unsigned(width),unsigned(height),1});
    commands.run(p.summary,{&p.partials},{{"labels",&p.displayLabels},{"observations",&p.observations}},p.parameters,{p.parameters.padding[3][3],1,1});
    commands.run(p.reduce,{&p.totals},{{"partials",&p.partials}},p.parameters,{11,1,1});
    const auto masks=commands.read(p.tiles),totals=commands.read(p.totals);p.requests.clear();
    for(int lev=0;lev<levels;++lev) { const auto& a=p.assessments[lev];auto& request=p.requests.emplace_back();request.scale=a.scale;request.tiles.resize(a.tileCount);
        for(int i=0;i<a.tileCount;++i) request.tiles[i]=masks[a.tileOffset+i]!=0.f; }
    for(int i=0;i<FluidCoarseningPreview::LabelCount;++i) counts[i]=int(totals[i]);
    statistics={int(totals[7]),int(totals[8]),int(totals[9]),int(totals[10])};labels.clear();++revision;
    if(trace) {
        *trace={};trace->levels.resize(levels);const auto values=commands.read(p.observations);
        for(int lev=0;lev<levels;++lev) {
            const auto& a=p.assessments[lev];auto& target=trace->levels[lev];target.scale=a.scale;target.width=a.width;target.height=a.height;target.cells.resize(a.width*a.height);
            target.tile=std::max(16,a.scale);target.tiles_x=(p.width+target.tile-1)/target.tile;target.tiles_y=(p.height+target.tile-1)/target.tile;
            target.padded=target.nested=p.requests[lev].tiles;
            for(int i=0;i<a.width*a.height;++i) { auto& c=target.cells[i];const auto* v=values.data()+24*(a.groupOffset+i);
                c.error={v[0],v[1],v[2],v[3]};c.source=std::uint8_t(v[4]);c.quiet=v[5]!=0.;c.observed_scale=int(v[6]);c.evaluated=v[7]!=0.;
                c.raw=std::uint8_t(v[12]);c.decision=std::uint8_t(v[13]);c.requested=v[14]!=0.;c.quiet_since=v[15];
            }
        }
    }
}
GasLayout FluidCoarseningGPU::proposeLayout(const FluidGridAdaptiveGPU& grid,Trace* trace) const {
    if(!impl || impl->requests.empty()) return grid.layout();
    return FluidCoarseningPreview::clusterRequests(grid.finestWidth(),grid.finestHeight(),impl->requests,trace);
}
void FluidCoarseningGPU::downloadLabels() {
    if(!impl) { labels.clear();return; }
    const auto values=impl->commands.read(impl->displayLabels);labels.resize(values.size());for(size_t i=0;i<values.size();++i) labels[i]=std::uint8_t(values[i]);
}
} // namespace fluid
