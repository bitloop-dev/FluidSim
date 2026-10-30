#include "FluidGridAdaptiveGPU.h"
#include "FluidCoarseningGPU.h"
#include "FluidCellView.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

using namespace fluid;
using Preview=FluidCoarseningPreview;
namespace {
void require(bool ok,const char* text) { if(!ok) throw std::runtime_error(text); }
struct Fixture {
    ElementCatalogue catalogue;
    FluidMixture liquid,gas;
    int n;
    Fixture(int n=64,double contrast=1.,bool wet=false):n(n) {
        if(wet) liquid.components.push_back({catalogue.add({"Water"}),100.});
        gas.components={{catalogue.add({"Air"}),1.},{catalogue.add({"Cloud"}),contrast}};
    }
    FluidGridAdaptive::State state() const {
        FluidGridAdaptive::State s;s.gas_mass={std::vector<double>(n*n,.65),std::vector<double>(n*n,.35*gas.components[1].density)};
        s.liquid_mass.assign(liquid.components.size(),std::vector<double>(n*n));s.liquid_fraction.resize(n*n);return s;
    }
};
struct Pair {
    Fixture fixture;
    FluidGridAdaptive cpu;
    FluidGridAdaptiveGPU gpu;
    Preview reference;
    FluidCoarseningGPU selection;
    Pair(int n,double contrast,bool wet,GasLayout layout):fixture(n,contrast,wet),
        cpu(n,n,1.,fixture.catalogue,fixture.liquid,fixture.gas,layout,16),gpu(n,n,1.,fixture.catalogue,fixture.liquid,fixture.gas,layout,16) {
        cpu.upload(fixture.state(),1e-4);gpu.upload(fixture.state(),1e-4);
        require(cpu.project(1e3,.25,{1.,-.4}).converged && gpu.project(1e3,.25,{1.,-.4}).converged,"Fixture pressure interval setup failed");
    }
};

// Tests prescribe the same float-valued native state in each engine. Production
// selection never downloads these fields, nor constructs a CPU shadow.
void prescribe(Pair& pair,int material,int velocity,int pressureMode=0,bool wet=false,bool exactVolume=true) {
    const int n=pair.fixture.n,nc=pair.gpu.ingredientCount(),nl=pair.gpu.liquidIngredientCount();
    auto cloud=[&](double x,double y) {
        if(material==0) return .35;
        if(material==1) return .2+.3*x/n+.05*y/n;
        if(material==2) return .4+.2*std::sin(.11*x)*std::cos(.09*y);
        if(material==3) return (int(x/2)+int(y/2))%2?.8:.2;
        return x>=24. && x<29. && y>=24. && y<40.?.8:.1;
    };
    auto speed=[&](double x,double y,int d) {
        if(velocity==0 || velocity==3) return 0.;
        if(velocity==1) return d?-.25:.75;
        if(velocity==2) return d?-.08+.004*x-.002*y:.2+.003*x+.002*y;
        return d?.12*std::sin(.1*x):.2*std::cos(.12*y);
    };
    for(size_t lev=0;lev<pair.gpu.levels().size();++lev) {
        const auto& g=*pair.gpu.levels()[lev];auto& c=*pair.cpu.levels()[lev];const double scale=g.geometry.cell_size;
        std::vector<float> q(g.qLayout.elements),p(g.pressure.bytes()/sizeof(float));
        for(size_t patch=0;patch<g.qLayout.patches.size();++patch) {
            const auto& b=g.qLayout.patches[patch];const auto cq=c.conserved.array(int(patch));const auto cp=c.pressure.array(int(patch));const auto& pb=g.pressureLayout.patches[patch];
            forCells(b.box,[&](int x,int y) {
                const double fx=(x+.5)*scale,fy=(y+.5)*scale;
                // Dyadic fractions isolate formula parity from cancellation of
                // almost-flat means; roundoff() retains that separate case.
                const double f=exactVolume?std::round(cloud(fx,fy)*65536.)/65536.:cloud(fx,fy);
                const bool liquid=wet && fx>=18. && fx<26. && fy>=28. && fy<36.;double rho=0.;
                for(int k=0;k<nc;++k) {
                    const double fraction=k<nl?double(liquid):liquid?0.:k==nl?1.-f:f;
                    const float value=float(fraction*pair.gpu.materialMixture().components[k].density);rho+=value;
                    cq(x,y,k)=value;q[b.offset+b.index(x,y)+k*b.plane]=value;
                }
                for(int d=0;d<2;++d) { const float value=float(rho*speed(fx,fy,d));cq(x,y,nc+d)=value;q[b.offset+b.index(x,y)+(nc+d)*b.plane]=value; }
                const float value=float(pair.cpu.pressureInterval()*(pressureMode==1?.02*fx+.03*fy:pressureMode==2?.001*(fx-32.)*(fy-24.):0.));
                cp(x,y)=value;p[pb.offset+pb.index(x,y)]=value;
            });
        }
        pair.gpu.commands().write(g.conserved,q);pair.gpu.commands().write(g.pressure,p);
        for(int d=0;d<2;++d) {
            std::vector<float> values(g.faceLayout[d].elements);
            for(size_t patch=0;patch<g.faceLayout[d].patches.size();++patch) {
                const auto& b=g.faceLayout[d].patches[patch];const auto out=c.velocity[d].array(int(patch));
                forCells(b.box,[&](int x,int y) {
                    const double fx=(x+(d?.5:0.))*scale,fy=(y+(d?0.:.5))*scale;
                    const float value=float(velocity==3?((x+y)%2?.8:-.8):speed(fx,fy,d));out(x,y)=value;values[b.offset+b.index(x,y)]=value;
                });
            }
            pair.gpu.commands().write(g.velocity[d],values);
        }
    }
}

double compare(Pair& pair,Preview::Tolerances tolerances,Preview::Stability stability,double time,double lookahead=0.,int maximum=32,Preview::Trace* out=nullptr) {
    Preview::Trace a,b;pair.reference.update(pair.cpu,tolerances,stability,time,lookahead,maximum,&a);pair.selection.update(pair.gpu,tolerances,stability,time,lookahead,maximum,&b);
    const auto ca=pair.reference.proposeLayout(pair.cpu,&a),cb=pair.selection.proposeLayout(pair.gpu,&b);
    require(a.levels.size()==b.levels.size(),"GPU assessment level count differs");double error=0.;
    for(size_t lev=0;lev<a.levels.size();++lev) {
        const auto& x=a.levels[lev];const auto& y=b.levels[lev];require(x.cells.size()==y.cells.size(),"GPU assessment footprint count differs");
        for(size_t i=0;i<x.cells.size();++i) {
            const auto& c=x.cells[i];const auto& g=y.cells[i];
            const std::array<double,4> v{c.error.velocity,c.error.pressure_gradient,c.error.density,c.error.ingredient},w{g.error.velocity,g.error.pressure_gradient,g.error.density,g.error.ingredient};
            for(int d=0;d<4;++d) error=std::max(error,std::abs(v[d]-w[d])/std::max(1.,std::abs(v[d])));
            if(c.source!=g.source || c.raw!=g.raw || c.decision!=g.decision || c.requested!=g.requested || c.quiet!=g.quiet || c.observed_scale!=g.observed_scale || c.evaluated!=g.evaluated) {
                std::cerr<<"mismatch scale="<<x.scale<<" xy="<<i%x.width<<','<<i/x.width<<" source="<<int(c.source)<<'/'<<int(g.source)<<" raw="<<int(c.raw)<<'/'<<int(g.raw)
                         <<" decision="<<int(c.decision)<<'/'<<int(g.decision)<<" quiet="<<c.quiet<<'/'<<g.quiet<<" errors="<<v[0]<<','<<v[1]<<','<<v[2]<<','<<v[3]<<" / "<<w[0]<<','<<w[1]<<','<<w[2]<<','<<w[3]<<'\n';
                throw std::runtime_error("GPU bounded criteria/history differs from CPU");
            }
            require(std::abs(c.quiet_since-g.quiet_since)<2e-6,"GPU quiet timestamp differs from CPU");
        }
        require(x.padded==y.padded && x.nested==y.nested,"GPU padded/nested tile masks differ from CPU");
    }
    require(error<3e-5,"GPU representation/pressure errors differ from CPU");require(ca==cb,"GPU proposed layout differs from CPU");
    pair.selection.downloadLabels();require(pair.reference.labels==pair.selection.labels,"GPU display labels differ from CPU");
    require(pair.reference.counts==pair.selection.counts && pair.reference.statistics==pair.selection.statistics,"GPU compact statistics differ from CPU");
    if(out) *out=std::move(b);return error;
}
void criteria() {
    for(double contrast:{1.,10.}) for(bool mixed:{false,true}) {
        const GasLayout layout=mixed?GasLayout{8,{{{8,8,48,48}},{{16,16,32,32}},{{20,20,24,24}}}}:GasLayout{};
        Pair pair(64,contrast,false,layout);double worst=0.;
        for(int material=0;material<5;++material) for(int velocity=0;velocity<5;++velocity) {
            pair.reference.reset();pair.selection.reset();prescribe(pair,material,velocity,material%3);
            try { worst=std::max(worst,compare(pair,{.137,.021,.053,.037},{.5,0.},0.)); }
            catch(...) { std::cerr<<"case contrast="<<contrast<<" mixed="<<mixed<<" material="<<material<<" velocity="<<velocity<<'\n';throw; }
        }
        std::cout<<"criteria contrast="<<contrast<<" mixed="<<mixed<<" representation error="<<worst<<'\n';
    }
}
void independentFacesAndLiquid() {
    Pair faces(64,1.,false,{});prescribe(faces,0,3);Preview::Trace trace;
    compare(faces,{.1,.1,.02,.02},{1.,0.},0.,0.,32,&trace);
    require(trace.levels.front().cells[10*32+10].error.velocity>.7,"Opposing MAC faces disappeared in cell momentum fit");
    require(faces.selection.proposeLayout(faces.gpu)==GasLayout{},"Broad detail coverage did not use the same base fallback");
    Pair wet(64,1.,true,{});prescribe(wet,0,0,0,true);compare(wet,{10.,10.,1.,1.},{1.,0.},0.,0.,32,&trace);
    const auto layout=wet.selection.proposeLayout(wet.gpu);wet.gpu.regrid(layout);
    require(trace.levels.front().cells[15*32+10].source==Preview::LiquidBuffer,"Liquid did not retain finest cells and buffer");
    std::cout<<"independent_faces_and_liquid PASS\n";
}
void history() {
    Pair pair(128,1.,false,{2,{{{64,64,16,16}}}});prescribe(pair,0,0);Preview::Trace trace;
    for(double time:{0.,0.,.16,.32}) compare(pair,{},{1.,.24},time,0.,32,&trace);
    const int changed=2*4+2,unchanged=0;require(trace.levels.back().cells[changed].decision==Preview::Candidate,"Deep group did not settle");
    require(pair.cpu.regrid({2,{}}) && pair.gpu.regrid({2,{}}),"History geometry change failed");prescribe(pair,0,0);
    compare(pair,{},{1.,.24},.4,0.,32,&trace);
    require(trace.levels.back().cells[changed].decision==Preview::Detail && trace.levels.back().cells[unchanged].decision==Preview::Candidate,"Exact deep footprint reset changed the wrong histories");
    compare(pair,{},{1.,.24},.8);require(pair.selection.proposeLayout(pair.gpu)==GasLayout{32,{}},"Quiet selection failed to reach maximum coarse size");
    // Repeated quiet retention must not create new padding on either side of a regrid.
    for(int tick=0;tick<6;++tick) {
        compare(pair,{},{.5,.24},1.+.16*tick);const auto layout=pair.selection.proposeLayout(pair.gpu);pair.cpu.regrid(layout);pair.gpu.regrid(layout);prescribe(pair,0,0);
    }
    require(pair.gpu.layout()==GasLayout{32,{}},"Quiet padding propagated through history");std::cout<<"history footprint_reset quiet_delay padding PASS\n";
    compare(pair,{},{1.,.24},2.,0.,2);compare(pair,{},{1.,.24},2.32,0.,2);
}
void lookahead() {
    Pair pair(64,1.,false,{8,{{{8,8,48,48}},{{16,16,32,32}},{{20,20,24,24}}}});prescribe(pair,4,1);Preview::Trace trace;
    compare(pair,{10.,1.,.02,.02},{.5,.24},0.,4.,32,&trace);int approaching=0;
    for(const auto& lev:trace.levels) for(const auto& c:lev.cells) approaching+=c.raw==Preview::Approaching;
    require(approaching>0,"Swept velocity/acceleration lookahead did not reach coarse cells");
    compare(pair,{10.,1.,.02,.02},{.5,.24},.16,4.,32);compare(pair,{10.,1.,.02,.02},{.5,.24},.48,4.,32);
    std::cout<<"lookahead approaching="<<approaching<<" PASS\n";
}
void compensatedPressure() {
    for(bool mixed:{false,true}) {
        Pair pair(64,1.,false,mixed?GasLayout{8,{{{8,8,48,48}},{{16,16,32,32}},{{20,20,24,24}}}}:GasLayout{});
        for(int pattern=0;pattern<4;++pattern) {
            prescribe(pair,0,0);pair.reference.reset();pair.selection.reset();double lowMaximum=0.;
            for(size_t lev=0;lev<pair.gpu.levels().size();++lev) {
                const auto& g=*pair.gpu.levels()[lev];auto& c=*pair.cpu.levels()[lev];const double scale=g.geometry.cell_size;
                require(g.pressure.channels==2,"Pressure fixture requires the compensated high/low field");
                const int lowOffset=g.pressure.width*g.pressure.height;std::vector<float> values(g.pressure.bytes()/sizeof(float));
                for(size_t patch=0;patch<g.pressureLayout.patches.size();++patch) {
                    const auto& b=g.pressureLayout.patches[patch];const auto cp=c.pressure.array(int(patch));
                    forCells(b.box,[&](int x,int y) {
                        const double fx=(x+.5)*scale,fy=(y+.5)*scale;
                        const double detail=pattern==0?fx/4096.+fy/2048.:pattern==1?(fx-32.)*(fy-24.)/4096.:
                            pattern==2?.25*fx-.5*fy+((x+y)%2?1.:-1.)/8192.:((x+y)%2?1.:-1.)/1024.;
                        const double impulse=pair.cpu.pressureInterval()*(1048576.+detail);
                        const float high=float(impulse),low=float(impulse-double(high));const int i=b.offset+b.index(x,y);
                        values[i]=high;values[lowOffset+i]=low;cp(x,y)=double(high)+double(low);lowMaximum=std::max(lowMaximum,std::abs(double(low)));
                    });
                }
                pair.gpu.commands().write(g.pressure,values);
            }
            require(lowMaximum>0.,"Compensated pressure fixture did not exercise low bits");
            const Preview::Tolerances tolerances{10.,.00037,1.,1.};Preview::Trace observed,expected;
            compare(pair,tolerances,{.5,0.},0.,0.,32,&observed);
            Preview oracle;oracle.update(pair.cpu,tolerances,{.5,0.},0.,0.,32,&expected);double error=0.;int detailCount=0;
            for(size_t lev=0;lev<observed.levels.size();++lev) for(size_t i=0;i<observed.levels[lev].cells.size();++i) {
                const auto& a=expected.levels[lev].cells[i];const auto& b=observed.levels[lev].cells[i];
                error=std::max(error,std::abs(a.error.pressure_gradient-b.error.pressure_gradient));detailCount+=b.source==Preview::Detail || b.source==Preview::Refine;
            }
            require(error<2e-6,"Compensated pressure gradient lost its small resolved detail");
            if(pattern==3) require(detailCount>0,"A low-plane-only pressure pattern failed to request refinement");
            std::cout<<"compensated_pressure mixed="<<mixed<<" pattern="<<pattern<<" gradient_error="<<error<<" detail="<<detailCount<<'\n';
        }
    }
}
void benchmark(int n) {
    Pair pair(n,1.,false,{});prescribe(pair,2,4);
    const auto run=[&](double time) { const auto start=std::chrono::steady_clock::now();pair.selection.update(pair.gpu,{.137,.021,.053,.037},{.5,.24},time,.32,32);
        pair.selection.proposeLayout(pair.gpu);return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); };
    const double first=run(0.);std::vector<double> elapsed;for(int i=0;i<7;++i) elapsed.push_back(run(.32*(i+1)));std::sort(elapsed.begin(),elapsed.end());
    pair.selection.reset();const double reset=run(3.);
    std::cout<<"selection timing n="<<n<<" first_ms="<<first<<" reset_ms="<<reset<<" median_ms="<<elapsed[3]<<" range="<<elapsed.front()<<','<<elapsed.back()<<" includes_native_snapshot_masks_clustering excludes_setup_trace_display\n";
}
void roundoff() {
    Pair pair(64,1.,false,{});prescribe(pair,1,4,1,false,false);Preview::Trace trace;
    pair.selection.update(pair.gpu,{.137,.021,.053,.037},{.5,0.},0.,0.,32,&trace);
    FluidCellView original(pair.cpu);FluidCellView::MaterialPlane plane;original.reconstruct(plane,0,8,8,true);
    std::cout<<"original CPU velocity estimate="<<Preview::estimate(pair.cpu,0,8,8).velocity<<" GPU estimate="<<trace.levels[2].cells[8].error.velocity<<'\n';
    for(int d=0;d<2;++d) std::cout<<"CPU momentum axis="<<d<<" mean="<<plane.mean(2+d)<<" slope="<<plane.slopes[2+d].x<<','<<plane.slopes[2+d].y<<'\n';
    pair.gpu.regrid({8,{}});pair.gpu.regrid({1,{}});double local=0.,global=0.;
    for(const auto& entry:pair.gpu.levels()) {
        const auto& level=*entry;const auto values=pair.gpu.commands().read(level.conserved);
        for(const auto& patch:level.qLayout.patches) forCells(patch.box,[&](int x,int y) {
            const auto c=original.at(x,y);double rho=0.;for(int k=0;k<2;++k) rho+=values[patch.offset+patch.index(x,y)+k*patch.plane];
            for(int d=0;d<2;++d) { const double v=values[patch.offset+patch.index(x,y)+(2+d)*patch.plane]/rho;const double error=std::abs(v-c.velocity(d));
                global=std::max(global,error);if(x<8 && y>=8 && y<16) local=std::max(local,error); }
        });
    }
    std::cout<<"GPU actual 1->8->1 momentum velocity loss local="<<local<<" global="<<global<<'\n';
    require(std::abs(local-trace.levels[2].cells[8].error.velocity)<3e-6,"Float near-flat estimate did not match actual GPU reconstruction loss");
}
}
int main(int argc,char** argv) {
    try {
        std::cout<<std::unitbuf;std::cerr<<std::unitbuf;bl::Thread::setMaxThreads(8);const std::string mode=argc>1?argv[1]:"all";
        if(mode=="all" || mode=="criteria") criteria();if(mode=="all" || mode=="faces") independentFacesAndLiquid();
        if(mode=="all" || mode=="history") history();if(mode=="all" || mode=="lookahead") lookahead();
        if(mode=="all" || mode=="pressure") compensatedPressure();
        if(mode=="benchmark") benchmark(argc>2?std::stoi(argv[2]):256);
        if(mode=="all" || mode=="roundoff") roundoff();
        std::cout<<"PASS WebGPU automatic selection "<<mode<<'\n';return 0;
    } catch(const std::exception& error) { std::cerr<<"FAIL "<<error.what()<<'\n';return 1; }
}
