#include "CaseOracle.h"
#include "AmrPressureGPU.h"
#include <iomanip>

using fluid::reference::CaseOracle;
using fluid::reference::PressureCase;
using fluid::reference::Clock;
using fluid::reference::milliseconds;
using GPU=fluid::AmrPressureGPU;

namespace {
void upload(fluid::webgpu::Compute& compute,GPU::Field& target,const PressureCase::Field& source)
{
    for(int l=0;l<int(target.size());++l)
    {
        const int plane=target[l].width*target[l].height;
        std::vector<float> values(plane*target[l].channels);
        for(int i=0;i<int(source[l].size());++i) {
            values[i]=float(source[l][i]);
            if(target[l].channels==2)values[plane+i]=float(source[l][i]-double(values[i]));
        }
        compute.write(target[l],values);
    }
}
PressureCase::Field download(fluid::webgpu::Compute& compute,const GPU& solver,const GPU::Field& source)
{
    PressureCase::Field result(source.size());
    for(int l=0;l<int(source.size());++l)
    {
        auto values=compute.read(source[l]);const int plane=source[l].width*source[l].height;
        result[l].resize(solver.topology().grids[l].size);
        for(int i=0;i<int(result[l].size());++i)result[l][i]=double(values[i])+(source[l].channels==2?double(values[plane+i]):0.);
    }
    return result;
}
}

int main(int argc,char** argv)
{
    try
    {
        if(argc<2)throw std::runtime_error("Usage: FluidPressureWebGPUReference case.fpc [repeats] [portable]");
        std::cout<<std::setprecision(10)<<std::unitbuf;
        CaseOracle oracle(PressureCase::load(argv[1]));const auto& problem=oracle.problem;
        const int repeats=argc>2?std::stoi(argv[2]):3;
        auto start=Clock::now();
        fluid::webgpu::Compute compute(argc<4||std::string(argv[3])!="portable");
        std::cout<<"backend=WebGPU,subgroups="<<compute.gpu.device().HasFeature(wgpu::FeatureName::Subgroups)
                 <<",device_ms="<<milliseconds(start)<<",levels="<<problem.levels.size()<<",capture_solve="<<problem.solveIndex<<'\n';
        start=Clock::now();GPU solver(compute,problem.levels);
        auto pressure=solver.makeField(true),rhs=solver.makeField(),cycleRhs=solver.makeField(),output=solver.makeField();
        GPU::Faces beta{solver.makeField(),solver.makeField()};
        upload(compute,rhs,problem.rhs);upload(compute,cycleRhs,oracle.cycleRhs);
        for(int d=0;d<2;++d)upload(compute,beta[d],problem.beta[d]);
        solver.setCoefficients(beta);compute.read(rhs[0],0,sizeof(float));
        std::cout<<"setup_upload_ms="<<milliseconds(start)<<'\n';
        solver.precondition(output,cycleRhs);oracle.reportCycle(download(compute,solver,output));
        for(int sample=0;sample<repeats;++sample)
        {
            start=Clock::now();
            for(int i=0;i<8;++i)solver.precondition(output,cycleRhs);
            const double issue=milliseconds(start);const auto drain=Clock::now();
            compute.read(output[0],0,sizeof(float));
            std::cout<<"cycle,sample="<<sample<<",issue_ms="<<issue/8<<",drain_ms="<<milliseconds(drain)/8<<",ms="<<milliseconds(start)/8<<'\n';
        }
        for(int sample=-1;sample<repeats;++sample)
        {
            upload(compute,pressure,problem.pressure);compute.read(pressure[0],0,sizeof(float));
            start=Clock::now();const auto status=solver.solve(pressure,rhs,problem.tolerance,problem.maxIterations);
            const double elapsed=milliseconds(start);
            if(sample<0)continue;
            std::cout<<"solve,sample="<<sample<<",ms="<<elapsed<<",iterations="<<status.iterations<<",reported_residual="<<status.residual
                     <<",reported_converged="<<status.converged<<'\n';
            oracle.reportResidual(download(compute,solver,pressure));
        }
        return 0;
    }
    catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
