#include "CaseOracle.h"
#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_GMRES_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>
#include <iomanip>
#include <optional>

using namespace amrex;
using fluid::reference::CaseOracle;
using fluid::reference::PressureCase;
using fluid::reference::Clock;
using fluid::reference::milliseconds;

namespace {
struct Native
{
    Vector<Geometry> geometry;
    Vector<BoxArray> boxes;
    Vector<DistributionMapping> mapping;
    Vector<MultiFab> pressure, rhs, cycleRhs, output;
    Vector<Array<MultiFab, 2>> beta;
    std::unique_ptr<MLABecLaplacian> op;
    std::unique_ptr<MLMG> solver;
    std::unique_ptr<GMRESMLMG> gmres;
    const CaseOracle& oracle;

    explicit Native(const CaseOracle& input) : oracle(input)
    {
        const auto& levels = oracle.problem.levels;
        RealBox physical({Real(0), Real(0)}, {Real(levels[0].width * levels[0].cell_size), Real(levels[0].height * levels[0].cell_size)});
        Array<int, 2> periodic{0, 0};
        for (const auto& level : levels)
        {
            geometry.emplace_back(Box(IntVect(0, 0), IntVect(level.width - 1, level.height - 1)), &physical, 0, periodic.data());
            Vector<Box> regions;
            for (const auto& b : level.boxes) regions.emplace_back(IntVect(b.x, b.y), IntVect(b.x + b.width - 1, b.y + b.height - 1));
            boxes.emplace_back(regions.data(), int(regions.size())); mapping.emplace_back(boxes.back());
            pressure.emplace_back(boxes.back(), mapping.back(), 1, 1);
            rhs.emplace_back(boxes.back(), mapping.back(), 1, 0); cycleRhs.emplace_back(boxes.back(), mapping.back(), 1, 0);
            output.emplace_back(boxes.back(), mapping.back(), 1, 1); beta.emplace_back();
            for (int d = 0; d < 2; ++d) beta.back()[d].define(convert(boxes.back(), IntVect::TheDimensionVector(d)), mapping.back(), 1, 0);
        }
        op = std::make_unique<MLABecLaplacian>(geometry, boxes, mapping);
        op->setScalars(0., 1.);
        Array<LinOpBCType, 2> wall{LinOpBCType::Neumann, LinOpBCType::Neumann}; op->setDomainBC(wall, wall);
        for (int l = 0; l < int(levels.size()); ++l)
        {
            op->setLevelBC(l, nullptr);
            for (int d = 0; d < 2; ++d) uploadLevel(l, oracle.problem.beta[d][l], beta[l][d]);
            op->setBCoeffs(l, GetArrOfConstPtrs(beta[l]));
        }
        upload(oracle.problem.rhs, rhs); upload(oracle.cycleRhs, cycleRhs);
        solver = std::make_unique<MLMG>(*op); solver->setVerbose(0); solver->setBottomSolver(BottomSolver::smoother); solver->setThrowException(true);
        gmres = std::make_unique<GMRESMLMG>(*solver); gmres->setMaxIters(oracle.problem.maxIterations);
        Gpu::streamSynchronize();
    }

    void uploadLevel(int l, const std::vector<double>& values, MultiFab& target)
    {
        for (MFIter mfi(target); mfi.isValid(); ++mfi)
        {
            const auto& patch = oracle.topology.grids[l].patches[mfi.index()];
            FArrayBox host(target[mfi].box(), 1, The_Pinned_Arena()); std::fill_n(host.dataPtr(), host.nBytes() / sizeof(Real), Real(0));
            auto a = host.array();
            for (BoxIterator it(mfi.validbox()); it.ok(); ++it) a(it(), 0) = Real(values[patch.index(it()[0], it()[1])]);
            Gpu::htod_memcpy(target[mfi].dataPtr(), host.dataPtr(), target[mfi].nBytes());
        }
    }
    void upload(const PressureCase::Field& values, Vector<MultiFab>& target)
    {
        for (int l = 0; l < int(target.size()); ++l) uploadLevel(l, values[l], target[l]);
    }
    PressureCase::Field download(const Vector<MultiFab>& source)
    {
        auto result = oracle.cpu.makeField();
        for (int l = 0; l < int(source.size()); ++l)
            for (MFIter mfi(source[l]); mfi.isValid(); ++mfi)
            {
                const auto& patch = oracle.topology.grids[l].patches[mfi.index()];
                FArrayBox host(source[l][mfi].box(), 1, The_Pinned_Arena());
                Gpu::dtoh_memcpy(host.dataPtr(), source[l][mfi].dataPtr(), source[l][mfi].nBytes());
                auto a = host.const_array();
                for (BoxIterator it(mfi.validbox()); it.ok(); ++it) result[l][patch.index(it()[0], it()[1])] = double(a(it(), 0));
            }
        return result;
    }
};
}

int main(int argc, char** argv)
{
    try
    {
        if (argc < 2) throw std::runtime_error("Usage: FluidPressureNativeReference case.fpc [repeats]");
        std::cout << std::setprecision(10) << std::unitbuf;
        const auto problem = PressureCase::load(argv[1]); const int repeats = argc > 2 ? std::stoi(argv[2]) : 3;
        const bool asynchronous = argc < 4 || std::string(argv[3]) != "synchronized";
        // The pressure-only utility needs no RNG, signal handlers or asynchronous I/O.
        Init_minimal();
        std::cout << "gpu_launch_region=" << Gpu::inLaunchRegion() << ",device=" << Gpu::Device::deviceName() << char(10);
        {
            CaseOracle oracle(problem);
            std::cout << "backend=AMReX_HIP,precision_bits=" << 8 * sizeof(Real) << ",levels=" << problem.levels.size() << ",capture_solve=" << problem.solveIndex << '\n';
            auto start = Clock::now(); Native native(oracle);
            std::cout << "setup_upload_ms=" << milliseconds(start) << '\n';
            // AMReX supports a single-stream region without an MFIter wait after every operation.
            Gpu::SingleStreamRegion singleStream;
            std::optional<Gpu::NoSyncRegion> noSync;
            if (asynchronous) noSync.emplace();
            std::cout << "single_stream_no_implicit_sync=" << asynchronous << char(10);

            native.gmres->precond(native.output, native.cycleRhs); Gpu::streamSynchronize();
            oracle.reportCycle(native.download(native.output));
            for (int sample = 0; sample < repeats; ++sample)
            {
                start = Clock::now();
                for (int i = 0; i < 8; ++i) native.gmres->precond(native.output, native.cycleRhs);
                const double issue = milliseconds(start); const auto drain = Clock::now();
                Gpu::streamSynchronize();
                std::cout << "cycle,sample=" << sample << ",issue_ms=" << issue / 8 << ",drain_ms=" << milliseconds(drain) / 8 << ",ms=" << milliseconds(start) / 8 << '\n';
            }
            for (int sample = -1; sample < repeats; ++sample)
            {
                native.upload(problem.pressure, native.pressure); Gpu::streamSynchronize(); start = Clock::now();
                native.gmres->solve(GetVecOfPtrs(native.pressure), GetVecOfConstPtrs(native.rhs), Real(0), Real(problem.tolerance));
                Gpu::streamSynchronize(); const double elapsed = milliseconds(start);
                if (sample < 0) continue;
                std::cout << "solve,sample=" << sample << ",ms=" << elapsed << ",iterations=" << native.gmres->getNumIters() << '\n';
                oracle.reportResidual(native.download(native.pressure));
            }
        }
        Finalize_minimal(); return 0;
    }
    catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
