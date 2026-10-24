#include "AmrPressure2D.h"
#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_GMRES_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <omp.h>

using Pressure = fluid::AmrPressure2D;
using Clock = std::chrono::steady_clock;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Reference
{
    amrex::Vector<amrex::Geometry> geometry;
    amrex::Vector<amrex::BoxArray> boxes;
    amrex::Vector<amrex::DistributionMapping> mapping;
    amrex::Vector<amrex::MultiFab> pressure, rhs, output;
    amrex::Vector<amrex::Array<amrex::MultiFab, 2>> beta, flux;
    std::unique_ptr<amrex::MLABecLaplacian> op;
    std::unique_ptr<amrex::MLMG> solver;

    Reference(const std::vector<Pressure::Level>& levels)
    {
        using namespace amrex;
        RealBox physical({0., 0.}, {levels[0].width * levels[0].cell_size, levels[0].height * levels[0].cell_size});
        Array<int, 2> periodic{0, 0};
        for (const auto& level : levels)
        {
            Box domain(IntVect(0, 0), IntVect(level.width - 1, level.height - 1));
            geometry.emplace_back(domain, &physical, 0, periodic.data());
            Vector<Box> regions;
            for (const auto& b : level.boxes)
                regions.emplace_back(IntVect(b.x, b.y), IntVect(b.x + b.width - 1, b.y + b.height - 1));
            boxes.emplace_back(regions.data(), int(regions.size()));
            mapping.emplace_back(boxes.back());
            pressure.emplace_back(boxes.back(), mapping.back(), 1, 1);
            rhs.emplace_back(boxes.back(), mapping.back(), 1, 0);
            output.emplace_back(boxes.back(), mapping.back(), 1, 0);
            beta.emplace_back();
            flux.emplace_back();
            for (int d = 0; d < 2; ++d)
            {
                auto faces = convert(boxes.back(), IntVect::TheDimensionVector(d));
                beta.back()[d].define(faces, mapping.back(), 1, 0);
                flux.back()[d].define(faces, mapping.back(), 1, 0);
            }
            pressure.back().setVal(0.);
        }
        op = std::make_unique<MLABecLaplacian>(geometry, boxes, mapping);
        op->setScalars(0., 1.);
        Array<LinOpBCType, 2> wall{LinOpBCType::Neumann, LinOpBCType::Neumann};
        op->setDomainBC(wall, wall);
        for (int l = 0; l < int(levels.size()); ++l)
            op->setLevelBC(l, nullptr);
        solver = std::make_unique<MLMG>(*op);
        solver->setVerbose(0);
        solver->setBottomSolver(BottomSolver::smoother);
        solver->setThrowException(true);
    }

    void upload(const Pressure& custom, const Pressure::Field& values, amrex::Vector<amrex::MultiFab>& target)
    {
        for (int l = 0; l < int(target.size()); ++l)
            for (amrex::MFIter mfi(target[l]); mfi.isValid(); ++mfi)
            {
                const auto& p = custom.patches(l)[mfi.index()];
                auto a = target[l].array(mfi);
                for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it)
                    a(it(), 0) = values[l][p.index(it()[0], it()[1])];
            }
    }

    double error(const Pressure& custom, const Pressure::Field& values, const amrex::Vector<amrex::MultiFab>& target)
    {
        double error = 0.;
        for (int l = 0; l < int(target.size()); ++l)
            for (amrex::MFIter mfi(target[l]); mfi.isValid(); ++mfi)
            {
                const auto& p = custom.patches(l)[mfi.index()];
                auto a = target[l].const_array(mfi);
                for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it)
                    error = std::max(error, std::abs(a(it(), 0) - values[l][p.index(it()[0], it()[1])]));
            }
        return error;
    }
};

std::vector<Pressure::Level> layout(int n, int kind)
{
    if (kind == 0) return {{n, n, 1., {{0, 0, n, n}}}};
    std::vector<Pressure::Level> levels{{n / 2, n / 2, 2., {{0, 0, n / 2, n / 2}}}};
    std::vector<Pressure::Box> fine;
    if (kind == 1) fine = {{n / 4, n / 4, n / 2, n / 2}};
    if (kind == 2) fine = {{0, 0, n / 2, n / 2}, {n / 2, 0, n / 4, n / 4}};
    if (kind == 3)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                if ((x + y) % 2 == 0) fine.push_back({x * n / 4, y * n / 4, n / 4, n / 4});
    levels.push_back({n, n, 1., fine});
    if (kind == 4 || kind == 5)
    {
        levels[1].boxes = {{n / 4, n / 4, n / 2, n / 2}};
        levels.push_back({2 * n, 2 * n, .5, {{3 * n / 4, 3 * n / 4, n / 2, n / 2}}});
        if (kind == 5)
            for (int l = 3; l < 6; ++l)
            {
                int width = levels.back().width * 2;
                int origin = (width - n / 2) / 2;
                levels.push_back({width, width, levels.back().cell_size / 2., {{origin, origin, n / 2, n / 2}}});
            }
    }
    if (kind == 6)
        for (int y = 0; y < n; y += 4)
            for (int x = 0; x < n; x += 4)
                if ((7 * (x / 4) + 11 * (y / 4)) % 5 < 2) levels[1].boxes.push_back({x, y, 4, 4});
    return levels;
}

void runCase(int n, int kind, double contrast, bool benchmark, bool tiled)
{
    auto levels = layout(n, kind);
    if (tiled)
        for (auto& level : levels)
        {
            std::vector<Pressure::Box> boxes;
            for (const auto& b : level.boxes)
                for (int y = b.y; y < b.y + b.height; y += 64)
                    for (int x = b.x; x < b.x + b.width; x += 64)
                        boxes.push_back({x, y, std::min(64, b.x + b.width - x), std::min(64, b.y + b.height - y)});
            level.boxes = std::move(boxes);
        }
    auto begin = Clock::now();
    Pressure custom(levels);
    double setup_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    Reference reference(levels);
    auto phi = custom.makeField(), result = custom.makeField();
    Pressure::Faces coefficients{custom.makeField(), custom.makeField()};
    for (int l = 0; l < int(levels.size()); ++l)
    {
        double dx = levels[l].cell_size;
        for (int pindex = 0; pindex < int(custom.patches(l).size()); ++pindex)
        {
            const auto& p = custom.patches(l)[pindex];
            for (int y = p.box.y; y < p.box.y + p.box.height; ++y)
                for (int x = p.box.x; x < p.box.x + p.box.width; ++x)
                    phi[l][p.index(x, y)] = std::sin(.11 * (x + .5) * dx) * std::cos(.07 * (y + .5) * dx) +
                                            .01 * std::sin(1.7 * x + 2.3 * y);
            auto density = [&](double x, double y)
            {
                bool dense = kind == 6 ? std::sin(1.7 * x + 2.3 * y) > 0. : y + .3 * x < .6 * n;
                return (dense ? contrast : 1.) * (1. + .1 * std::cos(.03 * x));
            };
            for (int d = 0; d < 2; ++d)
            {
                auto a = reference.beta[l][d][pindex].array();
                for (int y = p.box.y; y < p.box.y + p.box.height + (d == 1); ++y)
                    for (int x = p.box.x; x < p.box.x + p.box.width + (d == 0); ++x)
                    {
                        double px = (x + .5) * dx, py = (y + .5) * dx;
                        double coefficient = 2. / (density(px, py) + density(px - (d == 0) * dx, py - (d == 1) * dx));
                        coefficients[d][l][p.index(x, y)] = coefficient;
                        a(x, y, 0) = coefficient;
                    }
            }
        }
        reference.op->setBCoeffs(l, amrex::GetArrOfConstPtrs(reference.beta[l]));
    }
    custom.setCoefficients(coefficients);
    reference.upload(custom, phi, reference.pressure);
    reference.solver->apply(amrex::GetVecOfPtrs(reference.rhs), amrex::GetVecOfPtrs(reference.pressure));
    custom.apply(result, phi);
    double operator_error = reference.error(custom, result, reference.rhs);
    std::cout << "operator,n=" << n << ",layout=" << kind << ",contrast=" << contrast << ",tiled=" << tiled
              << ",max_error=" << operator_error << std::endl;
    require(operator_error < 2.e-11, "Extracted composite operator differs from upstream");

    auto rhs = result;
    amrex::GMRESMLMG oracle(*reference.solver);
    oracle.precond(reference.output, reference.rhs);
    custom.precondition(result, rhs);
    double cycle_error = reference.error(custom, result, reference.output);
    std::cout << "cycle,max_error=" << cycle_error << std::endl;
    require(cycle_error < 2.e-9, "Extracted multigrid cycle differs from upstream");

    constexpr double tolerance = 1.e-8;
    auto solved = custom.makeField();
    for (auto& p : reference.pressure)
        p.setVal(0.);
    auto start = Clock::now();
    auto status = custom.solve(solved, rhs, tolerance);
    double own_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    start = Clock::now();
    oracle.solve(amrex::GetVecOfPtrs(reference.pressure), amrex::GetVecOfConstPtrs(reference.rhs), 0., tolerance);
    double reference_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << "solve,own_iterations=" << status.iterations << ",reference_iterations=" << oracle.getNumIters()
              << ",residual=" << status.residual << ",own_ms=" << own_ms << ",reference_ms=" << reference_ms
              << ",setup_ms=" << setup_ms << std::endl;
    require(status.converged, "Extracted pressure solve failed to converge");
    Pressure::Faces own_flux{custom.makeField(), custom.makeField()};
    custom.fluxes(own_flux, solved);
    amrex::Vector<amrex::Array<amrex::MultiFab*, 2>> flux;
    for (auto& f : reference.flux)
        flux.push_back(amrex::GetArrOfPtrs(f));
    reference.solver->prepareForFluxes(amrex::GetVecOfConstPtrs(reference.pressure));
    reference.solver->getFluxes(flux, amrex::GetVecOfPtrs(reference.pressure), amrex::MLMG::Location::FaceCenter);
    for (int l = int(levels.size()) - 1; l > 0; --l)
        amrex::average_down_faces(amrex::GetArrOfConstPtrs(reference.flux[l]),
                                  amrex::GetArrOfPtrs(reference.flux[l - 1]), amrex::IntVect(2), 0);
    double flux_error = 0.;
    for (int l = 0; l < int(levels.size()); ++l)
        for (int d = 0; d < 2; ++d)
            for (amrex::MFIter mfi(reference.flux[l][d]); mfi.isValid(); ++mfi)
            {
                auto a = reference.flux[l][d].const_array(mfi);
                const auto& p = custom.patches(l)[mfi.index()];
                for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it)
                    flux_error = std::max(flux_error, std::abs(a(it(), 0) - own_flux[d][l][p.index(it()[0], it()[1])]));
            }
    std::cout << "flux,max_error=" << flux_error << std::endl;
    require(flux_error < 2.e-8, "Extracted pressure correction differs from upstream");
    auto repeated = custom.solve(solved, rhs, tolerance);
    require(repeated.converged && repeated.iterations == 0, "Converged warm pressure was not retained");
    auto zero = custom.makeField();
    require(custom.solve(zero, zero, tolerance).iterations == 0, "Zero forcing should not iterate");
    auto limited = custom.makeField();
    auto limited_status = custom.solve(limited, rhs, 1.e-12, 1);
    require(!limited_status.converged && limited_status.iterations == 1, "Iteration cap did not report nonconvergence");
    if (benchmark)
        for (int repeat = 0; repeat < 4; ++repeat)
        {
            auto own = [&]
            {
                for (auto& p : solved)
                    std::fill(p.begin(), p.end(), 0.);
                auto t = Clock::now();
                status = custom.solve(solved, rhs, 1.e-5);
                return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
            };
            auto upstream = [&]
            {
                for (auto& p : reference.pressure)
                    p.setVal(0.);
                auto t = Clock::now();
                oracle.solve(amrex::GetVecOfPtrs(reference.pressure), amrex::GetVecOfConstPtrs(reference.rhs), 0.,
                             1.e-5);
                return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
            };
            if (repeat % 2)
            {
                reference_ms = upstream();
                own_ms = own();
            }
            else
            {
                own_ms = own();
                reference_ms = upstream();
            }
            require(status.converged, "Benchmark did not converge");
            std::cout << "timing,repeat=" << repeat << ",own_ms=" << own_ms << ",reference_ms=" << reference_ms
                      << ",own_iterations=" << status.iterations << ",reference_iterations=" << oracle.getNumIters()
                      << std::endl;
        }
}
} // namespace

int main(int argc, char** argv)
{
    int n = argc > 1 ? std::stoi(argv[1]) : 64;
    bool benchmark = argc > 1;
    bool tiled = argc > 2;
    amrex::Initialize(argc, argv, false);
    omp_set_num_threads(8);
    int result = 0;
    try
    {
        std::cout << std::setprecision(12);
        for (double contrast : {1., 1000.})
            for (int kind = 0; kind < (benchmark ? 4 : 7); ++kind)
                runCase(n, kind, contrast, benchmark, tiled);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        result = 1;
    }
    amrex::Finalize();
    return result;
}
