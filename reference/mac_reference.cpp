#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <hydro_MacProjector.H>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace amrex;
using Clock = std::chrono::steady_clock;

struct Hierarchy {
    Vector<Geometry> geometry;
    Vector<BoxArray> grids;
    Vector<DistributionMapping> mapping;
    Vector<Array<MultiFab, 2>> velocity, initial, expected, beta;
    Vector<MultiFab> divergence;
    Vector<iMultiFab> active;
    Vector<Box> patches;
    long long activeCells = 0, allocatedCells = 0;

    Hierarchy(int finest, const std::string& layout, int maxGrid) {
        const int levels = layout == "uniform" ? 1 : 2;
        geometry.resize(levels); grids.resize(levels); mapping.resize(levels);
        velocity.resize(levels); initial.resize(levels); expected.resize(levels); beta.resize(levels);
        divergence.resize(levels); active.resize(levels);
        RealBox physical({0., 0.}, {1., 1.});
        Array<int, 2> periodic{0, 0};
        for (int level = 0; level < levels; ++level) {
            const int n = levels == 2 && level == 0 ? finest / 2 : finest;
            Box domain(IntVect(0, 0), IntVect(n - 1, n - 1));
            geometry[level].define(domain, &physical, 0, periodic.data());
            if (level == 0) grids[level].define(domain);
            else {
                if (layout == "patch") patches.emplace_back(IntVect(n / 4, n / 4), IntVect(3 * n / 4 - 1, 3 * n / 4 - 1));
                else {
                    for (int y : {n / 8, 5 * n / 8}) for (int x : {n / 8, 5 * n / 8})
                        patches.emplace_back(IntVect(x, y), IntVect(x + n / 4 - 1, y + n / 4 - 1));
                }
                grids[level] = BoxArray(patches.data(), static_cast<int>(patches.size()));
            }
            grids[level].maxSize(maxGrid);
            mapping[level].define(grids[level]);
            allocatedCells += grids[level].numPts();
            divergence[level].define(grids[level], mapping[level], 1, 0);
            for (int d = 0; d < 2; ++d) {
                const auto faces = amrex::convert(grids[level], IntVect::TheDimensionVector(d));
                velocity[level][d].define(faces, mapping[level], 1, 0);
                initial[level][d].define(faces, mapping[level], 1, 0);
                expected[level][d].define(faces, mapping[level], 1, 0);
                beta[level][d].define(faces, mapping[level], 1, 0);
            }
        }
        for (int level = 0; level < levels; ++level) {
            if (level + 1 < levels) active[level] = makeFineMask(grids[level], mapping[level], grids[level + 1], IntVect(2), 1, 0);
            else { active[level].define(grids[level], mapping[level], 1, 0); active[level].setVal(1); }
            activeCells += grids[level].numPts();
            if (level + 1 < levels) activeCells -= grids[level + 1].numPts() / 4;
        }
    }

    void averageDown(Vector<Array<MultiFab, 2>>& fields) {
        for (int level = static_cast<int>(geometry.size()) - 1; level > 0; --level)
            average_down_faces(GetArrOfConstPtrs(fields[level]), GetArrOfPtrs(fields[level - 1]), IntVect(2), geometry[level - 1]);
    }

    void restore() {
        for (int level = 0; level < geometry.size(); ++level) for (int d = 0; d < 2; ++d)
            MultiFab::Copy(velocity[level][d], initial[level][d], 0, 0, 1, 0);
    }
};

double psi(double x, double y) {
    constexpr double pi = 3.14159265358979323846;
    const double sx = std::sin(pi * x), sy = std::sin(pi * y);
    return 0.1 * sx * sx * sy * sy;
}

void initialize(Hierarchy& hierarchy, const std::string& test) {
    const bool curl = test == "curl" || test == "curl_gradient";
    const bool variable = test == "variable_gradient";
    const bool quadratic = test == "quadratic_gradient";
    for (int level = 0; level < hierarchy.geometry.size(); ++level) {
        const int n = hierarchy.geometry[level].Domain().length(0);
        const double h = 1. / n;
        for (int d = 0; d < 2; ++d) {
            for (MFIter mfi(hierarchy.velocity[level][d]); mfi.isValid(); ++mfi) {
                auto u = hierarchy.velocity[level][d].array(mfi);
                auto b = hierarchy.beta[level][d].array(mfi);
                auto target = hierarchy.expected[level][d].array(mfi);
                for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                    const auto iv = bit();
                    const int i = iv[0], j = iv[1];
                    const double x = (i + (d == 0 ? 0. : 0.5)) * h;
                    const double y = (j + (d == 1 ? 0. : 0.5)) * h;
                    b(i, j, 0) = variable ? 1. + 0.25 * x + 0.125 * y : 1.;
                    // A nodal streamfunction gives a discrete curl; face averages telescope exactly between levels.
                    double circulation = curl ? (d == 0 ? (psi(i * h, (j + 1) * h) - psi(i * h, j * h)) / h
                                                        : -(psi((i + 1) * h, j * h) - psi(i * h, j * h)) / h) : 0.;
                    double gradient = 0.;
                    if (test == "hydro_x") gradient = d == 0 ? 0.3 : 0.;
                    else if (test == "hydro_y") gradient = d == 1 ? -0.2 : 0.;
                    else if (test != "curl") gradient = d == 0 ? 0.3 : -0.2;
                    if (quadratic) gradient = d == 0 ? 0.4 * x + 0.1 : -0.3 * y + 0.2;
                    // Closed walls constrain normal velocity before projection; pressure has homogeneous Neumann data.
                    if ((d == 0 && (i == 0 || i == n)) || (d == 1 && (j == 0 || j == n))) { circulation = 0.; gradient = 0.; }
                    target(i, j, 0) = circulation;
                    u(i, j, 0) = circulation + b(i, j, 0) * gradient;
                }
            }
        }
    }
    hierarchy.averageDown(hierarchy.velocity);
    hierarchy.averageDown(hierarchy.expected);
    for (int level = 0; level < hierarchy.geometry.size(); ++level) for (int d = 0; d < 2; ++d)
        MultiFab::Copy(hierarchy.initial[level][d], hierarchy.velocity[level][d], 0, 0, 1, 0);
}

struct Metrics { double maxDivergence = 0., integratedDivergence = 0., velocityError = 0., wallVelocity = 0., fluxMismatch = 0., subfaceContrast = 0.; };

double valueAt(const MultiFab& values, int i, int j) {
    const IntVect point(i, j);
    for (MFIter mfi(values); mfi.isValid(); ++mfi) if (mfi.validbox().contains(point)) return values.const_array(mfi)(i, j, 0);
    amrex::Abort("Reference diagnostic requested an absent fine face");
    return 0.;
}

Metrics measure(Hierarchy& hierarchy) {
    Metrics result;
    for (int level = 0; level < hierarchy.geometry.size(); ++level) {
        const int n = hierarchy.geometry[level].Domain().length(0);
        const double area = 1. / (static_cast<double>(n) * n);
        computeDivergence(hierarchy.divergence[level], GetArrOfConstPtrs(hierarchy.velocity[level]), hierarchy.geometry[level]);
        for (MFIter mfi(hierarchy.divergence[level]); mfi.isValid(); ++mfi) {
            const auto div = hierarchy.divergence[level].const_array(mfi);
            const auto active = hierarchy.active[level].const_array(mfi);
            for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) if (active(bit()[0], bit()[1], 0)) {
                const double v = div(bit()[0], bit()[1], 0);
                result.maxDivergence = std::max(result.maxDivergence, std::abs(v));
                result.integratedDivergence += v * area;
            }
        }
        for (int d = 0; d < 2; ++d) for (MFIter mfi(hierarchy.velocity[level][d]); mfi.isValid(); ++mfi) {
            const auto u = hierarchy.velocity[level][d].const_array(mfi), target = hierarchy.expected[level][d].const_array(mfi);
            for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                const auto iv = bit();
                result.velocityError = std::max(result.velocityError, std::abs(u(iv[0], iv[1], 0) - target(iv[0], iv[1], 0)));
                if (iv[d] == 0 || iv[d] == n) result.wallVelocity = std::max(result.wallVelocity, std::abs(u(iv[0], iv[1], 0)));
            }
        }
    }
    // Check every actual coarse/fine boundary face, including its two independent fine subfaces.
    for (const auto& patch : hierarchy.patches) for (int d = 0; d < 2; ++d) {
        const int t = 1 - d;
        for (int side : {patch.smallEnd(d), patch.bigEnd(d) + 1}) for (int k = patch.smallEnd(t); k <= patch.bigEnd(t); k += 2) {
            const int i = d == 0 ? side : k, j = d == 1 ? side : k;
            const double a = valueAt(hierarchy.velocity[1][d], i, j);
            const double b = valueAt(hierarchy.velocity[1][d], i + (d == 1), j + (d == 0));
            const double c = valueAt(hierarchy.velocity[0][d], i / 2, j / 2);
            result.fluxMismatch = std::max(result.fluxMismatch, std::abs(c - 0.5 * (a + b)));
            result.subfaceContrast = std::max(result.subfaceContrast, std::abs(a - b));
        }
    }
    return result;
}

double milliseconds(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }

bool run(int finest, const std::string& layout, const std::string& test, int repeats, int maxGrid, double tolerance) {
    const auto setupStart = Clock::now();
    Hierarchy hierarchy(finest, layout, maxGrid);
    initialize(hierarchy, test);
    Hydro::MacProjector projector(GetVecOfArrOfPtrs(hierarchy.velocity), GetVecOfArrOfConstPtrs(hierarchy.beta), hierarchy.geometry);
    const Array<LinOpBCType, 2> boundary{LinOpBCType::Neumann, LinOpBCType::Neumann};
    projector.setDomainBC(boundary, boundary);
    projector.setVerbose(0);
    projector.getMLMG().setMaxIter(200);
    const double setupMs = milliseconds(setupStart);
    const auto before = measure(hierarchy);
    // Cold solve includes MLMG's lazy setup; repeated solves reuse the hierarchy and start with zero pressure.
    auto start = Clock::now();
    projector.project(0., tolerance);
    const double coldMs = milliseconds(start);
    const int coldIterations = projector.getMLMG().getNumIters();
    std::vector<double> times;
    int iterations = 0, bottomIterations = 0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        hierarchy.restore();
        start = Clock::now();
        projector.project(0., tolerance);
        times.push_back(milliseconds(start));
        iterations += projector.getMLMG().getNumIters();
        const auto& bottom = projector.getMLMG().getNumCGIters();
        bottomIterations += std::accumulate(bottom.begin(), bottom.end(), 0);
    }
    const auto after = measure(hierarchy);
    std::sort(times.begin(), times.end());
    const double mean = std::accumulate(times.begin(), times.end(), 0.) / repeats;
    const bool passed = after.maxDivergence < 2. * tolerance && after.velocityError < 2.e-8
                     && after.fluxMismatch < 1.e-14 && after.wallVelocity < 1.e-14 && std::abs(after.integratedDivergence) < 1.e-12;
    std::cout << finest << ',' << layout << ',' << test << ',' << hierarchy.activeCells << ',' << hierarchy.allocatedCells << ','
              << before.maxDivergence << ',' << after.maxDivergence << ',' << after.velocityError << ',' << after.fluxMismatch << ','
              << after.integratedDivergence << ',' << after.wallVelocity << ',' << after.subfaceContrast << ','
              << setupMs << ',' << coldMs << ',' << coldIterations << ',' << mean << ',' << times[times.size() / 2] << ','
              << static_cast<double>(iterations) / repeats << ',' << static_cast<double>(bottomIterations) / repeats << ','
              << projector.getMLMG().getFinalResidual() << ',' << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

int main(int argc, char* argv[]) {
    amrex::SetVerbose(0);
    amrex::Initialize(argc, argv);
    bool passed = true;
    {
        ParmParse pp;
        int finest = 128, repeats = 3, maxGrid = 64;
        double tolerance = 1.e-10;
        std::string selectedLayout = "all", selectedCase = "all";
        pp.query("n_cell", finest); pp.query("repeats", repeats); pp.query("max_grid_size", maxGrid); pp.query("tolerance", tolerance);
        pp.query("layout", selectedLayout); pp.query("case", selectedCase);
        std::cout << std::setprecision(12);
        std::cout << "finest,layout,case,active_cells,allocated_cells,initial_divergence,final_divergence,velocity_error,interface_flux_error,integrated_divergence,wall_velocity,subface_contrast,setup_ms,cold_project_ms,cold_iterations,mean_project_ms,median_project_ms,iterations,bottom_iterations,solver_residual,status\n";
        for (const std::string layout : {"uniform", "patch", "four"}) if (selectedLayout == "all" || selectedLayout == layout)
            for (const std::string test : {"curl", "hydro_x", "hydro_y", "curl_gradient", "variable_gradient", "quadratic_gradient"})
                if (selectedCase == "all" || selectedCase == test) passed &= run(finest, layout, test, repeats, maxGrid, tolerance);
    }
    amrex::Finalize();
    return passed ? 0 : 1;
}
