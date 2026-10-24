#pragma once
#include "FluidGridTypes.h"
#include "ElementCatalogue.h"
#include "FluidMixture.h"

// Windows RPC headers define `small`; AMReX uses that name for box corners.
#pragma push_macro("small")
#undef small
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BCRec.H>
#include <hydro_MacProjector.H>
#pragma pop_macro("small")
#include <array>
#include <memory>
#include <vector>

namespace fluid::reference {

struct GasPatch { int x, y, width, height; bool operator==(const GasPatch&) const = default; };
struct GasLayout {
    int baseRatio = 1;
    // Lists are nonempty and add successive 2:1 levels in finest-cell coordinates.
    // Project-controlled layouts are aligned, disjoint and properly nested,
    // including the coarse stencil needed by two fine ghost layers.
    std::vector<std::vector<GasPatch>> refinedPatches;
    bool operator==(const GasLayout&) const = default;
};

struct GasStepResult {
    // One final projection per physical step; substeps counts only transport.
    FluidProjectionResult projection;
    int substeps = 0;
    double transportMs = 0., projectionMs = 0.;
};

struct GasTotals {
    std::vector<double> ingredientMass;
    std::array<double, 2> momentum{};
    double kineticEnergy = 0., minMassDensity = 0., maxVolumeError = 0., maxSpeed = 0.;
    bool finite = true;
};

// One AMR hierarchy for gas and optional liquid phases. Conserved lanes are
// liquid ingredients, gas ingredients, then cell momentum. AMReX must be
// initialized by the application before constructing this engine.
class AmrGas {
public:
    using State = FluidGridState<double>;
    struct Level {
        amrex::Geometry geometry;
        // Owned state plus two refreshed stencil layers; no full-state copy.
        amrex::MultiFab conserved, slopes, pressure, divergence;
        amrex::MultiFab phaseGeometry, splitWeights;
        amrex::Array<amrex::MultiFab, 2> velocity, beta, flux, predictedVelocity;
        amrex::iMultiFab active;
    };

    AmrGas(int width, int height, double cellSize, ElementCatalogue catalogue, FluidMixture gasMixture, GasLayout layout, int maxGridSize = 64);
    AmrGas(int width, int height, double cellSize, ElementCatalogue catalogue, FluidMixture liquidMixture, FluidMixture gasMixture, GasLayout layout, int maxGridSize = 64);
    ~AmrGas();
    AmrGas(const AmrGas&) = delete;
    AmrGas& operator=(const AmrGas&) = delete;

    void upload(const State& state, double tolerance = 1.e-10);
    // Conservative state transfer only; project after a true return before advancing.
    bool regrid(GasLayout layout);
    // CFL-subcycle transport with fixed projected faces, then apply damping and
    // acceleration and project once. dt selects the physical update interval.
    GasStepResult step(double dt, double divergenceTolerance, double damping = 0., std::array<double, 2> acceleration = {});
    FluidProjectionResult project(double tolerance, double dt = 0., std::array<double, 2> acceleration = {});
    void transport(double dt);
    // Donor outflow on a single resolution; conservative reconstruction bound
    // on mixed levels. Projected faces stay fixed throughout transport.
    double stableTimeStep() const;
    double maxDivergence();
    GasTotals totals() const;

    const std::vector<std::unique_ptr<Level>>& levels() const { return data; }
    const ElementCatalogue& elements() const { return catalogue; }
    const FluidMixture& gasMixture() const { return gasPhase; }
    const FluidMixture& materialMixture() const { return mixture; }
    int liquidIngredientCount() const { return liquidComponents; }
    std::array<double,2> acceleration() const { return lastAcceleration; }
    const GasLayout& layout() const { return gridLayout; }
    int ingredientCount() const { return components; }
    long long activeCellCount() const { return activeCells; }
    long long allocatedCellCount() const { return allocatedCells; }
    int finestWidth() const { return width; }
    int finestHeight() const { return height; }
    double finestCellSize() const { return cellSize; }
    // pressure stores the last MAC correction impulse. dt=0 is a geometric
    // restoration, with no physical pressure equivalent for preview purposes.
    double pressureInterval() const { return lastPressureInterval; }

private:
    int width, height, components, maxGridSize, liquidComponents;
    double cellSize;
    bool equalIngredientDensities;
    double lastPressureInterval = 0.;
    std::array<double,2> lastAcceleration{};
    ElementCatalogue catalogue;
    FluidMixture mixture, liquidPhase, gasPhase;
    GasLayout gridLayout;
    long long activeCells = 0, allocatedCells = 0;
    std::vector<std::unique_ptr<Level>> data;
    amrex::Vector<amrex::BCRec> boundary;
    std::unique_ptr<Hydro::MacProjector> projector;
    struct BoundaryFill { amrex::MultiFab coarse, fine; };
    std::vector<std::unique_ptr<BoundaryFill>> boundaryFill;

    void fillGhosts();
    void transportPhases(double dt);
    void restrictState();
    void averageFaces(amrex::Array<amrex::MultiFab, 2> Level::* field);
};

}
