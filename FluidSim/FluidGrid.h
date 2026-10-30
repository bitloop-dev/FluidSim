#pragma once
#include "FluidGridTypes.h"
#include "FluidGridEdit.h"
#include "PressureMultigrid.h"
#include "ElementCatalogue.h"
#include "FluidMixture.h"
#include <optional>

#include <bitloop/core/types.h>

namespace fluid
{

struct FluidGrid
{
    using Scalar = double;
    using State = FluidGridState<Scalar>;
    using ProjectionResult = FluidProjectionResult;
    using Edit = FluidGridEdit<FluidGrid>;

    static constexpr bool requires_render_thread = false;

    ElementCatalogue elements;
    FluidMixture liquid_mixture;
    FluidMixture gas_mixture;

    std::vector<std::vector<Scalar>> liquid_mass;
    std::vector<std::vector<Scalar>> gas_mass;

    int width = 0;
    int height = 0;
    double cell_size = 1.0;
    bl::DVec2 origin = {};

    double gravity_y = 100.0;
    double gas_damping_rate = 0.0; // per second; zero disables optional drag toward stillness

    // Opt-in stationary gas reservoir at the upper face. Configure before upload().
    // Pressure is prescribed at that face; fractions follow gas_mixture order.
    struct AmbientBoundary
    {
        double pressure = 0.0;
        std::vector<double> gas_volume_fractions;
    };
    std::optional<AmbientBoundary> top_ambient;

    // Cumulative exchange into the domain, reset by upload(). Momentum here is
    // advective exchange only; pressure forces and wall impulses are separate.
    std::vector<double> boundary_liquid_mass, boundary_gas_mass;
    bl::DVec2 boundary_momentum{};

    enum class PressurePreconditioner { Diagonal, Multigrid };
    PressurePreconditioner pressure_preconditioner = PressurePreconditioner::Multigrid;

    // staggered face velocities and the evolving velocities during transport
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> next_u;
    std::vector<double> next_v;
    std::vector<double> pressure;

    // Geometry of all liquid ingredients combined; gas fills the remaining volume.
    std::vector<double> liquid_fraction;
    std::vector<double> next_liquid_fraction;

    // Cached density and pressure coefficients, rebuilt from component masses.
    std::vector<double> density;
    std::vector<double> inv_density_u;
    std::vector<double> inv_density_v;
    std::vector<double> inv_pressure_diagonal;

    FluidGrid(int width, int height, double cell_size);

    void upload(const State& state);
    State download() const;

    // Replace selected cells with a pure configured ingredient; existing velocities are retained.
    Edit beginEdit() { return Edit(*this); }

    // Call after editing component masses or derived density/pressure coefficients directly.
    // upload() and engine transport invalidate these caches automatically; velocities are not cached.
    void invalidateCaches();

    // CPU storage is released by the destructor; there are no device resources to destroy
    void destroy() {}

    using StepTimings = FluidStepTimings;

    // Optional timings observe the same ordered substep; null performs no clock reads.
    ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6, StepTimings* timings = nullptr);

    // transport requires a divergence-free velocity field; step() supplies the previous projection
    void advect(double dt);
    void dampGasVelocity(double dt);
    void applyGravity(double dt);

    // divergence_tolerance is the maximum permitted cell divergence, in inverse seconds
    ProjectionResult project(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6);
    double maxDivergence() const;

    constexpr int uIndex(int x, int y) const { return y * (width + 1) + x; }
    constexpr int vIndex(int x, int y) const { return y * width + x; }
    bl::DVec2 sampleVelocity(bl::DVec2 pos) const;

    void rebuildInterfaces();

    void addVortex(bl::DVec2 center, double radius, double strength);

private:
    friend Edit;
    void commitEdits(detail::CellEditBatch& batch);

    PressureMultigrid pressure_multigrid;
    bool density_valid = false;
    bool pressure_coefficients_valid = false;
    bool multigrid_valid = false;

    std::vector<double> previous_pressure;
    std::vector<double> pressure_rhs;
    std::vector<double> pressure_residual;
    std::vector<double> pressure_correction;
    std::vector<double> pressure_direction;
    std::vector<double> pressure_product;

    std::vector<bl::DVec2> interface_normals;
    std::vector<double>    interface_offsets;
    std::vector<unsigned char> split_weights;
    bool transport_has_liquid = true, transport_has_gas = true;

    std::vector<std::vector<Scalar>> liquid_correction_weights;
    std::vector<std::vector<Scalar>> gas_correction_weights;
    std::vector<Scalar> mass_correction_weights;

    // Signed liquid volume crossing each face, measured in cell-volume units.
    std::vector<double> liquid_transfer_u;
    std::vector<double> liquid_transfer_v;

    // Total transported mass per cell volume, used by momentum.
    std::vector<double> mass_transfer_u;
    std::vector<double> mass_transfer_v;

    // Reused for one ingredient and one direction at a time.
    std::vector<Scalar> component_transfer;

    std::vector<double> momentum_u;
    std::vector<double> momentum_v;

    bl::DVec2 interfaceNormal(int x, int y) const;

    static double componentMassTransfer(
        double volume_transfer,
        double donor_fraction,
        double donor_mass);

    double liquidTransferX(int x, int y, double step) const;
    double liquidTransferY(int x, int y, double step) const;

    void rebuildPressureCoefficients();
    void rebuildTransfersX(double step);
    void rebuildTransfersY(double step);

    void rebuildComponentTransfersX(
        double step, const std::vector<Scalar>& mass, bool is_liquid);

    void rebuildComponentTransfersY(
        double step, const std::vector<Scalar>& mass, bool is_liquid, double ambient_mass);

    void freezeComponentCorrections();
    void advectComponents(double step, bool horizontal);
    double transportStepLimit(double dt) const;
    void advectDirection(double step, bool horizontal);

    void rebuildDensity();
    void rebuildMomentum();

    static double momentumTransfer(
        double mass_transfer,
        double velocity_before,
        double velocity_after);

    void advectMomentumX(double step, const std::vector<double>& flow_u);
    void advectMomentumY(double step, const std::vector<double>& flow_v);
    void recoverVelocity();

    static void removeMean(std::vector<double>& values);
    static double maxAbs(const std::vector<double>& values);
    // Also returns input dot output, used as the CG denominator.
    double applyPressureOperator(const std::vector<double>& input, std::vector<double>& output) const;

    struct ResidualStats
    {
        double sum = 0.0;
        double maximum = 0.0;
    };

    ResidualStats updatePressureAndResidual(double alpha);
    ResidualStats rebuildPressureResidual();
    double applyPreconditioner(const std::vector<double>& residual, std::vector<double>& correction, double residual_mean);
    int solvePressure(int max_iterations, double residual_tolerance);
};

}
