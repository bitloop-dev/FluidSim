#pragma once
#include "FluidGridBase.h"
#include "PressureMultigrid.h"
#include "SubstanceCatalogue.h"
#include "FluidMixture.h"
#include "FluidSurfaceTensionCPU.h"

#include <bitloop/core/types.h>
#include <optional>

namespace fluid
{

struct FluidGridCPU final : FluidGridBase
{
    using Scalar = double;
    using State = FluidGridState<Scalar>;
    using ProjectionResult = FluidProjectionResult;
    using Edit = FluidGridBase::Edit;

    static constexpr bool requires_render_thread = false;

    SubstanceCatalogue substances;
    FluidMixture liquid_mixture;
    FluidMixture gas_mixture;

    std::vector<std::vector<Scalar>> liquid_mass;
    std::vector<std::vector<Scalar>> gas_mass;

    int width = 0;
    int height = 0;
    double cell_size = 1.0;
    bl::DVec2 origin = {};

    double gravity_y = 100.0;

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

    FluidGridCPU(int width, int height, double cell_size);

    int getWidth() const override { return width; }
    int getHeight() const override { return height; }
    double getCellSize() const override { return cell_size; }
    bl::DVec2 getOrigin() const override { return origin; }
    void setOrigin(bl::DVec2 position) override { origin = position; }
    bool requiresRenderThread() const override { return false; }
    void configureMaterials(SubstanceCatalogue catalogue, FluidMixture liquid, FluidMixture gas) override
    {
        substances = std::move(catalogue);
        liquid_mixture = std::move(liquid);
        gas_mixture = std::move(gas);
    }
    const SubstanceCatalogue& getSubstances() const override { return substances; }
    const FluidMixture& getLiquidMixture() const override { return liquid_mixture; }
    const FluidMixture& getGasMixture() const override { return gas_mixture; }
    double getGravityY() const override { return gravity_y; }
    void setGravityY(double acceleration) override { gravity_y = acceleration; }
    double getSurfaceTension() const override { return surface_tension; }
    void setSurfaceTension(double coefficient) override;
    double capillaryStepLimit() const override;
    void setAccelerationField(const FluidField& field) override;
    void clearAccelerationField() override { acceleration.reset(); }

    void upload(const State& state) override;
    State download() const override;
    void writeColorPixels(std::vector<float>& pixels, bool liquid_fraction_alpha = false) const override;

    // Call after editing component masses or derived density/pressure coefficients directly.
    // upload() and engine transport invalidate these caches automatically; velocities are not cached.
    void invalidateCaches();

    // Physical arrays remain owned until destruction; release the external field binding.
    void destroy() override { clearAccelerationField(); }

    // one physics substep: transport material and momentum, apply forces, then project
    ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6, StepTimings* timings = nullptr) override;

    // transport requires a divergence-free velocity field; step() supplies the previous projection
    void advect(double dt);
    void applyGravity(double dt);

    // divergence_tolerance is the maximum permitted cell divergence, in inverse seconds
    ProjectionResult project(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6);
    double maxDivergence() const override;

    constexpr int uIndex(int x, int y) const { return y * (width + 1) + x; }
    constexpr int vIndex(int x, int y) const { return y * width + x; }
    bl::DVec2 sampleVelocity(bl::DVec2 pos) const;

    void rebuildInterfaces();

    void addVortex(bl::DVec2 center, double radius, double strength);

private:
    friend Edit;
    void commitEdits(detail::CellEditBatch& batch) override;
    void applyAcceleration(double dt);
    std::optional<FluidField> acceleration;
    double surface_tension = 0.;
    std::optional<FluidSurfaceTensionCPU> capillary;
    void applySurfaceTension(double dt);

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
        double step, const std::vector<Scalar>& mass, bool is_liquid);

    void freezeComponentCorrections();
    void advectComponents(double step, bool horizontal);

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
