#pragma once
#include <limits>
#include <cstddef>
#include <vector>

namespace fluid
{

// host-side state uses each engine's native precision; empty fields initialize to zero
template <typename Scalar> struct FluidGridState
{
    // Total liquid volume fraction, including every liquid ingredient.
    std::vector<Scalar> liquid_fraction;
    std::vector<Scalar> u;
    std::vector<Scalar> v;
    std::vector<Scalar> pressure;

    std::vector<std::vector<Scalar>> liquid_mass;
    std::vector<std::vector<Scalar>> gas_mass;

    // Symmetric log(B), B = F F^T, carried by condensed material. All empty means relaxed.
    std::vector<Scalar> elastic_log_xx;
    std::vector<Scalar> elastic_log_xy;
    std::vector<Scalar> elastic_log_yy;
};

template <typename To, typename From> FluidGridState<To> convertFluidState(const FluidGridState<From>& source)
{
    FluidGridState<To> result;

    auto copy = [](auto& target, const auto& values)
    {
        target.resize(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            target[i] = static_cast<To>(values[i]);
    };

    copy(result.liquid_fraction, source.liquid_fraction);
    copy(result.u, source.u);
    copy(result.v, source.v);
    copy(result.pressure, source.pressure);
    copy(result.elastic_log_xx, source.elastic_log_xx);
    copy(result.elastic_log_xy, source.elastic_log_xy);
    copy(result.elastic_log_yy, source.elastic_log_yy);

    result.liquid_mass.resize(source.liquid_mass.size());
    result.gas_mass.resize(source.gas_mass.size());

    for (std::size_t k = 0; k < source.liquid_mass.size(); ++k)
        copy(result.liquid_mass[k], source.liquid_mass[k]);
    for (std::size_t k = 0; k < source.gas_mass.size(); ++k)
        copy(result.gas_mass[k], source.gas_mass[k]);

    return result;
}

struct FluidStepTimings
{
    double transport_ms = 0;
    double viscosity_ms = 0;
    double forces_ms = 0;
    double pressure_ms = 0;
    double selection_ms = 0;
    double regrid_ms = 0;
    double elasticity_ms = 0;
};

struct FluidProjectionResult
{
    int iterations = 0;
    double max_divergence = std::numeric_limits<double>::infinity();
    bool converged = false;

    // Actual complete physics steps and transport subdivisions, where available.
    int physical_steps = 1;
    int transport_steps = 0;

    // An intentionally approximate, numerically valid step is distinct from convergence.
    bool relaxed = false;
    double max_volume_error = 0.0;
    double rms_volume_error = 0.0;
    int viscosity_iterations = 0; // Dispatched slots; convergence is observed in batches of four.

    // Convergence accepted with a local floating-point allowance (pressure or coupled momentum).
    bool precision_limited = false;
    int elasticity_iterations = 0; // Coupled velocity/pressure MINRES updates.

    bool accepted() const { return converged || relaxed || precision_limited; }
};

}
