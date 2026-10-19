#pragma once
#include <limits>
#include <vector>

namespace fluid
{

// host-side state uses each engine's native precision; empty fields initialize to zero
template<typename Scalar>
struct FluidGridState
{
    // Total liquid volume fraction, including every liquid ingredient.
    std::vector<Scalar> liquid_fraction;
    std::vector<Scalar> u;
    std::vector<Scalar> v;
    std::vector<Scalar> pressure;

    std::vector<std::vector<Scalar>> liquid_mass;
    std::vector<std::vector<Scalar>> gas_mass;
};

struct FluidProjectionResult
{
    int iterations = 0;
    double max_divergence = std::numeric_limits<double>::infinity();
    bool converged = false;
};

}
