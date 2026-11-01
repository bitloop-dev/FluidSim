#pragma once
#include "FluidGridTypes.h"
#include <array>
#include <vector>

namespace fluid
{
struct AdaptiveStepResult
{
    // One final projection per physical step; substeps counts only transport.
    FluidProjectionResult projection;
    int substeps = 0;
    double transportMs = 0.0;
    double projectionMs = 0.0;
};

struct AdaptiveTotals
{
    std::vector<double> ingredientMass;
    std::array<double, 2> momentum{};
    double kineticEnergy = 0.0;
    double minMassDensity = 0.0;
    double maxVolumeError = 0.0;
    double maxSpeed = 0.0;
    bool finite = true;
};

}
