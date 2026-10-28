#pragma once
#include "Adaptive/FluidCoarseningPreview.h"

namespace fluid
{
// The scene chooses quality and simulated-time cadence; the backend executes it.
struct FluidAdaptiveSettings
{
    bool automatic_refinement = true;
    FluidCoarseningPreview::Tolerances tolerances;
    FluidCoarseningPreview::Stability stability;
    double check_interval = 0.32;
    // Include the scene's observation/frame interval as well as the check interval.
    double prediction_horizon = 0.34;
    bool operator==(const FluidAdaptiveSettings&) const = default;
};
}
