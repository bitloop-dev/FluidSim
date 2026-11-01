#pragma once
#include "FluidMixture.h"
#include "SubstanceCatalogue.h"
#include <vector>

namespace fluid
{

enum class BoundarySide
{
    Left,
    Right,
    Top,
    Bottom
};

// Half-open face interval [first, first + count). Vertical walls count downwards;
// horizontal walls count left to right. Unspecified faces remain closed/free-slip.
struct BoundaryOpening
{
    BoundarySide side;
    int first;
    int count;
    bool operator==(const BoundaryOpening&) const = default;
};

struct BoundaryInflow
{
    BoundaryOpening opening;
    Phase phase;
    SubstanceId substance;
    double speed; // Nonnegative normal speed into the domain; zero tangential velocity.
    bool operator==(const BoundaryInflow&) const = default;
};

struct BoundaryOutflow
{
    BoundaryOpening opening;
    double speed; // Nonnegative normal speed out; carries the local material mixture.
    bool operator==(const BoundaryOutflow&) const = default;
};

// Prescribed flow, not an exterior pressure condition. Total inflow and outflow
// must balance in this incompressible box. Openings must not overlap.
struct BoundaryFlows
{
    std::vector<BoundaryInflow> inflows;
    std::vector<BoundaryOutflow> outflows;

    bool empty() const { return inflows.empty() && outflows.empty(); }

    bool operator==(const BoundaryFlows&) const = default;
};

}
