#pragma once
#include <bitloop.h>
#include <cstdint>
#include <limits>

SIM_BEG;

// Stable until that node is removed; its slot may then be reused.
using NodeId = std::uint32_t;

struct ElasticNode
{
    bl::DVec2 position{};
    bl::DVec2 velocity{};
    bool active = true;
    bool fixed = false;
};

struct ElasticBond
{
    NodeId a;
    NodeId b;

    double rest_length = 10.0;
    double stiffness = 1.0;
    double damping = 0.99;

    // Tensile strain limit: 0.30 permits 30% extension; infinity keeps the bond unbreakable.
    double breaking_strain = std::numeric_limits<double>::infinity();

    // Explicit collision segment; spring, damping and fracture parameters do not apply.
    bool boundary = false;
};

SIM_END;
