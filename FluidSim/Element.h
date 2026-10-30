#pragma once
#include <cstdint>
#include <string>

namespace fluid
{
    using ElementId = std::uint32_t;

    // Chemical building-block metadata; no individual atoms are simulated.
    struct Element
    {
        std::string name;
        std::string symbol;
        double atomic_mass; // Atomic mass in u; numerically the molar mass in g/mol.
    };

    struct ElementCount
    {
        ElementId element;
        std::uint32_t count;
    };
}
