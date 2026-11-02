#pragma once
#include "SubstanceCatalogue.h"
#include <vector>

namespace fluid
{

enum class Phase
{
    Liquid,
    Gas
};

struct MixtureComponent
{
    SubstanceId substance;

    // Density of this ingredient under this mixture's model.
    double density;

    // Dynamic viscosity in scene units, specific to this ingredient's phase.
    double dynamic_viscosity = 0.0;

    // Persistent incompressible shear stiffness in scene stress units.
    double shear_modulus = 0.0;
};

struct FluidMixture
{
    std::vector<MixtureComponent> components;

    bool hasElasticity() const
    {
        for (const auto& component : components)
        {
            if (component.shear_modulus != 0.0)
                return true;
        }

        return false;
    }

    bool hasViscosity() const
    {
        for (const auto& component : components)
        {
            if (component.dynamic_viscosity != 0.0)
                return true;
        }

        return false;
    }
};
}
