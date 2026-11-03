#pragma once
#include "SubstanceCatalogue.h"
#include <vector>
#include <cstddef>

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

    size_t elasticComponentIndex() const
    {
        for (size_t i = 0; i < components.size(); ++i)
        {
            if (components[i].shear_modulus != 0.0)
                return i;
        }

        return components.size();
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
