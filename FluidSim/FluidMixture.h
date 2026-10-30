#pragma once
#include "SubstanceCatalogue.h"
#include <vector>

namespace fluid
{
    struct MixtureComponent
    {
        SubstanceId substance;

        // Density of this ingredient under this mixture's model.
        double density;
    };

    struct FluidMixture
    {
        std::vector<MixtureComponent> components;
    };
}