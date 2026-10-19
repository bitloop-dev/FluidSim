#pragma once
#include "ElementCatalogue.h"
#include <vector>

namespace fluid
{
    struct MixtureComponent
    {
        ElementId element;

        // Density of this ingredient under this mixture's model.
        double density;
    };

    struct FluidMixture
    {
        std::vector<MixtureComponent> components;
    };
}