#pragma once
#include "FluidMixture.h"

namespace FluidSim
{
// Shared chemical definitions for the scenarios. Colours are diagnostic labels.
struct ChemicalMaterials
{
    fluid::SubstanceCatalogue substances;
    fluid::SubstanceId hydrogen, water, glycerol, helium;
    fluid::FluidMixture liquid, gas;

    ChemicalMaterials(std::array<float, 3> hydrogen_color, std::array<float, 3> helium_color)
    {
        // Abridged standard atomic masses: https://ciaaw.org/abridged-atomic-weights.htm
        const auto h = substances.addElement({"Hydrogen", "H", 1.0080});
        const auto he = substances.addElement({"Helium", "He", 4.0026});
        const auto c = substances.addElement({"Carbon", "C", 12.011});
        const auto o = substances.addElement({"Oxygen", "O", 15.999});

        hydrogen = substances.add({"Hydrogen", hydrogen_color, {{h, 2}}});
        water = substances.add({"Water", {0.1f, 0.35f, 1.0f}, {{h, 2}, {o, 1}}});
        glycerol = substances.add({"Glycerol", {0.95f, 0.25f, 0.15f}, {{c, 3}, {h, 8}, {o, 3}}});
        helium = substances.add({"Helium", helium_color, {{he, 1}}});

        // Fixed liquid reference densities, not a runtime equation of state or thermal model.
        // .01 scene density units per kg/m^3 keeps water near the original value 10.
        constexpr double density_scale = 0.01;

        // Water at 20 C: NIST GLP 10; glycerol: NIST STAR material 174 (1.2613 g/cm^3).
        liquid.components = {{water, 998.20675 * density_scale}, {glycerol, 1261.3 * density_scale}};

        // Deliberately equal gas densities, matching the old atmosphere for performance comparison.
        constexpr double atmosphere_density = 0.1;
        gas.components = {{hydrogen, atmosphere_density}, {helium, atmosphere_density}};
    }
};
}
