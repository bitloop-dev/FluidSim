#pragma once
#include "Element.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fluid
{
    using SubstanceId = std::uint32_t;

    struct Substance
    {
        std::string name;
        std::array<float, 3> color{ 1.0f, 1.0f, 1.0f };
        // Empty means unspecified chemistry, useful for abstract materials and old snapshots.
        std::vector<ElementCount> composition;
    };

    class SubstanceCatalogue
    {
        std::vector<Element> elements;
        std::vector<Substance> definitions;

    public:

        ElementId addElement(Element element)
        {
            const auto id = static_cast<ElementId>(elements.size());
            elements.push_back(std::move(element));
            return id;
        }

        const Element& getElement(ElementId id) const { return elements[id]; }
        std::size_t elementCount() const { return elements.size(); }

        SubstanceId add(Substance substance)
        {
            const auto id = static_cast<SubstanceId>(definitions.size());
            definitions.push_back(std::move(substance));
            return id;
        }

        const Substance& operator[](SubstanceId id) const
        {
            return definitions[id];
        }

        std::size_t size() const
        {
            return definitions.size();
        }

        // For substances with specified composition. Called on demand, never during transport.
        double molarMass(SubstanceId id) const
        {
            double mass = 0.;
            for (const auto& constituent : definitions[id].composition)
                mass += constituent.count * elements[constituent.element].atomic_mass;
            return mass;
        }
    };
}
