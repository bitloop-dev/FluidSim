#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fluid
{
    using ElementId = std::uint32_t;

    struct Element
    {
        std::string name;
        std::array<float, 3> color{ 1.0f, 1.0f, 1.0f };
    };

    class ElementCatalogue
    {
        std::vector<Element> definitions;

    public:

        ElementId add(Element element)
        {
            const auto id = static_cast<ElementId>(definitions.size());
            definitions.push_back(std::move(element));
            return id;
        }

        const Element& operator[](ElementId id) const
        {
            return definitions[id];
        }

        std::size_t size() const
        {
            return definitions.size();
        }
    };
}