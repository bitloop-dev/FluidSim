#pragma once
#include "AmrPressureTopology2D.h"
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fluid::reference {

// Frozen solver inputs, independent of either executor's buffers and runtime.
struct PressureCase
{
    using Field = std::vector<std::vector<double>>;
    std::vector<AmrPressureTopology2D::Level> levels;
    Field pressure, rhs;
    std::array<Field, 2> beta;
    double tolerance = 0;
    int maxIterations = 0, solveIndex = 0;

    void save(const std::string& path) const
    {
        std::ofstream out(path, std::ios::binary);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        auto number = [&]<class T>(T value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
        number(std::uint32_t(0x46504331)); number(std::uint32_t(levels.size()));
        number(tolerance); number(maxIterations); number(solveIndex);
        for (int l = 0; l < int(levels.size()); ++l)
        {
            const auto& level = levels[l];
            number(level.width); number(level.height); number(level.cell_size); number(std::uint32_t(level.boxes.size()));
            for (const auto& box : level.boxes) { number(box.x); number(box.y); number(box.width); number(box.height); }
            for (const auto* field : {&pressure, &rhs, &beta[0], &beta[1]})
            {
                number(std::uint32_t((*field)[l].size()));
                out.write(reinterpret_cast<const char*>((*field)[l].data()), (*field)[l].size() * sizeof(double));
            }
        }
    }

    static PressureCase load(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        in.exceptions(std::ios::failbit | std::ios::badbit);
        auto number = [&]<class T>() { T value; in.read(reinterpret_cast<char*>(&value), sizeof(value)); return value; };
        if (number.template operator()<std::uint32_t>() != 0x46504331) throw std::runtime_error("Unknown pressure case format");
        const auto count = number.template operator()<std::uint32_t>();
        PressureCase result;
        result.tolerance = number.template operator()<double>();
        result.maxIterations = number.template operator()<int>(); result.solveIndex = number.template operator()<int>();
        result.levels.resize(count); result.pressure.resize(count); result.rhs.resize(count);
        for (auto& field : result.beta) field.resize(count);
        for (std::uint32_t l = 0; l < count; ++l)
        {
            auto& level = result.levels[l];
            level.width = number.template operator()<int>(); level.height = number.template operator()<int>();
            level.cell_size = number.template operator()<double>(); level.boxes.resize(number.template operator()<std::uint32_t>());
            for (auto& box : level.boxes)
            {
                box.x = number.template operator()<int>(); box.y = number.template operator()<int>();
                box.width = number.template operator()<int>(); box.height = number.template operator()<int>();
            }
            for (auto* field : {&result.pressure, &result.rhs, &result.beta[0], &result.beta[1]})
            {
                auto& values = (*field)[l]; values.resize(number.template operator()<std::uint32_t>());
                in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));
            }
        }
        return result;
    }
};
} // namespace fluid::reference
