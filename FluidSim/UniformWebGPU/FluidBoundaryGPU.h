#pragma once
#include "Shared/WebGPU/FluidWebGPUCompute.h"
#include "FluidBoundaryFlows.h"
#include "FluidMixture.h"
#include <cmath>

namespace fluid::webgpu
{

// Perimeter records: signed inward speed, incoming liquid fraction, density,
// packed component index + 1. Outflow uses the interior donor instead.
inline std::vector<float> packBoundaryFlows(
    const BoundaryFlows& flows, int width, int height, const FluidMixture& liquid, const FluidMixture& gas)
{
    std::vector<float> records(size_t(2 * (width + height)) * 4, 0.0f);
    std::vector<bool> used(2 * (width + height), false);
    double incoming = 0.0;
    double outgoing = 0.0;

    auto opening = [&](const BoundaryOpening& opening, double speed, float fraction, float density, int component)
    {
        const int side = int(opening.side);
        if (side < 0 || side > 3)
            throw std::invalid_argument("Unknown boundary side");

        const int extent = side < 2 ? height : width;
        if (opening.first < 0 || opening.count <= 0 || opening.first > extent - opening.count)
            throw std::invalid_argument("Boundary opening lies outside the wall");

        if (!std::isfinite(float(speed)))
            throw std::invalid_argument("Boundary speed must be finite");

        const int offset = side < 2 ? side * height : 2 * height + (side - 2) * width;

        for (int i = opening.first; i < opening.first + opening.count; ++i)
        {
            const size_t index = size_t(offset + i);
            if (used[index])
                throw std::invalid_argument("Boundary openings overlap");

            used[index] = true;
            records[4 * index] = float(speed);
            records[4 * index + 1] = fraction;
            records[4 * index + 2] = density;
            records[4 * index + 3] = float(component + 1);
        }
    };

    for (const auto& inlet : flows.inflows)
    {
        if (inlet.speed < 0.0)
            throw std::invalid_argument("Inflow speed must be nonnegative");

        if (inlet.phase != Phase::Liquid && inlet.phase != Phase::Gas)
            throw std::invalid_argument("Unknown inflow phase");

        const auto& mixture = inlet.phase == Phase::Liquid ? liquid : gas;
        const auto found = std::find_if(mixture.components.begin(), mixture.components.end(), [&](const auto& component)
        {
            return component.substance == inlet.substance;
        });

        if (found == mixture.components.end())
            throw std::invalid_argument("Inflow substance is not in its phase mixture");

        const int component =
            int(found - mixture.components.begin()) + (inlet.phase == Phase::Gas ? int(liquid.components.size()) : 0);
        opening(inlet.opening, inlet.speed, inlet.phase == Phase::Liquid ? 1.0f : 0.0f, float(found->density), component);
        incoming += inlet.speed * inlet.opening.count;
    }

    for (const auto& outlet : flows.outflows)
    {
        if (outlet.speed < 0.0)
            throw std::invalid_argument("Outflow speed must be nonnegative");

        opening(outlet.opening, -outlet.speed, 0.0f, 0.0f, -1);
        outgoing += outlet.speed * outlet.opening.count;
    }

    // A net volume source has no solution with prescribed velocities everywhere.
    if (std::abs(incoming - outgoing) > 1e-7 * std::max(incoming, outgoing))
        throw std::invalid_argument("Incompressible boundary inflow and outflow must balance");

    return records;
}

struct BoundaryGPU
{
    Kernel apply;
    Field records;
    float max_speed = 0.0f;

    BoundaryGPU(Compute& compute, int width, int height) : records(compute.field(2 * (width + height), 1, 4))
    {
        compute.build(apply, "fluid prescribed boundary velocity", 2, {"boundary"},
            R"wgsl(
                fn execute()
                {
                    let i = cell().x;
                    let w = gridSize().x;
                    let h = gridSize().y;
                    let speed = boundary(cell()).x;
                    if (i < h)
                    {
                        write0(vec2i(0, i), vec4f(speed));
                    }
                    else if (i < 2 * h)
                    {
                        write0(vec2i(w, i - h), vec4f(-speed));
                    }
                    else if (i < 2 * h + w)
                    {
                        write1(vec2i(i - 2 * h, 0), vec4f(speed));
                    }
                    else
                    {
                        write1(vec2i(i - 2 * h - w, h), vec4f(-speed));
                    }
                }
            )wgsl",
            {}, false, {64, 1});
    }

    void upload(Compute& compute, const std::vector<float>& values)
    {
        max_speed = 0.0f;
        for (size_t i = 0; i < values.size(); i += 4)
            max_speed = std::max(max_speed, std::abs(values[i]));
        compute.write(records, values);
    }

    void enforce(Compute& compute, Field& u, Field& v, int width, int height)
    {
        Parameters params;
        params.grid = {width, height, 0, 0};
        compute.run(apply, {&u, &v}, {{"boundary", &records}}, params, {unsigned(records.width), 1, 1});
    }
};

}
