#pragma once
#include "FluidCheckpoint.h"

// Construct the historical on-disk schema, not just a current checkpoint with an old version number.
inline void makeLegacyCheckpoint(fluid::FluidCheckpoint& checkpoint, int version)
{
    auto& meta = checkpoint.metadata;
    meta["elements"] = meta.at("substances");
    meta.erase("substances");
    for (auto& entry : meta["elements"]) entry.erase("composition");
    for (const char* phase : {"liquid_mixture", "gas_mixture"})
    {
        for (auto& component : meta[phase])
        {
            component["element"] = component.at("substance");
            component.erase("substance");
        }
    }
    if (version < 3)
    {
        if (meta.at("acceleration_field").get<bool>())
            checkpoint.data.resize(checkpoint.data.size() -
                2 * meta.at("width").get<size_t>() * meta.at("height").get<size_t>() * sizeof(float));
        meta.erase("acceleration_field");
    }
    if (version < 2) meta.erase("surface_tension");
    meta["version"] = version;
}
