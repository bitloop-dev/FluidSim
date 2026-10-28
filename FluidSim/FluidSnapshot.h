#pragma once
#include "FluidCheckpoint.h"
#include <bitloop/core/types.h>
#include <filesystem>
#include <span>
#include <string>

namespace fluid
{

struct FluidSnapshot
{
    int width = 0, height = 0;
    double cell_size = 1;
    bl::JSON::json scene;
    FluidCheckpoint checkpoint;
};

// One versioned, lossless packet for WebP XMP, independent of filesystem checkpoints.
std::string encodeFluidSnapshot(const FluidSnapshot& snapshot);
FluidSnapshot decodeFluidSnapshot(std::string_view payload);
FluidSnapshot readFluidSnapshot(const std::filesystem::path& image);
bl::bytebuf fluidSnapshotPixels(std::span<const float> colors);

}
