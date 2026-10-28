#pragma once
#include "FluidCheckpoint.h"
#include <span>

#ifdef BITLOOP_DEV_MODE
#include <bitloop/core/sim_checkpoint.h>
#endif

namespace fluid
{

// Raw lossless Brotli, without text/base64 expansion. Quality is 0..11.
std::vector<std::byte> compressCheckpointData(std::span<const std::byte> data, int quality);
std::vector<std::byte> decompressCheckpointData(std::span<const std::byte> data, size_t original_bytes);

#ifdef BITLOOP_DEV_MODE
// Compression belongs to storage, independently of each backend's native state format.
// -1 leaves the payload uncompressed; reading detects the codec from the chunk manifest.
void writeFluidCheckpoint(bl::SimCheckpointWriter& writer, const FluidCheckpoint& checkpoint, int brotli_quality = -1);
FluidCheckpoint readFluidCheckpoint(const bl::SimCheckpointReader& reader);
#endif

}
