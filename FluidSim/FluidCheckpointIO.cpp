#include "FluidCheckpointIO.h"
#include <brotli/encode.h>
#include <brotli/decode.h>
#include <stdexcept>

namespace fluid
{

std::vector<std::byte> compressCheckpointData(std::span<const std::byte> data, int quality)
{
    if (quality < BROTLI_MIN_QUALITY || quality > BROTLI_MAX_QUALITY)
        throw std::invalid_argument("Brotli checkpoint quality must be 0..11");
    size_t bytes = BrotliEncoderMaxCompressedSize(data.size());
    std::vector<std::byte> encoded(bytes);
    if (!BrotliEncoderCompress(quality, BROTLI_DEFAULT_WINDOW, BROTLI_MODE_GENERIC,
        data.size(), reinterpret_cast<const uint8_t*>(data.data()),
        &bytes, reinterpret_cast<uint8_t*>(encoded.data())))
        throw std::runtime_error("Brotli checkpoint compression failed");
    encoded.resize(bytes);
    return encoded;
}

std::vector<std::byte> decompressCheckpointData(std::span<const std::byte> data, size_t original_bytes)
{
    std::vector<std::byte> decoded(original_bytes);
    size_t bytes = decoded.size();
    const auto result = BrotliDecoderDecompress(data.size(), reinterpret_cast<const uint8_t*>(data.data()),
        &bytes, reinterpret_cast<uint8_t*>(decoded.data()));
    if (result != BROTLI_DECODER_RESULT_SUCCESS || bytes != original_bytes)
        throw std::runtime_error("Invalid/truncated Brotli checkpoint or incorrect uncompressed size");
    return decoded;
}

#ifdef BITLOOP_DEV_MODE
void writeFluidCheckpoint(bl::SimCheckpointWriter& writer, const FluidCheckpoint& checkpoint, int brotli_quality)
{
    if (brotli_quality < -1 || brotli_quality > BROTLI_MAX_QUALITY)
        throw std::invalid_argument("Checkpoint Brotli quality must be -1 (off) or 0..11");
    writer.writeJson("fluid-metadata", checkpoint.metadata);
    if (brotli_quality == -1)
    {
        writer.writeTrivialVector("fluid-state", checkpoint.data);
        return;
    }
    const auto encoded = compressCheckpointData(checkpoint.data, brotli_quality);
    writer.writeTrivialVector("fluid-state", encoded, {
        {"compression", "brotli"}, {"uncompressed_bytes", checkpoint.data.size()}, {"quality", brotli_quality}
    });
}

FluidCheckpoint readFluidCheckpoint(const bl::SimCheckpointReader& reader)
{
    FluidCheckpoint checkpoint;
    checkpoint.metadata = reader.readJson("fluid-metadata");
    const auto metadata = reader.chunkMetadata("fluid-state");
    auto encoded = reader.readTrivialVector<std::byte>("fluid-state");
    const auto compression = metadata.value("compression", "none");
    if (compression == "none")
        checkpoint.data = std::move(encoded);
    else if (compression == "brotli")
        checkpoint.data = decompressCheckpointData(encoded, metadata.at("uncompressed_bytes").get<size_t>());
    else
        throw std::runtime_error("Unsupported checkpoint compression: " + compression);
    return checkpoint;
}
#endif

}
