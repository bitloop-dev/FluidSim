#include "FluidSnapshot.h"
#include "FluidCheckpointIO.h"
#include <bitloop/core/capture_manager.h>
#include <bitloop/util/compression.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace fluid
{
namespace
{
constexpr std::string_view prefix = "FluidSimSnapshot/1:";
}

std::string encodeFluidSnapshot(const FluidSnapshot& snapshot)
{
    const bl::JSON::json header = {
        {"width", snapshot.width}, {"height", snapshot.height}, {"cell_size", snapshot.cell_size},
        {"scene", snapshot.scene}, {"engine", snapshot.checkpoint.metadata}
    };
    std::string packet = header.dump();
    packet.push_back('\0'); // JSON cannot contain a literal NUL; the remaining bytes are native state.
    packet.append(reinterpret_cast<const char*>(snapshot.checkpoint.data.data()), snapshot.checkpoint.data.size());
    const auto encoded = compressCheckpointData(std::as_bytes(std::span(packet)), 1);
    const std::string_view bytes(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    return std::string(prefix) + std::to_string(packet.size()) + ":" + compression::b64_encode(bytes);
}

FluidSnapshot decodeFluidSnapshot(std::string_view payload)
{
    if (!payload.starts_with(prefix))
        throw std::runtime_error("This image does not contain a supported FluidSim snapshot");
    payload.remove_prefix(prefix.size());
    const size_t separator = payload.find(':');
    if (separator == std::string_view::npos)
        throw std::runtime_error("Incomplete FluidSim snapshot header");
    size_t original_bytes = 0;
    const auto length = std::from_chars(payload.data(), payload.data() + separator, original_bytes);
    if (length.ec != std::errc{} || length.ptr != payload.data() + separator)
        throw std::runtime_error("Invalid FluidSim snapshot size");
    const auto encoded = compression::b64_decode(payload.substr(separator + 1));
    auto packet = decompressCheckpointData(std::as_bytes(std::span(encoded)), original_bytes);
    const auto end = std::find(packet.begin(), packet.end(), std::byte{0});
    if (end == packet.end()) throw std::runtime_error("Missing FluidSim snapshot metadata");
    const size_t header_bytes = size_t(end - packet.begin());
    const auto header = bl::JSON::json::parse(
        reinterpret_cast<const char*>(packet.data()), reinterpret_cast<const char*>(packet.data()) + header_bytes);
    FluidSnapshot snapshot;
    snapshot.width = header.at("width").get<int>();
    snapshot.height = header.at("height").get<int>();
    snapshot.cell_size = header.at("cell_size").get<double>();
    if (snapshot.width <= 0 || snapshot.height <= 0 || !std::isfinite(snapshot.cell_size) || snapshot.cell_size <= 0)
        throw std::runtime_error("Invalid saved fluid grid dimensions");
    snapshot.scene = header.at("scene");
    snapshot.checkpoint.metadata = header.at("engine");
    snapshot.checkpoint.data.assign(end + 1, packet.end());
    return snapshot;
}

FluidSnapshot readFluidSnapshot(const std::filesystem::path& image)
{
    std::ifstream file(image, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open snapshot: " + image.string());
    const bl::bytebuf webp{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    std::string payload;
    if (!bl::webp_extract_save_from_xmp(webp, payload))
        throw std::runtime_error("This WebP does not contain saved simulation state");
    return decodeFluidSnapshot(payload);
}

bl::bytebuf fluidSnapshotPixels(std::span<const float> colors)
{
    bl::bytebuf pixels(colors.size());
    std::transform(colors.begin(), colors.end(), pixels.begin(), [](float value)
    {
        return uint8_t(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    });
    return pixels;
}
}
