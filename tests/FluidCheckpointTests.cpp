#include "FluidCheckpointIO.h"
#include "FluidSnapshot.h"
#include <bitloop/core/capture_manager.h>
#include <webp/decode.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace fluid;
namespace fs = std::filesystem;
namespace
{
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

template<class Operation> double milliseconds(Operation&& operation)
{
    const auto start = std::chrono::steady_clock::now();
    operation();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

template<class Operation> void requireRejected(Operation&& operation)
{
    bool rejected = false;
    try { operation(); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "Invalid checkpoint encoding was accepted");
}

void roundTrips(const fs::path& root)
{
    std::mt19937 random(7);
    for (int kind = 0; kind < 3; ++kind)
    {
        FluidCheckpoint checkpoint;
        checkpoint.metadata = {{"backend", "test"}, {"version", 1}, {"fixture", kind}};
        checkpoint.data.resize(kind == 0 ? 0 : 8192);
        for (size_t i = 0; i < checkpoint.data.size(); ++i)
            checkpoint.data[i] = std::byte(kind == 1 ? i % 8 : random() & 255);
        for (int quality : {-1, 0, 1, 4, 6, 9, 11})
        {
            const auto path = root / (std::to_string(kind) + "-" + std::to_string(quality));
            bl::SimCheckpointWriter writer(path);
            writeFluidCheckpoint(writer, checkpoint, quality);
            writer.finalize();
            const auto loaded = readFluidCheckpoint(bl::SimCheckpointReader(path));
            require(loaded.metadata == checkpoint.metadata && loaded.data == checkpoint.data,
                "Checkpoint file round trip changed native metadata/bytes");
        }
    }

    const std::vector<std::byte> source(8192, std::byte{42});
    auto encoded = compressCheckpointData(source, 4);
    requireRejected([&] { decompressCheckpointData(encoded, source.size() + 1); });
    requireRejected([&] { decompressCheckpointData(encoded, source.size() - 1); });
    encoded.resize(encoded.size() / 2);
    requireRejected([&] { decompressCheckpointData(encoded, source.size()); });
    requireRejected([&] { compressCheckpointData(source, -1); });
    requireRejected([&] { compressCheckpointData(source, 12); });

    const auto invalid = root / "invalid";
    for (bool unknown : {true, false})
    {
        bl::SimCheckpointWriter writer(invalid);
        writer.writeJson("fluid-metadata", {{"backend", "test"}});
        writer.writeTrivialVector("fluid-state", source, {{"compression", unknown ? "unknown" : "brotli"}});
        writer.finalize();
        requireRejected([&] { readFluidCheckpoint(bl::SimCheckpointReader(invalid)); });
    }
    std::cout << "PASS raw/legacy and Brotli0/1/4/6/9/11 file round trips: empty, repetitive, random; invalid codec/size/quality/truncated stream rejection\n";
}

void imageRoundTrip(const fs::path& root)
{
    FluidSnapshot saved;
    saved.width = 3;
    saved.height = 2;
    saved.cell_size = 2.5;
    saved.scene = {{"simulation_time", 2.02}, {"name", "quotes\" & <XML>"}};
    saved.checkpoint.metadata = {{"backend", "test"}, {"version", 1}};
    saved.checkpoint.data.resize(8192);
    for (size_t i = 0; i < saved.checkpoint.data.size(); ++i) saved.checkpoint.data[i] = std::byte(i & 255);
    const auto payload = encodeFluidSnapshot(saved);
    const auto decoded = decodeFluidSnapshot(payload);
    require(decoded.width == 3 && decoded.height == 2 && decoded.cell_size == 2.5 &&
        decoded.scene == saved.scene && decoded.checkpoint.metadata == saved.checkpoint.metadata &&
        decoded.checkpoint.data == saved.checkpoint.data, "Image packet changed state");

    const std::vector<float> colors = {
        1,0,0,1, 0,1,0,1, 0,0,1,1,
        .25f,.5f,.75f,1, 1,1,1,1, 0,0,0,1
    };
    const auto pixels = fluidSnapshotPixels(colors);
    bl::bytebuf webp;
    require(bl::webp_snapshot_encode_rgba(pixels.data(), 3, 2, 100, 6, true, 100, 100, webp), "Image encode failed");
    require(bl::webp_set_save_string_as_xmp_inplace(webp, payload), "Image XMP embedding failed");
    std::string extracted;
    require(bl::webp_extract_save_from_xmp(webp, extracted) && extracted == payload, "XMP extraction changed the packet");
    int width = 0, height = 0;
    auto* image = WebPDecodeRGBA(webp.data(), webp.size(), &width, &height);
    require(image && width == 3 && height == 2 && std::equal(pixels.begin(), pixels.end(), image), "Native image pixels/orientation changed");
    WebPFree(image);
    fs::create_directories(root);
    const auto path = root / "round-trip.webp";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(webp.data()), std::streamsize(webp.size()));
    }
    require(readFluidSnapshot(path).checkpoint.data == saved.checkpoint.data, "File image resume changed bytes");
    requireRejected([&] { decodeFluidSnapshot("a Mandelbrot bookmark"); });
    requireRejected([&] { decodeFluidSnapshot("FluidSimSnapshot/1:wrong:AAAA"); });
    requireRejected([&] { decodeFluidSnapshot(payload.substr(0, payload.size() / 2)); });
    requireRejected([&] { readFluidSnapshot(root / "missing.webp"); });
    std::cout << "PASS versioned scene/native image packet, lossless native pixels/orientation, WebP XMP/file round trip and invalid image rejection\n";
}

void benchmark(const fs::path& source, const fs::path& output)
{
    const auto checkpoint = readFluidCheckpoint(bl::SimCheckpointReader(source));
    fs::create_directories(output);
    std::ofstream csv(output / "samples.csv");
    csv << "round,quality,raw_bytes,stored_bytes,encode_ms,decode_ms,save_ms,load_ms\n";
    // Alternating orders reduce systematic clock/cache effects; retain all samples.
    const std::array<std::array<int, 7>, 3> orders = {{
        {{-1, 0, 1, 4, 6, 9, 11}}, {{11, 9, 6, 4, 1, 0, -1}}, {{4, -1, 9, 0, 11, 1, 6}}
    }};
    for (size_t round = 0; round < orders.size(); ++round)
    for (int quality : orders[round])
    {
        std::vector<std::byte> encoded, decoded;
        double encode_ms = 0, decode_ms = 0;
        if (quality != -1)
        {
            encode_ms = milliseconds([&] { encoded = compressCheckpointData(checkpoint.data, quality); });
            decode_ms = milliseconds([&] { decoded = decompressCheckpointData(encoded, checkpoint.data.size()); });
            require(decoded == checkpoint.data, "Benchmark codec round trip changed bytes");
        }
        const auto target = output / ("quality-" + std::to_string(quality));
        const double save_ms = milliseconds([&]
        {
            bl::SimCheckpointWriter writer(target);
            writeFluidCheckpoint(writer, checkpoint, quality);
            writer.finalize();
        });
        FluidCheckpoint loaded;
        const double load_ms = milliseconds([&] { loaded = readFluidCheckpoint(bl::SimCheckpointReader(target)); });
        require(loaded.metadata == checkpoint.metadata && loaded.data == checkpoint.data, "Benchmark file round trip changed state");
        const auto bytes = fs::file_size(target / "fluid-state.bin");
        csv << round << ',' << quality << ',' << checkpoint.data.size() << ',' << bytes << ','
            << encode_ms << ',' << decode_ms << ',' << save_ms << ',' << load_ms << '\n';
        csv.flush();
        std::cout << "round=" << round << " quality=" << quality << " bytes=" << bytes
                  << " encode_ms=" << encode_ms << " decode_ms=" << decode_ms
                  << " save_ms=" << save_ms << " load_ms=" << load_ms << '\n';
    }
}

void compare(const fs::path& a, const fs::path& b)
{
    const bl::SimCheckpointReader reader_a(a), reader_b(b);
    const auto original = readFluidCheckpoint(reader_a);
    const auto loaded = readFluidCheckpoint(reader_b);
    require(original.metadata == loaded.metadata && original.data == loaded.data,
        "Compressed checkpoint differs from uncompressed reference");
    if (reader_a.hasChunk("scene") && reader_b.hasChunk("scene"))
        require(reader_a.readJson("scene") == reader_b.readJson("scene"), "Checkpoint scene clocks/settings differ");
    std::cout << "PASS exact decoded native fields/history/metadata and scene state, bytes=" << loaded.data.size() << '\n';
}

void compareImage(const fs::path& image, const fs::path& checkpoint)
{
    const auto snapshot = readFluidSnapshot(image);
    const bl::SimCheckpointReader reader(checkpoint);
    const auto original = readFluidCheckpoint(reader);
    require(snapshot.checkpoint.metadata == original.metadata && snapshot.checkpoint.data == original.data &&
        snapshot.scene == reader.readJson("scene"), "Image state differs from reference checkpoint");
    require(snapshot.width == original.metadata.at("width") && snapshot.height == original.metadata.at("height"), "Image grid dimensions differ");
    std::cout << "PASS exact image/native fields/history/scene metadata and geometry, bytes=" << snapshot.checkpoint.data.size() << '\n';
}
}

int main(int argc, char** argv)
{
    try
    {
        std::cout << std::unitbuf;
        if (argc == 4 && std::string(argv[1]) == "--benchmark") benchmark(argv[2], argv[3]);
        else if (argc == 4 && std::string(argv[1]) == "--compare") compare(argv[2], argv[3]);
        else if (argc == 4 && std::string(argv[1]) == "--compare-image") compareImage(argv[2], argv[3]);
        else
        {
            const auto root = argc == 2 ? fs::path(argv[1]) : fs::path("checkpoint-io-tests");
            roundTrips(root);
            imageRoundTrip(root);
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
