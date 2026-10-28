#include "FluidSim.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace bl;
using namespace fluid;
namespace fs = std::filesystem;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct TestProject : BasicProject
{
    SceneHandle<FluidSim::FluidSim_Scene> source;
    void projectPrepare(Layout&) override
    {
        source = create<FluidSim::FluidSim_Scene>();
    }
};

void writeImage(const fs::path& path, const FluidSnapshot& snapshot)
{
    // These tests exercise the state/drop path; native image pixels are covered separately.
    bytebuf image;
    const bytebuf pixel(4, 255);
    require(webp_snapshot_encode_rgba(pixel.data(), 1, 1, 100, 6, true, 100, 100, image), "Image encode failed");
    require(webp_set_save_string_as_xmp_inplace(image, encodeFluidSnapshot(snapshot)), "XMP write failed");
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(image.data()), std::streamsize(image.size()));
}

void drop(TestProject& project, const fs::path& path)
{
    EventData event;
    event.type = EventType::DROP_FILE;
    event.drop.data = path.string();
    project.queueHeadlessInputEvent(event);
    project.processHeadlessFrame(.02);
}

void run(const fs::path& root)
{
    fs::create_directories(root);
    ImDebugLog log;
    CaptureContext capture;
    TestProject project;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();
    auto& scene = *project.source.get();

    FluidSnapshot saved;
    saved.width = 32;
    saved.height = 24;
    saved.cell_size = .5;
    saved.scene = scene.captureSceneState();
    saved.scene["simulation_time"] = 1.25;
    saved.scene["real_time"] = 4.5;
    saved.scene["projection_tolerance"] = .125;
    FluidGrid native_grid(saved.width, saved.height, saved.cell_size);
    native_grid.setDevice(FluidDevice::GPU);
    native_grid.configureMaterials(scene.fluid.getElements(), scene.fluid.getLiquidMixture(), scene.fluid.getGasMixture());
    native_grid.upload({});
    require(native_grid.saveCheckpoint(saved.checkpoint), "Small native checkpoint unsupported");
    const auto image = root / "native_grid.webp";
    writeImage(image, saved);
    scene.fluid.destroy();
    scene.fluid.setDevice(FluidDevice::CPU);
    scene.fluid_device = FluidDevice::CPU;
    drop(project, image);
    require(scene.snapshot_error.empty(), scene.snapshot_error.c_str());
    require(scene.grid_w == 32 && scene.grid_h == 24 && scene.cell_size == .5 &&
        scene.world_w == 16 && scene.world_h == 12, "Saved geometry was not recreated");
    require(scene.captureSceneState() == saved.scene, "Saved scene policy/clocks changed");
    FluidCheckpoint restored;
    require(scene.fluid.saveCheckpoint(restored) && restored.metadata == saved.checkpoint.metadata &&
        restored.data == saved.checkpoint.data, "Dropped image changed native state");

    const auto assertUnchanged = [&]
    {
        FluidCheckpoint after;
        require(!scene.snapshot_error.empty(), "Invalid image did not report an error");
        require(scene.captureSceneState() == saved.scene && scene.fluid.saveCheckpoint(after) &&
            after.metadata == restored.metadata && after.data == restored.data,
            "Invalid image changed the active simulation");
    };
    drop(project, root / "missing.webp");
    assertUnchanged();
    auto broken = saved;
    broken.checkpoint.data.pop_back();
    writeImage(root / "truncated.webp", broken);
    drop(project, root / "truncated.webp");
    assertUnchanged();
    broken = saved;
    broken.scene["scene_elements"][0] = 999;
    writeImage(root / "bad-element.webp", broken);
    drop(project, root / "bad-element.webp");
    assertUnchanged();
    broken = saved;
    broken.scene["projection_tolerance"] = 0;
    writeImage(root / "bad-quality.webp", broken);
    drop(project, root / "bad-quality.webp");
    assertUnchanged();
    drop(project, image);
    require(scene.snapshot_error.empty(), "Successful resume did not clear the error");
    project.stopHeadless();
    std::cout << "PASS file-drop resume, saved geometry/backend/clocks/quality, atomic rejection and recovery\n";
}
}

int main(int argc, char** argv)
{
    try
    {
        run(argc == 2 ? fs::path(argv[1]) : fs::path("snapshot-scene-tests"));
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
