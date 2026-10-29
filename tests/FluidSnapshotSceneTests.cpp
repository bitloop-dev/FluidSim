#include "FluidSim.h"
#include "MouseAttraction.h"
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
    bool mount_scene = false;
    Viewport* viewport = nullptr;

    void projectPrepare(Layout& layout) override
    {
        source = create<FluidSim::FluidSim_Scene>();
        if (mount_scene)
            viewport = &layout.add(source, "main");
    }
};

void checkMouseTarget()
{
    ImDebugLog log;
    CaptureContext capture;
    TestProject project;
    project.mount_scene = true;
    project.configureHeadless(0, &log, {800, 600}, &capture);
    project.startHeadless();
    auto& scene = *project.source.get();
    scene.simulation_failed = true; // Exercise input without advancing the large fluid fixture.
    scene.camera.setZoom(1.5);
    scene.camera.setRotation(0.3);

    const DVec2 pointer(250.0, 180.0);
    const auto expected = scene.camera.transformFor(*project.viewport).toWorld<double>(pointer);
    InputEmulation input(IVec2(800, 600));
    input.mouseMove(0, pointer);
    input.queueFrame(project, 0);
    project.processHeadlessFrame(.02);
    require(std::abs(scene.mouse_target.x - expected.x) < 1e-8 &&
        std::abs(scene.mouse_target.y - expected.y) < 1e-8,
        "Mouse target did not use the viewport camera transform");

    input.mouseMove(1, DVec2(850.0, 650.0));
    input.queueFrame(project, 1);
    project.processHeadlessFrame(.02);
    require(std::abs(scene.mouse_target.x - expected.x) < 1e-8 &&
        std::abs(scene.mouse_target.y - expected.y) < 1e-8,
        "Leaving the viewport changed the last mouse target");
    project.stopHeadless();
    std::cout << "PASS camera-transformed mouse target and leaving the viewport\n";
}

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
    saved.scene["device"] = int(FluidDevice::GPU); // This fixture writes a native WebGPU checkpoint.
    saved.scene["adaptive_mode"] = int(AdaptiveMode::DISABLED);
    saved.scene["simulation_time"] = 1.25;
    saved.scene["real_time"] = 4.5;
    saved.scene["projection_tolerance"] = .125;
    saved.scene["mouse_attraction"] = true;
    saved.scene["mouse_target"] = {6.0, 4.0};
    FluidGrid native_grid(saved.width, saved.height, saved.cell_size);
    native_grid.setDevice(FluidDevice::GPU);
    native_grid.setSurfaceTension(10000);
    native_grid.configureMaterials(scene.fluid.getElements(), scene.fluid.getLiquidMixture(), scene.fluid.getGasMixture());
    native_grid.upload({});
    const std::array<float, 4> parameters[] = {{6.f, 4.f, 500.f, 20.f}};
    const auto context = native_grid.getWebGPUContext();
    FluidField acceleration({saved.width, saved.height, saved.cell_size, {}, 2}, context);
    FluidFieldWebGPUGenerator generator(context, mouse_attraction_shader);
    generator.generate(acceleration, parameters);
    native_grid.setAccelerationField(acceleration);
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
    require(scene.surface_tension == 10000 && scene.fluid.getSurfaceTension() == 10000,
        "Snapshot lost the scene/native surface tension setting");
    require(scene.mouse_attraction && scene.mouse_target.x == 6 && scene.mouse_target.y == 4,
        "Snapshot lost the scene's field generator settings");
    FluidCheckpoint restored;
    require(scene.fluid.saveCheckpoint(restored) && restored.metadata == saved.checkpoint.metadata &&
        restored.data == saved.checkpoint.data, "Dropped image changed native state");
    scene.updateAccelerationField();
    require(scene.mouse_field && scene.fluid.saveCheckpoint(restored) && restored.data == saved.checkpoint.data,
        "Restored mouse producer did not rebuild equivalent values on its new context");

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
    scene.updateAccelerationField(); // The previous producer belonged to the replaced device.
    scene.restartSimulation();
    require(scene.surface_tension == 10000 && scene.fluid.getSurfaceTension() == 10000,
        "Restart lost surface tension");
    scene.mouse_attraction = false;
    scene.updateAccelerationField();
    FluidCheckpoint cleared;
    scene.fluid.saveCheckpoint(cleared);
    require(!cleared.metadata.at("acceleration_field").get<bool>(),
        "Disabling the restored mouse controller left its acceleration field active");
    scene.mouse_field_generator.reset();
    scene.mouse_field.reset();
    scene.fluid.destroy();
    scene.setSurfaceTension(0);
    scene.fluid.setDevice(FluidDevice::CPU);
    scene.fluid_device = FluidDevice::CPU;
    scene.restartSimulation();
    scene.mouse_attraction = true;
    scene.updateAccelerationField();
    require(scene.mouse_field && !scene.mouse_field->isWebGPU() && !scene.mouse_field_generator,
        "CPU mouse attraction created GPU storage or a GPU generator");
    const auto samples = scene.mouse_field->read();
    scene.mouse_target.x += 2.;
    scene.updateAccelerationField();
    require(scene.mouse_field->read() != samples, "CPU mouse field ignored a changed target");
    scene.mouse_attraction = false;
    scene.updateAccelerationField();
    require(!scene.mouse_field_active, "CPU mouse controller did not disable");
    project.stopHeadless();
    std::cout << "PASS file-drop resume, saved geometry/backend/clocks/quality, atomic rejection and recovery\n";
}
}

int main(int argc, char** argv)
{
    try
    {
        run(argc == 2 ? fs::path(argv[1]) : fs::path("snapshot-scene-tests"));
        checkMouseTarget();
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
