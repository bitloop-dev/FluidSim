#pragma once
#include "FluidSim.h"
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <fstream>
#include <iostream>

namespace fluid::test
{
struct AppearanceProject : bl::BasicProject
{
    bl::SceneHandle<FluidSim::FluidSim_Scene> source;
    void projectPrepare(bl::Layout&) override { source = create<FluidSim::FluidSim_Scene>(); }
};

inline void liquidAppearance(const std::filesystem::path& output, const std::filesystem::path& saved_wave = {})
{
    auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    for (auto device : {FluidDevice::CPU, FluidDevice::GPU})
    for (auto mode : {AdaptiveMode::DISABLED, AdaptiveMode::MAX_4})
    for (int count : {1, 18})
    {
        FluidGrid grid(64, 48, .5);
        grid.setDevice(device);
        grid.setAdaptiveMode(mode);
        ElementCatalogue elements;
        FluidMixture liquid, gas;
        for (int k = 0; k < count; ++k)
            liquid.components.push_back({elements.add({"Liquid", {.1f, .35f, 1.f}}), 10. + k});
        gas.components.push_back({elements.add({"Gas", {.015f, .025f, .06f}}), .1});
        grid.configureMaterials(elements, liquid, gas);
        FluidGrid::State state;
        state.liquid_fraction.assign(64 * 48, .37);
        for (auto& ingredient : liquid.components)
            state.liquid_mass.emplace_back(64 * 48, .37 * ingredient.density * .25 / count);
        state.gas_mass.emplace_back(64 * 48, .63 * .1 * .25);
        grid.upload(state);
        const auto before = grid.download();
        std::vector<float> opaque, coverage;
        grid.writeColorPixels(opaque);
        grid.writeColorPixels(coverage, true);
        for (size_t i = 0; i < opaque.size(); ++i)
            check(i % 4 == 3 ? opaque[i] == 1.f && std::abs(coverage[i] - .37f) < 1e-5f : opaque[i] == coverage[i],
                "Liquid coverage changed RGB or disagrees across backends");
        const auto after = grid.download();
        check(before.liquid_mass == after.liquid_mass && before.gas_mass == after.gas_mass &&
            before.liquid_fraction == after.liquid_fraction && before.u == after.u && before.v == after.v && before.pressure == after.pressure,
            "Appearance observation changed physical state");
    }

    check(SDL_Init(SDL_INIT_VIDEO), "SDL initialization failed");
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    auto* window = SDL_CreateWindow("Liquid appearance tests", 32, 32, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    check(window != nullptr, "GL test window failed");
    auto context = SDL_GL_CreateContext(window);
    check(context != nullptr, "GL test context failed");
    check(gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress) != 0, "GL loader failed");

    FluidGridDisplay display;
    bl::ShaderSurface surface;
    surface.setFragmentSource(FluidSim::FluidSim_Scene::fluid_shader_src);
    auto render = [&](int w, int h, const std::vector<float>& rgba, float range = 4.f, float brightness = .6f)
    {
        surface.render(w, h, [&](const bl::ShaderSurface& shader)
        {
            shader.bindTexture2D("u_color", display.texture(w, h, rgba));
            shader.bindTexture2D("u_proximity", display.proximityTexture(w, h, rgba, range));
            shader.setUniform1f("u_deep_brightness", brightness);
        });
        check(surface.texture() != 0, "Actual scene GLSL failed to compile");
        bl::bytebuf result(size_t(w) * h * 4);
        glBindTexture(GL_TEXTURE_2D, surface.texture());
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, result.data());
        for (size_t i = 3; i < result.size(); i += 4) check(result[i] == 255, "Water revealed the background");
        return result;
    };
    auto save = [&](const char* name, int w, int h, const bl::bytebuf& rgba)
    {
        bl::bytebuf image;
        check(bl::webp_snapshot_encode_rgba(rgba.data(), w, h, 100, 6, true, 100, 100, image), "WebP encode failed");
        std::ofstream file(output / name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(image.data()), std::streamsize(image.size()));
    };
    std::filesystem::create_directories(output);
    constexpr int w = 63, h = 79;
    std::vector<float> pixels(size_t(w) * h * 4);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
    {
        const bool gas = y < 16 || (x - 32) * (x - 32) + (y - 45) * (y - 45) < 16;
        float* p = pixels.data() + (y * w + x) * 4;
        p[0] = gas ? .015f : .1f; p[1] = gas ? .025f : .35f; p[2] = gas ? .06f : 1.f;
        p[3] = gas ? 0.f : 1.f;
    }
    const auto shaded = render(w, h, pixels);
    auto value = [&](int x, int y, int c) { return shaded[(y * w + x) * 4 + c]; };
    for (int c = 0; c < 3; ++c)
    {
        check(value(3, 18, c) > value(3, 22, c) && value(3, 22, c) > value(3, 27, c), "Liquid does not darken with depth");
        check(value(37, 45, c) > value(57, 45, c), "Bubble did not lighten nearby liquid");
        check(std::abs(int(value(32, 45, c)) - int(std::lround(pixels[(45 * w + 32) * 4 + c] * 255))) <= 1,
            "Gas colour changed");
    }
    save("pool-bubble.webp", w, h, shaded);
    // Resize after a gas-containing image: no stale seeds or artificial light at tank walls.
    pixels.resize(3 * 5 * 4);
    for (size_t i = 0; i < pixels.size(); i += 4)
    { pixels[i] = .1f; pixels[i + 1] = .35f; pixels[i + 2] = 1.f; pixels[i + 3] = 1.f; }
    const auto full = render(3, 5, pixels);
    for (size_t i = 0; i < full.size(); ++i)
        check(std::abs(int(full[i]) - int(i % 4 == 3 ? 255 : std::lround(pixels[i] * .6f * 255))) <= 1,
            "Full tank did not retain its uniform deep tint");

    // Every representable gas fraction contributes, with no discontinuity at50%.
    pixels.resize(size_t(w) * h * 4);
    for (size_t i = 0; i < pixels.size(); i += 4)
    { pixels[i] = .1f; pixels[i + 1] = .35f; pixels[i + 2] = 1.f; pixels[i + 3] = 1.f; }
    std::vector<float> proximity(size_t(w) * h);
    float previous = 0.f, below_half = 0.f;
    for (float gas : {1e-6f, 1e-4f, .01f, .49f, .5f, .51f, 1.f})
    {
        pixels[(39 * w + 31) * 4 + 3] = 1.f - gas;
        glBindTexture(GL_TEXTURE_2D, display.proximityTexture(w, h, pixels, 4.f));
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, proximity.data());
        const float influence = proximity[39 * w + 32];
        check(influence > previous, "Tiny gas coverage disappeared or responded discontinuously");
        if (gas == .49f) below_half = influence;
        if (gas == .51f) check(influence / below_half < 1.05f, "Old50% bubble cutoff still flickers");
        previous = influence;
        check(influence > 100.f * proximity[39 * w + 61], "Bubble influence is not localized");
    }
    float minimum_peak = 1.f, maximum_peak = 0.f;
    for (int frame = 0; frame <= 32; ++frame)
    {
        const float t = frame / 32.f;
        pixels[(39 * w + 31) * 4 + 3] = 1.f - .25f * (1.f - t);
        pixels[(39 * w + 32) * 4 + 3] = 1.f - .25f * t;
        glBindTexture(GL_TEXTURE_2D, display.proximityTexture(w, h, pixels, 4.f));
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, proximity.data());
        const float peak = *std::max_element(proximity.begin(), proximity.end());
        minimum_peak = std::min(minimum_peak, peak);
        maximum_peak = std::max(maximum_peak, peak);
    }
    check(minimum_peak > 0.f && maximum_peak / minimum_peak < 1.15f,
        "Moving a subcell bubble between cells caused an illumination jump");

    if (!saved_wave.empty())
    {
        const auto snapshot = readFluidSnapshot(saved_wave);
        FluidGrid grid(snapshot.width, snapshot.height, snapshot.cell_size);
        grid.setDevice(FluidDevice::GPU);
        check(grid.loadCheckpoint(snapshot.checkpoint), "Could not load reference wave");
        grid.writeColorPixels(pixels);
        save("wave-flat.webp", snapshot.width, snapshot.height, fluidSnapshotPixels(pixels));

        bl::ImDebugLog log;
        bl::CaptureContext capture;
        AppearanceProject project;
        project.configureHeadless(0, &log, {800, 600}, &capture);
        project.startHeadless();
        auto& scene = *project.source.get();
        scene.resumeFromImage(saved_wave);
        check(scene.simulation_time == snapshot.scene.at("simulation_time").get<double>(), "Snapshot did not restore the exact moment");
        grid.writeColorPixels(pixels, true);
        const auto expected = render(snapshot.width, snapshot.height, pixels,
            scene.liquid_tint_depth, scene.liquid_deep_brightness);
        save("wave-shaded.webp", snapshot.width, snapshot.height, expected);
        // Native snapshots must render fully even after a viewport leaves a clip.
        glEnable(GL_SCISSOR_TEST);
        glScissor(0, 0, 1, 1);
        bl::CaptureFrame captured;
        check(scene.captureFrame(bl::CapturePreset{}, captured), "Shaded native capture failed");
        check(glIsEnabled(GL_SCISSOR_TEST), "Snapshot changed the viewport clip state");
        glDisable(GL_SCISSOR_TEST);
        check(captured.frame.data == expected, "Native snapshot pixels differ from the actual scene GLSL");
        check(captured.frame.data != fluidSnapshotPixels(pixels), "Native capture still bypasses the shader");
        const auto restored = decodeFluidSnapshot(captured.frame.payload);
        check(restored.checkpoint.data == snapshot.checkpoint.data && restored.checkpoint.metadata == snapshot.checkpoint.metadata,
            "Shaded snapshot changed the exact native checkpoint");
        check(restored.scene.at("liquid_tint_depth") == scene.liquid_tint_depth &&
            restored.scene.at("liquid_deep_brightness") == scene.liquid_deep_brightness,
            "Snapshot did not save appearance controls");
        bl::bytebuf webp;
        check(bl::webp_snapshot_encode_rgba(captured.frame.data.data(), snapshot.width, snapshot.height, 100, 6, true, 100, 100, webp), "Capture encode failed");
        check(bl::webp_set_save_string_as_xmp_inplace(webp, captured.frame.payload), "Capture XMP encode failed");
        const auto image = output / "snapshot-shaded.webp";
        std::ofstream file(image, std::ios::binary);
        file.write(reinterpret_cast<const char*>(webp.data()), std::streamsize(webp.size()));
        file.close();
        scene.resumeFromImage(image);
        bl::CaptureFrame recaptured;
        check(scene.captureFrame(bl::CapturePreset{}, recaptured) && recaptured.frame.data == captured.frame.data &&
            recaptured.frame.payload == captured.frame.payload, "Saved shaded snapshot did not round-trip exactly");
        project.stopHeadless();
        std::cout << "PASS supplied XMP moment, shaded capture pixels, clip restoration and exact image/state round-trip\n";
    }
    check(glGetError() == GL_NO_ERROR, "Appearance produced an OpenGL error");
    display.destroy();
    surface.destroy();
    SDL_GL_DestroyContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << "PASS four-backend coverage, unchanged RGB/state, continuous localized bubbles, opacity and full-tank resize\n";
}
}
