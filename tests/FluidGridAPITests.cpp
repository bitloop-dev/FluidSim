#include "FluidGrid.h"
#include "FluidGridCPU.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace
{
using namespace fluid;

static_assert(std::is_abstract_v<FluidGridBase>);
static_assert(std::has_virtual_destructor_v<FluidGridBase>);
static_assert(!std::is_abstract_v<FluidGrid> && !std::is_abstract_v<FluidGridCPU>);
static_assert(std::is_same_v<FluidGrid::State, FluidGridCPU::State>);
static_assert(std::is_same_v<FluidGrid::Edit, FluidGridCPU::Edit>);

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

bool sameState(const FluidGridBase::State& a, const FluidGridBase::State& b)
{
    return a.liquid_fraction == b.liquid_fraction && a.u == b.u && a.v == b.v &&
           a.pressure == b.pressure && a.liquid_mass == b.liquid_mass && a.gas_mass == b.gas_mass;
}

void sameResult(const FluidProjectionResult& a, const FluidProjectionResult& b)
{
    require(a.iterations == b.iterations && a.max_divergence == b.max_divergence && a.converged == b.converged,
            "Wrapper changed the projection result");
}

FluidGridBase::State configure(FluidGridBase& grid)
{
    SubstanceCatalogue substances;
    const auto water = substances.add({"Water", {.1f, .35f, 1.f}});
    const auto red = substances.add({"Red", {.95f, .25f, .15f}});
    const auto air = substances.add({"Air", {.015f, .025f, .06f}});
    const auto cloud = substances.add({"Cloud", {0.f, 1.f, 1.f}});
    FluidMixture liquid, gas;
    liquid.components = {{water, 100.}, {red, 300.}};
    gas.components = {{air, .1}, {cloud, .1}};
    grid.configureMaterials(std::move(substances), liquid, gas);
    grid.setOrigin({-7., 11.});
    grid.setGravityY(80.);

    const int width = grid.getWidth(), height = grid.getHeight();
    const double area = grid.getCellSize() * grid.getCellSize();
    FluidGridBase::State state;
    state.liquid_fraction.resize(width * height);
    state.liquid_mass.assign(2, std::vector<double>(width * height));
    state.gas_mass.assign(2, std::vector<double>(width * height));
    for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
    {
        const int i = y * width + x;
        const double f = std::clamp(y + 1. - (x + height * .5), 0., 1.);
        const double red_share = x < width / 4 ? 1. : 0.;
        const double cloud_share = .15 + .1 * std::sin(.2 * x);
        state.liquid_fraction[i] = f;
        state.liquid_mass[0][i] = 100. * f * area * (1. - red_share);
        state.liquid_mass[1][i] = 300. * f * area * red_share;
        state.gas_mass[0][i] = .1 * (1. - f) * area * (1. - cloud_share);
        state.gas_mass[1][i] = .1 * (1. - f) * area * cloud_share;
    }
    grid.upload(state);
    return state;
}

void timingsPreserveAllBackends()
{
    for (const auto device : {FluidDevice::CPU, FluidDevice::GPU})
    for (const auto mode : {AdaptiveMode::DISABLED, AdaptiveMode::MAX_4})
    {
        FluidGrid plain(32, 32, 1.), measured(32, 32, 1.);
        FluidAdaptiveSettings settings;
        settings.check_interval = .02;
        settings.stability.quiet_seconds = 0.;
        for (auto* grid : {&plain, &measured})
        {
            grid->setAdaptiveSettings(settings);
            grid->setDevice(device);
            grid->setAdaptiveMode(mode);
            configure(*grid);
        }
        FluidGridBase& api = measured;
        FluidGridBase::StepTimings timings;
        for (int tick = 0; tick < 8; ++tick)
        {
            const auto a = plain.step(.01, 2000, 1e-3);
            const auto b = api.step(.01, 2000, 1e-3, &timings);
            require(a.converged && b.converged, "Timed comparison did not converge");
            sameResult(a, b);
            require(sameState(plain.download(), api.download()), "Timing changed backend state or adaptive observation");
            require(timings.transport_ms > 0. && timings.pressure_ms >= 0., "Missing completed stage measurements");
            // A skipped policy check must clear the previous substep's selection cost.
            if (mode != AdaptiveMode::DISABLED && tick == 1)
                require(timings.selection_ms == 0. && timings.regrid_ms == 0., "Skipped refinement retained stale timings");
        }
        api.step(0., 2000, 1e-3, &timings);
        require(timings.transport_ms == 0. && timings.forces_ms == 0. && timings.pressure_ms == 0. &&
                timings.selection_ms == 0. && timings.regrid_ms == 0., "Zero-time step retained stale timings");
        std::cout << "PASS timed/untimed exact state/results: device=" << int(device) << " adaptive=" << int(mode) << '\n';
    }
}

void configurationAndSelection()
{
    FluidGrid grid(33, 25, 2.5);
    FluidGridBase& api = grid;
    configure(api);
    require(api.getWidth() == 33 && api.getHeight() == 25 && api.getCellSize() == 2.5,
            "Wrapper changed grid geometry");
    require(api.getOrigin().x == -7. && api.getOrigin().y == 11. && api.getGravityY() == 80.,
            "Common configuration did not reach the backend");
    require(api.getSubstances().size() == 4 && api.getSubstances()[1].name == "Red" &&
            api.getLiquidMixture().components.size() == 2 && api.getGasMixture().components.size() == 2,
            "Common materials did not reach the backend");
    require(!api.requiresRenderThread(), "CPU wrapper requested the render owner");
    const auto before = api.download();
    grid.setDevice(FluidDevice::CPU);
    grid.setAdaptiveMode(AdaptiveMode::DISABLED);
    bool rejected = false;
    try { grid.setDevice(FluidDevice::GPU); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Live GPU selection silently reset initialized state");
    for (auto mode : {AdaptiveMode::MAX_2, AdaptiveMode::MAX_4, AdaptiveMode::MAX_8,
                      AdaptiveMode::MAX_16, AdaptiveMode::MAX_32})
    {
        rejected = false;
        try { grid.setAdaptiveMode(mode); }
        catch (const std::logic_error&) { rejected = true; }
        require(rejected, "Live adaptive selection silently reset initialized state");
    }
    require(grid.getDevice() == FluidDevice::CPU && grid.getAdaptiveMode() == AdaptiveMode::DISABLED &&
            sameState(before, api.download()), "Selection reset or mutated the CPU state");
}

void steppingMatchesDirectCPU()
{
    FluidGridCPU cpu(33, 25, 2.5);
    FluidGrid facade(33, 25, 2.5);
    FluidGridBase& direct = cpu;
    FluidGridBase& api = facade;
    const auto initial = configure(direct);
    configure(api);
    require(sameState(direct.download(), api.download()), "Common upload differs from CPU upload");
    sameResult(direct.step(0.), api.step(0.));
    for (int step = 0; step < 64; ++step)
    {
        if (step == 32) { direct.setGravityY(110.); api.setGravityY(110.); }
        const auto expected = direct.step(.004, 2000, 1e-8);
        const auto actual = api.step(.004, 2000, 1e-8);
        require(actual.converged, "Wrapper physics failed to converge");
        sameResult(expected, actual);
        require(sameState(direct.download(), api.download()), "Wrapper changed committed CPU fields");
    }
    require(direct.maxDivergence() == api.maxDivergence(), "Common divergence differs from CPU");
    const auto expected = direct.step(.01, 0, 1e-12);
    const auto actual = api.step(.01, 0, 1e-12);
    sameResult(expected, actual);
    require(!actual.converged && sameState(direct.download(), api.download()), "Wrapper hid projection failure");
    direct.destroy(); api.destroy();
    direct.upload(initial); api.upload(initial);
    require(sameState(direct.download(), api.download()), "Wrapper initialization reset differs from CPU");
}

void editingAndColorsUseCommonAPI()
{
    FluidGridCPU cpu(33, 25, 2.5);
    FluidGrid facade(33, 25, 2.5);
    FluidGridBase& direct = cpu;
    FluidGridBase& api = facade;
    configure(direct); configure(api);
    sameResult(direct.step(.004, 2000, 1e-8), api.step(.004, 2000, 1e-8));
    auto expected_edit = direct.beginEdit();
    auto edit = api.beginEdit();
    const auto before = api.download();
    for (auto* session : {&expected_edit, &edit})
    {
        session->reserve(4);
        session->setCell(3, 2, Phase::Liquid, 0);
        session->setCell(4, 2, Phase::Liquid, 1);
        session->setCell(4, 2, Phase::Gas, 3);
    }
    require(sameState(before, api.download()), "Staged common edits touched physical state");
    auto moved = std::move(edit);
    expected_edit.commit(); moved.commit();
    require(moved.empty() && sameState(direct.download(), api.download()), "Common edit forwarding changed replacements");
    std::vector<float> expected_pixels, pixels(7, -1.f);
    direct.writeColorPixels(expected_pixels); api.writeColorPixels(pixels);
    require(pixels == expected_pixels && pixels.size() == 33 * 25 * 4, "Wrapper changed derived display pixels");
    for (int channel = 0; channel < 3; ++channel)
    {
        require(pixels[(2 * 33 + 3) * 4 + channel] == api.getSubstances()[0].color[channel], "Liquid edit colour is incorrect");
        require(pixels[(2 * 33 + 4) * 4 + channel] == api.getSubstances()[3].color[channel], "Last-write gas colour is incorrect");
    }
    bool rejected = false;
    moved.setCell(-1, 2, Phase::Liquid, 0);
    const auto committed = api.download();
    try { moved.commit(); }
    catch (const std::out_of_range&) { rejected = true; }
    require(rejected && moved.size() == 1 && sameState(committed, api.download()), "Common edit lost batch atomicity");
    moved.clear(); moved.commit();
    sameResult(direct.step(.004, 2000, 1e-8), api.step(.004, 2000, 1e-8));
    require(sameState(direct.download(), api.download()), "Common edit changed pressure/cache history");
    api.configureMaterials({}, {}, {}); api.upload({});
    api.writeColorPixels(pixels);
    for (size_t i = 0; i < pixels.size(); ++i)
        require(pixels[i] == (i % 4 == 3 ? 1.f : 0.f), "Empty catalogue left stale display colours");
}
}

int main()
{
    try
    {
        configurationAndSelection(); std::cout << "PASS common configuration and explicit CPU selection\n";
        steppingMatchesDirectCPU(); std::cout << "PASS exact polymorphic CPU step/state/failure parity\n";
        editingAndColorsUseCommonAPI(); std::cout << "PASS common staged edits, history and colour output\n";
        timingsPreserveAllBackends();
        return 0;
    }
    catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
