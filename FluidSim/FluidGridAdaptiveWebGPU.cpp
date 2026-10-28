#include "FluidGridAdaptiveWebGPU.h"
#include "AdaptiveWebGPU/FluidGridAdaptiveGPU.h"
#include "AdaptiveWebGPU/FluidCoarseningGPU.h"
#include "Adaptive/FluidAdaptiveRaster.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace fluid
{
FluidGridAdaptiveWebGPU::FluidGridAdaptiveWebGPU(int w, int h, double dx, int scale, FluidAdaptiveSettings value)
    : width(w), height(h), max_cell_scale(scale), cell_size(dx), settings(value)
{
    // Project geometry is aligned: the maximum scale divides both dimensions.
    if (scale != 2 && scale != 4 && scale != 8 && scale != 16 && scale != 32)
        throw std::invalid_argument("Unsupported adaptive WebGPU cell scale");
}
FluidGridAdaptiveWebGPU::~FluidGridAdaptiveWebGPU() = default;
void FluidGridAdaptiveWebGPU::configureMaterials(ElementCatalogue catalogue, FluidMixture liquid_mixture, FluidMixture gas_mixture)
{
    destroy(); elements = std::move(catalogue); liquid = std::move(liquid_mixture); gas = std::move(gas_mixture);
}
const FluidGridAdaptiveGPU& FluidGridAdaptiveWebGPU::native() const
{
    if (!engine) throw std::logic_error("Initialize the adaptive WebGPU grid with upload before using its device state");
    return *engine;
}
void FluidGridAdaptiveWebGPU::setAdaptiveSettings(const FluidAdaptiveSettings& value)
{
    if (settings == value) return;
    settings = value; if (selector) selector->reset(); next_refinement_time = simulation_time;
}
void FluidGridAdaptiveWebGPU::upload(const State& state)
{
    auto initialized = std::make_unique<FluidGridAdaptiveGPU>(width, height, cell_size, elements, liquid, gas, GasLayout{});
    initialized->upload(state); // Preserve the native 1e-4 initialization projection policy.
    selector.reset(); engine = std::move(initialized); selector = std::make_unique<FluidCoarseningGPU>();
    simulation_time = next_refinement_time = 0.; projection_tolerance = 1e-4; refinement_checks = 0;
}
void FluidGridAdaptiveWebGPU::destroy()
{
    selector.reset(); engine.reset();
    simulation_time = next_refinement_time = 0.; refinement_checks = 0;
}
FluidGridBase::ProjectionResult FluidGridAdaptiveWebGPU::step(double dt, int max_iterations, double tolerance, StepTimings* timings)
{
    if (timings) *timings = {};
    native();
    if (dt == 0.) return engine->step(0., tolerance).projection;
    projection_tolerance = tolerance;
    if (settings.automatic_refinement && simulation_time >= next_refinement_time)
    {
        using Clock = std::chrono::steady_clock;
        const auto selection_started = timings ? Clock::now() : Clock::time_point{};
        selector->update(*engine, settings.tolerances, settings.stability, simulation_time,
                         settings.prediction_horizon, max_cell_scale);
        const auto layout = selector->proposeLayout(*engine);
        if (timings)
            timings->selection_ms = std::chrono::duration<double, std::milli>(Clock::now() - selection_started).count();
        const auto regrid_started = timings ? Clock::now() : Clock::time_point{};
        next_refinement_time = simulation_time + settings.check_interval; ++refinement_checks;
        if (engine->regrid(layout))
        {
            try
            {
                auto restored = engine->project(tolerance, 0., {}, max_iterations);
                if (!restored.converged) return restored;
            }
            catch (const FluidAdaptiveGPUProjectionFailure& failure)
            {
                return {failure.iterations, engine->maxDivergence(), false};
            }
        }
        if (timings)
        {
            engine->synchronize();
            timings->regrid_ms = std::chrono::duration<double, std::milli>(Clock::now() - regrid_started).count();
        }
    }
    ProjectionResult result;
    try
    {
        const auto stepped = engine->step(dt, tolerance, 0., {0., gravity_y}, max_iterations, timings != nullptr);
        result = stepped.projection;
        if (timings)
        {
            timings->transport_ms = stepped.transportMs;
            timings->pressure_ms = stepped.projectionMs;
        }
    }
    catch (const FluidAdaptiveGPUProjectionFailure& failure)
    {
        result = {failure.iterations, engine->maxDivergence(), false};
    }
    simulation_time += dt;
    return result;
}
double FluidGridAdaptiveWebGPU::maxDivergence() const { native(); return engine->maxDivergence(); }
void FluidGridAdaptiveWebGPU::writeColorPixels(std::vector<float>& pixels) const
{
    native(); engine->writeColorPixels(pixels);
}

FluidGridBase::State FluidGridAdaptiveWebGPU::download() const
{
    const auto& gpu = native();
    struct Observation { std::vector<double> q, pressure, u, v; std::vector<int> active; };
    // Only explicit download constructs these temporary native-resolution arrays.
    // They are never used by physics, refinement or colour rendering.
    std::vector<Observation> observed(gpu.levels().size());
    std::vector<FluidCellView::Patch> patches;
    auto read = [&](const webgpu::Field& field, std::vector<double>& values)
    {
        const auto floats = gpu.commands().read(field); values.assign(floats.begin(), floats.end());
    };
    auto array = [](const AmrFieldLayout2D::Patch& p, const std::vector<double>& values)
    {
        return AmrArray2D<const double>{values.data() + p.offset, p.box.x - p.ghosts, p.box.y - p.ghosts, p.stride, p.plane};
    };
    for (size_t lev = 0; lev < observed.size(); ++lev)
    {
        const auto& level = *gpu.levels()[lev]; auto& data = observed[lev];
        read(level.conserved, data.q); read(level.velocity[0], data.u); read(level.velocity[1], data.v);
        const auto mask = gpu.commands().read(level.active); data.active.resize(mask.size());
        std::transform(mask.begin(), mask.end(), data.active.begin(), [](float value) { return int(value != 0.f); });
        const auto pressure = gpu.commands().read(level.pressure);
        const size_t plane = size_t(level.pressure.width) * level.pressure.height;
        data.pressure.resize(plane);
        for (size_t i = 0; i < plane; ++i) data.pressure[i] = double(pressure[i]) + pressure[i + plane];
        const int scale = int(std::lround(level.geometry.cell_size / cell_size));
        for (size_t index = 0; index < level.qLayout.patches.size(); ++index)
        {
            const auto& q = level.qLayout.patches[index]; const auto& active = level.cellLayout.patches[index];
            FluidCellView::Patch p{q.box.x * scale, q.box.y * scale, q.box.width * scale, q.box.height * scale, scale};
            p.q = array(q, data.q); p.pressure = array(level.pressureLayout.patches[index], data.pressure);
            p.u = array(level.faceLayout[0].patches[index], data.u); p.v = array(level.faceLayout[1].patches[index], data.v);
            p.active = {data.active.data() + active.offset, active.box.x - active.ghosts, active.box.y - active.ghosts, active.stride, active.plane};
            patches.push_back(p);
        }
    }
    const FluidCellView view(width, height, cell_size, gpu.materialMixture(), gpu.elements(), std::move(patches),
                             gpu.acceleration(), gpu.pressureInterval(), gpu.liquidIngredientCount(), &gpu);
    return adaptiveRaster(view);
}

void FluidGridAdaptiveWebGPU::refineForEdits(const detail::CellEditBatch& batch)
{
    if (engine->layout().baseRatio == 1) return;
    const auto& levels = engine->levels();
    auto isFine = [&](const auto& cell)
    {
        for (const auto& level : levels)
            if (std::lround(level->geometry.cell_size / cell_size) == 1)
                for (const auto& box : level->boxes) if (box.contains(cell.x, cell.y)) return true;
        return false;
    };
    if (std::all_of(batch.cells.begin(), batch.cells.end(), isFine)) return;
    std::vector<FluidCoarseningPreview::Requests> requests;
    for (int scale = 2; scale <= max_cell_scale; scale *= 2)
    {
        const int tile = std::max(16, scale), nx = (width + tile - 1) / tile;
        auto& request = requests.emplace_back(scale, std::vector<std::uint8_t>(size_t(nx) * ((height + tile - 1) / tile)));
        auto mark = [&](int x, int y) { request.tiles[(y / tile) * nx + x / tile] = 1; };
        // Retain native resolved footprints. Geometry alone identifies them;
        // no material/velocity state or active-mask readback is required.
        for (const auto& level : levels)
        {
            const int actual = int(std::lround(level->geometry.cell_size / cell_size));
            if (actual >= scale) continue;
            for (const auto& box : level->boxes)
                for (int y = box.y * actual; y < (box.y + box.height) * actual; y += actual)
                    for (int x = box.x * actual; x < (box.x + box.width) * actual; x += actual) mark(x, y);
        }
        for (const auto& cell : batch.cells) mark(cell.x, cell.y);
    }
    if (engine->regrid(FluidCoarseningPreview::clusterRequests(width, height, requests)))
        if (!engine->project(projection_tolerance).converged)
            throw std::runtime_error("Adaptive WebGPU edit refinement projection did not converge");
}
void FluidGridAdaptiveWebGPU::commitEdits(detail::CellEditBatch& batch)
{
    native(); batch.prepare(width, height, cell_size * cell_size, liquid, gas);
    refineForEdits(batch); engine->replaceMaterials(batch.cells);
    selector->reset(); next_refinement_time = simulation_time;
}
}
