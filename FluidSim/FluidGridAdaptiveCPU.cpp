#include "FluidGridAdaptiveCPU.h"
#include "Adaptive/FluidAdaptiveRaster.h"
#include "Adaptive/FluidGridAdaptive.h"
#include "Adaptive/FluidCellView.h"
#include "Adaptive/AmrGasRuntime.h"
#include <bitloop/core/threads.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace fluid
{
FluidGridAdaptiveCPU::FluidGridAdaptiveCPU(int w, int h, double dx, int scale, FluidAdaptiveSettings value)
    : width(w), height(h), max_cell_scale(scale), cell_size(dx), settings(value),
      runtime(std::make_unique<AmrGasRuntime>())
{
    // Project geometry is aligned: the maximum scale divides both dimensions.
    if (scale != 2 && scale != 4 && scale != 8 && scale != 16 && scale != 32)
        throw std::invalid_argument("Unsupported adaptive CPU cell scale");
}
FluidGridAdaptiveCPU::~FluidGridAdaptiveCPU() = default;

void FluidGridAdaptiveCPU::configureMaterials(ElementCatalogue catalogue, FluidMixture liquid_mixture, FluidMixture gas_mixture)
{
    elements = std::move(catalogue); liquid = std::move(liquid_mixture); gas = std::move(gas_mixture);
    destroy(); // Membership/order changes require initialization, as on the common API.
}
void FluidGridAdaptiveCPU::setAdaptiveSettings(const FluidAdaptiveSettings& value)
{
    if (settings == value) return;
    settings = value; selector.reset(); next_refinement_time = simulation_time;
}
const FluidGridAdaptive& FluidGridAdaptiveCPU::native() const
{
    if (!engine) throw std::logic_error("Initialize the adaptive grid with upload before using its state");
    return *engine;
}
void FluidGridAdaptiveCPU::upload(const State& state)
{
    // Start fully resolved. The existing selector may subsequently release quiet gas.
    auto initialized = std::make_unique<FluidGridAdaptive>(width, height, cell_size, elements, liquid, gas, GasLayout{});
    initialized->upload(state);
    engine = std::move(initialized);
    simulation_time = next_refinement_time = 0.; refinement_checks = 0; selector.reset();
}
void FluidGridAdaptiveCPU::destroy()
{
    engine.reset(); selector.reset();
    simulation_time = next_refinement_time = 0.; refinement_checks = 0;
}
FluidGridBase::ProjectionResult FluidGridAdaptiveCPU::step(double dt, int max_iterations, double tolerance, StepTimings* timings)
{
    if (timings) *timings = {};
    native();
    if (dt == 0.) return engine->step(0., tolerance).projection;
    projection_tolerance = tolerance;
    if (settings.automatic_refinement && simulation_time >= next_refinement_time)
    {
        using Clock = std::chrono::steady_clock;
        const auto selection_started = timings ? Clock::now() : Clock::time_point{};
        selector.update(*engine, settings.tolerances, settings.stability, simulation_time,
                        settings.prediction_horizon, max_cell_scale);
        const auto layout = selector.proposeLayout(*engine);
        if (timings)
            timings->selection_ms = std::chrono::duration<double, std::milli>(Clock::now() - selection_started).count();
        const auto regrid_started = timings ? Clock::now() : Clock::time_point{};
        next_refinement_time = simulation_time + settings.check_interval;
        ++refinement_checks;
        // Regrid changes the predictor; restore projected faces before any transport.
        if (engine->regrid(layout))
        {
            try
            {
                auto restored = engine->project(tolerance, 0., {}, max_iterations);
                if (!restored.converged) return restored;
            }
            catch (const FluidAdaptiveProjectionFailure& failure)
            {
                return {failure.iterations, engine->maxDivergence(), false};
            }
        }
        if (timings)
        {
            timings->regrid_ms = std::chrono::duration<double, std::milli>(Clock::now() - regrid_started).count();
        }
    }
    ProjectionResult result;
    try
    {
        const auto stepped = engine->step(dt, tolerance, 0., {0., gravity_y}, max_iterations);
        result = stepped.projection;
        if (timings)
        {
            timings->transport_ms = stepped.transportMs;
            timings->pressure_ms = stepped.projectionMs;
        }
    }
    catch (const FluidAdaptiveProjectionFailure& failure)
    {
        result = {failure.iterations, engine->maxDivergence(), false};
    }
    simulation_time += dt;
    return result;
}
double FluidGridAdaptiveCPU::maxDivergence() const
{
    native(); return engine->maxDivergence();
}

FluidGridBase::State FluidGridAdaptiveCPU::download() const
{
    return adaptiveRaster(FluidCellView(native()));
}

void FluidGridAdaptiveCPU::writeColorPixels(std::vector<float>& pixels) const
{
    const FluidCellView view(native());
    pixels.resize(size_t(width) * height * 4);
    const int patch_count = int(view.patches.size());
    // Same bounded material plane and traversal as the development display.
    bl::Thread::forEachBatch(patch_count, [&](int first, int last)
    {
        FluidCellView::MaterialPlane plane;
        for (int index = first; index < last; ++index)
        {
            const auto& p = view.patches[index];
            for (int y = p.y; y < p.y + p.height; y += p.scale)
            for (int x = p.x; x < p.x + p.width; x += p.scale)
            {
                if (!p.active(x / p.scale, y / p.scale)) continue;
                if (p.scale > 1) view.reconstruct(plane, x, y, p.scale);
                for (int dy = 0; dy < p.scale; ++dy) for (int dx = 0; dx < p.scale; ++dx)
                {
                    float color[4] = {0, 0, 0, 1};
                    for (int k = 0; k < view.components(); ++k)
                    {
                        const auto& component = view.mixture->components[k];
                        const auto& c = (*view.catalogue)[component.element].color;
                        const double q = p.scale == 1 ? p.q(x, y, k) : plane.value(k, (dx + .5) / p.scale - .5, (dy + .5) / p.scale - .5);
                        for (int channel = 0; channel < 3; ++channel) color[channel] += float(q / component.density * c[channel]);
                    }
                    std::copy_n(color, 4, pixels.data() + size_t((y + dy) * width + x + dx) * 4);
                }
            }
        }
    }, std::min({8, int(bl::Thread::threadCount()), patch_count}));
}

void FluidGridAdaptiveCPU::refineForEdits(const detail::CellEditBatch& batch)
{
    if (engine->layout().baseRatio == 1) return;
    const FluidCellView view(native());
    if (std::none_of(batch.cells.begin(), batch.cells.end(), [&](const auto& cell)
        { return view.at(cell.x, cell.y).patch->scale > 1; })) return;
    std::vector<FluidCoarseningPreview::Requests> requests;
    for (int scale = 2; scale <= max_cell_scale; scale *= 2)
    {
        const int tile = std::max(16, scale), nx = (width + tile - 1) / tile;
        auto& request = requests.emplace_back(scale, std::vector<std::uint8_t>(size_t(nx) * ((height + tile - 1) / tile)));
        auto mark = [&](int x, int y) { request.tiles[(y / tile) * nx + x / tile] = 1; };
        // Preserve every currently resolved footprint; editing never discards detail elsewhere.
        for (const auto& p : view.patches) if (p.scale < scale)
            for (int y = p.y; y < p.y + p.height; y += p.scale)
                for (int x = p.x; x < p.x + p.width; x += p.scale)
                    if (p.active(x / p.scale, y / p.scale)) mark(x, y);
        for (const auto& cell : batch.cells) mark(cell.x, cell.y);
    }
    if (engine->regrid(FluidCoarseningPreview::clusterRequests(width, height, requests)))
        if (!engine->project(projection_tolerance).converged)
            throw std::runtime_error("Adaptive edit refinement projection did not converge");
}

void FluidGridAdaptiveCPU::commitEdits(detail::CellEditBatch& batch)
{
    native();
    batch.prepare(width, height, cell_size * cell_size, liquid, gas);
    refineForEdits(batch);
    const FluidCellView view(*engine);
    const int nc = view.components();
    std::vector<double> delta(nc + 2);
    for (const auto& edit : batch.cells)
    {
        const auto c = view.at(edit.x, edit.y); // Locally refined to one finest cell.
        const double rho = edit.mass / (cell_size * cell_size);
        for (int k = 0; k < nc; ++k) delta[k] = (k == edit.component ? rho : 0.) - c.density(k);
        for (int d = 0; d < 2; ++d) delta[nc + d] = rho * c.velocity(d) - c.momentum(d);
        // Apply the exact fine replacement and its conservative contribution to
        // covered parents. Existing projected faces stay unchanged, as on CPU.
        for (const auto& level : engine->levels())
        {
            const int scale = int(std::lround(level->geometry.cell_size / cell_size));
            const int x = edit.x / scale, y = edit.y / scale;
            for (int p = 0; p < level->conserved.size(); ++p)
            {
                const auto& b = level->conserved.box(p);
                if (x < b.smallEnd(0) || x > b.bigEnd(0) || y < b.smallEnd(1) || y > b.bigEnd(1)) continue;
                const auto q = level->conserved.array(p);
                for (int k = 0; k < nc + 2; ++k) q(x, y, k) += delta[k] / (scale * scale);
            }
        }
    }
    selector.reset(); next_refinement_time = simulation_time;
}
}
