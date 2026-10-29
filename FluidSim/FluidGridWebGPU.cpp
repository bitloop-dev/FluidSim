#include "FluidGridWebGPU.h"
#include "UniformWebGPU/FluidGridGPU.h"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fluid
{
FluidGridWebGPU::FluidGridWebGPU(int width, int height, double cell_size)
    : engine(std::make_unique<FluidGridGPU>(width, height, cell_size)) {}
FluidGridWebGPU::~FluidGridWebGPU() = default;
int FluidGridWebGPU::getWidth() const { return engine->width; }
int FluidGridWebGPU::getHeight() const { return engine->height; }
double FluidGridWebGPU::getCellSize() const { return engine->cell_size; }
bl::DVec2 FluidGridWebGPU::getOrigin() const { return engine->origin; }
void FluidGridWebGPU::setOrigin(bl::DVec2 origin) { engine->origin = origin; }
void FluidGridWebGPU::configureMaterials(ElementCatalogue elements, FluidMixture liquid, FluidMixture gas)
{
    engine->elements = std::move(elements);
    engine->liquid_mixture = std::move(liquid);
    engine->gas_mixture = std::move(gas);
}
const ElementCatalogue& FluidGridWebGPU::getElements() const { return engine->elements; }
const FluidMixture& FluidGridWebGPU::getLiquidMixture() const { return engine->liquid_mixture; }
const FluidMixture& FluidGridWebGPU::getGasMixture() const { return engine->gas_mixture; }
double FluidGridWebGPU::getGravityY() const { return engine->gravity_y; }
void FluidGridWebGPU::setGravityY(double acceleration) { engine->gravity_y = acceleration; }
double FluidGridWebGPU::getSurfaceTension() const { return engine->getSurfaceTension(); }
void FluidGridWebGPU::setSurfaceTension(double coefficient) { engine->setSurfaceTension(coefficient); }
void FluidGridWebGPU::setAccelerationField(const FluidField& field)
{
    requireUploaded();
    engine->setAccelerationField(field);
}

void FluidGridWebGPU::setWebGPUContext(const FluidWebGPUContext& context)
{
    engine->setWebGPUContext(context);
}

FluidWebGPUContext FluidGridWebGPU::getWebGPUContext() { return engine->getWebGPUContext(); }

void FluidGridWebGPU::clearAccelerationField()
{
    engine->clearAccelerationField();
}

void FluidGridWebGPU::upload(const State& state) { engine->upload(convertFluidState<float>(state)); }
FluidGridBase::State FluidGridWebGPU::download() const
{
    requireUploaded(); return convertFluidState<double>(engine->download());
}
void FluidGridWebGPU::destroy() { engine->destroy(); }
bool FluidGridWebGPU::saveCheckpoint(FluidCheckpoint& checkpoint) const
{
    requireUploaded();
    return engine->saveCheckpoint(checkpoint);
}
bool FluidGridWebGPU::loadCheckpoint(const FluidCheckpoint& checkpoint) { return engine->loadCheckpoint(checkpoint); }
void FluidGridWebGPU::requireUploaded() const
{
    if (!engine->gpu) throw std::logic_error("Initialize the WebGPU grid with upload before using its device state");
}
FluidGridBase::ProjectionResult FluidGridWebGPU::step(double dt, int max_iterations, double divergence_tolerance, StepTimings* timings)
{
    if (timings) *timings = {};
    requireUploaded();
    // The native API returns an empty result at zero time. The common contract
    // reports current divergence without advancing or changing physical fields.
    if (dt == 0.)
    {
        const double divergence = engine->maxDivergence();
        return {0, divergence, std::isfinite(divergence) && divergence <= divergence_tolerance, 0, 0};
    }
    return engine->step(dt, max_iterations, divergence_tolerance, timings);
}
double FluidGridWebGPU::maxDivergence() const { requireUploaded(); return engine->maxDivergence(); }
void FluidGridWebGPU::writeColorPixels(std::vector<float>& pixels) const
{
    requireUploaded(); engine->writeColorPixels(pixels);
}
void FluidGridWebGPU::commitEdits(detail::CellEditBatch& batch) { engine->commitEdits(batch); }
}
