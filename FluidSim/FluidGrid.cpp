#include "FluidGrid.h"

#include "FluidGridCPU.h"
#include "FluidGridAdaptiveCPU.h"
#include "FluidGridWebGPU.h"
#include "FluidGridAdaptiveWebGPU.h"

#include <stdexcept>
#include <utility>

namespace fluid
{

FluidGrid::FluidGrid(int width, int height, double cell_size)
    : backend(std::make_unique<FluidGridCPU>(width, height, cell_size)) {}

FluidGrid::~FluidGrid() = default;

void FluidGrid::setDevice(FluidDevice value)
{
    if (value == device)
        return;

    selectBackend(value, adaptive_mode);
}

void FluidGrid::setAdaptiveMode(AdaptiveMode mode)
{
    if (mode == adaptive_mode) 
        return;

    selectBackend(device, mode);
}

void FluidGrid::selectBackend(FluidDevice value, AdaptiveMode mode)
{
    if (initialized)
        throw std::logic_error("Destroy the initialized grid before changing its device or adaptive mode");

    std::unique_ptr<FluidGridBase> selected;
    FluidGridAdaptiveCPU* adaptive = nullptr;
    FluidGridAdaptiveWebGPU* adaptive_webgpu = nullptr;

    if (value == FluidDevice::GPU)
    {
        if (mode == AdaptiveMode::DISABLED)
            selected = std::make_unique<FluidGridWebGPU>(getWidth(), getHeight(), getCellSize());
        else
        {
            auto engine = std::make_unique<FluidGridAdaptiveWebGPU>(getWidth(), getHeight(), getCellSize(), int(mode), adaptive_settings);
            adaptive_webgpu = engine.get();
            selected = std::move(engine);
        }
    }
    else if (value != FluidDevice::CPU)
    {
        throw std::invalid_argument("Unknown fluid device");
    }
    else if (mode == AdaptiveMode::DISABLED)
    {
        selected = std::make_unique<FluidGridCPU>(getWidth(), getHeight(), getCellSize());
    }
    else
    {
        auto engine = std::make_unique<FluidGridAdaptiveCPU>(getWidth(), getHeight(), getCellSize(), int(mode), adaptive_settings);
        adaptive = engine.get(); 
        selected = std::move(engine);
    }

    selected->configureMaterials(getElements(), getLiquidMixture(), getGasMixture());
    selected->setOrigin(getOrigin()); 
    selected->setGravityY(getGravityY());
    selected->setSurfaceTension(getSurfaceTension());

    backend = std::move(selected); 

    adaptive_backend = adaptive; 
    adaptive_webgpu_backend = adaptive_webgpu;
    device = value;
    adaptive_mode = mode;
}

void FluidGrid::setAdaptiveSettings(const FluidAdaptiveSettings& settings)
{
    adaptive_settings = settings;

    if (adaptive_backend) 
        adaptive_backend->setAdaptiveSettings(settings);
    if (adaptive_webgpu_backend)
        adaptive_webgpu_backend->setAdaptiveSettings(settings);
}

int FluidGrid::getWidth() const { return backend->getWidth(); }
int FluidGrid::getHeight() const { return backend->getHeight(); }

double FluidGrid::getCellSize() const { return backend->getCellSize(); }
bl::DVec2 FluidGrid::getOrigin() const { return backend->getOrigin(); }
void FluidGrid::setOrigin(bl::DVec2 origin) { backend->setOrigin(origin); }

bool FluidGrid::requiresRenderThread() const { return backend->requiresRenderThread(); }

void FluidGrid::configureMaterials(ElementCatalogue elements, FluidMixture liquid, FluidMixture gas)
{
    backend->configureMaterials(std::move(elements), std::move(liquid), std::move(gas));
}

const ElementCatalogue& FluidGrid::getElements() const { return backend->getElements(); }
const FluidMixture& FluidGrid::getLiquidMixture() const { return backend->getLiquidMixture(); }
const FluidMixture& FluidGrid::getGasMixture() const { return backend->getGasMixture(); }
double FluidGrid::getGravityY() const { return backend->getGravityY(); }
void FluidGrid::setGravityY(double acceleration) { backend->setGravityY(acceleration); }
double FluidGrid::getSurfaceTension() const { return backend->getSurfaceTension(); }
void FluidGrid::setSurfaceTension(double coefficient) { backend->setSurfaceTension(coefficient); }
void FluidGrid::setAccelerationField(const FluidField& field)
{
    backend->setAccelerationField(field);
}

void FluidGrid::setWebGPUContext(const FluidWebGPUContext& context)
{
    backend->setWebGPUContext(context);
}

FluidWebGPUContext FluidGrid::getWebGPUContext() { return backend->getWebGPUContext(); }

void FluidGrid::clearAccelerationField()
{
    backend->clearAccelerationField();
}

void FluidGrid::upload(const State& state) { backend->upload(state); initialized = true; }
FluidGrid::State FluidGrid::download() const { return backend->download(); }
bool FluidGrid::saveCheckpoint(FluidCheckpoint& checkpoint) const { return backend->saveCheckpoint(checkpoint); }
bool FluidGrid::loadCheckpoint(const FluidCheckpoint& checkpoint)
{
    if (!backend->loadCheckpoint(checkpoint)) return false;
    initialized = true;
    return true;
}
void FluidGrid::destroy() { backend->destroy(); initialized = false; }

FluidGrid::ProjectionResult FluidGrid::step(double dt, int max_iterations, double divergence_tolerance, StepTimings* timings)
{
    return backend->step(dt, max_iterations, divergence_tolerance, timings);
}

double FluidGrid::maxDivergence() const { return backend->maxDivergence(); }
void FluidGrid::writeColorPixels(std::vector<float>& pixels) const { backend->writeColorPixels(pixels); }
void FluidGrid::commitEdits(detail::CellEditBatch& batch) { backend->commitEdits(batch); }

}
