#include "FluidGridHybrid.h"

namespace fluid
{

FluidGridHybrid::FluidGridHybrid(int width, int height, double cell_size) :
    width(width), height(height), cell_size(cell_size), gpu(width, height, cell_size) {}

void FluidGridHybrid::configure()
{
    gpu.gravity_y = gravity_y;
    gpu.pressure_preconditioner = pressure_preconditioner;
    gpu.origin = origin;
}

void FluidGridHybrid::upload(const State& state)
{
    configure();
    gpu.elements = elements;
    gpu.liquid_mixture = liquid_mixture;
    gpu.gas_mixture = gas_mixture;
    gpu.upload(state);
    gpu.useCPUCoarsePressure(nullptr, 0);
    cpu.reset();
    if (cpu_pressure_cells != 0)
    {
        int w = width, h = height;
        while (w * h > cpu_pressure_cells && w * h > 16) { w = (w + 1) / 2; h = (h + 1) / 2; }
        cpu = std::make_unique<PressureMultigrid>(w, h);
        gpu.useCPUCoarsePressure(cpu.get(), cpu_pressure_cells);
    }
}

void FluidGridHybrid::destroy()
{
    gpu.destroy();
    cpu.reset();
}

FluidGridHybrid::ProjectionResult FluidGridHybrid::step(double dt, int max_iterations, double divergence_tolerance)
{
    configure();
    return gpu.step(dt, max_iterations, divergence_tolerance);
}

void FluidGridHybrid::advect(double dt) { configure(); gpu.advect(dt); }
void FluidGridHybrid::applyGravity(double dt) { configure(); gpu.applyGravity(dt); }
FluidGridHybrid::ProjectionResult FluidGridHybrid::project(double dt, int max_iterations, double divergence_tolerance)
{
    configure();
    return gpu.project(dt, max_iterations, divergence_tolerance);
}

void FluidGridHybrid::commitEdits(detail::CellEditBatch& batch)
{
    gpu.liquid_mixture = liquid_mixture;
    gpu.gas_mixture = gas_mixture;
    gpu.commitEdits(batch);
}

}
