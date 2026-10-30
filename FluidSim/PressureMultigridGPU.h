#pragma once
#include "FluidWebGPUCompute.h"

namespace fluid::webgpu {

// Fixed symmetric V-cycle, with transpose restriction and a GPU-resident tiny Cholesky solve.
class PressureMultigridGPU
{
public:
    PressureMultigridGPU(Compute& compute, int width, int height);
    void rebuild(const Field& fine_coefficients);
    const Field& apply(const Field& residual, const Field& residual_stats);

private:
    static constexpr int smoothing_sweeps = 5;
    struct Level
    {
        int width, height, spacing;
        Field coefficients, rhs, solution, next_solution, residual, prolong_map, restrict_map;
        int prolong_offset = 0, restrict_offset = 0, tile_offset = 0;
        Level(Compute& compute, int width, int height, int spacing, bool finest);
    };
    Compute& compute;
    std::vector<Level> levels;
    Field coarse_factor;
    const Field* fine_coefficients = nullptr;
    const Field* residual_stats = nullptr;
    std::array<float, smoothing_sweeps> smoothing_weights;
    Kernel coarsen, smooth_pair, polynomial_smooth, residual, restrict_pass, prolong, factor, coarse_solve;
    int tile_level = -1;
    Field tile_coefficients, tile_maps;
    Kernel pack_coefficients, tile_cycle;
    void buildTransfers(Level& fine, const Level& coarse, std::vector<float>* packed_maps);
    void run(Kernel& kernel, int level, std::initializer_list<const Field*> outputs,
             std::initializer_list<Compute::Input> inputs, int mode = 0, float relaxation = 0);
    const Field& coefficients(int level) const;
    void smoothLevel(int level, const Field& rhs, bool reverse, bool from_zero, int sweep);
    void cycle(int level, const Field& rhs);
};

} // namespace fluid::webgpu
