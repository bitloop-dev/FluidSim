#pragma once
#include "Shared/WebGPU/FluidWebGPUCompute.h"

namespace fluid::webgpu
{

// Fixed symmetric V-cycle, with transpose restriction and a GPU-resident tiny Cholesky solve.
class PressureMultigridGPU
{
public:
    PressureMultigridGPU(Compute& compute, int width, int height);
    void rebuild(const Field& fine_coefficients);

    // Negative tolerance applies the full cycle for direct preconditioner use.
    const Field& apply(const Field& residual, const Field& residual_stats, float stop_tolerance = -1);

private:
    friend struct PressureMultigridGPUTestAccess;
    static constexpr int smoothing_sweeps = 5;

    struct Level
    {
        int width;
        int height;
        int spacing;
        Field coefficients;
        Field rhs;
        Field solution;
        Field next_solution;
        Field residual;
        Field prolong_map;
        Field restrict_map;
        int prolong_offset = 0;
        int restrict_offset = 0;
        int tile_offset = 0;

        Level(Compute& compute, int width, int height, int spacing, bool finest);
    };

    Compute& compute;
    std::vector<Level> levels;
    Field coarse_factor;
    float stop_tolerance = -1;

    // The regression reference can select the original global sweeps.
    bool use_shared_smoothing = true;
    bool use_paired_smoothing = true;
    Kernel shared_smooth;
    Kernel polynomial_pair;
    const Field* fine_coefficients = nullptr;
    const Field* residual_stats = nullptr;
    std::array<float, smoothing_sweeps> smoothing_weights;
    Kernel coarsen;
    Kernel smooth_pair;
    Kernel polynomial_smooth;
    Kernel residual;
    Kernel restrict_pass;
    Kernel prolong;
    Kernel factor;
    Kernel coarse_solve;
    int tile_level = -1;
    Field tile_coefficients;
    Field tile_maps;
    Kernel pack_coefficients;
    Kernel tile_cycle;

    void buildTransfers(Level& fine, const Level& coarse, std::vector<float>* packed_maps);
    void run(Kernel& kernel, int level, std::initializer_list<const Field*> outputs,
        std::initializer_list<Compute::Input> inputs, int mode = 0, float relaxation = 0, float second_relaxation = 0);
    const Field& coefficients(int level) const;
    void smoothLevel(int level, const Field& rhs, bool reverse, bool from_zero, int sweep);
    void smoothSweeps(int level, const Field& rhs, bool reverse);
    void cycle(int level, const Field& rhs);
};

} // namespace fluid::webgpu
