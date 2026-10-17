#pragma once
#include "FluidGridGPUInternal.h"

namespace fluid::gpu_detail
{

// The same geometric V-cycle as the CPU, expressed as reusable fragment passes.
class PressureMultigridGPU
{
public:
    PressureMultigridGPU(PassRunner& runner, int width, int height);
    void rebuild(const Texture& fine_coefficients);
    const Texture& apply(const Texture& residual, const Texture& residual_stats);

private:
    static constexpr int smoothing_sweeps = 5;
    struct Level
    {
        int width, height, spacing;
        Texture coefficients, rhs, solution, next_solution, residual;
        Texture prolong_map, restrict_map;
        Level(int width, int height, int spacing, bool finest);
    };

    PassRunner& runner;
    std::vector<Level> levels;
    Texture coarse_factor;
    const Texture* fine_coefficients = nullptr;
    const Texture* residual_stats = nullptr;
    std::array<float, smoothing_sweeps> smoothing_weights;
    Program coarsen, smooth_pair, polynomial_smooth, residual, restrict_pass, prolong, factor, coarse_solve;

    void buildTransfers(Level& fine, const Level& coarse);
    void run(Program& program, int level, std::initializer_list<const Texture*> outputs,
             std::initializer_list<PassRunner::Input> inputs, int mode = 0, float relaxation = 0);
    const Texture& coefficients(int level) const;
    void smoothLevel(int level, const Texture& rhs, bool reverse, bool from_zero, int sweep);
    void cycle(int level, const Texture& rhs);
};

}
