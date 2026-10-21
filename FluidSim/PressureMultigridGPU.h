#pragma once
#include "FluidGridGPUInternal.h"
#include <memory>

namespace fluid { class PressureMultigrid; }

namespace fluid::gpu_detail
{

// The same geometric V-cycle as the CPU, expressed as reusable fragment passes.
class PressureMultigridGPU
{
public:
    PressureMultigridGPU(PassRunner& runner, int width, int height);
    ~PressureMultigridGPU();
    void rebuild(const Texture& fine_coefficients);
    const Texture& apply(const Texture& residual, const Texture& residual_stats, float tolerance = 0);
    void useCPUCoarsePressure(int max_cells);
    bool usesCPUCoarsePressure() const { return cpu_coarse != nullptr; }
    bool cpuConverged() const { return cpu_converged; }

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
    Program coarsen, smooth_pair, polynomial_smooth, residual, restrict_pass, restrict_cpu, prolong, factor, coarse_solve;
    std::unique_ptr<PressureMultigrid> cpu_coarse;
    int cpu_level = 0;
    bool cpu_matrix_stale = true;
    bool cpu_converged = false;
    float cpu_tolerance = 0;
    Texture cpu_readback;
    std::vector<float> cpu_pixels;
    std::vector<double> cpu_rhs, cpu_solution, cpu_right, cpu_down;

    void readCPU(const Texture& texture);
    void rebuildCPU();
    void solveCPU(const Texture& rhs);

    void buildTransfers(Level& fine, const Level& coarse);
    void run(Program& program, int level, std::initializer_list<const Texture*> outputs,
             std::initializer_list<PassRunner::Input> inputs, int mode = 0, float relaxation = 0);
    const Texture& coefficients(int level) const;
    void smoothLevel(int level, const Texture& rhs, bool reverse, bool from_zero, int sweep);
    void cycle(int level, const Texture& rhs);
};

}
