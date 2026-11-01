#pragma once
#include <array>
#include <vector>

namespace fluid
{

// CPU pressure preconditioner; the fine equation still belongs to FluidGridCPU.
class PressureMultigrid
{
public:
    PressureMultigrid(int width, int height);
    bool rebuild(const std::vector<double>& inv_density_u, const std::vector<double>& inv_density_v);

    // Apply one fixed V-cycle; the caller removes the correction's arbitrary constant.
    void apply(const std::vector<double>& residual, std::vector<double>& correction, double residual_mean);

private:
    struct Interpolation
    {
        int first = 0;
        int second = 0;
        double second_weight = 0.0;
    };

    struct Restriction
    {
        int count = 0;
        std::array<int, 4> indices{};
        std::array<double, 4> weights{};
    };

    struct Level
    {
        int width;
        int height;
        int stride;
        std::vector<double> center_x;
        std::vector<double> center_y;

        // Cell-aligned right/bottom face weights; wall weights and the padded halo stay zero.
        std::vector<double> right;
        std::vector<double> down;
        std::vector<double> inverse_diagonal;
        std::vector<double> rhs;
        std::vector<double> solution;
        std::vector<double> residual;
        std::vector<Interpolation> prolong_x;
        std::vector<Interpolation> prolong_y;
        std::vector<Restriction> restrict_x;
        std::vector<Restriction> restrict_y;

        Level(int width, int height, int spacing, int domain_width, int domain_height);
        void rebuildDiagonal();
    };

    std::vector<Level> levels;
    std::vector<double> coarse_factor;
    std::vector<double> coarse_rhs;

    static void buildTransfers(Level& fine, const Level& coarse);
    static void coarsen(const Level& fine, Level& coarse);
    static void smooth(Level& level, bool reverse, bool from_zero = false);
    static void restrictResidual(Level& fine, Level& coarse);
    static void prolongCorrection(const Level& coarse, Level& fine);
    bool factorCoarsest();
    void solveCoarsest();
    void cycle(int level);
};

}
