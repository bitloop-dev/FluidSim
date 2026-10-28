#pragma once
#include "AmrPressureTopology2D.h"
#include <array>
#include <vector>

namespace fluid
{

// The regular 2D pressure subset of AMReX; see reference/pressure2d/README.md.
class AmrPressure2D
{
public:
    using Box = AmrPressureTopology2D::Box;
    using Level = AmrPressureTopology2D::Level;
    using Patch = AmrPressureTopology2D::Patch;
    using Field = std::vector<std::vector<double>>;
    using Faces = std::array<Field, 2>;
    struct Result
    {
        int iterations;
        double residual;
        bool converged;
    };

    // Levels are properly nested, disjoint 2:1 patches; level zero covers the domain.
    explicit AmrPressure2D(const std::vector<Level>& levels);
    Field makeField() const;
    const std::vector<Patch>& patches(int level) const { return grids[level].patches; }
    void setCoefficients(const Faces& coefficients);
    void apply(Field& output, Field& pressure);
    void fluxes(Faces& output, Field& pressure);
    void precondition(Field& output, const Field& rhs, bool bottomKrylov = false);
    Result solve(Field& pressure, const Field& rhs, double tolerance, int max_iterations = 2000);
    Result solveMultigrid(Field& pressure, const Field& rhs, double tolerance, int max_iterations = 200);
    double norm(const Field& field) const;

private:
    struct Grid : AmrPressureTopology2D::Grid
    {
        std::array<std::vector<double>, 2> beta;
        std::vector<double> delta, diagonal;
    };

    std::vector<Grid> grids, multigrid;
    Field residual, correction, held, solution, rhs_work, scratch;
    std::vector<std::vector<double>> mg_rhs, mg_correction, mg_residual;

    static void fillBoundary(const Grid& grid, std::vector<double>& field, const std::vector<double>* coarse);
    static void updateDiagonal(Grid& grid);
    static void applyGrid(const Grid& grid, std::vector<double>& output, std::vector<double>& pressure,
                          const std::vector<double>* coarse = nullptr);
    static void smooth(const Grid& grid, std::vector<double>& pressure, const std::vector<double>& rhs, int iterations);
    static void restrictCells(const Grid& fine, std::vector<double>& coarse, const std::vector<double>& values);
    static void subtractMean(const Grid& grid, std::vector<double>& field);
    void restrictField(Field& field) const;
    void residualLevel(int level, Field& output, Field& pressure, const Field& rhs);
    void reflux(int coarse_level, Field& output, Field& pressure, double sign);
    void cycleBase(bool bottomKrylov);
    static bool solveBottom(const Grid& grid, std::vector<double>& pressure, const std::vector<double>& rhs);
    void increment(Field& destination, const Field& source, double scale) const;
    void scale(Field& field, double factor) const;
    double dot(const Field& first, const Field& second) const;
};

} // namespace fluid
