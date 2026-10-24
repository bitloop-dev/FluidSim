#pragma once
#include <array>
#include <vector>

namespace fluid
{

// The regular 2D pressure subset of AMReX; see reference/pressure2d/README.md.
class AmrPressure2D
{
public:
    struct Box
    {
        int x, y, width, height;
    };
    struct Level
    {
        int width, height;
        double cell_size;
        std::vector<Box> boxes;
    };
    struct Patch
    {
        Box box;
        int offset, stride;
        int index(int x, int y) const { return offset + (y - box.y + 1) * stride + x - box.x + 1; }
    };
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
    struct Row
    {
        int patch, y, begin, end;
    };
    struct Ghost
    {
        int target, first, second, source;
        double first_weight, second_weight, boundary_weight;
        std::array<int, 3> coarse{};
        std::array<double, 3> weights{};
        int count = 0;
    };
    struct Average
    {
        int coarse;
        std::array<int, 4> fine;
    };
    struct FaceAverage
    {
        int coarse;
        std::array<int, 2> fine;
    };
    struct Interpolation
    {
        int fine;
        std::array<int, 4> coarse;
    };
    struct Join
    {
        int coarse, coarse_step, coarse_face, fine, fine_step, fine_face, axis, sign;
    };
    struct Grid
    {
        int width, height, size = 0, cells = 0;
        double cell_size;
        std::vector<Patch> patches;
        std::vector<Row> rows;
        std::vector<Ghost> ghosts;
        std::vector<Average> averages;
        std::array<std::vector<FaceAverage>, 2> face_averages;
        std::vector<Interpolation> interpolation;
        std::vector<Join> joins;
        std::vector<unsigned char> active;
        std::array<std::vector<double>, 2> beta;
        std::vector<double> delta, diagonal;
    };

    std::vector<Grid> grids, multigrid;
    Field residual, correction, held, solution, rhs_work, scratch;
    std::vector<std::vector<double>> mg_rhs, mg_correction, mg_residual;

    static Grid makeGrid(const Level& level);
    static int findCell(const Grid& grid, int x, int y);
    static void buildBoundary(Grid& grid, const Grid* coarse);
    static void buildTransfer(Grid& fine, const Grid& coarse);
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
