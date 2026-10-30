#pragma once
#include <array>
#include <vector>

namespace fluid
{

// Geometry recipes shared by the CPU and GPU executors; no numerical field storage.
class AmrPressureTopology2D
{
public:
    struct Box { int x, y, width, height; };
    struct Level { int width, height; double cell_size; std::vector<Box> boxes; };
    struct Patch
    {
        Box box;
        int offset, stride;
        int index(int x, int y) const { return offset + (y - box.y + 1) * stride + x - box.x + 1; }
    };
    struct Row { int patch, y, begin, end; };
    struct Ghost
    {
        int target, first, second, source;
        double first_weight, second_weight, boundary_weight;
        std::array<int, 3> coarse{};
        std::array<double, 3> weights{};
        int count = 0;
    };
    struct Average { int coarse; std::array<int, 4> fine; };
    struct FaceAverage { int coarse; std::array<int, 2> fine; };
    struct Interpolation { int fine; std::array<int, 4> coarse; };
    struct Join { int coarse, coarse_step, coarse_face, fine, fine_step, fine_face, axis, sign; };
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
    };

    explicit AmrPressureTopology2D(const std::vector<Level>& levels);
    std::vector<Grid> grids, multigrid;

private:
    static Grid makeGrid(const Level& level);
    static int findCell(const Grid& grid, int x, int y);
    static void buildBoundary(Grid& grid, const Grid* coarse);
    static void buildTransfer(Grid& fine, const Grid& coarse);
};

} // namespace fluid
