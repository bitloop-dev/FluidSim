#pragma once
#include <array>
#include <vector>

namespace fluid
{

// Geometry recipes shared by the CPU and GPU executors; no numerical field storage.
class AmrPressureTopology2D
{
public:
    struct Box
    {
        int x;
        int y;
        int width;
        int height;
    };

    struct Level
    {
        int width;
        int height;
        double cell_size;
        std::vector<Box> boxes;
    };

    struct Patch
    {
        Box box;
        int offset;
        int stride;

        int index(int x, int y) const { return offset + (y - box.y + 1) * stride + x - box.x + 1; }
    };

    struct Row
    {
        int patch;
        int y;
        int begin;
        int end;
    };

    struct Ghost
    {
        int target;
        int first;
        int second;
        int source;
        double first_weight;
        double second_weight;
        double boundary_weight;
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
        int coarse;
        int coarse_step;
        int coarse_face;
        int fine;
        int fine_step;
        int fine_face;
        int axis;
        int sign;
    };

    struct Grid
    {
        int width;
        int height;
        int size = 0;
        int cells = 0;
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
    std::vector<Grid> grids;
    std::vector<Grid> multigrid;

private:
    static Grid makeGrid(const Level& level);
    static int findCell(const Grid& grid, int x, int y);
    static void buildBoundary(Grid& grid, const Grid* coarse);
    static void buildTransfer(Grid& fine, const Grid& coarse);
};

} // namespace fluid
