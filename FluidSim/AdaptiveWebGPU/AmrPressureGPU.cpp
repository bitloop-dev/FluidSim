// Derived from the same AMReX subset as AmrPressure2D; see reference/pressure2d/.
#include "AmrPressureGPU.h"
#include "AmrPressureGPUShaders.h"
#include "AmrPressureGPUCoarseShaders.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace fluid
{
namespace
{
using Buffer = webgpu::Field;
using Parameters = webgpu::Parameters;
using Topology = AmrPressureTopology2D;
constexpr int scalarCount = 1152;
constexpr int restart = 30;
constexpr int reductionWidth = 128;
constexpr int reductionBlock = 4 * reductionWidth;
constexpr int reductionGroups = 1024; // Bound partial storage and finish in one workgroup.
constexpr unsigned patchThreads = 128;

std::array<unsigned, 3> workExtent(unsigned count, unsigned threads = patchThreads)
{
    const unsigned groups = (count + threads - 1) / threads, columns = std::min(groups, 256u);
    return {columns * threads, (groups + columns - 1) / columns, 1};
}

std::array<unsigned, 3> reductionExtent(int count)
{
    const unsigned groups = std::min((count + reductionBlock - 1) / reductionBlock, reductionGroups);
    const unsigned columns = std::min(groups, 256u);
    return {columns * reductionWidth, (groups + columns - 1) / columns, 1};
}

Buffer allocate(webgpu::Compute& compute, int count, int channels = 1, const char* label = "AMR pressure field")
{
    const int width = std::min(std::max(count, 1), 4096);
    return compute.field(width, (count + width - 1) / width, channels, label);
}

float bits(int value) { return std::bit_cast<float>(int32_t(value)); }
float bits(unsigned value) { return std::bit_cast<float>(uint32_t(value)); }

std::string paired(std::string_view extra, std::initializer_list<std::pair<const char*, int>> fields, bool output = false)
{
    namespace shader = webgpu::amrpressure;
    std::string code = std::string(shader::compensated) + std::string(extra);
    for (const auto& [name, slot] : fields)
    {
        const std::string field = name;
        code += "fn " + field + "Pair(i:u32)->vec2f{let info=params.fields[" + std::to_string(slot) + "];var value=vec2f(" + field
              + "Data[i],0.);if(info.z==2u){value.y=" + field + "Data[info.x*info.y+i];}return value;}\n";
    }
    if (output)
        code += "fn storePair(i:u32,value:vec2f){output0Data[i]=value.x;let info=params.fields[0];if(info.z==2u){output0Data[info.x*info.y+i]=value.y;}}\n";
    return code;
}

Buffer uploadMetadata(webgpu::Compute& compute, const std::vector<float>& words, const char* label)
{
    auto buffer = allocate(compute, int(words.size()), 1, label);
    compute.write(buffer, words);
    return buffer;
}

Buffer packGrid(webgpu::Compute& compute, const Topology::Grid& grid, std::array<float, 4>& boundaryWork, std::array<float, 4>& patchWork)
{
    // Four compact planes hold flags/stride/recipe offsets; irregular data lives only at its destinations.
    std::vector<float> words(size_t(grid.size) * 4);
    auto set = [&](int cell, int plane, int value) { words[size_t(plane) * grid.size + cell] = bits(value); };
    for (const auto& row : grid.rows)
    {
        const auto& patch = grid.patches[row.patch];
        for (int i = row.begin; i < row.end; ++i)
        {
            const int color = (patch.box.x + i - row.begin + row.y) & 1;
            set(i, 0, 1 | (color << 1) | ((grid.active[i] != 0) << 2));
            set(i, 1, patch.stride);
        }
    }
    for (const auto& patch : grid.patches)
        for (int axis = 0; axis < 2; ++axis)
            for (int y = patch.box.y; y < patch.box.y + patch.box.height + (axis == 1); ++y)
                for (int x = patch.box.x; x < patch.box.x + patch.box.width + (axis == 0); ++x)
                {
                    const int i = patch.index(x, y);
                    set(i, 0, std::bit_cast<int32_t>(words[i]) | (8 << axis));
                    set(i, 1, patch.stride);
                }
    std::vector<const Topology::Ghost*> diagonal;
    for (const auto& g : grid.ghosts) if (g.source < 0) diagonal.push_back(&g);
    std::stable_sort(diagonal.begin(), diagonal.end(), [](auto a, auto b) { return a->first < b->first; });
    for (size_t begin = 0, end; begin < diagonal.size(); begin = end)
    {
        for (end = begin + 1; end < diagonal.size() && diagonal[end]->first == diagonal[begin]->first; ++end) {}
        set(diagonal[begin]->first, 2, int(words.size())); words.push_back(bits(int(end - begin)));
        for (size_t j = begin; j < end; ++j)
        {
            const auto* g = diagonal[j];
            int stride = g->first - g->target;
            words.push_back(bits(std::abs(stride) == 1 ? 0 : 1));
            words.push_back(bits(stride > 0 ? g->first : g->target));
            words.push_back(float(g->first_weight));
        }
    }
    for (const auto& a : grid.interpolation)
    {
        set(a.fine, 3, int(words.size()));
        for (int coarse : a.coarse) words.push_back(bits(coarse));
    }
    std::vector<const Topology::Ghost*> ghosts;
    for (const auto& ghost : grid.ghosts) ghosts.push_back(&ghost);
    std::stable_sort(ghosts.begin(), ghosts.end(), [](auto a, auto b) { return a->target < b->target; });
    boundaryWork = {bits(int(words.size())), 0., 0., 0.};
    for (size_t begin = 0, end; begin < ghosts.size(); begin = end)
    {
        for (end = begin + 1; end < ghosts.size() && ghosts[end]->target == ghosts[begin]->target; ++end) {}
        const auto& g = *ghosts[end - 1];
        for (int value : {g.target, g.source, g.first, g.second, g.count}) words.push_back(bits(value));
        for (double value : {g.first_weight, g.second_weight, g.boundary_weight}) words.push_back(float(value));
        for (int value : g.coarse) words.push_back(bits(value));
        for (double value : g.weights) words.push_back(float(value));
        ++boundaryWork[1];
    }
    std::vector<std::array<int, 2>> full, colors;
    for (const auto& patch : grid.patches)
    {
        const auto& b = patch.box; const int descriptor = int(words.size());
        for (int value : {patch.index(b.x, b.y), patch.stride, b.width, b.height, (b.x + b.y) & 1}) words.push_back(bits(value));
        for (int i = 0; i < b.width * b.height; i += patchThreads) full.push_back({descriptor, i});
        for (int i = 0; i < ((b.width + 1) / 2) * b.height; i += patchThreads) colors.push_back({descriptor, i});
    }
    patchWork = {bits(int(words.size())), float(full.size()), 0., float(colors.size())};
    for (auto job : full) for (int value : job) words.push_back(bits(value));
    patchWork[2] = bits(int(words.size()));
    for (auto job : colors) for (int value : job) words.push_back(bits(value));
    return uploadMetadata(compute, words, "AMR pressure cell recipes");
}

Buffer packTransfer(webgpu::Compute& compute, const Topology::Grid& fine, int coarseSize, std::array<std::array<int, 2>, 4>& work)
{
    std::vector<float> words;
    std::vector<int> selected(coarseSize, -1);
    for (int j = 0; j < int(fine.averages.size()); ++j) selected[fine.averages[j].coarse] = j;
    for (int j : selected) if (j >= 0)
    {
        const auto& a = fine.averages[j]; words.push_back(bits(a.coarse));
        for (int cell : a.fine) words.push_back(bits(cell));
        ++work[0][1];
    }
    // Compact unique destinations retain the CPU's last recipe at duplicate faces.
    for (int axis = 0; axis < 2; ++axis)
    {
        auto& range = work[axis + 1]; range[0] = int(words.size());
        std::fill(selected.begin(), selected.end(), -1);
        const auto& faces = fine.face_averages[axis];
        for (int j = 0; j < int(faces.size()); ++j) selected[faces[j].coarse] = j;
        for (int j : selected) if (j >= 0)
        {
            const auto& a = faces[j]; words.push_back(bits(a.coarse));
            for (int cell : a.fine) words.push_back(bits(cell));
            ++range[1];
        }
    }
    std::vector<const Topology::Join*> joins;
    for (const auto& join : fine.joins) joins.push_back(&join);
    std::stable_sort(joins.begin(), joins.end(), [](auto a, auto b) { return a->coarse < b->coarse; });
    std::vector<float> headers;
    for (size_t begin = 0, end; begin < joins.size(); begin = end)
    {
        for (end = begin + 1; end < joins.size() && joins[end]->coarse == joins[begin]->coarse; ++end) {}
        for (int value : {joins[begin]->coarse, int(words.size()), int(end - begin)}) headers.push_back(bits(value));
        for (size_t k = begin; k < end; ++k)
        {
            const auto* j = joins[k];
            for (int value : {j->coarse, j->coarse_step, j->coarse_face, j->fine, j->fine_step, j->fine_face, j->axis, j->sign})
                words.push_back(bits(value));
        }
    }
    work[3] = {int(words.size()), int(headers.size() / 3)};
    words.insert(words.end(), headers.begin(), headers.end());
    return uploadMetadata(compute, words, "AMR pressure transfer recipes");
}
} // namespace

struct AmrPressureGPU::Impl
{
    struct Grid
    {
        const Topology::Grid* plan;
        Buffer metadata, transfer, diagonal;
        std::array<float, 4> boundaryWork{}, patchWork{};
        std::array<std::array<int, 2>, 4> transferWork{};
        Buffer partials, productPartials;
        std::array<Buffer, 2> beta;
    };

    webgpu::Compute& compute;
    Topology topology;
    std::vector<Grid> grids, multigrid;
    Buffer scalars, bottomWork;
    Field residual, correction, held, rhsWork;
    Field mgRhs, mgCorrection, mgResidual;
    Field error, update, work, right, current, candidate;
    std::vector<Field> basis; // Each tile packs up to four basis vectors per AMR level.
    int basisWidth = 4;
    webgpu::Kernel vectorKernel, boundaryKernel, diagonalKernel, applyKernel, smoothKernel, transferKernel;
    webgpu::Kernel interpolateKernel, fluxKernel, refluxKernel, reductionKernel, finishKernel, scalarKernel, bottomKernel;
    webgpu::Kernel productsKernel, productsFinishKernel, normalizeKernel, basisUpdateKernel;
    webgpu::Kernel coarsePackKernel, coarseCycleKernel;
    Buffer coarseRecipes, coarseCoefficients;
    Parameters coarseSettings;
    int coarseLevel = -1;
    bool coarseDirty = true;

    Impl(webgpu::Compute& owner, const std::vector<Level>& levels) : compute(owner), topology(levels)
    {
        namespace shader = webgpu::amrpressure;
        compute.build(vectorKernel, "AMR pressure vectors", 1, {"source", "geometry", "scalar"}, shader::vector,
                      paired(shader::metadata, {{"output0", 0}, {"source", 1}}, true), false, {patchThreads, 1});
        compute.build(boundaryKernel, "AMR pressure ghosts", 1, {"geometry", "coarse"}, shader::boundary,
                      paired(shader::metadata, {{"output0", 0}, {"coarse", 2}}, true), false, {patchThreads, 1});
        compute.build(diagonalKernel, "AMR pressure diagonal", 1, {"geometry", "bx", "by"}, shader::diagonal, shader::metadata, false, {patchThreads, 1});
        compute.build(applyKernel, "AMR pressure operator", 1, {"pressure", "geometry", "bx", "by", "rhs"}, shader::apply,
                      paired(shader::metadata, {{"pressure", 1}}), false, {patchThreads, 1});
        compute.build(smoothKernel, "AMR red black smoother", 1, {"geometry", "bx", "by", "diagonal", "rhs"}, shader::smooth, shader::metadata, false, {patchThreads, 1});
        compute.build(transferKernel, "AMR pressure restriction", 1, {"fine", "recipe"}, shader::transfer, shader::index, false, {patchThreads, 1});
        compute.build(interpolateKernel, "AMR pressure interpolation", 3, {"coarse", "geometry"}, shader::interpolate, shader::metadata, false, {patchThreads, 1});
        compute.build(fluxKernel, "AMR pressure flux", 2, {"pressure", "geometry", "bx", "by"}, shader::flux,
                      paired(shader::metadata, {{"pressure", 2}}), false, {patchThreads, 1});
        compute.build(refluxKernel, "AMR pressure reflux", 1, {"coarse", "fine", "cbx", "cby", "fbx", "fby", "recipe"}, shader::reflux,
                      paired(shader::index, {{"coarse", 1}, {"fine", 2}}), false, {patchThreads, 1});
        const bool subgroups = compute.gpu.device().HasFeature(wgpu::FeatureName::Subgroups);
        const std::string reduction = std::string(shader::reductionLayout) + std::string(subgroups ? shader::reductionSubgroups : shader::reductionShared);
        const std::string products = reduction + std::string(shader::productsLayout) + std::string(subgroups ? shader::productsSubgroups : shader::productsShared);
        compute.build(reductionKernel, "AMR pressure reduction", 1, {"first", "second", "geometry"}, shader::reduction,
                      std::string(shader::metadata) + reduction, true, {reductionWidth, 1}, subgroups);
        compute.build(finishKernel, "AMR pressure reduction finish", 1, {"partial"}, shader::reduceFinish, reduction, true, {reductionWidth, 1}, subgroups);
        compute.build(productsKernel, "AMR batched Krylov products", 1, {"source", "basis", "geometry"}, shader::products,
                      std::string(shader::metadata) + products, true, {reductionWidth, 1}, subgroups);
        compute.build(productsFinishKernel, "AMR batched Krylov reduction", 1, {"partial"}, shader::productsFinish,
                      products, true, {reductionWidth, 1}, subgroups);
        compute.build(normalizeKernel, "AMR normalize and store basis", 2, {"geometry", "scalar"}, shader::normalizeBasis, shader::metadata, false, {patchThreads, 1});
        compute.build(basisUpdateKernel, "AMR batched Krylov update", 1, {"basis", "geometry", "scalar"}, shader::basisUpdate, shader::metadata, false, {patchThreads, 1});
        compute.build(scalarKernel, "AMR pressure Krylov scalars", 1, {}, shader::scalar, shader::index);
        compute.build(bottomKernel, "AMR normalized bottom BiCGStab", 2, {"geometry", "bx", "by", "diagonal", "rhs"}, shader::bottom, shader::metadata);
        scalars = allocate(compute, scalarCount, 1, "AMR Krylov scalars");
        auto create = [&](const auto& plans, auto& target, bool base)
        {
            for (int i = 0; i < int(plans.size()); ++i)
            {
                const auto& plan = plans[i];
                Grid grid{&plan};
                grid.metadata = packGrid(compute, plan, grid.boundaryWork, grid.patchWork);
                if (base && i + 1 < int(plans.size())) grid.transfer = packTransfer(compute, plan, plans[i + 1].size, grid.transferWork);
                else if (!base && i) grid.transfer = packTransfer(compute, plan, plans[i - 1].size, grid.transferWork);
                grid.diagonal = allocate(compute, plan.size, 2);
                const int count = std::min((plan.size + reductionBlock - 1) / reductionBlock, reductionGroups);
                grid.partials = allocate(compute, count, 2);
                for (auto& beta : grid.beta) beta = allocate(compute, plan.size);
                target.push_back(std::move(grid));
            }
        };
        create(topology.grids, grids, false); create(topology.multigrid, multigrid, true);
        residual = makeField(); correction = makeField(); held = makeField(); rhsWork = makeField();
        error = makeField(); update = makeField(); work = makeField(); right = makeField();
        for (const auto& grid : multigrid)
        {
            mgRhs.push_back(allocate(compute, grid.plan->size));
            mgCorrection.push_back(allocate(compute, grid.plan->size));
            mgResidual.push_back(allocate(compute, grid.plan->size));
        }
        bottomWork = allocate(compute, 5 * multigrid.back().plan->size, 1, "AMR bottom Krylov workspace");
        createCoarseCycle();
    }

    void createCoarseCycle()
    {
        namespace shader = webgpu::amrpressure;
        int count = 0; const int last = int(multigrid.size()) - 1;
        for (int m = last; m >= 0; --m)
        {
            const auto& plan = *multigrid[m].plan;
            if (count + plan.size > shader::coarseCapacity || last - m >= int(coarseSettings.vectors.size()) || plan.cells != plan.width * plan.height) break;
            if (m < last && (int(plan.averages.size()) != multigrid[m + 1].plan->cells || int(plan.interpolation.size()) != plan.cells)) break;
            count += plan.size; coarseLevel = m;
        }
        if (coarseLevel < 0) return;
        std::vector<float> words(size_t(count) * 12);
        for (int i = 0; i < count; ++i) {
            words[12 * i + 2] = bits(-2);
            for (int k = 7; k < 12; ++k) words[12 * i + k] = bits(-1);
        }
        int offset = 0;
        for (int m = coarseLevel; m <= last; ++m)
        {
            const auto& plan = *multigrid[m].plan;
            coarseSettings.vectors[m - coarseLevel] = {float(plan.size), float(offset), float(1. / (plan.cell_size * plan.cell_size)), 0.};
            auto set = [&](int i, int k, int value) { words[12 * (offset + i) + k] = bits(value); };
            for (const auto& row : plan.rows)
            {
                const auto& patch = plan.patches[row.patch];
                for (int i = row.begin; i < row.end; ++i) {
                    set(i, 0, patch.stride); set(i, 1, (patch.box.x + i - row.begin + row.y) & 1);
                }
            }
            for (const auto& ghost : plan.ghosts) {
                set(ghost.target, 2, ghost.source); set(ghost.target, 3, ghost.first); set(ghost.target, 4, ghost.second);
                words[12 * (offset + ghost.target) + 5] = float(ghost.first_weight);
                words[12 * (offset + ghost.target) + 6] = float(ghost.second_weight);
            }
            for (const auto& interpolation : plan.interpolation) set(interpolation.fine, 7, interpolation.coarse[0]);
            if (m > coarseLevel)
                for (const auto& average : multigrid[m - 1].plan->averages)
                    for (int k = 0; k < 4; ++k) set(average.coarse, 8 + k, average.fine[k]);
            offset += plan.size;
        }
        coarseSettings.grid = {last - coarseLevel + 1, count, 0, 0};
        coarseRecipes = uploadMetadata(compute, words, "AMR shared coarse recipes");
        coarseCoefficients = allocate(compute, count, 6, "AMR shared coarse coefficients");
        compute.build(coarsePackKernel, "AMR shared coarse coefficients", 1, {"geometry", "bx", "by", "diagonal"}, shader::coarsePack,
                      shader::metadata, false, {shader::coarseThreads, 1});
        compute.build(coarseCycleKernel, "AMR shared coarse cycle", 1, {"rhs", "recipes", "coefficients"}, shader::coarseCycle,
                      {}, true, {shader::coarseThreads, 1});
    }
    void refreshCoarseCoefficients()
    {
        if (!coarseDirty) return;
        for (int m = coarseLevel; m < int(multigrid.size()); ++m)
        {
            auto& grid = multigrid[m]; auto p = settings(grid); p.grid[1] = int(coarseSettings.vectors[m - coarseLevel][1]);
            compute.run(coarsePackKernel, {&coarseCoefficients}, {{"geometry", &grid.metadata}, {"bx", &grid.beta[0]}, {"by", &grid.beta[1]},
                        {"diagonal", &grid.diagonal}}, p, {unsigned(grid.plan->size), 1, 1});
        }
        coarseDirty = false;
    }

    Field makeField(bool compensated = false) const
    {
        Field result;
        for (const auto& grid : grids) result.push_back(allocate(compute, grid.plan->size, compensated ? 2 : 1));
        return result;
    }

    static Parameters settings(const Grid& grid)
    {
        Parameters p;
        p.grid[0] = grid.plan->size; p.grid[1] = grid.plan->cells;
        p.value[0] = float(1. / (grid.plan->cell_size * grid.plan->cell_size));
        p.vectors[10] = grid.patchWork; p.vectors[11] = grid.boundaryWork;
        return p;
    }

    void vector(const Grid& grid, Buffer& destination, const Buffer& source, int operation, float factor = 0., int scalar = -1, bool gated = false)
    {
        auto p = settings(grid); p.grid[2] = operation; p.grid[3] = scalar; p.value[0] = factor; p.level[3] = float(gated);
        // Unused source bindings stay distinct from the writable destination.
        const auto* input = destination.buffer.Get() == source.buffer.Get() ? &scalars : &source;
        compute.run(vectorKernel, {&destination}, {{"source", input}, {"geometry", &grid.metadata}, {"scalar", &scalars}}, p);
    }
    void clear(Field& field) { for (int l = 0; l < int(grids.size()); ++l) vector(grids[l], field[l], scalars, 0); }
    void copy(Field& destination, const Field& source)
    {
        for (int l = 0; l < int(grids.size()); ++l) vector(grids[l], destination[l], source[l], 1);
    }
    void increment(Field& destination, const Field& source, float factor, int scalar = -1, bool gated = false)
    {
        for (int l = 0; l < int(grids.size()); ++l) vector(grids[l], destination[l], source[l], 2, factor, scalar, gated);
    }
    void scale(Field& field, float factor, int scalar = -1)
    {
        for (int l = 0; l < int(grids.size()); ++l) vector(grids[l], field[l], scalars, 3, factor, scalar);
    }
    void boundary(const Grid& grid, Buffer& pressure, const Buffer* coarse = nullptr)
    {
        if (grid.boundaryWork[1] == 0.) return;
        auto p = settings(grid); p.grid[2] = coarse != nullptr;
        compute.run(boundaryKernel, {&pressure}, {{"geometry", &grid.metadata}, {"coarse", coarse ? coarse : &scalars}}, p,
                    workExtent(unsigned(grid.boundaryWork[1])));
    }
    void applyGrid(const Grid& grid, Buffer& output, Buffer& pressure, const Buffer* coarse = nullptr, const Buffer* rhs = nullptr)
    {
        boundary(grid, pressure, coarse);
        auto p = settings(grid);
        // An in-place residual reads the writable output, avoiding aliased WebGPU bindings.
        const bool inPlace = rhs && rhs->buffer.Get() == output.buffer.Get();
        p.grid[2] = rhs ? (inPlace ? 2 : 1) : 0;
        compute.run(applyKernel, {&output}, {{"pressure", &pressure}, {"geometry", &grid.metadata}, {"bx", &grid.beta[0]}, {"by", &grid.beta[1]},
                    {"rhs", rhs && !inPlace ? rhs : &grid.metadata}}, p,
                    workExtent(unsigned(grid.patchWork[1]) * patchThreads));
    }
    void smooth(const Grid& grid, Buffer& pressure, const Buffer& rhs, int iterations, bool zeroStart = false)
    {
        for (int iteration = 0; iteration < iterations; ++iteration)
            for (int color = 0; color < 2; ++color)
            {
                // A cleared correction already has the zero homogeneous ghosts needed by its first colour.
                if (!zeroStart || iteration || color) boundary(grid, pressure);
                auto p = settings(grid); p.grid[2] = color;
                compute.run(smoothKernel, {&pressure}, {{"geometry", &grid.metadata}, {"bx", &grid.beta[0]}, {"by", &grid.beta[1]}, {"diagonal", &grid.diagonal}, {"rhs", &rhs}}, p,
                            workExtent(unsigned(grid.patchWork[3]) * patchThreads));
            }
    }
    void restrictCells(const Grid& fine, const Grid& coarse, Buffer& destination, const Buffer& source, int mode = 0)
    {
        const auto& range = fine.transferWork[mode]; if (!range[1]) return;
        auto p = settings(coarse); p.grid[1] = range[0]; p.grid[2] = mode; p.grid[3] = range[1];
        compute.run(transferKernel, {&destination}, {{"fine", &source}, {"recipe", &fine.transfer}}, p, workExtent(range[1]));
    }
    void restrictField(Field& field)
    {
        for (int l = int(grids.size()) - 1; l > 0; --l) restrictCells(grids[l], grids[l - 1], field[l - 1], field[l]);
    }
    void setCoefficients(const Faces& coefficients)
    {
        for (int l = 0; l < int(grids.size()); ++l)
            for (int d = 0; d < 2; ++d) vector(grids[l], grids[l].beta[d], coefficients[d][l], 1);
        for (int l = int(grids.size()) - 1; l > 0; --l)
            for (int d = 0; d < 2; ++d) restrictCells(grids[l], grids[l - 1], grids[l - 1].beta[d], grids[l].beta[d], d + 1);
        auto diagonal = [&](Grid& grid)
        {
            compute.run(diagonalKernel, {&grid.diagonal}, {{"geometry", &grid.metadata}, {"bx", &grid.beta[0]}, {"by", &grid.beta[1]}}, settings(grid));
        };
        for (auto& grid : grids) diagonal(grid);
        for (int d = 0; d < 2; ++d) vector(multigrid[0], multigrid[0].beta[d], grids[0].beta[d], 1);
        for (int m = 0; m < int(multigrid.size()); ++m)
        {
            if (m)
                for (int d = 0; d < 2; ++d) restrictCells(multigrid[m - 1], multigrid[m], multigrid[m].beta[d], multigrid[m - 1].beta[d], d + 1);
            diagonal(multigrid[m]);
        }
        coarseDirty = true;
    }
    void reflux(int coarseLevel, Field& output, Field& pressure, float sign)
    {
        int l = coarseLevel + 1;
        auto& coarse = grids[coarseLevel]; auto& fine = grids[l];
        const auto& range = fine.transferWork[3]; if (!range[1]) return;
        // Callers have filled both pressure boundaries; reflux only changes the output residual.
        auto p = settings(coarse); p.value = {float(coarse.plan->cell_size), float(fine.plan->cell_size), sign, 0.};
        p.grid[1] = range[0]; p.grid[3] = range[1];
        compute.run(refluxKernel, {&output[coarseLevel]}, {{"coarse", &pressure[coarseLevel]}, {"fine", &pressure[l]}, {"cbx", &coarse.beta[0]}, {"cby", &coarse.beta[1]},
                    {"fbx", &fine.beta[0]}, {"fby", &fine.beta[1]}, {"recipe", &fine.transfer}}, p, workExtent(range[1]));
    }
    void apply(Field& output, Field& pressure)
    {
        for (int l = 0; l < int(grids.size()); ++l) applyGrid(grids[l], output[l], pressure[l], l ? &pressure[l - 1] : nullptr);
        for (int l = int(grids.size()) - 2; l >= 0; --l) reflux(l, output, pressure, 1.);
        restrictField(output);
    }
    void fluxes(Faces& output, Field& pressure)
    {
        for (int l = 0; l < int(grids.size()); ++l)
        {
            auto& grid = grids[l]; boundary(grid, pressure[l], l ? &pressure[l - 1] : nullptr);
            auto p = settings(grid); p.value[0] = float(1. / grid.plan->cell_size);
            compute.run(fluxKernel, {&output[0][l], &output[1][l]}, {{"pressure", &pressure[l]}, {"geometry", &grid.metadata}, {"bx", &grid.beta[0]}, {"by", &grid.beta[1]}}, p);
        }
        for (int l = int(grids.size()) - 1; l > 0; --l)
            for (int d = 0; d < 2; ++d) restrictCells(grids[l], grids[l - 1], output[d][l - 1], output[d][l], d + 1);
    }
    void scalar(int operation, int index = 0, float value = 0., float tolerance = 0., int limit = 0)
    {
        Parameters p; p.grid[0] = index; p.grid[2] = operation; p.value[0] = value; p.value[2] = tolerance; p.value[3] = float(limit);
        compute.run(scalarKernel, {&scalars}, {}, p, {1, 1, 1});
    }
    void reduceGrid(Grid& grid, const Buffer& first, const Buffer& second, int mode, int destination, bool accumulate)
    {
        auto p = settings(grid); p.grid[2] = mode;
        compute.run(reductionKernel, {&grid.partials}, {{"first", &first}, {"second", &second}, {"geometry", &grid.metadata}}, p, reductionExtent(grid.plan->size));
        const int count = std::min((grid.plan->size + reductionBlock - 1) / reductionBlock, reductionGroups);
        p.grid = {count, destination, mode == 2 || mode == 3, accumulate};
        compute.run(finishKernel, {&scalars}, {{"partial", &grid.partials}}, p, {reductionWidth, 1, 1});
    }
    void reduce(const Field& first, const Field& second, int mode, int destination)
    {
        for (int l = 0; l < int(grids.size()); ++l) reduceGrid(grids[l], first[l], second[l], mode, destination, l != 0);
    }
    void normTo(const Field& field, int destination = 0)
    {
        reduce(field, field, 0, destination); scalar(1, destination);
    }
    void subtractMean(Grid& grid, Buffer& field)
    {
        reduceGrid(grid, field, field, 1, 1, false); scalar(2, 1, float(grid.plan->cells)); vector(grid, field, scalars, 4, 0., 1);
    }
    void prepareRhs(Field& destination, const Field& rhs)
    {
        copy(destination, rhs); restrictField(destination);
        reduceGrid(grids[0], destination[0], destination[0], 1, 1, false); scalar(2, 1, float(grids[0].plan->cells));
        for (int l = 0; l < int(grids.size()); ++l) vector(grids[l], destination[l], scalars, 4, 0., 1);
    }
    void residualLevel(int l, Field& output, Field& pressure, const Field& rhs)
    {
        applyGrid(grids[l], output[l], pressure[l], l ? &pressure[l - 1] : nullptr, &rhs[l]);
    }
    void cycleBase(bool bottomKrylov)
    {
        const int last = int(multigrid.size()) - 1;
        const int stop = !bottomKrylov && coarseLevel >= 0 ? coarseLevel : last;
        vector(multigrid[0], mgRhs[0], residual[0], 1);
        for (int m = 0; m < stop; ++m)
        {
            auto& grid = multigrid[m]; vector(grid, mgCorrection[m], scalars, 0);
            smooth(grid, mgCorrection[m], mgRhs[m], 2, true); applyGrid(grid, mgResidual[m], mgCorrection[m], nullptr, &mgRhs[m]);
            restrictCells(grid, multigrid[m + 1], mgRhs[m + 1], mgResidual[m]);
        }
        if (!bottomKrylov && coarseLevel >= 0)
        {
            refreshCoarseCoefficients();
            compute.run(coarseCycleKernel, {&mgCorrection[stop]}, {{"rhs", &mgRhs[stop]}, {"recipes", &coarseRecipes}, {"coefficients", &coarseCoefficients}},
                        coarseSettings, {webgpu::amrpressure::coarseThreads, 1, 1});
        }
        else
        {
            auto& bottom = multigrid[last]; vector(bottom, mgCorrection[last], scalars, 0);
            if (bottomKrylov)
                compute.run(bottomKernel, {&mgCorrection[last], &bottomWork}, {{"geometry", &bottom.metadata}, {"bx", &bottom.beta[0]}, {"by", &bottom.beta[1]},
                            {"diagonal", &bottom.diagonal}, {"rhs", &mgRhs[last]}}, settings(bottom), {1, 1, 1});
            else smooth(bottom, mgCorrection[last], mgRhs[last], 8, true);
        }
        for (int m = stop - 1; m >= 0; --m)
        {
            auto p = settings(multigrid[m]); p.grid[2] = 0;
            compute.run(interpolateKernel, {&mgCorrection[m], &mgRhs[m], &mgResidual[m]}, {{"coarse", &mgCorrection[m + 1]}, {"geometry", &multigrid[m].metadata}}, p,
                        workExtent(unsigned(multigrid[m].patchWork[1]) * patchThreads));
            smooth(multigrid[m], mgCorrection[m], mgRhs[m], 2);
        }
        vector(grids[0], correction[0], mgCorrection[0], 1);
    }
    void precondition(Field& output, const Field& rhs, bool bottomKrylov)
    {
        // As in MLMG's solution alias, the caller's float workspace holds the correction directly.
        Field& solution = output;
        prepareRhs(rhsWork, rhs); clear(solution); copy(residual, rhsWork);
        const int last = int(grids.size()) - 1;
        for (int l = last; l > 0; --l)
        {
            auto& grid = grids[l]; vector(grid, correction[l], scalars, 0); smooth(grid, correction[l], residual[l], 2, true);
            vector(grid, solution[l], correction[l], 2, 1.); residualLevel(l - 1, residual, solution, rhsWork);
            applyGrid(grid, residual[l], correction[l], nullptr, &residual[l]);
            boundary(grid, solution[l], &solution[l - 1]);
            reflux(l - 1, residual, solution, -1.); restrictCells(grid, grids[l - 1], residual[l - 1], residual[l]);
            if (l != last) held[l].swap(correction[l]);
        }
        subtractMean(grids[0], residual[0]); cycleBase(bottomKrylov);
        vector(grids[0], solution[0], correction[0], 2, 1.);
        for (int l = 1; l <= last; ++l)
        {
            auto& grid = grids[l]; auto p = settings(grid); p.grid[2] = 1; p.grid[3] = l != last;
            compute.run(interpolateKernel, {&correction[l], &solution[l], &held[l]}, {{"coarse", &correction[l - 1]}, {"geometry", &grid.metadata}}, p,
                        workExtent(unsigned(grid.patchWork[1]) * patchThreads));
            applyGrid(grid, residual[l], correction[l], &correction[l - 1], &residual[l]);
            vector(grid, correction[l], scalars, 0); smooth(grid, correction[l], residual[l], 2, true);
            vector(grid, solution[l], correction[l], 2, 1.);
            if (l != last) vector(grid, correction[l], held[l], 2, 1.);
        }
        restrictField(solution);
    }

    std::vector<float> status() { return compute.read(scalars, 0, 8 * sizeof(float)); }

    void prepareBasis()
    {
        if (!basis.empty()) return;
        wgpu::Limits limits; compute.gpu.device().GetLimits(&limits);
        // Scalar work fields already fit; cap tile width by the same binding limit.
        for (const auto& field : work) basisWidth = std::min(basisWidth, int(limits.maxStorageBufferBindingSize / field.bytes()));
        for (int first = 0; first < restart; first += basisWidth)
        {
            Field tile;
            for (const auto& grid : grids) tile.push_back(allocate(compute, grid.plan->size, std::min(basisWidth, restart - first)));
            basis.push_back(std::move(tile));
        }
        current = makeField(); candidate = makeField();
        for (auto& grid : grids)
        {
            const int count = std::min((grid.plan->size + reductionBlock - 1) / reductionBlock, reductionGroups);
            grid.productPartials = allocate(compute, 5 * count);
        }
    }
    void normalizeBasis(Field& field, int index)
    {
        if (index == restart) { scale(field, 0., 2); return; }
        for (int l = 0; l < int(grids.size()); ++l)
        {
            auto p = settings(grids[l]); p.grid[1] = index % basisWidth;
            compute.run(normalizeKernel, {&field[l], &basis[index / basisWidth][l]}, {{"geometry", &grids[l].metadata}, {"scalar", &scalars}}, p,
                        workExtent(unsigned(grids[l].patchWork[1]) * patchThreads));
        }
    }
    void basisProducts(const Field& field, int last)
    {
        for (int first = 0; first <= last; first += basisWidth)
            for (int l = 0; l < int(grids.size()); ++l)
            {
                auto& grid = grids[l]; const int products = std::min(basisWidth, last - first + 1);
                auto p = settings(grid); p.grid[1] = products;
                compute.run(productsKernel, {&grid.productPartials}, {{"source", &field[l]}, {"basis", &basis[first / basisWidth][l]},
                            {"geometry", &grid.metadata}}, p, reductionExtent(grid.plan->size));
                const int count = std::min((grid.plan->size + reductionBlock - 1) / reductionBlock, reductionGroups);
                p.grid = {count, first, products, l != 0};
                compute.run(productsFinishKernel, {&scalars}, {{"partial", &grid.productPartials}}, p, {reductionWidth, 1, 1});
            }
    }
    void updateBasis(Field& field, int count, bool combine = false)
    {
        for (int first = 0; first < count; first += basisWidth)
            for (int l = 0; l < int(grids.size()); ++l)
            {
                auto p = settings(grids[l]); p.grid = {p.grid[0], std::min(basisWidth, count - first), combine, first};
                compute.run(basisUpdateKernel, {&field[l]}, {{"basis", &basis[first / basisWidth][l]}, {"geometry", &grids[l].metadata}, {"scalar", &scalars}}, p,
                            workExtent(unsigned(grids[l].patchWork[1]) * patchThreads));
            }
    }

    Result solveCorrection(Field& pressure, const Field& rhs, double tolerance, int maxIterations)
    {
        prepareBasis();
        scalar(9); clear(update); apply(error, pressure); increment(error, rhs, -1.); normTo(error);
        auto state = status(); int iterations = 0;
        while (state[0] >= tolerance && iterations < maxIterations && state[6] == 0.)
        {
            apply(current, update); scale(current, -1.); increment(current, error, 1.); normTo(current);
            scalar(3, 0, 0., float(tolerance)); normalizeBasis(current, 0);
            const int count = std::min(restart, maxIterations - iterations);
            int submitted = 0;
            for (int k = 0; k < count; ++k)
            {
                precondition(work, current, false); apply(candidate, work);
                for (int pass = 0; pass < 2; ++pass)
                {
                    // Every product uses the unchanged vector before any subtraction.
                    basisProducts(candidate, k); scalar(4, k); updateBasis(candidate, k + 1);
                }
                normTo(candidate); scalar(5); normalizeBasis(candidate, k + 1); scalar(6, k, 0., float(tolerance));
                current.swap(candidate);
                submitted = k + 1;
                state = status();
                if (state[3] == 0. || state[6] != 0.) break;
            }
            for (int k = submitted - 1; k >= 0; --k) scalar(7, k);
            updateBasis(candidate, submitted, true);
            precondition(work, candidate, false); increment(update, work, 1.);
            state = status(); iterations = int(state[5]);
            // As on CPU, a Krylov breakdown returns before committing the accumulated impulse.
            if (state[6] != 0.) return {iterations, std::numeric_limits<double>::infinity(), false};
            if (state[7] < tolerance || state[4] == 0.) break;
            // The next restart recomputes its residual from the accumulated correction.
            state[0] = state[7];
        }
        increment(pressure, update, -1.); apply(work, pressure); increment(work, rhs, -1.); normTo(work);
        state = status();
        const double magnitude = state[6] == 0. ? double(state[0]) : std::numeric_limits<double>::infinity();
        return {int(state[5]), magnitude, std::isfinite(magnitude) && magnitude <= tolerance};
    }

    Result solve(Field& pressure, const Field& rhs, double tolerance, int maxIterations)
    {
        int iterations = 0;
        for (;;)
        {
            auto result = solveCorrection(pressure, rhs, tolerance, maxIterations - iterations);
            iterations += result.iterations;
            // Float Krylov work can reach its estimated target before the true
            // residual. Correct that residual while retaining pressure low bits.
            if (result.converged || !std::isfinite(result.residual) || iterations >= maxIterations || result.iterations == 0 || pressure[0].channels == 1)
                return {iterations, result.residual, result.converged};
        }
    }

    Result solveMultigrid(Field& pressure, const Field& rhs, double tolerance, int maxIterations)
    {
        scalar(9); prepareRhs(right, rhs);
        auto measure = [&]
        {
            apply(error, pressure); scale(error, -1.); increment(error, right, 1.); reduce(error, error, 2, 0);
            scalar(10, 0, 0., float(tolerance), maxIterations);
        };
        measure(); auto state = status();
        while (state[3] != 0.)
        {
            const int count = std::min(8, maxIterations - int(state[5]));
            for (int k = 0; k < count; ++k)
            {
                precondition(update, error, true); increment(pressure, update, 1., -1, true); scalar(11); measure();
            }
            state = status();
        }
        const double magnitude = state[6] == 0. ? double(state[0]) : std::numeric_limits<double>::infinity();
        return {int(state[5]), magnitude, std::isfinite(magnitude) && magnitude <= tolerance};
    }
};

AmrPressureGPU::AmrPressureGPU(webgpu::Compute& compute, const std::vector<Level>& levels) : impl(std::make_unique<Impl>(compute, levels)) {}
AmrPressureGPU::~AmrPressureGPU() = default;
AmrPressureGPU::Field AmrPressureGPU::makeField(bool compensated) const { return impl->makeField(compensated); }
const std::vector<AmrPressureGPU::Patch>& AmrPressureGPU::patches(int level) const { return impl->topology.grids[level].patches; }
const AmrPressureTopology2D& AmrPressureGPU::topology() const { return impl->topology; }
void AmrPressureGPU::setCoefficients(const Faces& coefficients) { impl->setCoefficients(coefficients); }
void AmrPressureGPU::apply(Field& output, Field& pressure) { impl->apply(output, pressure); }
void AmrPressureGPU::fluxes(Faces& output, Field& pressure) { impl->fluxes(output, pressure); }
void AmrPressureGPU::precondition(Field& output, const Field& rhs, bool bottomKrylov) { impl->precondition(output, rhs, bottomKrylov); }
AmrPressureGPU::Result AmrPressureGPU::solve(Field& pressure, const Field& rhs, double tolerance, int maxIterations) { return impl->solve(pressure, rhs, tolerance, maxIterations); }
AmrPressureGPU::Result AmrPressureGPU::solveMultigrid(Field& pressure, const Field& rhs, double tolerance, int maxIterations) { return impl->solveMultigrid(pressure, rhs, tolerance, maxIterations); }
double AmrPressureGPU::norm(const Field& field) { impl->normTo(field); return impl->status()[0]; }

} // namespace fluid
