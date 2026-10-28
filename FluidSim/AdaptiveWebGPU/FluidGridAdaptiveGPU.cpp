#include "FluidGridAdaptiveGPU.h"
#include "FluidGridAdaptiveGPUEdit.h"
#include "FluidLinearReconstructionShaders.h"
#include "FluidGridAdaptiveGPUShaders.h"
#include "AmrPressureGPU.h"
#include "../UniformWebGPU/FluidWebGPUShaders.h"
#include <bit>
#include <chrono>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace fluid {
using webgpu::Field;
using webgpu::Kernel;
using webgpu::Parameters;
namespace {
float bits(int i) {
    return std::bit_cast<float>(i);
}
int address(const AmrFieldLayout2D::Patch& p, int x, int y) {
    return p.offset + p.index(x, y);
}
std::array<unsigned, 3> linearExtent(int n) {
    const unsigned w = std::min(std::max(n, 1), 4096);
    return {w, unsigned((std::max(n, 1) + w - 1) / w), 1};
}
struct Recipe {
    Field field;
    int count = 0;
};
} // namespace

struct FluidGridAdaptiveGPU::Level::Storage {
    AmrFieldLayout2D slopesLayout, weightsLayout, geometryLayout;
    std::array<AmrFieldLayout2D, 2> fluxLayout;
    Field metadata, slopes, weights, phaseGeometry, scalar;
    std::array<Field, 2> beta, flux, predicted;
    Recipe ghosts, restriction;
    std::array<Recipe, 2> faceAverage, fluxAverage, faceSync, fluxSync;
    unsigned extentX = 0, extentY = 0;
};
FluidGridAdaptiveGPU::Level::Level() = default;
FluidGridAdaptiveGPU::Level::~Level() = default;

struct FluidGridAdaptiveGPU::Impl {
    int width, height, maxGridSize, components, liquidComponents;
    double cellSize, lastPressureInterval = 0.;
    std::array<double, 2> lastAcceleration{};
    bool equalDensity;
    ElementCatalogue catalogue;
    FluidMixture liquidPhase, gasPhase, mixture;
    GasLayout gridLayout;
    long long activeCells = 0, allocatedCells = 0;
    webgpu::Compute runner{true};
    Field density, dummy, palette, color, status;
    Field editCommands;
    Kernel editKernel;
    std::vector<std::unique_ptr<Level>> data;
    std::unique_ptr<AmrPressureGPU> pressureSolver;
    AmrPressureGPU::Field pressure, rhs;
    AmrPressureGPU::Faces pressureBeta, pressureFlux;
    Kernel slopeKernel, gasFluxKernel, gasUpdateKernel, weightsKernel, geometryKernel, phaseFluxKernel, phaseUpdateKernel;
    Kernel predictorKernel, divergenceKernel, betaKernel, correctFacesKernel, momentumKernel, dampingKernel, cflKernel;
    Kernel maxKernel, storeScalarKernel, zeroKernel, copyKernel, ghostKernel, averageKernel, syncKernel, overlapKernel, refineKernel, colorKernel;
    std::vector<Kernel*> kernels;
    std::vector<Field> reductions;

    Impl(int w, int h, double dx, ElementCatalogue cat, FluidMixture liquid, FluidMixture gas, GasLayout layout, int maximum)
        : width(w), height(h), maxGridSize(maximum), components(int(liquid.components.size() + gas.components.size())),
          liquidComponents(int(liquid.components.size())), cellSize(dx), catalogue(std::move(cat)), liquidPhase(std::move(liquid)), gasPhase(std::move(gas)),
          mixture(liquidPhase), gridLayout(std::move(layout)) {
        mixture.components.insert(mixture.components.end(), gasPhase.components.begin(), gasPhase.components.end());
        equalDensity =
            std::all_of(mixture.components.begin(), mixture.components.end(), [&](const auto& c) { return c.density == mixture.components.front().density; });
        normalizeGasLayout(gridLayout);
        dummy = packed(1, "adaptive empty input");
        density = packed(components, "adaptive ingredient density");
        std::vector<float> densities;
        for (const auto& c : mixture.components)
            densities.push_back(float(c.density));
        runner.write(density, densities);
        palette = packed(components * 4, "adaptive display palette");
        std::vector<float> colors;
        for (const auto& c : mixture.components) {
            const auto& rgb = catalogue[c.element].color;
            colors.insert(colors.end(), {rgb[0], rgb[1], rgb[2], float(c.density)});
        }
        runner.write(palette, colors);
        reductions.reserve(32);
        buildKernels();
        makeHierarchy(gridLayout);
        makePressure();
    }

    Field packed(int n, const char* name, int channels = 1) {
        auto e = linearExtent(n);
        return runner.field(e[0], e[1], channels, name);
    }
    Recipe recipe(const std::vector<float>& values, int words) {
        Recipe r{packed(int(values.size()), "adaptive topology recipe"), int(values.size()) / words};
        runner.write(r.field, values);
        return r;
    }
    Parameters params(const Level& l, double dt = 0., int d = 0) const {
        Parameters p;
        p.grid = {l.geometry.width, l.geometry.height, components, liquidComponents};
        p.value = {float(l.geometry.cell_size), float(dt), 0, 0};
        p.level = {float(d), 0, 0, float(equalDensity)};
        return p;
    }
    std::array<unsigned, 3> extent(const Level& l, int growth = 0) const {
        return {l.storage->extentX + unsigned(growth), l.storage->extentY + unsigned(growth), unsigned(l.boxes.size())};
    }
    void buildKernels() {
        using namespace amr_gpu;
        const std::string common = std::string(webgpu::geometry) + std::string(addressing);
        auto build = [&](Kernel& k, const char* name, int outputs, std::initializer_list<const char*> inputs, std::string_view body, const std::string& extra) {
            runner.build(k, name, outputs, inputs, body, extra);
            kernels.push_back(&k);
        };
        build(slopeKernel, "adaptive bounded planes", 1, {"state", "metadata"}, slopes, common + std::string(qLimiter) + std::string(webgpu::linearReconstruction));
        build(colorKernel, "adaptive bounded material display", 1, {"state", "metadata", "mask", "palette"}, amr_gpu::color, common + std::string(qLimiter) + std::string(webgpu::linearReconstruction));
        build(gasFluxKernel, "adaptive gas flux", 1, {"state", "velocity", "slopes", "metadata"}, gasFlux, common);
        build(gasUpdateKernel, "adaptive gas conserved update", 1, {"fx", "fy", "metadata"}, gasUpdate, common);
        build(weightsKernel, "adaptive frozen phase correction", 1, {"state", "density", "metadata"}, splitWeights, common);
        build(geometryKernel, "adaptive minority PLIC", 1, {"state", "density", "metadata"}, phaseGeometry, common);
        build(phaseFluxKernel, "adaptive geometric flux", 1, {"state", "velocity", "geometry", "metadata"}, phaseFlux, common);
        build(phaseUpdateKernel, "adaptive split update", 1, {"flux", "velocity", "weights", "metadata"}, phaseUpdate, common);
        build(predictorKernel, "adaptive MAC predictor", 2, {"state", "metadata"}, predictor, common);
        build(divergenceKernel, "adaptive actual divergence", 2, {"ux", "uy", "mask", "metadata"}, amr_gpu::divergence, common);
        build(betaKernel, "adaptive pressure coefficients", 1, {"beta", "metadata"}, packBeta, common);
        build(correctFacesKernel, "adaptive pressure faces", 1, {"predicted", "flux", "metadata"}, correctFaces, common);
        build(momentumKernel, "adaptive MAC momentum correction", 1, {"ux", "uy", "px", "py", "metadata"}, correctMomentum, common);
        build(dampingKernel, "adaptive gas damping", 1, {"metadata"}, damping, common);
        build(cflKernel, "adaptive donor CFL", 1, {"ux", "uy", "metadata"}, donorCfl, common);
        build(maxKernel, "adaptive maximum reduction", 1, {"source"}, reduceMax, {});
        build(storeScalarKernel, "adaptive compact status", 1, {"source"}, "fn execute(){output0Data[params.grid.x]=sourceData[0];}", {});
        build(zeroKernel, "adaptive clear", 1, {}, amr_gpu::zero, {});
        build(copyKernel, "adaptive copy", 1, {"source"}, amr_gpu::copy, {});
        build(ghostKernel, "adaptive conserved boundary", 1, {"parent", "recipes"}, ghost, std::string(webgpu::geometry) + std::string(webgpu::linearReconstruction));
        build(averageKernel, "adaptive conservative restriction", 1, {"source", "recipes"}, amr_gpu::average, {});
        build(syncKernel, "adaptive canonical faces", 1, {"recipes"}, amr_gpu::sync, {});
        build(overlapKernel, "adaptive exact overlap", 1, {"source", "recipes"}, overlap, {});
        build(refineKernel, "adaptive bounded refinement", 1, {"source", "recipes"}, amr_gpu::refine, std::string(webgpu::geometry) + std::string(webgpu::linearReconstruction));
    }
    Recipe ghostRecipe(const Level& level, const Level* coarse) {
        const auto& q = level.qLayout;
        const auto c = coarse ? coarse->qLayout : AmrFieldLayout2D{};
        AmrBoundary2D boundary(q, level.geometry, c);
        std::vector<float> values;
        for (int p = 0; p < int(q.patches.size()); ++p)
            for (const auto& g : boundary.recipes()[p]) {
                const auto& target = q.patches[p];
                const auto& source = (g.coarse ? c : q).patches[g.patch];
                values.insert(values.end(), {bits(target.offset + g.target), bits(source.offset + g.source), bits(source.plane), bits(source.stride),
                                             bits(int(g.coarse) + 2 * int(g.reflectX) + 4 * int(g.reflectY)), float(g.dx), float(g.dy), bits(target.plane)});
            }
        return recipe(values, 8);
    }
    Recipe averageRecipe(const AmrFieldLayout2D& fine, const AmrFieldLayout2D& coarse, const std::vector<AmrBox2D>& fineBoxes,
                         const std::vector<AmrBox2D>& coarseBoxes, int ratio = 2, int axis = -1) {
        AmrTransfer2D plan(fineBoxes, coarseBoxes, ratio, axis);
        std::map<int, std::array<float, 8>> unique;
        for (int p = 0; p < int(coarse.patches.size()); ++p)
            for (const auto& region : plan.recipes()[p]) {
                const auto& out = coarse.patches[p];
                const auto& in = fine.patches[region.fine];
                forCells(region.box, [&](int x, int y) {
                    const int target = address(out, x, y);
                    unique[target] = {bits(target),
                                      bits(address(in, x * ratio, y * ratio)),
                                      bits(out.plane),
                                      bits(in.plane),
                                      bits(in.stride),
                                      bits(axis == 0 ? 1 : ratio),
                                      bits(axis == 1 ? 1 : ratio),
                                      0};
                });
            }
        std::vector<float> values;
        for (const auto& [key, row] : unique)
            values.insert(values.end(), row.begin(), row.end());
        return recipe(values, 8);
    }
    Recipe syncRecipe(const AmrFieldLayout2D& layout, const std::vector<AmrBox2D>& boxes, int axis) {
        AmrFaceSync2D plan(boxes, axis);
        std::map<int, std::array<float, 4>> unique;
        for (int p = 0; p < int(layout.patches.size()); ++p)
            for (const auto& region : plan.recipes()[p]) {
                const auto& out = layout.patches[p];
                const auto& in = layout.patches[region.source];
                forCells(region.box, [&](int x, int y) {
                    const int target = address(out, x, y);
                    unique[target] = {bits(target), bits(address(in, x, y)), bits(out.plane), bits(in.plane)};
                });
            }
        std::vector<float> values;
        for (const auto& [key, row] : unique)
            values.insert(values.end(), row.begin(), row.end());
        return recipe(values, 4);
    }
    std::unique_ptr<Level> makeLevel(AmrGeometry2D geometry, std::vector<AmrBox2D> boxes, const Level* parent, bool physical = true) {
        auto l = std::make_unique<Level>();
        l->geometry = geometry;
        l->boxes = std::move(boxes);
        l->storage = std::make_unique<Level::Storage>();
        auto& s = *l->storage;
        l->qLayout = {l->boxes, components + 2, 2};
        l->conserved = packed(l->qLayout.elements, "adaptive conserved state");
        s.ghosts = ghostRecipe(*l, parent);
        if (!physical)
            return l;
        l->cellLayout = {l->boxes, 1, 0};
        l->pressureLayout = {l->boxes, 1, 1};
        l->active = packed(l->cellLayout.elements, "adaptive active mask");
        l->pressure = packed(l->pressureLayout.elements, "adaptive pressure impulse high and low planes", 2);
        s.slopesLayout = {l->boxes, 2 * components, 1};
        s.weightsLayout = {l->boxes, components + 2, 0};
        s.geometryLayout = {l->boxes, 5, 1};
        s.slopes = packed(s.slopesLayout.elements, "adaptive slopes");
        s.scalar = packed(l->cellLayout.elements, "adaptive reduction samples");
        if (liquidComponents) {
            s.weights = packed(s.weightsLayout.elements, "adaptive frozen split weights");
            s.phaseGeometry = packed(s.geometryLayout.elements, "adaptive PLIC geometry");
        }
        for (int d = 0; d < 2; ++d) {
            auto faces = l->boxes;
            for (auto& b : faces)
                b = b.faces(d);
            l->faceLayout[d] = {faces, 1, 0};
            s.fluxLayout[d] = {faces, components + 2, 0};
            l->velocity[d] = packed(l->faceLayout[d].elements, "adaptive MAC velocity");
            s.beta[d] = packed(l->faceLayout[d].elements, "adaptive inverse face density");
            s.predicted[d] = packed(l->faceLayout[d].elements, "adaptive predicted faces");
            s.flux[d] = packed(s.fluxLayout[d].elements, "adaptive ingredient momentum flux");
            s.faceSync[d] = syncRecipe(l->faceLayout[d], l->boxes, d);
            s.fluxSync[d] = syncRecipe(s.fluxLayout[d], l->boxes, d);
            if (parent) {
                s.faceAverage[d] = averageRecipe(l->faceLayout[d], parent->faceLayout[d], l->boxes, parent->boxes, 2, d);
                s.fluxAverage[d] = averageRecipe(s.fluxLayout[d], parent->storage->fluxLayout[d], l->boxes, parent->boxes, 2, d);
            }
        }
        if (parent)
            s.restriction = averageRecipe(l->qLayout, parent->qLayout, l->boxes, parent->boxes);
        std::vector<float> metadata;
        for (int p = 0; p < int(l->boxes.size()); ++p) {
            const auto& b = l->boxes[p];
            const auto& q = l->qLayout.patches[p];
            const auto& sl = s.slopesLayout.patches[p];
            const auto& w = s.weightsLayout.patches[p];
            const auto& fx = l->faceLayout[0].patches[p];
            const auto& fy = l->faceLayout[1].patches[p];
            const auto& pr = l->pressureLayout.patches[p];
            const int row[] = {b.x,
                               b.y,
                               b.width,
                               b.height,
                               q.offset,
                               q.stride,
                               q.plane,
                               sl.offset,
                               sl.stride,
                               sl.plane,
                               w.offset,
                               w.plane,
                               fx.offset,
                               fx.plane,
                               fy.offset,
                               fy.plane,
                               s.fluxLayout[0].patches[p].offset,
                               s.fluxLayout[1].patches[p].offset,
                               pr.offset,
                               pr.stride,
                               s.geometryLayout.patches[p].offset,
                               l->cellLayout.patches[p].offset,
                               0,
                               0};
            for (int value : row)
                metadata.push_back(bits(value));
            s.extentX = std::max(s.extentX, unsigned(b.width));
            s.extentY = std::max(s.extentY, unsigned(b.height));
        }
        s.metadata = packed(int(metadata.size()), "adaptive patch addresses");
        runner.write(s.metadata, metadata);
        return l;
    }
    void makeHierarchy(const GasLayout& layout) {
        AmrHierarchy2D hierarchy(width, height, cellSize, layout, maxGridSize);
        for (auto& l : hierarchy.levels)
            data.push_back(makeLevel(l.geometry, std::move(l.boxes), data.empty() ? nullptr : data.back().get()));
        activeCells = hierarchy.activeCells;
        allocatedCells = hierarchy.allocatedCells;
        status = packed(2 * int(data.size()), "adaptive compact status");
        for (int lev = 0; lev < int(data.size()); ++lev) {
            auto& l = *data[lev];
            std::vector<float> mask(l.cellLayout.elements, 1.f);
            if (lev + 1 < int(data.size()))
                for (const auto& p : l.cellLayout.patches)
                    for (const auto& b : data[lev + 1]->boxes)
                        forCells(intersect(p.box, b.coarse(2)), [&](int x, int y) { mask[address(p, x, y)] = 0.f; });
            runner.write(l.active, mask);
        }
    }
    void makePressure() {
        std::vector<AmrPressureGPU::Level> levels;
        for (const auto& l : data) {
            auto& v = levels.emplace_back(l->geometry.width, l->geometry.height, l->geometry.cell_size);
            for (auto b : l->boxes)
                v.boxes.push_back({b.x, b.y, b.width, b.height});
        }
        pressureSolver = std::make_unique<AmrPressureGPU>(runner, levels);
        pressure.clear();
        for (const auto& l : data)
            pressure.push_back(l->pressure);
        rhs = pressureSolver->makeField();
        pressureBeta = {pressureSolver->makeField(), pressureSolver->makeField()};
        pressureFlux = {pressureSolver->makeField(), pressureSolver->makeField()};
    }
    void clearBindings() {
        runner.flush();
        for (auto* k : kernels)
            k->bindings.clear();
    }
    void fill(Level& l, const Level* coarse) {
        const auto& r = l.storage->ghosts;
        if (!r.count)
            return;
        Parameters p;
        p.grid = {r.count, 0, components, 0};
        p.level[3] = float(equalDensity);
        runner.run(ghostKernel, {&l.conserved}, {{"parent", coarse ? &coarse->conserved : &dummy}, {"recipes", &r.field}}, p, linearExtent(r.count));
    }
    void fillGhosts() {
        for (int lev = 0; lev < int(data.size()); ++lev)
            fill(*data[lev], lev ? data[lev - 1].get() : nullptr);
    }
    void average(Field& target, const Field& source, const Recipe& r, int count) {
        if (!r.count)
            return;
        Parameters p;
        p.grid = {r.count, 0, count, 0};
        runner.run(averageKernel, {&target}, {{"source", &source}, {"recipes", &r.field}}, p, linearExtent(r.count));
    }
    void sync(Field& field, const Recipe& r, int count) {
        if (!r.count)
            return;
        Parameters p;
        p.grid = {r.count, 0, count, 0};
        runner.run(syncKernel, {&field}, {{"recipes", &r.field}}, p, linearExtent(r.count));
    }
    void restrictState() {
        for (int lev = int(data.size()) - 1; lev > 0; --lev)
            average(data[lev - 1]->conserved, data[lev]->conserved, data[lev]->storage->restriction, components + 2);
    }
    void averageFaces(bool flux, int only = -1) {
        for (int lev = int(data.size()) - 1; lev > 0; --lev)
            for (int d = 0; d < 2; ++d)
                if (only < 0 || only == d) {
                    auto& fine = *data[lev];
                    auto& coarse = *data[lev - 1];
                    if (flux)
                        average(coarse.storage->flux[d], fine.storage->flux[d], fine.storage->fluxAverage[d], components + 2);
                    else
                        average(coarse.velocity[d], fine.velocity[d], fine.storage->faceAverage[d], 1);
                }
    }
    const Field& reduceMaximum(const Field& source, int count) {
        const Field* current = &source;
        int n = count;
        size_t level = 0;
        do {
            const int next = (n + 255) / 256;
            if (level == reductions.size())
                reductions.push_back(packed(next, "adaptive maximum scratch"));
            else if (reductions[level].width * reductions[level].height < next)
                reductions[level] = packed(next, "adaptive maximum scratch");
            Parameters p;
            p.grid[0] = n;
            runner.run(maxKernel, {&reductions[level]}, {{"source", current}}, p, linearExtent(next));
            current = &reductions[level];
            n = next;
            ++level;
        } while (n > 1);
        return *current;
    }
    void maximumStatus(const Field& source, int count, int slot) {
        const auto& maximum = reduceMaximum(source, count);
        Parameters p;
        p.grid[0] = slot;
        runner.run(storeScalarKernel, {&status}, {{"source", &maximum}}, p, {1, 1, 1});
    }
    void copy(Field& target, const Field& source) { runner.run(copyKernel, {&target}, {{"source", &source}}, {}); }
    void zero(Field& target) { runner.run(zeroKernel, {&target}, {}, {}); }
    double divergence() {
        for (int lev = 0; lev < int(data.size()); ++lev) {
            auto& l = *data[lev];
            auto& s = *l.storage;
            runner.run(divergenceKernel, {&rhs[lev], &s.scalar},
                       {{"ux", &l.velocity[0]}, {"uy", &l.velocity[1]}, {"mask", &l.active}, {"metadata", &s.metadata}}, params(l), extent(l));
            maximumStatus(s.scalar, l.cellLayout.elements, lev);
        }
        const auto values = runner.read(status, 0, data.size() * sizeof(float));
        return *std::max_element(values.begin(), values.end());
    }
    double stableTimeStep() {
        double rate = 0.;
        if (data.size() == 1) {
            auto& l = *data[0];
            auto& s = *l.storage;
            runner.run(cflKernel, {&s.scalar}, {{"ux", &l.velocity[0]}, {"uy", &l.velocity[1]}, {"metadata", &s.metadata}}, params(l), extent(l));
            rate = runner.read(reduceMaximum(s.scalar, l.cellLayout.elements), 0, sizeof(float))[0];
        } else {
            for (int lev = 0; lev < int(data.size()); ++lev)
                for (int d = 0; d < 2; ++d)
                    maximumStatus(data[lev]->velocity[d], data[lev]->faceLayout[d].elements, 2 * lev + d);
            const auto values = runner.read(status, 0, 2 * data.size() * sizeof(float));
            for (int lev = 0; lev < int(data.size()); ++lev)
                rate = std::max(rate, (double(values[2 * lev]) + values[2 * lev + 1]) / data[lev]->geometry.cell_size);
        }
        return rate > 0. ? 0.45 / rate : std::numeric_limits<double>::infinity();
    }
    void transport(double dt) {
        if (dt == 0.)
            return;
        if (liquidComponents) {
            transportPhases(dt);
            return;
        }
        fillGhosts();
        for (auto& l : data) {
            auto& s = *l->storage;
            runner.run(slopeKernel, {&s.slopes}, {{"state", &l->conserved}, {"metadata", &s.metadata}}, params(*l, dt), extent(*l, 2));
            for (int d = 0; d < 2; ++d) {
                runner.run(gasFluxKernel, {&s.flux[d]},
                           {{"state", &l->conserved}, {"velocity", &l->velocity[d]}, {"slopes", &s.slopes}, {"metadata", &s.metadata}}, params(*l, dt, d),
                           extent(*l, 1));
                sync(s.flux[d], s.fluxSync[d], components + 2);
            }
        }
        averageFaces(true);
        for (auto& l : data) {
            auto& s = *l->storage;
            runner.run(gasUpdateKernel, {&l->conserved}, {{"fx", &s.flux[0]}, {"fy", &s.flux[1]}, {"metadata", &s.metadata}}, params(*l, dt), extent(*l));
        }
        restrictState();
    }
    void transportPhases(double dt) {
        for (auto& l : data) {
            auto& s = *l->storage;
            runner.run(weightsKernel, {&s.weights}, {{"state", &l->conserved}, {"density", &density}, {"metadata", &s.metadata}}, params(*l, dt), extent(*l));
        }
        for (int d = 0; d < 2; ++d) {
            fillGhosts();
            for (auto& l : data) {
                auto& s = *l->storage;
                runner.run(geometryKernel, {&s.phaseGeometry}, {{"state", &l->conserved}, {"density", &density}, {"metadata", &s.metadata}}, params(*l, dt),
                           extent(*l, 2));
                runner.run(phaseFluxKernel, {&s.flux[d]},
                           {{"state", &l->conserved}, {"velocity", &l->velocity[d]}, {"geometry", &s.phaseGeometry}, {"metadata", &s.metadata}},
                           params(*l, dt, d), extent(*l, 1));
                sync(s.flux[d], s.fluxSync[d], components + 2);
            }
            averageFaces(true, d);
            for (auto& l : data) {
                auto& s = *l->storage;
                runner.run(phaseUpdateKernel, {&l->conserved},
                           {{"flux", &s.flux[d]}, {"velocity", &l->velocity[d]}, {"weights", &s.weights}, {"metadata", &s.metadata}}, params(*l, dt, d),
                           extent(*l));
            }
            restrictState();
        }
    }
    FluidProjectionResult project(double tolerance, double dt, std::array<double, 2> acceleration, int maxIterations) {
        lastPressureInterval = dt;
        if (dt > 0.)
            lastAcceleration = acceleration;
        fillGhosts();
        for (auto& l : data) {
            auto& s = *l->storage;
            for (int d = 0; d < 2; ++d) {
                auto p = params(*l, dt, d);
                p.vectors[0] = {float(dt * acceleration[0]), float(dt * acceleration[1]), 0, 0};
                runner.run(predictorKernel, {&l->velocity[d], &s.beta[d]}, {{"state", &l->conserved}, {"metadata", &s.metadata}}, p, extent(*l, 1));
                sync(l->velocity[d], s.faceSync[d], 1);
            }
        }
        averageFaces(false);
        const double predicted = divergence();
        const bool needs = predicted > tolerance;
        int iterations = 0;
        if (needs) {
            for (int lev = 0; lev < int(data.size()); ++lev) {
                auto& l = *data[lev];
                auto& s = *l.storage;
                for (int d = 0; d < 2; ++d) {
                    copy(s.predicted[d], l.velocity[d]);
                    runner.run(betaKernel, {&pressureBeta[d][lev]}, {{"beta", &s.beta[d]}, {"metadata", &s.metadata}}, params(l, 0, d), extent(l, 1));
                }
            }
            pressureSolver->setCoefficients(pressureBeta);
            const auto result = equalDensity ? pressureSolver->solveMultigrid(pressure, rhs, tolerance, maxIterations < 0 ? (liquidComponents ? 2000 : 200) : maxIterations)
                                             : pressureSolver->solve(pressure, rhs, tolerance, maxIterations < 0 ? (liquidComponents ? 2000 : 200) : maxIterations);
            if (!result.converged) {
                std::ostringstream message;
                message << "Adaptive WebGPU pressure solver did not converge (iterations=" << result.iterations
                        << ", residual=" << result.residual << ", tolerance=" << tolerance << ')';
                throw FluidAdaptiveGPUProjectionFailure(result.iterations, message.str());
            }
            iterations = result.iterations;
            pressureSolver->fluxes(pressureFlux, pressure);
            for (int lev = 0; lev < int(data.size()); ++lev) {
                auto& l = *data[lev];
                auto& s = *l.storage;
                for (int d = 0; d < 2; ++d)
                    runner.run(correctFacesKernel, {&l.velocity[d]},
                               {{"predicted", &s.predicted[d]}, {"flux", &pressureFlux[d][lev]}, {"metadata", &s.metadata}}, params(l, 0, d), extent(l, 1));
            }
            averageFaces(false);
        } else {
            for (auto& l : data)
                zero(l->pressure);
            if (dt * acceleration[0] == 0. && dt * acceleration[1] == 0.)
                return {0, predicted, true};
        }
        for (auto& l : data) {
            auto& s = *l->storage;
            auto p = params(*l, dt);
            p.value[3] = float(needs);
            p.vectors[0] = {float(dt * acceleration[0]), float(dt * acceleration[1]), 0, 0};
            runner.run(momentumKernel, {&l->conserved},
                       {{"ux", &l->velocity[0]}, {"uy", &l->velocity[1]}, {"px", &s.predicted[0]}, {"py", &s.predicted[1]}, {"metadata", &s.metadata}}, p,
                       extent(*l));
        }
        restrictState();
        const double actual = needs ? divergence() : predicted;
        return {iterations, actual, std::isfinite(actual) && actual <= 1.01 * tolerance};
    }

    void copyOverlap(Level& target, const Level& source) {
        std::vector<float> values;
        for (const auto& out : target.qLayout.patches)
            for (const auto& in : source.qLayout.patches)
                forCells(intersect(out.box, in.box), [&](int x, int y) {
                    values.insert(values.end(), {bits(address(out, x, y)), bits(address(in, x, y)), bits(out.plane), bits(in.plane)});
                });
        auto r = recipe(values, 4);
        if (!r.count)
            return;
        Parameters p;
        p.grid = {r.count, 0, components + 2, 0};
        runner.run(overlapKernel, {&target.conserved}, {{"source", &source.conserved}, {"recipes", &r.field}}, p, linearExtent(r.count));
    }
    void refine(Level& target, const Level& source) {
        std::vector<float> values;
        for (const auto& out : target.qLayout.patches)
            for (const auto& in : source.qLayout.patches)
                forCells(intersect(out.box.coarse(2), in.box), [&](int x, int y) {
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                            values.insert(values.end(), {bits(address(out, 2 * x + dx, 2 * y + dy)), bits(address(in, x, y)), bits(out.plane), bits(in.plane),
                                                         bits(in.stride), dx ? .25f : -.25f, dy ? .25f : -.25f, 0});
                });
        auto r = recipe(values, 8);
        if (!r.count)
            return;
        Parameters p;
        p.grid = {r.count, 0, components, 0};
        p.level[3] = float(equalDensity);
        runner.run(refineKernel, {&target.conserved}, {{"source", &source.conserved}, {"recipes", &r.field}}, p, linearExtent(r.count));
    }
    bool regrid(GasLayout nextLayout) {
        normalizeGasLayout(nextLayout);
        if (nextLayout == gridLayout)
            return false;
        restrictState();
        fillGhosts();
        clearBindings();
        auto old = std::move(data);
        const int oldRatio = gridLayout.baseRatio;
        makeHierarchy(nextLayout);
        auto refineAndOverlap = [&](Level& target, const Level& source, int ratio) {
            refine(target, source);
            for (int lev = 0; lev < int(old.size()); ++lev)
                if ((oldRatio >> lev) == ratio)
                    copyOverlap(target, *old[lev]);
        };
        std::vector<std::unique_ptr<Level>> bridges;
        const Level* source = old[0].get();
        for (int ratio = oldRatio / 2; ratio > nextLayout.baseRatio; ratio /= 2) {
            AmrGeometry2D geometry{width / ratio, height / ratio, cellSize * ratio};
            auto bridge = makeLevel(geometry, splitAmrBoxes({geometry.Domain()}, maxGridSize), nullptr, false);
            refineAndOverlap(*bridge, *source, ratio);
            fill(*bridge, nullptr);
            source = bridge.get();
            bridges.push_back(std::move(bridge));
        }
        for (int lev = 0; lev < int(data.size()); ++lev) {
            auto& target = *data[lev];
            const int ratio = nextLayout.baseRatio >> lev;
            if (ratio >= oldRatio) {
                if (ratio == oldRatio)
                    copyOverlap(target, *old[0]);
                else {
                    auto r = averageRecipe(old[0]->qLayout, target.qLayout, old[0]->boxes, target.boxes, ratio / oldRatio);
                    average(target.conserved, old[0]->conserved, r, components + 2);
                }
            } else
                refineAndOverlap(target, lev ? *data[lev - 1] : *source, ratio);
            fill(target, lev ? data[lev - 1].get() : nullptr);
        }
        restrictState();
        clearBindings();
        gridLayout = std::move(nextLayout);
        makePressure();
        lastPressureInterval = 0.;
        return true;
    }
};

FluidGridAdaptiveGPU::FluidGridAdaptiveGPU(int w, int h, double dx, ElementCatalogue cat, FluidMixture gas, GasLayout layout, int maximum)
    : FluidGridAdaptiveGPU(w, h, dx, std::move(cat), {}, std::move(gas), std::move(layout), maximum) {}
FluidGridAdaptiveGPU::FluidGridAdaptiveGPU(int w, int h, double dx, ElementCatalogue cat, FluidMixture liquid, FluidMixture gas, GasLayout layout, int maximum)
    : impl(std::make_unique<Impl>(w, h, dx, std::move(cat), std::move(liquid), std::move(gas), std::move(layout), maximum)) {}
FluidGridAdaptiveGPU::~FluidGridAdaptiveGPU() {
    impl->runner.flush();
}

void FluidGridAdaptiveGPU::upload(const State& state, double tolerance) {
    auto& g = *impl;
    if (!g.liquidComponents && std::any_of(state.liquid_fraction.begin(), state.liquid_fraction.end(), [](double v) { return v != 0.; }))
        throw std::invalid_argument("Liquid input requires a configured liquid mixture");
    if (!g.liquidComponents)
        for (const auto& lane : state.liquid_mass)
            if (std::any_of(lane.begin(), lane.end(), [](double v) { return v != 0.; }))
                throw std::invalid_argument("FluidGridAdaptiveGPU cannot discard liquid ingredient mass");
    const double fineArea = g.cellSize * g.cellSize;
    for (int lev = 0; lev < int(g.data.size()); ++lev) {
        auto& l = *g.data[lev];
        std::vector<float> values(l.qLayout.elements);
        const int ratio = g.gridLayout.baseRatio >> lev;
        for (const auto& p : l.qLayout.patches)
            forCells(p.box, [&](int i, int j) {
                std::vector<double> q(g.components + 2);
                for (int y = j * ratio; y < (j + 1) * ratio; ++y)
                    for (int x = i * ratio; x < (i + 1) * ratio; ++x) {
                        const int index = y * g.width + x;
                        double rho = 0.;
                        for (int n = 0; n < g.components; ++n) {
                            const double density =
                                (n < g.liquidComponents ? state.liquid_mass[n][index] : state.gas_mass[n - g.liquidComponents][index]) / fineArea;
                            q[n] += density / (ratio * ratio);
                            rho += density;
                        }
                        const double u = state.u.empty() ? 0. : .5 * (state.u[y * (g.width + 1) + x] + state.u[y * (g.width + 1) + x + 1]);
                        const double v = state.v.empty() ? 0. : .5 * (state.v[y * g.width + x] + state.v[(y + 1) * g.width + x]);
                        q[g.components] += rho * u / (ratio * ratio);
                        q[g.components + 1] += rho * v / (ratio * ratio);
                    }
                for (int n = 0; n < g.components + 2; ++n)
                    values[address(p, i, j) + n * p.plane] = float(q[n]);
            });
        g.runner.write(l.conserved, values);
        g.zero(l.pressure);
    }
    g.restrictState();
    if (!project(tolerance).converged)
        throw std::runtime_error("Initial adaptive WebGPU projection did not converge");
}
bool FluidGridAdaptiveGPU::regrid(GasLayout layout) {
    return impl->regrid(std::move(layout));
}
void FluidGridAdaptiveGPU::transport(double dt) {
    impl->transport(dt);
}
FluidProjectionResult FluidGridAdaptiveGPU::project(double tolerance, double dt, std::array<double, 2> acceleration, int maxIterations) {
    return impl->project(tolerance, dt, acceleration, maxIterations);
}
double FluidGridAdaptiveGPU::stableTimeStep() const {
    return impl->stableTimeStep();
}
double FluidGridAdaptiveGPU::maxDivergence() {
    return impl->divergence();
}
std::vector<float> FluidGridAdaptiveGPU::colorPixels() const {
    std::vector<float> pixels;
    writeColorPixels(pixels);
    return pixels;
}
void FluidGridAdaptiveGPU::writeColorPixels(std::vector<float>& pixels) const {
    auto& g = *impl;
    g.fillGhosts();
    if (!g.color.buffer)
        g.color = g.runner.field(g.width, g.height, 4, "adaptive display color");
    for (const auto& l : g.data) {
        auto p = g.params(*l);
        p.level[1] = float(std::lround(l->geometry.cell_size / g.cellSize));
        g.runner.run(g.colorKernel, {&g.color}, {{"state", &l->conserved}, {"metadata", &l->storage->metadata}, {"mask", &l->active}, {"palette", &g.palette}},
                     p, g.extent(*l));
    }
    g.runner.read(g.color, pixels);
}
void FluidGridAdaptiveGPU::synchronize() const {
    impl->runner.read(impl->data.front()->conserved, 0, sizeof(float));
}
GasStepResult FluidGridAdaptiveGPU::step(double dt, double tolerance, double damping, std::array<double, 2> acceleration, int maxIterations, bool measureTimings) {
    GasStepResult result;
    auto& g = *impl;
    if (dt == 0.) {
        const double divergence = maxDivergence();
        result.projection = {0, divergence, std::isfinite(divergence) && divergence <= tolerance};
        return result;
    }
    using Clock = std::chrono::steady_clock;
    if (measureTimings) synchronize();
    auto started = measureTimings ? Clock::now() : Clock::time_point{};
    const double limit = stableTimeStep();
    double remaining = dt;
    while (remaining > 0.) {
        const double subDt = std::min(remaining, limit);
        transport(subDt);
        ++result.substeps;
        remaining -= subDt;
    }
    if (damping != 0.)
        for (auto& l : g.data) {
            auto p = g.params(*l, dt);
            p.value[3] = float(std::exp(-damping * dt));
            g.runner.run(g.dampingKernel, {&l->conserved}, {{"metadata", &l->storage->metadata}}, p, g.extent(*l));
        }
    result.transportMs = result.projectionMs = -1.;
    if (measureTimings)
    {
        synchronize();
        const auto now = Clock::now();
        result.transportMs = std::chrono::duration<double, std::milli>(now - started).count();
        started = now;
    }
    result.projection = project(tolerance, dt, acceleration, maxIterations);
    synchronize();
    if (measureTimings)
        result.projectionMs = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    return result;
}
GasTotals FluidGridAdaptiveGPU::totals() const {
    auto& g = *impl;
    GasTotals result;
    result.ingredientMass.resize(g.components);
    result.minMassDensity = std::numeric_limits<double>::infinity();
    // Explicit diagnostic observation; execution and selection never use these downloads.
    for (const auto& l : g.data) {
        const auto q = g.runner.read(l->conserved);
        const auto mask = g.runner.read(l->active);
        const double area = l->geometry.cell_size * l->geometry.cell_size;
        for (int p = 0; p < int(l->qLayout.patches.size()); ++p) {
            const auto& patch = l->qLayout.patches[p];
            const auto& active = l->cellLayout.patches[p];
            forCells(patch.box, [&](int x, int y) {
                if (!mask[address(active, x, y)])
                    return;
                const int i = address(patch, x, y);
                double rho = 0., volume = 0.;
                for (int n = 0; n < g.components; ++n) {
                    const double v = q[i + n * patch.plane];
                    result.finite &= std::isfinite(v);
                    result.ingredientMass[n] += v * area;
                    result.minMassDensity = std::min(result.minMassDensity, v);
                    rho += v;
                    volume += v / g.mixture.components[n].density;
                }
                const double mx = q[i + g.components * patch.plane], my = q[i + (g.components + 1) * patch.plane];
                result.finite &= std::isfinite(mx) && std::isfinite(my);
                result.momentum[0] += mx * area;
                result.momentum[1] += my * area;
                result.kineticEnergy += .5 * (mx * mx + my * my) / rho * area;
                result.maxSpeed = std::max(result.maxSpeed, std::hypot(mx, my) / rho);
                result.maxVolumeError = std::max(result.maxVolumeError, std::abs(volume - 1.));
            });
        }
    }
    return result;
}
webgpu::Compute& FluidGridAdaptiveGPU::commands() const {
    return impl->runner;
}
const std::vector<std::unique_ptr<FluidGridAdaptiveGPU::Level>>& FluidGridAdaptiveGPU::levels() const {
    return impl->data;
}
const ElementCatalogue& FluidGridAdaptiveGPU::elements() const {
    return impl->catalogue;
}
const FluidMixture& FluidGridAdaptiveGPU::gasMixture() const {
    return impl->gasPhase;
}
const FluidMixture& FluidGridAdaptiveGPU::materialMixture() const {
    return impl->mixture;
}
int FluidGridAdaptiveGPU::liquidIngredientCount() const {
    return impl->liquidComponents;
}
int FluidGridAdaptiveGPU::ingredientCount() const {
    return impl->components;
}
const GasLayout& FluidGridAdaptiveGPU::layout() const {
    return impl->gridLayout;
}
long long FluidGridAdaptiveGPU::activeCellCount() const {
    return impl->activeCells;
}
long long FluidGridAdaptiveGPU::allocatedCellCount() const {
    return impl->allocatedCells;
}
int FluidGridAdaptiveGPU::finestWidth() const {
    return impl->width;
}
int FluidGridAdaptiveGPU::finestHeight() const {
    return impl->height;
}
double FluidGridAdaptiveGPU::finestCellSize() const {
    return impl->cellSize;
}
double FluidGridAdaptiveGPU::pressureInterval() const {
    return impl->lastPressureInterval;
}
std::array<double, 2> FluidGridAdaptiveGPU::acceleration() const {
    return impl->lastAcceleration;
}

} // namespace fluid

namespace fluid {
void FluidGridAdaptiveGPU::replaceMaterials(std::span<const detail::ResolvedCellEdit> cells) {
    auto& g = *impl;
    if (!g.editKernel.pipeline) {
        g.runner.build(g.editKernel, "adaptive common material replacement", 1, {"edits"}, amr_gpu::replaceMaterial);
        g.kernels.push_back(&g.editKernel);
    }
    // The adapter has refined these footprints first. No physical-state readback.
    auto& level = *g.data.back();
    std::vector<float> commands;
    commands.reserve(cells.size() * 4);
    for (const auto& cell : cells) {
        const auto found = std::find_if(level.qLayout.patches.begin(), level.qLayout.patches.end(),
            [&](const auto& p) { return p.box.contains(cell.x, cell.y); });
        const auto& p = *found;
        commands.insert(commands.end(), {bits(address(p, cell.x, cell.y)), bits(p.plane),
            bits(int(cell.component)), float(cell.mass / (g.cellSize * g.cellSize))});
    }
    if (!g.editCommands.buffer || g.editCommands.bytes() < commands.size() * sizeof(float))
        g.editCommands = g.packed(int(commands.size()), "adaptive staged material edits");
    g.runner.write(g.editCommands, commands);
    Parameters p;
    p.grid = {int(cells.size()), 0, g.components, 0};
    g.runner.run(g.editKernel, {&level.conserved}, {{"edits", &g.editCommands}}, p, linearExtent(int(cells.size())));
    g.restrictState(); // Synchronize covered parents, preserving projected MAC faces/pressure.
    g.runner.flush();
}
}
