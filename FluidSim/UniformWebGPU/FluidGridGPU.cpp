#include "FluidGridGPU.h"
#include "UniformPressureReduction.h"
#include "FluidUniformWebGPUKernels.h"
#include "FluidSurfaceTensionKernels.h"
#include "FluidAccelerationKernels.h"
#include "FluidFieldInternal.h"
#include "PressureMultigridGPU.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fluid
{
using namespace webgpu;
using detail::FluidFieldAccess;

struct FluidGridGPU::Impl : UniformKernels
{
    const int width, height;
    float cell_size = 1, gravity_y = 100, tolerance = 0, gas_damping_rate = 0;
    FluidWebGPUContext context;
    Compute& runner;
    Field u, v, next_u, next_v, liquid_fraction, next_liquid_fraction;
    // RG holds liquid mass / cell volume and gas mass / cell volume separately.
    Field density, next_density;
    Field color, next_color;
    // Each RGBA group holds four ingredients, liquid first and then gas.
    struct ComponentGroup
    {
        Field mass, next_mass, correction;
        std::array<float, 4> liquid_mask = {}, gas_mask = {};
        ComponentGroup(Compute& runner, int w, int h) : mass(runner.field(w,h,4)), next_mass(runner.field(w,h,4)), correction(runner.field(w,h,4)) {}
    };
    std::vector<ComponentGroup> components;
    std::vector<float> edit_upload;
    size_t liquid_count = 0, gas_count = 0;
    bool density_valid = false;
    Field momentum_u, momentum_v, next_momentum_u, next_momentum_v;
    Field interfaces, split_weights, transfers_u, transfers_v;
    Field pressure, next_pressure, previous_pressure, pressure_base, pressure_rhs, pressure_residual, next_residual;
    Field pressure_correction, pressure_direction, next_direction, pressure_product;
    Field pressure_coefficients;
    Field residual_stats, correction_stats, denominator, solver_state, next_solver_state;
    std::vector<Field> reduction_levels;
    UniformPressureReduction pressure_reduction;
    PressureMultigridGPU multigrid;
    bool use_multigrid = true;
    int previous_solve_iterations = 7;
    float cached_outflow = 0;
    bool velocity_stats_valid = false;
    int transport_steps = 0;
    struct Capillary
    {
        SurfaceTensionKernels kernels;
        Field heights, raw_curvature, curvature;
        Capillary(Compute& compute, int w, int h)
            : heights(compute.field(w,h,4)), raw_curvature(compute.field(w,h,2)), curvature(compute.field(w,h,2))
        { kernels.build(compute); }
    };
    std::unique_ptr<Capillary> capillary;
    float surface_tension = 0;
    std::optional<FluidField> acceleration;
    std::unique_ptr<AccelerationKernels> acceleration_kernels;

    Impl(int w, int h, const FluidWebGPUContext& context);
    void configure(const FluidGridGPU& grid)
    {
        if (cell_size != float(grid.cell_size)) density_valid = false;
        cell_size = float(grid.cell_size);
        gravity_y = float(grid.gravity_y);
        gas_damping_rate = float(grid.gas_damping_rate);
        surface_tension = float(grid.getSurfaceTension());
        use_multigrid = grid.pressure_preconditioner == FluidGridGPU::PressurePreconditioner::Multigrid;
    }
    using Input = std::pair<const char*, const Field*>;
    void run(Kernel& program, std::initializer_list<const Field*> outputs,
             std::initializer_list<Input> inputs = {}, float dt = 0, int mode = 0)
    {
        Parameters params = program.parameters;
        params.grid[0] = width; params.grid[1] = height; params.grid[2] = mode;
        params.value[0] = cell_size; params.value[1] = dt; params.value[2] = tolerance;
        runner.run(program, outputs, inputs, params);
    }
    void reduce(const Field& input, Field& result, int mode = 0, const Field* other = nullptr, size_t first_level = 0)
    {
        if (mode != 4)
        {
            pressure_reduction.reduce(input, result, mode, other);
            return;
        }
        // Preserve the existing divergence/CFL statistics and their reduction order.
        const Field* source = &input;
        for (size_t i = first_level; i < reduction_levels.size(); ++i)
        {
            const Field& level = reduction_levels[i];
            run(reduce_pass, { &level }, { {"source", source}, {"other", other ? other : source} }, 0, mode);
            source = &level;
            mode = mode == 4 ? 4 : 3;
        }
        run(reduce_pass, { &result }, { {"source", source}, {"other", other ? other : source} }, 0, mode);
    }
    std::array<float, 4> readScalar(const Field& field)
    {
        const auto values = runner.read(field);
        return {values[0], values[1], values[2], values[3]};
    }
    void removeMean(Field& field, Field& scratch)
    {
        reduce(field, residual_stats);
        run(subtract_mean, { &scratch }, { {"source", &field}, {"stats", &residual_stats} });
        field.swap(scratch);
    }
    void uploadComponents(const FluidGridGPU& grid, const State& state);
    void rebuildDensity();
    void rebuildColor(const FluidGridGPU& grid);
    void sumGroups(Kernel& program, Field& result, Field& scratch, size_t first);
    void advectComponents(double dt, int direction);
    void rebuildPressureCoefficients();
    void applyPressureOperator(const Field& input);
    void rebuildResidual();
    void applyPreconditioner(bool restart);
    int solvePressure(int max_iterations, float residual_tolerance, bool refinement);
    FluidGridGPU::ProjectionResult project(double dt, int max_iterations, double divergence_tolerance, bool with_capillarity = false);
    void rebuildCurvature();
    double maxDivergence();
    void dampGasVelocity(double dt);
    void applyGravity(double dt);
    void applyAcceleration(double dt);
    void rebuildInterfaces();
    void rebuildTransfersX(double dt);
    void rebuildTransfersY(double dt);
    void rebuildMomentum();
    void advectMomentumX(double dt);
    void advectMomentumY(double dt);
    void recoverVelocity();
    double transportStepLimit(double dt);
    void freezeComponentCorrections();
    void advectDirection(double step, bool horizontal);
    void advect(double dt);
};

FluidGridGPU::Impl::Impl(int w, int h, const FluidWebGPUContext& context) :
    width(w), height(h),
    context(context), runner(FluidFieldAccess::commands(context)),
    u(runner.field(w + 1, h)), v(runner.field(w, h + 1)), next_u(runner.field(w + 1, h)), next_v(runner.field(w, h + 1)),
    liquid_fraction(runner.field(w, h)), next_liquid_fraction(runner.field(w, h)), density(runner.field(w, h, 2)), next_density(runner.field(w, h, 2)),
    color(runner.field(w, h, 4)), next_color(runner.field(w, h, 4)),
    momentum_u(runner.field(w + 1, h)), momentum_v(runner.field(w, h + 1)), next_momentum_u(runner.field(w + 1, h)), next_momentum_v(runner.field(w, h + 1)),
    interfaces(runner.field(w, h, 4)), split_weights(runner.field(w, h, 2)), transfers_u(runner.field(w + 1, h, 2)), transfers_v(runner.field(w, h + 1, 2)),
    pressure(runner.field(w, h)), next_pressure(runner.field(w, h)), previous_pressure(runner.field(w, h)), pressure_base(runner.field(w, h)), pressure_rhs(runner.field(w, h)),
    pressure_residual(runner.field(w, h)), next_residual(runner.field(w, h)), pressure_correction(runner.field(w, h)),
    pressure_direction(runner.field(w, h)), next_direction(runner.field(w, h)), pressure_product(runner.field(w, h)),
    pressure_coefficients(runner.field(w, h, 4)),
    residual_stats(runner.field(1, 1, 4)), correction_stats(runner.field(1, 1, 4)), denominator(runner.field(1, 1, 4)),
    solver_state(runner.field(1, 1, 4)), next_solver_state(runner.field(1, 1, 4)),
    pressure_reduction(runner, w * h), multigrid(runner, w, h)
{
    for (int rw = (w + 3) / 4, rh = (h + 3) / 4; rw > 1 || rh > 1; rw = (rw + 3) / 4, rh = (rh + 3) / 4)
        reduction_levels.push_back(runner.field(rw,rh,4));
    build(runner);
}

void FluidGridGPU::upload(const State& state)
{
    if (!gpu) gpu = std::make_unique<Impl>(width, height, getWebGPUContext());
    gpu->configure(*this);
    auto upload = [&](Field& field, const std::vector<float>& values)
    {
        if (values.empty()) { gpu->run(gpu->fill,{&field}); return; }
        if (values.size() != size_t(field.width)*field.height)
            throw std::invalid_argument("FluidGridGPU upload dimensions do not match the grid");
        gpu->runner.write(field,values);
    };
    upload(gpu->liquid_fraction,state.liquid_fraction);
    upload(gpu->u,state.u); upload(gpu->v,state.v);
    upload(gpu->pressure,state.pressure); upload(gpu->previous_pressure,state.pressure);
    gpu->uploadComponents(*this,state);
    gpu->previous_solve_iterations=7;
    gpu->cached_outflow=0;
    gpu->velocity_stats_valid=state.u.empty()&&state.v.empty();
    gpu->runner.flush();
}

void FluidGridGPU::commitEdits(detail::CellEditBatch& batch)
{
    batch.prepare(width,height,cell_size*cell_size,liquid_mixture,gas_mixture);
    if (!gpu||gpu->liquid_count!=liquid_mixture.components.size()||gpu->gas_count!=gas_mixture.components.size())
        throw std::logic_error("Upload the configured fluid mixtures before committing edits");
    const auto& cells=batch.cells;
    for (size_t first=0; first<cells.size();)
    {
        const size_t offset=size_t(cells[first].y)*width+cells[first].x;
        size_t last=first+1;
        while (last<cells.size()&&size_t(cells[last].y)*width+cells[last].x==offset+last-first) ++last;
        const size_t count=last-first;
        auto& data=gpu->edit_upload;
        data.assign(count*(1+4*gpu->components.size()),0.0f);
        for (size_t i=0; i<count; ++i)
        {
            const auto& cell=cells[first+i];
            data[i]=cell.component<gpu->liquid_count?1.0f:0.0f;
            data[count+(cell.component/4)*count*4+i*4+cell.component%4]=float(cell.mass);
        }
        gpu->runner.write(gpu->liquid_fraction,std::span<const float>(data.data(),count),offset);
        for (size_t group=0; group<gpu->components.size(); ++group)
            gpu->runner.write(gpu->components[group].mass,std::span<const float>(data.data()+count*(1+4*group),count*4),offset*4);
        first=last;
    }
    gpu->density_valid=false;
}

FluidGridGPU::State FluidGridGPU::download() const
{
    State state{gpu->runner.read(gpu->liquid_fraction),gpu->runner.read(gpu->u),gpu->runner.read(gpu->v),gpu->runner.read(gpu->pressure)};
    const size_t count=size_t(width)*height;
    state.liquid_mass.assign(gpu->liquid_count,std::vector<float>(count));
    state.gas_mass.assign(gpu->gas_count,std::vector<float>(count));
    for (size_t group=0; group<gpu->components.size(); ++group)
    {
        const auto packed=gpu->runner.read(gpu->components[group].mass);
        for (size_t lane=0; lane<4; ++lane)
        {
            const size_t index=group*4+lane;
            if (index>=gpu->liquid_count+gpu->gas_count) break;
            auto& mass=index<gpu->liquid_count?state.liquid_mass[index]:state.gas_mass[index-gpu->liquid_count];
            for (size_t i=0; i<count; ++i) mass[i]=packed[4*i+lane];
        }
    }
    return state;
}

bool FluidGridGPU::saveCheckpoint(FluidCheckpoint& checkpoint) const
{
    if (!gpu) throw std::logic_error("Upload before saving a WebGPU checkpoint");
    using bl::JSON::json;
    json catalogue = json::array();
    for (ElementId id = 0; id < elements.size(); ++id)
        catalogue.push_back({{"name", elements[id].name}, {"color", elements[id].color}});
    auto mixtureMetadata = [](const FluidMixture& mixture)
    {
        json result = json::array();
        for (const auto& component : mixture.components)
            result.push_back({{"element", component.element}, {"density", component.density}});
        return result;
    };

    checkpoint.metadata = {
        {"backend", "webgpu-uniform"}, {"version", 3}, {"scalar", "float32"},
        {"byte_order", std::endian::native == std::endian::little ? "little" : "big"},
        {"width", width}, {"height", height}, {"cell_size", cell_size},
        {"origin", {origin.x, origin.y}}, {"gravity_y", gravity_y}, {"gas_damping_rate", gas_damping_rate},
        {"surface_tension", surface_tension},
        {"acceleration_field", gpu->acceleration.has_value()},
        {"preconditioner", int(pressure_preconditioner)}, {"elements", std::move(catalogue)},
        {"liquid_mixture", mixtureMetadata(liquid_mixture)}, {"gas_mixture", mixtureMetadata(gas_mixture)},
        {"previous_solve_iterations", gpu->previous_solve_iterations}, {"solver_tolerance", gpu->tolerance},
        {"cached_outflow", gpu->cached_outflow}, {"velocity_stats_valid", gpu->velocity_stats_valid}
    };

    // Fixed order: fraction, face velocities, two pressure histories, packed component groups.
    // Density, momentum, colour and pressure workspaces are derived at the next operation.
    const size_t cells = size_t(width) * height;
    const bool has_acceleration = gpu->acceleration.has_value();
    const size_t values = (3 + 4 * gpu->components.size() + (has_acceleration ? 2 : 0)) * cells
        + size_t(width + 1) * height + size_t(width) * (height + 1);
    checkpoint.data.resize(values * sizeof(float));
    size_t offset = 0;
    auto append = [&](const Field& field)
    {
        const auto values = gpu->runner.read(field);
        const size_t bytes = values.size() * sizeof(float);
        std::memcpy(checkpoint.data.data() + offset, values.data(), bytes);
        offset += bytes;
    };
    append(gpu->liquid_fraction);
    append(gpu->u);
    append(gpu->v);
    append(gpu->pressure);
    append(gpu->previous_pressure);
    for (const auto& group : gpu->components) append(group.mass);
    if (has_acceleration) append(FluidFieldAccess::gpuField(*gpu->acceleration));
    return true;
}

bool FluidGridGPU::loadCheckpoint(const FluidCheckpoint& checkpoint)
{
    const auto& meta = checkpoint.metadata;
    const char* byte_order = std::endian::native == std::endian::little ? "little" : "big";
    if (meta.at("backend") != "webgpu-uniform" ||
        (meta.at("version") != 1 && meta.at("version") != 2 && meta.at("version") != 3) ||
        meta.at("scalar") != "float32" || meta.at("byte_order") != byte_order)
        throw std::runtime_error("Incompatible uniform WebGPU checkpoint format");
    if (meta.at("width") != width || meta.at("height") != height || meta.at("cell_size") != cell_size)
        throw std::runtime_error("Checkpoint geometry does not match the fluid grid");

    // Validate file structure before replacing live state. Physical model inputs keep their normal contract.
    ElementCatalogue restored_elements;
    for (const auto& element : meta.at("elements"))
        restored_elements.add({element.at("name").get<std::string>(), element.at("color").get<std::array<float, 3>>()});
    auto readMixture = [&](const bl::JSON::json& entries)
    {
        FluidMixture result;
        for (const auto& entry : entries)
        {
            const auto element = entry.at("element").get<ElementId>();
            if (element >= restored_elements.size())
                throw std::runtime_error("Checkpoint mixture references a missing element");
            result.components.push_back({element, entry.at("density").get<double>()});
        }
        return result;
    };
    auto liquid = readMixture(meta.at("liquid_mixture"));
    auto gas = readMixture(meta.at("gas_mixture"));
    const auto restored_origin = meta.at("origin").get<std::array<double, 2>>();
    const double restored_gravity = meta.at("gravity_y").get<double>();
    const double restored_damping = meta.at("gas_damping_rate").get<double>();
    const double restored_tension = meta.at("version") == 1 ? 0. : meta.at("surface_tension").get<double>();
    if (!std::isfinite(float(restored_tension)) || restored_tension < 0.)
        throw std::runtime_error("Invalid saved surface tension");
    const int preconditioner = meta.at("preconditioner").get<int>();
    if (preconditioner != int(PressurePreconditioner::Diagonal) && preconditioner != int(PressurePreconditioner::Multigrid))
        throw std::runtime_error("Unknown checkpoint pressure preconditioner");
    const int iterations = meta.at("previous_solve_iterations").get<int>();
    const float tolerance = meta.at("solver_tolerance").get<float>();
    const float outflow = meta.at("cached_outflow").get<float>();
    const bool stats_valid = meta.at("velocity_stats_valid").get<bool>();
    const bool has_acceleration = meta.at("version") == 3 && meta.at("acceleration_field").get<bool>();
    const size_t groups = (liquid.components.size() + gas.components.size() + 3) / 4;
    const size_t cells = size_t(width) * height;
    const size_t count = (3 + 4 * groups + (has_acceleration ? 2 : 0)) * cells
        + size_t(width + 1) * height + size_t(width) * (height + 1);
    if (checkpoint.data.size() != count * sizeof(float))
        throw std::runtime_error("Checkpoint fluid payload has the wrong size");
    std::vector<float> values(count);
    std::memcpy(values.data(), checkpoint.data.data(), checkpoint.data.size());

    elements = std::move(restored_elements);
    liquid_mixture = std::move(liquid);
    gas_mixture = std::move(gas);
    origin = {restored_origin[0], restored_origin[1]};
    gravity_y = restored_gravity;
    gas_damping_rate = restored_damping;
    surface_tension = restored_tension;
    pressure_preconditioner = PressurePreconditioner(preconditioner);
    upload({}); // Allocate/reuse fields and rebuild component masks for the saved mixtures.

    size_t offset = 0;
    auto restore = [&](const Field& field)
    {
        const size_t count = size_t(field.width) * field.height * field.channels;
        gpu->runner.write(field, std::span<const float>(values.data() + offset, count));
        offset += count;
    };
    restore(gpu->liquid_fraction);
    restore(gpu->u);
    restore(gpu->v);
    restore(gpu->pressure);
    restore(gpu->previous_pressure);
    for (const auto& group : gpu->components) restore(group.mass);
    clearAccelerationField();
    if (has_acceleration)
    {
        FluidField field({width, height, cell_size, origin, 2}, getWebGPUContext());
        restore(FluidFieldAccess::gpuField(field));
        setAccelerationField(field);
    }
    gpu->previous_solve_iterations = iterations;
    gpu->tolerance = tolerance;
    gpu->cached_outflow = outflow;
    gpu->velocity_stats_valid = stats_valid;
    gpu->density_valid = false;
    gpu->runner.flush();
    return true;
}

void FluidGridGPU::destroy()
{
    if (gpu) gpu->runner.flush();
    gpu.reset();
    context.reset();
}

void FluidGridGPU::setWebGPUContext(const FluidWebGPUContext& shared_context)
{
    if (gpu) throw std::logic_error("Set the WebGPU context before grid upload");
    context = shared_context;
}

FluidWebGPUContext FluidGridGPU::getWebGPUContext()
{
    if (!context) context.emplace();
    return *context;
}

void FluidGridGPU::setAccelerationField(const FluidField& field)
{
    if (!gpu) throw std::logic_error("Upload before setting acceleration");
    const auto& layout = field.layout();
    if (layout.width != width || layout.height != height || layout.cell_size != cell_size ||
        layout.origin.x != origin.x || layout.origin.y != origin.y || layout.channels != 2)
        throw std::invalid_argument("Acceleration requires a matching cell-centred XY field");
    if (!FluidFieldAccess::usesContext(field, gpu->context))
        throw std::invalid_argument("Upload the field to the grid's WebGPU context before binding");

    if (!gpu->acceleration_kernels)
        gpu->acceleration_kernels = std::make_unique<AccelerationKernels>(gpu->runner);
    if (!gpu->acceleration || FluidFieldAccess::gpuField(*gpu->acceleration).buffer.Get() != FluidFieldAccess::gpuField(field).buffer.Get())
        gpu->acceleration_kernels->apply.bindings.clear();
    gpu->acceleration = field;
}

void FluidGridGPU::clearAccelerationField()
{
    if (!gpu) return;
    gpu->acceleration.reset();
    if (gpu->acceleration_kernels) gpu->acceleration_kernels->apply.bindings.clear();
}

void FluidGridGPU::setSurfaceTension(double coefficient)
{
    if (!std::isfinite(float(coefficient)) || coefficient < 0.)
        throw std::invalid_argument("Surface tension must be finite, nonnegative and representable on the GPU");
    surface_tension = coefficient;
}

double FluidGridGPU::capillaryStepLimit() const
{
    if (surface_tension == 0. || liquid_mixture.components.empty() || gas_mixture.components.empty())
        return std::numeric_limits<double>::infinity();
    auto minimum = [](const FluidMixture& mixture)
    {
        double rho = mixture.components.front().density;
        for (const auto& component : mixture.components) rho = std::min(rho,component.density);
        return rho;
    };
    // Constant coefficient and configured density bounds avoid a device reduction/readback.
    constexpr double pi = 3.14159265358979323846;
    return .5 * std::sqrt((minimum(liquid_mixture)+minimum(gas_mixture))*cell_size*cell_size*cell_size/(2.*pi*surface_tension));
}

void FluidGridGPU::Impl::rebuildCurvature()
{
    if (!capillary) capillary = std::make_unique<Capillary>(runner,width,height);
    auto& c = *capillary;
    run(c.kernels.heights,{&c.heights},{{"liquid",&liquid_fraction}});
    run(c.kernels.curvature,{&c.raw_curvature},{{"liquid",&liquid_fraction},{"height_field",&c.heights}});
    run(c.kernels.complete,{&c.curvature},{{"liquid",&liquid_fraction},{"raw_curvature",&c.raw_curvature}});
}

FluidGridGPU::ProjectionResult FluidGridGPU::step(double dt, int max_iterations, double divergence_tolerance, FluidStepTimings* timings)
{
    if (timings) *timings = {};
    if (dt<=0.0) return {0,std::numeric_limits<double>::infinity(),false,0,0};

    using Clock = std::chrono::steady_clock;
    if (timings) synchronize(); // Exclude earlier queued work from this substep's stages.
    auto previous = timings ? Clock::now() : Clock::time_point{};
    auto elapsed = [&]
    {
        synchronize();
        const auto now = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - previous).count();
        previous = now;
        return ms;
    };

    gpu->configure(*this);
    const bool tension = surface_tension > 0.;
    const double subdivisions = tension ? std::max(1.,std::ceil(dt/capillaryStepLimit())) : 1.;
    // A user-supplied coefficient can make the required timestep unrepresentably small.
    if (subdivisions > std::numeric_limits<int>::max())
        throw std::runtime_error("Surface tension requires too many physical substeps for this timestep");
    const int count = int(subdivisions);
    const double step = dt / count;
    ProjectionResult result;
    result.physical_steps = 0;
    for (int i=0; i<count; ++i)
    {
        gpu->advect(step);
        result.transport_steps += gpu->transport_steps;
        if (timings) timings->transport_ms += elapsed();

        gpu->dampGasVelocity(step);
        gpu->applyAcceleration(step);
        if (tension) gpu->rebuildCurvature();
        if (timings) timings->forces_ms += elapsed();

        const auto projection=gpu->project(step,max_iterations,divergence_tolerance,tension);
        gpu->runner.flush();
        if (timings) timings->pressure_ms += elapsed();
        result.iterations += projection.iterations;
        result.max_divergence = projection.max_divergence;
        result.converged = projection.converged;
        ++result.physical_steps;
        if (!result.converged) break;
    }
    return result;
}

void FluidGridGPU::advect(double dt)
{
    if (dt<=0.0) return;
    gpu->configure(*this);
    gpu->advect(dt); gpu->runner.flush();
}

void FluidGridGPU::dampGasVelocity(double dt)
{
    gpu->configure(*this); gpu->dampGasVelocity(dt); gpu->runner.flush();
}

void FluidGridGPU::applyGravity(double dt)
{
    gpu->configure(*this); gpu->applyGravity(dt); gpu->runner.flush();
}

FluidGridGPU::ProjectionResult FluidGridGPU::project(double dt, int max_iterations, double divergence_tolerance)
{
    if (dt<=0.0) return {};
    gpu->configure(*this); const auto result=gpu->project(dt,max_iterations,divergence_tolerance);
    gpu->runner.flush();
    return result;
}

double FluidGridGPU::maxDivergence() const { gpu->configure(*this); return gpu->maxDivergence(); }

std::vector<float> FluidGridGPU::colorPixels() const
{
    std::vector<float> pixels;
    writeColorPixels(pixels);
    return pixels;
}

void FluidGridGPU::writeColorPixels(std::vector<float>& pixels) const
{
    gpu->configure(*this); gpu->rebuildColor(*this); gpu->runner.read(gpu->color, pixels);
}

void FluidGridGPU::synchronize() const { if (gpu) gpu->runner.read(gpu->liquid_fraction,0,sizeof(float)); }

void FluidGridGPU::Impl::rebuildPressureCoefficients()
{
    rebuildDensity();
    run(pressure_coefficient, { &pressure_coefficients }, { {"density", &density} });
    if (use_multigrid) multigrid.rebuild(pressure_coefficients);
}

void FluidGridGPU::Impl::applyPressureOperator(const Field& input)
{
    run(pressure_operator, { &pressure_product }, { {"source", &input}, {"weights", &pressure_coefficients} });
}

void FluidGridGPU::Impl::rebuildResidual()
{
    applyPressureOperator(pressure);
    run(residual, { &pressure_residual }, { {"rhs", &pressure_rhs}, {"product", &pressure_product} });
    reduce(pressure_residual, residual_stats);
}

void FluidGridGPU::Impl::applyPreconditioner(bool restart)
{
    const Field* correction = &pressure_correction;
    if (use_multigrid)
        correction = &multigrid.apply(pressure_residual, residual_stats, tolerance);
    else
        run(precondition, { &pressure_correction }, {
            {"residual", &pressure_residual}, {"stats", &residual_stats}, {"weights", &pressure_coefficients} });
    reduce(*correction, correction_stats, 2, &pressure_residual);
    run(coefficients, { &next_solver_state }, {
        {"residual_stats", &residual_stats}, {"correction_stats", &correction_stats},
        {"previous", &solver_state}, {"denominator", &denominator} }, 0, restart ? 1 : 0);
    run(direction, { &next_direction }, {
        {"correction", correction}, {"old_direction", &pressure_direction},
        {"state", &next_solver_state}, {"previous", &solver_state} }, 0, restart ? 1 : 0);
    pressure_direction.swap(next_direction);
    solver_state.swap(next_solver_state);
}

int FluidGridGPU::Impl::solvePressure(int max_iterations, float residual_tolerance, bool refinement)
{
    tolerance = residual_tolerance;
    // solve a correction from zero so small updates are not added to a large warm-start pressure
    run(fill, { &pressure });
    rebuildResidual();
    auto stats = readScalar(residual_stats);
    if (stats[1] <= tolerance || max_iterations == 0 || stats[3] != 0) return 0;

    run(fill, { &solver_state });
    run(fill, { &pressure_direction });
    run(fill, { &denominator });
    applyPreconditioner(true);
    int submitted = 0;
    int batch = use_multigrid ? (refinement ? 2 : std::clamp(previous_solve_iterations + 1, 2, 64)) : 8;
    int extra_batch = 4;
    std::array<float, 4> state = {};
    while (submitted < max_iterations)
    {
        // Predict the check point from the previous solve; frozen updates cannot overshoot convergence.
        const int end = std::min(submitted + batch, max_iterations);
        for (; submitted < end; ++submitted)
        {
            applyPressureOperator(pressure_direction);
            reduce(pressure_direction, denominator, 1, &pressure_product);
            run(update_pressure, { &next_pressure, &next_residual }, {
                {"pressure", &pressure}, {"residual", &pressure_residual}, {"direction", &pressure_direction},
                {"product", &pressure_product}, {"state", &solver_state}, {"denominator", &denominator} });
            pressure.swap(next_pressure);
            pressure_residual.swap(next_residual);
            reduce(pressure_residual, residual_stats);
            applyPreconditioner(false);
        }
        state = readScalar(solver_state);
        // project verifies the true remaining error on velocity before accepting this correction
        if (state[0] < 0.0 || state[2] <= tolerance) break;
        batch = use_multigrid ? extra_batch : 8;
        extra_batch = std::min(extra_batch * 2, 128);
    }
    removeMean(pressure, next_pressure);
    return int(state[3] + 0.5f);
}

FluidGridGPU::ProjectionResult FluidGridGPU::Impl::project(double dt, int max_iterations, double divergence_tolerance, bool with_capillarity)
{
    velocity_stats_valid = false;
    rebuildPressureCoefficients();

    run(predict_pressure, { &pressure_base }, { {"latest", &pressure}, {"previous", &previous_pressure} });
    previous_pressure.swap(pressure);
    removeMean(pressure_base, next_pressure);
    // apply the warm start first, then solve the remaining divergence as a smaller pressure correction
    if (with_capillarity)
    {
        auto& kernel=capillary->kernels.predict;
        kernel.parameters.value[3]=surface_tension;
        run(kernel,{&next_u},{{"velocity",&u},{"pressure",&pressure_base},{"weights",&pressure_coefficients},
            {"liquid",&liquid_fraction},{"curvature_field",&capillary->curvature}},float(dt));
        run(kernel,{&next_v},{{"velocity",&v},{"pressure",&pressure_base},{"weights",&pressure_coefficients},
            {"liquid",&liquid_fraction},{"curvature_field",&capillary->curvature}},float(dt),1);
    }
    else
    {
        run(correct_velocity, { &next_u }, { {"velocity", &u}, {"pressure", &pressure_base}, {"weights", &pressure_coefficients} }, float(dt));
        run(correct_velocity, { &next_v }, { {"velocity", &v}, {"pressure", &pressure_base}, {"weights", &pressure_coefficients} }, float(dt), 1);
    }
    u.swap(next_u);
    v.swap(next_v);
    ProjectionResult result;
    for (;;)
    {
        run(divergence, { &pressure_rhs }, { {"u", &u}, {"v", &v} }, float(dt), 1);
        removeMean(pressure_rhs, pressure_product);
        // leave room for float rounding when applying the pressure gradient to velocity
        const int iterations = solvePressure(max_iterations - result.iterations, float(0.25 * divergence_tolerance * cell_size * cell_size / dt), result.iterations != 0);
        if (result.iterations == 0) previous_solve_iterations = iterations;
        result.iterations += iterations;
        run(correct_velocity, { &next_u }, { {"velocity", &u}, {"pressure", &pressure}, {"weights", &pressure_coefficients} }, float(dt));
        run(correct_velocity, { &next_v }, { {"velocity", &v}, {"pressure", &pressure}, {"weights", &pressure_coefficients} }, float(dt), 1);
        u.swap(next_u);
        v.swap(next_v);
        run(add_pressure, { &next_pressure }, { {"base", &pressure_base}, {"correction", &pressure} });
        pressure_base.swap(next_pressure);

        // measure actual divergence; refine from zero if float recurrence drift left too much error
        result.max_divergence = maxDivergence();
        result.converged = result.max_divergence <= divergence_tolerance;
        if (result.converged || iterations == 0 || result.iterations >= max_iterations) break;
    }
    // retain total pressure for next step's prediction, after applying the corrections separately
    pressure.swap(pressure_base);
    removeMean(pressure, next_pressure);
    return result;
}

double FluidGridGPU::Impl::maxDivergence()
{
    if (reduction_levels.empty())
        run(velocity_stats, { &residual_stats }, { {"u", &u}, {"v", &v} });
    else
    {
        run(velocity_stats, { &reduction_levels.front() }, { {"u", &u}, {"v", &v} });
        reduce(reduction_levels.front(), residual_stats, 4, nullptr, 1);
    }
    const auto result = readScalar(residual_stats);
    cached_outflow = result[2];
    velocity_stats_valid = result[3] == 0;
    return velocity_stats_valid ? result[1] : std::numeric_limits<double>::infinity();
}

void FluidGridGPU::Impl::dampGasVelocity(double dt)
{
    if (gas_damping_rate == 0 || dt <= 0) return;
    const float retention = float(std::exp(-gas_damping_rate * dt));
    run(damping, {&next_u}, {{"velocity", &u}, {"liquid", &liquid_fraction}}, retention, 0);
    run(damping, {&next_v}, {{"velocity", &v}, {"liquid", &liquid_fraction}}, retention, 1);
    u.swap(next_u); v.swap(next_v);
    velocity_stats_valid = false;
}

void FluidGridGPU::Impl::applyGravity(double dt)
{
    velocity_stats_valid = false;
    run(gravity, { &next_v }, { {"velocity", &v} }, float(gravity_y * dt));
    v.swap(next_v);
}

void FluidGridGPU::Impl::applyAcceleration(double dt)
{
    if (!acceleration)
    {
        applyGravity(dt);
        return;
    }

    velocity_stats_valid = false;
    const auto& values = FluidFieldAccess::gpuField(*acceleration);
    run(acceleration_kernels->apply, {&next_u}, {{"velocity", &u}, {"acceleration", &values}}, float(dt), 0);
    run(acceleration_kernels->apply, {&next_v}, {{"velocity", &v}, {"acceleration", &values}}, float(dt), 1);
    u.swap(next_u);
    v.swap(next_v);
}

void FluidGridGPU::Impl::uploadComponents(const FluidGridGPU& grid, const State& state)
{
    liquid_count = grid.liquid_mixture.components.size();
    gas_count = grid.gas_mixture.components.size();
    const size_t cell_count = size_t(width) * height;
    auto validate = [&](const auto& fields, size_t count)
    {
        if (!fields.empty() && fields.size() != count)
            throw std::invalid_argument("FluidGridGPU upload component count does not match the mixture");
        for (const auto& field : fields)
            if (!field.empty() && field.size() != cell_count)
                throw std::invalid_argument("FluidGridGPU upload component dimensions do not match the grid");
    };
    validate(state.liquid_mass, liquid_count);
    validate(state.gas_mass, gas_count);
    const size_t group_count = (liquid_count + gas_count + 3) / 4;
    while (components.size() > group_count) components.pop_back();
    while (components.size() < group_count) components.emplace_back(runner,width,height);
    std::vector<float> packed(cell_count * 4);
    for (size_t group = 0; group < group_count; ++group)
    {
        auto& component = components[group];
        component.liquid_mask = component.gas_mask = {};
        std::fill(packed.begin(), packed.end(), 0.0f);
        for (size_t lane = 0; lane < 4; ++lane)
        {
            const size_t index = group * 4 + lane;
            if (index >= liquid_count + gas_count) break;
            const bool liquid = index < liquid_count;
            (liquid ? component.liquid_mask : component.gas_mask)[lane] = 1;
            const auto& fields = liquid ? state.liquid_mass : state.gas_mass;
            const size_t k = liquid ? index : index - liquid_count;
            if (fields.empty() || fields[k].empty()) continue;
            for (size_t i = 0; i < cell_count; ++i) packed[4 * i + lane] = fields[k][i];
        }
        runner.write(component.mass, packed);
    }
    density_valid = false;
}

void FluidGridGPU::Impl::rebuildDensity()
{
    if (density_valid) return;
    if (components.empty()) run(fill, { &density });
    for (size_t first = 0; first < components.size(); first += 4)
    {
        std::array<std::array<float, 4>, 4> liquid_masks = {}, gas_masks = {};
        for (size_t j = 0; j < 4 && first + j < components.size(); ++j)
        {
            liquid_masks[j] = components[first + j].liquid_mask;
            gas_masks[j] = components[first + j].gas_mask;
        }
        for (size_t j=0; j<4; ++j) { density_pass.parameters.vectors[j]=liquid_masks[j]; density_pass.parameters.vectors[j+4]=gas_masks[j]; }
        sumGroups(density_pass, density, next_density, first);
    }
    density_valid = true;
}

void FluidGridGPU::Impl::rebuildColor(const FluidGridGPU& grid)
{
    if (components.empty()) run(fill, { &color }, {}, 0, 1);
    for (size_t first = 0; first < components.size(); first += 4)
    {
        std::array<std::array<float, 4>, 4> channels[3] = {};
        for (size_t group = 0; group < 4 && first + group < components.size(); ++group)
        for (size_t lane = 0; lane < 4; ++lane)
        {
            const size_t index = (first + group) * 4 + lane;
            if (index >= liquid_count + gas_count) break;
            const auto& component = index < liquid_count ? grid.liquid_mixture.components[index]
                : grid.gas_mixture.components[index - liquid_count];
            const auto& rgb = grid.elements[component.element].color;
            for (size_t c = 0; c < 3; ++c)
                channels[c][group][lane] = float(rgb[c] / (component.density * grid.cell_size * grid.cell_size));
        }
        for (size_t c=0; c<3; ++c) for (size_t j=0; j<4; ++j) color_pass.parameters.vectors[c*4+j]=channels[c][j];
        sumGroups(color_pass, color, next_color, first);
    }
}

void FluidGridGPU::Impl::sumGroups(Kernel& program, Field& result, Field& scratch, size_t first)
{
    // Four packed inputs cover sixteen ingredients per dispatch.
    const size_t count = std::min(size_t(4), components.size() - first);
    const Field* inputs[4];
    for (size_t j = 0; j < 4; ++j) inputs[j] = &components[first + std::min(j, count - 1)].mass;
    program.parameters.grid[3] = int(count);
    run(program, { &scratch }, { {"masses0", inputs[0]}, {"masses1", inputs[1]},
        {"masses2", inputs[2]}, {"masses3", inputs[3]}, {"previous", &result} }, 0, first != 0);
    result.swap(scratch);
}


FluidGridGPU::FluidGridGPU(int width, int height, double cell_size) : width(width), height(height), cell_size(cell_size) {}
FluidGridGPU::~FluidGridGPU() = default;


void FluidGridGPU::Impl::rebuildInterfaces()
{
    run(reconstruct, { &interfaces }, { {"liquid", &liquid_fraction} });
}

void FluidGridGPU::Impl::rebuildTransfersX(double dt)
{
    run(transfers, { &transfers_u }, { {"liquid", &liquid_fraction}, {"interfaces", &interfaces}, {"flow", &u}, {"density", &density} }, float(dt));
}

void FluidGridGPU::Impl::rebuildTransfersY(double dt)
{
    run(transfers, { &transfers_v }, { {"liquid", &liquid_fraction}, {"interfaces", &interfaces}, {"flow", &v}, {"density", &density} }, float(dt), 1);
}

void FluidGridGPU::Impl::advectComponents(double dt, int direction)
{
    for (size_t index = 0; index < components.size(); ++index)
    {
        auto& component = components[index];
        const bool update_fraction = index == 0;
        Kernel& program = direction == 0
            ? (update_fraction ? fused_transport.fraction_x : transport_components_x)
            : (update_fraction ? fused_transport.fraction_y : transport_components_y);
        program.parameters.vectors[0] = component.liquid_mask;
        program.parameters.vectors[1] = component.gas_mask;
        const std::initializer_list<Input> inputs = {
            {"liquid", &liquid_fraction}, {"masses", &component.mass}, {"corrections", &component.correction},
            {"transfers", direction == 0 ? &transfers_u : &transfers_v},
            {"flow", direction == 0 ? &u : &v}, {"weights", &split_weights}
        };
        if (direction == 0)
        {
            if (update_fraction)
                run(program, {&component.next_mass, &component.correction, &next_liquid_fraction}, inputs, float(dt), direction);
            else
                run(program, {&component.next_mass, &component.correction}, inputs, float(dt), direction);
        }
        else
        {
            if (update_fraction)
                run(program, {&component.next_mass, &next_liquid_fraction}, inputs, float(dt), direction);
            else
                run(program, {&component.next_mass}, inputs, float(dt), direction);
        }
        component.mass.swap(component.next_mass);
    }
    if (components.empty())
    {
        run(transport_liquid, {&next_liquid_fraction}, {{"liquid", &liquid_fraction},
            {"transfers", direction == 0 ? &transfers_u : &transfers_v},
            {"flow", direction == 0 ? &u : &v}, {"weights", &split_weights}}, float(dt), direction);
    }
    // All groups must consume the old fraction before its replacement becomes current.
    liquid_fraction.swap(next_liquid_fraction);
    density_valid = false;
}

void FluidGridGPU::Impl::rebuildMomentum()
{
    rebuildDensity();
    // Initialize donor velocities alongside momentum, without multiplying and dividing them by density.
    run(momentum, { &momentum_u, &next_u }, { {"density", &density}, {"source", &u} });
    run(momentum, { &momentum_v, &next_v }, { {"density", &density}, {"source", &v} }, 0, 1);
}

void FluidGridGPU::Impl::recoverVelocity()
{
    rebuildDensity();
    run(recover, { &next_u }, { {"density", &density}, {"source", &momentum_u} }, 0, 2);
    run(recover, { &next_v }, { {"density", &density}, {"source", &momentum_v} }, 0, 3);
}

void FluidGridGPU::Impl::advectMomentumX(double dt)
{
    run(transport_momentum, { &next_momentum_u }, {
        {"momentum", &momentum_u}, {"donor_velocity", &next_u}, {"original_velocity", &u},
        {"flow", &u}, {"weights", &split_weights}, {"transfers", &transfers_u} }, float(dt), 0);
    run(transport_momentum, { &next_momentum_v }, {
        {"momentum", &momentum_v}, {"donor_velocity", &next_v}, {"original_velocity", &v},
        {"flow", &u}, {"weights", &split_weights}, {"transfers", &transfers_u} }, float(dt), 1);
    momentum_u.swap(next_momentum_u);
    momentum_v.swap(next_momentum_v);
}

void FluidGridGPU::Impl::advectMomentumY(double dt)
{
    run(transport_momentum, { &next_momentum_u }, {
        {"momentum", &momentum_u}, {"donor_velocity", &next_u}, {"original_velocity", &u},
        {"flow", &v}, {"weights", &split_weights}, {"transfers", &transfers_v} }, float(dt), 2);
    run(transport_momentum, { &next_momentum_v }, {
        {"momentum", &momentum_v}, {"donor_velocity", &next_v}, {"original_velocity", &v},
        {"flow", &v}, {"weights", &split_weights}, {"transfers", &transfers_v} }, float(dt), 3);
    momentum_u.swap(next_momentum_u);
    momentum_v.swap(next_momentum_v);
}

double FluidGridGPU::Impl::transportStepLimit(double dt)
{
    if (!velocity_stats_valid && !std::isfinite(maxDivergence()))
        throw std::runtime_error("FluidGridGPU encountered non-finite transport velocity");
    return cached_outflow > 0 ? 0.5 * cell_size / cached_outflow : dt;
}

void FluidGridGPU::Impl::freezeComponentCorrections()
{
    run(freeze_weights, {&split_weights}, {{"liquid", &liquid_fraction}, {"density", &density}});
}

void FluidGridGPU::Impl::advectDirection(double step, bool horizontal)
{
    rebuildInterfaces();
    if (horizontal) { rebuildTransfersX(step); advectMomentumX(step); }
    else { rebuildTransfersY(step); advectMomentumY(step); }
    // GPU momentum consumes the old component/density buffers before they swap.
    advectComponents(step, horizontal ? 0 : 1);
    recoverVelocity();
}

void FluidGridGPU::Impl::advect(double dt)
{
    transport_steps = 0;
    rebuildMomentum();
    const double max_step = transportStepLimit(dt);
    double remaining = dt;
    while (remaining > 0)
    {
        const double step = std::min(remaining, max_step);
        freezeComponentCorrections();
        advectDirection(step, true);
        advectDirection(step, false);
        ++transport_steps;
        remaining -= step;
    }
    u.swap(next_u);
    v.swap(next_v);
    velocity_stats_valid = false;
}

}
