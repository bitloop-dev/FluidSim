#include "FluidFieldInternal.h"
#include <optional>

namespace fluid
{
using detail::FluidFieldAccess;

FluidWebGPUContext::FluidWebGPUContext() : commands(std::make_shared<webgpu::Compute>()) {}

struct FluidField::Storage
{
    Layout layout;
    std::vector<float> cpu;
    std::optional<FluidWebGPUContext> context;
    webgpu::Field gpu;

    explicit Storage(Layout layout) : layout(layout)
    {
        if (layout.channels < 1 || layout.channels > 4)
            throw std::invalid_argument("FluidField supports one to four channels");
    }
    size_t size() const { return size_t(layout.width) * layout.height * layout.channels; }
};

FluidField::FluidField(Layout layout) : storage(std::make_shared<Storage>(layout))
{
    storage->cpu.resize(storage->size());
}

FluidField::FluidField(Layout layout, const FluidWebGPUContext& context)
    : storage(std::make_shared<Storage>(layout))
{
    storage->context = context;
    storage->gpu = FluidFieldAccess::commands(context).field(layout.width, layout.height, layout.channels);
}

const FluidField::Layout& FluidField::layout() const { return storage->layout; }
bool FluidField::isWebGPU() const { return storage->context.has_value(); }

std::span<float> FluidField::values()
{
    if (isWebGPU()) throw std::logic_error("Download a GPU field before accessing CPU values");
    return storage->cpu;
}

std::span<const float> FluidField::values() const
{
    if (isWebGPU()) throw std::logic_error("Download a GPU field before accessing CPU values");
    return storage->cpu;
}

void FluidField::write(std::span<const float> values)
{
    if (values.size() != storage->size()) throw std::invalid_argument("FluidField value count does not match its layout");
    if (isWebGPU())
        FluidFieldAccess::commands(*storage->context).write(storage->gpu, values);
    else
        std::copy(values.begin(), values.end(), storage->cpu.begin());
}

std::vector<float> FluidField::read() const
{
    if (isWebGPU()) return FluidFieldAccess::commands(*storage->context).read(storage->gpu);
    return storage->cpu;
}

FluidField FluidField::upload(const FluidWebGPUContext& context) const
{
    const auto source = values();
    FluidField result(layout(), context);
    result.write(source);
    return result;
}

FluidField FluidField::download() const
{
    FluidField result(layout());
    result.write(read());
    return result;
}

void generateFluidField(FluidField& field,
    const std::function<std::array<float, 4>(bl::DVec2)>& evaluate)
{
    auto values = field.values();
    const auto& layout = field.layout();
    for (int y = 0; y < layout.height; ++y)
    {
        for (int x = 0; x < layout.width; ++x)
        {
            const bl::DVec2 position = {
                layout.origin.x + (x + .5) * layout.cell_size,
                layout.origin.y + (y + .5) * layout.cell_size
            };
            const auto value = evaluate(position);
            const size_t offset = (size_t(y) * layout.width + x) * layout.channels;
            std::copy_n(value.begin(), layout.channels, values.begin() + offset);
        }
    }
}

const webgpu::Field& FluidFieldAccess::gpuField(const FluidField& field)
{
    if (!field.isWebGPU()) throw std::logic_error("Upload the CPU field before GPU consumption");
    return field.storage->gpu;
}

bool FluidFieldAccess::usesContext(const FluidField& field, const FluidWebGPUContext& context)
{
    return field.storage->context && *field.storage->context == context;
}

struct FluidFieldWebGPUGenerator::Impl
{
    FluidWebGPUContext context;
    webgpu::Kernel kernel;
    wgpu::Buffer output;

    Impl(const FluidWebGPUContext& context, std::string_view shader) : context(context)
    {
        FluidFieldAccess::commands(context).build(kernel, "fluid generate field", 1, {}, R"wgsl(
fn execute() {
    let position = params.level.xy + (vec2f(cell()) + vec2f(0.5)) * cellSize();
    write0(cell(), fieldValue(position));
}
)wgsl", std::string(R"wgsl(
fn fieldParameter(index: u32) -> vec4f { return params.vectors[index]; }
)wgsl") + std::string(shader));
    }
};

FluidFieldWebGPUGenerator::FluidFieldWebGPUGenerator(const FluidWebGPUContext& context, std::string_view shader)
    : impl(std::make_unique<Impl>(context, shader)) {}
FluidFieldWebGPUGenerator::~FluidFieldWebGPUGenerator() = default;

void FluidFieldWebGPUGenerator::generate(FluidField& field, std::span<const std::array<float, 4>> parameters)
{
    if (!FluidFieldAccess::usesContext(field, impl->context))
        throw std::invalid_argument("Field generator and output must share a WebGPU context");
    webgpu::Parameters settings;
    if (parameters.size() > settings.vectors.size())
        throw std::invalid_argument("Field generators support up to 12 vec4 parameters");
    const auto& layout = field.layout();
    settings.grid = {layout.width, layout.height, 0, 0};
    settings.value[0] = float(layout.cell_size);
    settings.level[0] = float(layout.origin.x);
    settings.level[1] = float(layout.origin.y);
    std::copy(parameters.begin(), parameters.end(), settings.vectors.begin());
    const auto& output = FluidFieldAccess::gpuField(field);
    if (impl->output.Get() != output.buffer.Get())
    {
        // Do not retain every previous destination through the binding cache.
        impl->kernel.bindings.clear();
        impl->output = output.buffer;
    }
    auto& commands = FluidFieldAccess::commands(impl->context);
    commands.run(impl->kernel, {&output}, {}, settings);
    commands.flush();
}
}
