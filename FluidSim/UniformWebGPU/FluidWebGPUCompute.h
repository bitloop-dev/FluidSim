#pragma once
#include <bitloop/gpu/webgpu.h>
#ifndef __EMSCRIPTEN__
#include <SDL3/SDL_filesystem.h>
#endif
#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fluid::webgpu {

struct Field
{
    wgpu::Buffer buffer;
    int width = 0, height = 0, channels = 1;
    size_t bytes() const { return size_t(width) * height * channels * sizeof(float); }
    void swap(Field& other) { std::swap(*this, other); }
};

struct alignas(16) Parameters
{
    std::array<int, 4> grid = {};
    std::array<float, 4> value = {};
    std::array<float, 4> level = {};
    std::array<std::array<unsigned, 4>, 8> fields = {};
    std::array<std::array<float, 4>, 12> vectors = {};
    std::array<unsigned, 4> extent = {};
    std::array<std::array<unsigned, 4>, 8> padding = {};
};
static_assert(sizeof(Parameters) == 512);

struct Kernel
{
    struct Bindings
    {
        wgpu::BindGroup group;
        // Emdawn's C++ proxy may be recycled even while the JS bind group owns
        // its underlying buffer. Retain the handles used by the pointer key.
        std::array<wgpu::Buffer, 8> buffers;
    };
    wgpu::ComputePipeline pipeline;
    wgpu::ShaderModule module;
    wgpu::PipelineLayout pipeline_layout;
    std::string label;
    std::array<unsigned,8> pipeline_channels = {};
    std::map<std::array<unsigned,8>,wgpu::ComputePipeline> pipelines;
    wgpu::BindGroupLayout layout;
    std::vector<std::string> inputs;
    size_t output_count = 0;
    std::array<unsigned,2> workgroup = {8,8};
    Parameters parameters;
    std::map<std::array<uintptr_t, 8>, Bindings> bindings;
};

// Field bindings and parameter slices are reused; only scalar observation ends a batch.
class Compute
{
public:
    using Input = std::pair<const char*, const Field*>;
    bl::WebGpuDevice gpu;

    explicit Compute(bool preferSubgroups = false)
    {
        bl::WebGpuDevice::Options options;
#ifndef __EMSCRIPTEN__
        if (char* directory = SDL_GetPrefPath("bitloop", "FluidSim"))
        {
            options.cache_directory = std::string(directory) + "webgpu-cache";
            SDL_free(directory);
        }
#endif
        const std::array optional = {wgpu::FeatureName::Subgroups};
        if (preferSubgroups) options.features = optional;
        gpu.initialize(options);
        if (!gpu.waitReady())
        {
            // Retry only unsupported optional features, preserving genuine device failures.
            if (!preferSubgroups || !gpu.adapter() || gpu.adapter().HasFeature(wgpu::FeatureName::Subgroups))
                throw std::runtime_error("FluidSim WebGPU: " + gpu.error());
            gpu.reset(); options.features = {}; gpu.initialize(options);
            if (!gpu.waitReady()) throw std::runtime_error("FluidSim WebGPU: " + gpu.error());
        }
        wgpu::Limits limits;
        gpu.device().GetLimits(&limits);
        stride = (sizeof(Parameters) + limits.minUniformBufferOffsetAlignment - 1) / limits.minUniformBufferOffsetAlignment * limits.minUniformBufferOffsetAlignment;
        uniforms = gpu.buffer(stride * batch_capacity, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "fluid dispatch parameters");
        parameters.reserve(stride * batch_capacity);
    }

    Field field(int width, int height, int channels = 1, const char* label = "fluid field")
    {
        Field result{{}, width, height, channels};
        result.buffer = gpu.buffer(result.bytes(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst, label);
        return result;
    }

    void build(Kernel& kernel, const char* label, size_t outputs, std::initializer_list<const char*> inputs, std::string_view body,
               std::string_view extra = {}, bool full_workgroups = false, std::array<unsigned,2> workgroup = {8,8}, bool subgroups = false)
    {
        kernel.output_count = outputs;
        kernel.workgroup = workgroup;
        kernel.label = label;
        for (auto name : inputs) kernel.inputs.emplace_back(name);
        if (outputs + inputs.size() > 8) throw std::logic_error("FluidSim kernel exceeds portable storage-buffer limit");
        std::string code = R"wgsl(
struct Parameters {
    grid: vec4i, value: vec4f, level: vec4f,
    fields: array<vec4u, 8>, vectors: array<vec4f, 12>, extent: vec4u, padding: array<vec4u, 8>
}
@group(0) @binding(0) var<uniform> params: Parameters;
var<private> point: vec2i;
var<private> invocationZ: u32;
fn cell() -> vec2i { return point; }
fn slice() -> u32 { return invocationZ; }
fn gridSize() -> vec2i { return params.grid.xy; }
fn mode() -> i32 { return params.grid.z; }
fn cellSize() -> f32 { return params.value.x; }
fn timeStep() -> f32 { return params.value.y; }
fn tolerance() -> f32 { return params.value.z; }
fn finite(x: f32) -> bool { return abs(x) <= 3.402823466e+38; }
)wgsl";
        if (subgroups) code = "enable subgroups;\nvar<private> subgroupLane:u32;\n" + code;
        std::vector<wgpu::BindGroupLayoutEntry> entries(outputs + inputs.size() + 1);
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[0].buffer.hasDynamicOffset = true;
        entries[0].buffer.minBindingSize = sizeof(Parameters);
        for (size_t i = 0; i < outputs + inputs.size(); ++i)
        {
            const bool output = i < outputs;
            const std::string name = output ? "output" + std::to_string(i) : kernel.inputs[i - outputs];
            const std::string index = std::to_string(i);
            const std::string channels = "bufferChannels" + index;
            code += "override " + channels + ": u32;\n";
            code += "@group(0) @binding(" + std::to_string(i + 1) + ") var<storage, " + (output ? "read_write" : "read") + "> " + name + "Data: array<f32>;\n";
            // Fixed component lanes let the shader compiler discard unused scalar/vector work.
            const char* lane[] = {"x", "y", "z", "w"};
            if (output)
            {
                code += "fn write" + index + "(p: vec2i, value: vec4f) { let info=params.fields[" + index + "]; let start=(u32(p.y)*info.x+u32(p.x))*" + channels + ";\n";
                for (unsigned c=0;c<4;++c)
                    code += "if (" + channels + " > " + std::to_string(c) + "u) { " + name + "Data[start+" + std::to_string(c) + "u]=value." + lane[c] + "; }\n";
                code += "}\n";
            }
            else
            {
                code += "fn " + name + "Size() -> vec2i { return vec2i(params.fields[" + index + "].xy); }\nfn " + name + "(p: vec2i) -> vec4f { let info=params.fields[" + index + "]; let q=clamp(p,vec2i(0),vec2i(info.xy)-1); let start=(u32(q.y)*info.x+u32(q.x))*" + channels + "; var value=vec4f(0);\n";
                for (unsigned c=0;c<4;++c)
                    code += "if (" + channels + " > " + std::to_string(c) + "u) { value." + lane[c] + "=" + name + "Data[start+" + std::to_string(c) + "u]; }\n";
                code += "return value; }\n";
            }
            auto& entry = entries[i + 1];
            entry.binding = uint32_t(i + 1);
            entry.visibility = wgpu::ShaderStage::Compute;
            entry.buffer.type = output ? wgpu::BufferBindingType::Storage : wgpu::BufferBindingType::ReadOnlyStorage;
        }
        code += extra;
        code += body;
        code += "\n@compute @workgroup_size("+std::to_string(workgroup[0])+", "+std::to_string(workgroup[1])+")\n";
        code += "fn main(@builtin(global_invocation_id) id: vec3u";
        if (subgroups) code += ", @builtin(subgroup_invocation_id) lane:u32";
        code += ") {\n";
        if (subgroups) code += "subgroupLane=lane;\n";
        // Barrier kernels dispatch full workgroups; a divergent tail return would invalidate their barriers.
        if (!full_workgroups) code += "    if (any(id >= params.extent.xyz)) { return; }\n";
        code += R"wgsl(
    point = vec2i(id.xy);
    invocationZ = id.z;
    execute();
}
)wgsl";
        kernel.module = gpu.shader(code, label);
        wgpu::BindGroupLayoutDescriptor group_descriptor;
        group_descriptor.entryCount = entries.size();
        group_descriptor.entries = entries.data();
        kernel.layout = gpu.device().CreateBindGroupLayout(&group_descriptor);
        wgpu::PipelineLayoutDescriptor layout_descriptor;
        layout_descriptor.bindGroupLayoutCount = 1;
        layout_descriptor.bindGroupLayouts = &kernel.layout;
        kernel.pipeline_layout = gpu.device().CreatePipelineLayout(&layout_descriptor);
    }

    void run(Kernel& kernel, std::initializer_list<const Field*> outputs, std::initializer_list<Input> inputs, Parameters settings,
             std::array<unsigned, 3> extent = {})
    {
        if (outputs.size() != kernel.output_count) throw std::logic_error("FluidSim kernel output count mismatch");
        std::array<const Field*, 8> fields = {};
        std::copy(outputs.begin(), outputs.end(), fields.begin());
        size_t field_count = outputs.size();
        for (const auto& name : kernel.inputs)
        {
            const auto found = std::find_if(inputs.begin(), inputs.end(), [&](auto input) { return name == input.first; });
            if (found == inputs.end()) throw std::logic_error("FluidSim kernel missing input " + name);
            fields[field_count++] = found->second;
        }
        std::array<uintptr_t, 8> key = {};
        std::array<unsigned,8> channels = {};
        for (size_t i = 0; i < field_count; ++i)
        {
            const auto& f = *fields[i];
            settings.fields[i] = {unsigned(f.width), unsigned(f.height), unsigned(f.channels), 0};
            key[i] = reinterpret_cast<uintptr_t>(f.buffer.Get());
            channels[i] = unsigned(f.channels);
        }
        if (channels != kernel.pipeline_channels)
        {
            // The few scalar/vector layouts are compiled once; steady dispatches reuse their pipeline.
            auto [variant,inserted] = kernel.pipelines.try_emplace(channels);
            if (inserted)
            {
                std::array<wgpu::ConstantEntry,8> constants;
                std::array<std::string,8> names;
                for (size_t i=0;i<field_count;++i)
                {
                    names[i]="bufferChannels"+std::to_string(i);
                    constants[i].key=names[i].c_str(); constants[i].value=channels[i];
                }
                wgpu::ComputePipelineDescriptor descriptor;
                descriptor.label=kernel.label.c_str(); descriptor.layout=kernel.pipeline_layout;
                descriptor.compute.module=kernel.module; descriptor.compute.entryPoint="main";
                descriptor.compute.constantCount=field_count; descriptor.compute.constants=constants.data();
                variant->second=gpu.device().CreateComputePipeline(&descriptor);
            }
            kernel.pipeline=variant->second; kernel.pipeline_channels=channels;
        }
        if (extent[0] == 0) extent = {unsigned(fields[0]->width), unsigned(fields[0]->height), 1};
        settings.extent = {extent[0], extent[1], extent[2], 0};
        auto [cached, inserted] = kernel.bindings.try_emplace(std::move(key));
        if (inserted)
        {
            std::array<wgpu::BindGroupEntry, 9> entries;
            entries[0].binding = 0; entries[0].buffer = uniforms; entries[0].size = sizeof(Parameters);
            for (size_t i = 0; i < field_count; ++i)
            {
                entries[i + 1].binding = uint32_t(i + 1);
                entries[i + 1].buffer = fields[i]->buffer;
                entries[i + 1].size = fields[i]->bytes();
                cached->second.buffers[i] = fields[i]->buffer;
            }
            wgpu::BindGroupDescriptor descriptor;
            descriptor.layout = kernel.layout;
            descriptor.entryCount = field_count + 1;
            descriptor.entries = entries.data();
            cached->second.group = gpu.device().CreateBindGroup(&descriptor);
        }
        if (parameters.size() == stride * batch_capacity) flush();
        if (!encoder) encoder = gpu.device().CreateCommandEncoder();
        if (!pass) pass = encoder.BeginComputePass();
        const uint32_t offset = uint32_t(parameters.size());
        parameters.resize(parameters.size() + stride);
        std::memcpy(parameters.data() + offset, &settings, sizeof(settings));
        pass.SetPipeline(kernel.pipeline);
        pass.SetBindGroup(0, cached->second.group, 1, &offset);
        pass.DispatchWorkgroups((extent[0] + kernel.workgroup[0]-1) / kernel.workgroup[0],
                                (extent[1] + kernel.workgroup[1]-1) / kernel.workgroup[1], extent[2]);
    }

    void flush()
    {
        if (!encoder) return;
        endPass();
        if (!parameters.empty()) gpu.queue().WriteBuffer(uniforms, 0, parameters.data(), parameters.size());
        auto commands = encoder.Finish();
        gpu.queue().Submit(1, &commands);
        encoder = nullptr;
        parameters.clear();
    }

    void write(const Field& field, std::span<const float> values, size_t offset = 0)
    {
        flush();
        if (!values.empty()) gpu.queue().WriteBuffer(field.buffer, offset * sizeof(float), values.data(), values.size_bytes());
    }

    std::vector<float> read(const Field& field)
    {
        return read(field, 0, field.bytes());
    }

    std::vector<float> read(const Field& field, size_t offset, size_t bytes)
    {
        std::vector<float> values;
        read(field, offset, bytes, values);
        return values;
    }

    void read(const Field& field, std::vector<float>& values)
    {
        read(field, 0, field.bytes(), values);
    }

    void read(const Field& field, size_t offset, size_t bytes, std::vector<float>& values)
    {
        if (!encoder) encoder = gpu.device().CreateCommandEncoder();
        endPass();
        readback.copy(gpu.device(), encoder, field.buffer, offset, bytes);
        flush();
        if (gpu.wait(readback.map()) != wgpu::WaitStatus::Success || !readback.ready())
            throw std::runtime_error("FluidSim WebGPU readback: " + readback.error() + gpu.error());
        values.resize(bytes / sizeof(float));
        std::memcpy(values.data(), readback.data().data(), bytes);
        readback.unmap();
        if (gpu.status() == bl::WebGpuDevice::Status::Failed) throw std::runtime_error("FluidSim WebGPU: " + gpu.error());
    }

private:
    static constexpr size_t batch_capacity = 4096;
    size_t stride = 0;
    wgpu::Buffer uniforms;
    wgpu::CommandEncoder encoder;
    wgpu::ComputePassEncoder pass;
    bl::WebGpuReadback readback;
    std::vector<std::byte> parameters;
    void endPass() { if (pass) { pass.End(); pass = nullptr; } }
};

} // namespace fluid::webgpu
