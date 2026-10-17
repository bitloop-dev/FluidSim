#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#ifdef __EMSCRIPTEN__
#include <GLES3/gl3.h>
#include <emscripten/html5.h>
#else
#include <glad/glad.h>
#endif

#include <vector>

namespace fluid::gpu_detail
{

// save graphics state once around a group of numerical passes, rather than around every draw
struct ComputeState
{
    GLint draw_fbo, read_fbo, program, vao, active_texture, pack_buffer, unpack_buffer;
    GLint viewport[4], textures[8], samplers[8];
    GLboolean color_mask[4];
    static constexpr GLenum capabilities[] = {
        GL_BLEND, GL_DEPTH_TEST, GL_STENCIL_TEST, GL_SCISSOR_TEST,
        GL_CULL_FACE, GL_RASTERIZER_DISCARD, GL_DITHER
    };
    static constexpr GLenum pixel_parameters[] = {
        GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH, GL_PACK_SKIP_ROWS, GL_PACK_SKIP_PIXELS,
        GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_ROWS, GL_UNPACK_SKIP_PIXELS
    };
    GLboolean enabled[std::size(capabilities)];
    GLint pixel_values[std::size(pixel_parameters)];
#ifdef __EMSCRIPTEN__
    const bool sampler_objects = true;
#else
    // Bitloop requests GL 3.2; separate sampler objects become core in GL 3.3
    const bool sampler_objects = GLAD_GL_VERSION_3_3 != 0;
#endif
#ifndef __EMSCRIPTEN__
    GLboolean srgb;
#endif

    ComputeState()
    {
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fbo);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo);
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        glGetIntegerv(GL_VIEWPORT, viewport);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &active_texture);
        glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &pack_buffer);
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &unpack_buffer);
        glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
        for (int i = 0; i < 8; ++i)
        {
            glActiveTexture(GL_TEXTURE0 + i);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &textures[i]);
            if (sampler_objects)
            {
                glGetIntegerv(GL_SAMPLER_BINDING, &samplers[i]);
                glBindSampler(i, 0);
            }
        }
        for (size_t i = 0; i < std::size(capabilities); ++i)
        {
            enabled[i] = glIsEnabled(capabilities[i]);
            glDisable(capabilities[i]);
        }
        for (size_t i = 0; i < std::size(pixel_parameters); ++i)
        {
            glGetIntegerv(pixel_parameters[i], &pixel_values[i]);
            glPixelStorei(pixel_parameters[i], i % 4 == 0 ? 1 : 0);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
#ifndef __EMSCRIPTEN__
        srgb = glIsEnabled(GL_FRAMEBUFFER_SRGB);
        glDisable(GL_FRAMEBUFFER_SRGB);
#endif
    }

    ~ComputeState()
    {
        for (int i = 0; i < 8; ++i)
        {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            if (sampler_objects) glBindSampler(i, samplers[i]);
        }
        glActiveTexture(active_texture);
        for (size_t i = 0; i < std::size(capabilities); ++i)
            if (enabled[i]) glEnable(capabilities[i]);
        for (size_t i = 0; i < std::size(pixel_parameters); ++i)
            glPixelStorei(pixel_parameters[i], pixel_values[i]);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pack_buffer);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpack_buffer);
        glColorMask(color_mask[0], color_mask[1], color_mask[2], color_mask[3]);
#ifndef __EMSCRIPTEN__
        if (srgb) glEnable(GL_FRAMEBUFFER_SRGB);
#endif
        glBindVertexArray(vao);
        glUseProgram(program);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fbo);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo);
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    }
};

struct Texture
{
    GLuint id = 0;
    mutable GLuint framebuffer = 0;
    int width = 0, height = 0, channels = 1;

    Texture() = default;
    Texture(int w, int h, int channels = 1) : width(w), height(h), channels(channels)
    {
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        const GLenum format = channels == 1 ? GL_RED : channels == 2 ? GL_RG : GL_RGBA;
        const GLint storage = channels == 1 ? GL_R32F : channels == 2 ? GL_RG32F : GL_RGBA32F;
        glTexImage2D(GL_TEXTURE_2D, 0, storage, w, h, 0, format, GL_FLOAT, nullptr);
    }
    ~Texture()
    {
        if (framebuffer) glDeleteFramebuffers(1, &framebuffer);
        if (id) glDeleteTextures(1, &id);
    }
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&& other) noexcept { swap(other); }
    Texture& operator=(Texture&& other) noexcept { swap(other); return *this; }
    void swap(Texture& other) noexcept
    {
        std::swap(id, other.id);
        std::swap(framebuffer, other.framebuffer);
        std::swap(width, other.width);
        std::swap(height, other.height);
        std::swap(channels, other.channels);
    }
};

inline const char* shaderHeader()
{
#ifdef __EMSCRIPTEN__
    return "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n"
           "#define OUT0 layout(location=0) out\n#define OUT1 layout(location=1) out\n";
#else
    return "#version 150\n#define OUT0 out\n#define OUT1 out\n";
#endif
}

inline GLuint compile(GLenum type, const std::string& source, std::string_view name)
{
    GLuint shader = glCreateShader(type);
    const char* text = source.c_str();
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(std::max(length, 1), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("FluidGridGPU " + std::string(name) + ": " + log);
    }
    return shader;
}

struct Program
{
    GLuint id = 0;
    std::unordered_map<std::string, GLint> locations;
    ~Program() { if (id) glDeleteProgram(id); }
    Program() = default;
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    GLint location(const char* name)
    {
        auto [it, added] = locations.try_emplace(name, -1);
        if (added) it->second = glGetUniformLocation(id, name);
        return it->second;
    }
    void scalar(const char* name, float value) { glUniform1f(location(name), value); }
    void integer(const char* name, int value) { glUniform1i(location(name), value); }
};

struct PassRunner
{
    using Input = std::pair<const char*, const Texture*>;
    GLuint framebuffer = 0, vertex_array = 0, vertex_shader = 0;
    PassRunner()
    {
        vertex_shader = compile(GL_VERTEX_SHADER, std::string(shaderHeader()) + R"(
            void main()
            {
                vec2 p = gl_VertexID == 0 ? vec2(-1, -1) : gl_VertexID == 1 ? vec2(3, -1) : vec2(-1, 3);
                gl_Position = vec4(p, 0, 1);
            }
        )", "vertex shader");
        glGenFramebuffers(1, &framebuffer);
        glGenVertexArrays(1, &vertex_array);
    }
    ~PassRunner()
    {
        glDeleteFramebuffers(1, &framebuffer);
        glDeleteVertexArrays(1, &vertex_array);
        glDeleteShader(vertex_shader);
    }
    void target(std::initializer_list<const Texture*> outputs) const
    {
        const Texture& first = **outputs.begin();
        if (outputs.size() == 1)
        {
            // Keep a framebuffer with its texture, so swaps never require reattaching it.
            if (!first.framebuffer)
            {
                glGenFramebuffers(1, &first.framebuffer);
                glBindFramebuffer(GL_FRAMEBUFFER, first.framebuffer);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, first.id, 0);
                const GLenum attachment = GL_COLOR_ATTACHMENT0;
                glDrawBuffers(1, &attachment);
            }
            else glBindFramebuffer(GL_FRAMEBUFFER, first.framebuffer);
            glViewport(0, 0, first.width, first.height);
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GLenum attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        int i = 0;
        for (const Texture* output : outputs)
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachments[i++], GL_TEXTURE_2D, output->id, 0);
        for (; i < 2; ++i)
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachments[i], GL_TEXTURE_2D, 0, 0);
        glDrawBuffers(GLsizei(outputs.size()), attachments);
        glViewport(0, 0, first.width, first.height);
    }
    void bind(Program& program, std::initializer_list<const Texture*> outputs, std::initializer_list<Input> inputs) const
    {
        target(outputs);
        glUseProgram(program.id);
        glBindVertexArray(vertex_array);
        int unit = 0;
        for (const auto& [name, texture] : inputs)
        {
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, texture->id);
            program.integer(name, unit++);
        }
    }
    void build(Program& program, const char* name, const std::string& common, const char* body) const
    {
        GLuint fragment = compile(GL_FRAGMENT_SHADER, std::string(shaderHeader()) + common + body, name);
        program.id = glCreateProgram();
        glAttachShader(program.id, vertex_shader);
        glAttachShader(program.id, fragment);
#ifndef __EMSCRIPTEN__
        glBindFragDataLocation(program.id, 0, "output0");
        glBindFragDataLocation(program.id, 1, "output1");
#endif
        glLinkProgram(program.id);
        glDeleteShader(fragment);
        GLint success = 0;
        glGetProgramiv(program.id, GL_LINK_STATUS, &success);
        if (!success)
        {
            GLint length = 0;
            glGetProgramiv(program.id, GL_INFO_LOG_LENGTH, &length);
            std::string log(std::max(length, 1), '\0');
            glGetProgramInfoLog(program.id, length, nullptr, log.data());
            throw std::runtime_error("FluidGridGPU " + std::string(name) + ": " + log);
        }
    }
    std::vector<float> read(const Texture& texture) const
    {
        target({ &texture });
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        // RGBA/FLOAT readback also works for float render targets in WebGL2
        std::vector<float> pixels(size_t(texture.width) * texture.height * 4);
        glReadPixels(0, 0, texture.width, texture.height, GL_RGBA, GL_FLOAT, pixels.data());
        return pixels;
    }
};

inline const std::string common = R"(
    uniform ivec2 grid_size;
    uniform float cell_size, air_density, water_density, time_step, tolerance;
    uniform int mode;
    OUT0 vec4 output0;
    ivec2 cell() { return ivec2(gl_FragCoord.xy); }
    float at(sampler2D field, ivec2 p) { return texelFetch(field, p, 0).r; }
    bool finite(float x) { return !isnan(x) && !isinf(x); }
    float densityOf(float fraction)
    {
        fraction = clamp(fraction, 0.0, 1.0);
        return fraction * water_density + (1.0 - fraction) * air_density;
    }
)";


}
