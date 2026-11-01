# FluidSim project instructions

## Learning and defensive checks

This is a learning project. Keep implementations, tutorials, and example code focused on the algorithm; speculative defensive checks make the underlying ideas harder to understand.

- Add guards for realistic failure modes, such as a zero timestep before division, numerical solver breakdown, or failure to converge.
- Treat project-controlled densities, `cell_size`, `width`, `height`, and `divergence_tolerance` as valid by default. Do not add non-finite or non-positive checks for them unless the actual input source or computation makes invalid values a plausible concern.
- A mathematical precondition alone does not justify runtime validation. Briefly state an assumption when useful instead of adding checks for almost impossible cases.
- Do not automatically replace unnecessary runtime guards with assertions, exceptions, or other validation boilerplate.
- When a non-obvious guard is warranted, explain the concrete risk with a short comment.

## Incremental development

- Follow [DEVELOPMENT.md](DEVELOPMENT.md) for the current engine boundary, optimisation map and small regression set. Future milestones are discussion, not permission to implement them.
- Take one agreed conceptual step at a time. Keep cleanup, new physics and performance changes separate; explain what an optimisation does and which algorithm it preserves.
- Keep experiment setup, frame timing and rendering in the scene, and the ordered physics substep in the engine. Stay strictly 2D; a Mandelbrot slope experiment means an in-plane force field, not a shallow-water model.
- Make physical modelling choices explicit. Automatic optimisation may choose how to execute a selected model, but must not silently replace simulated gas or trapped bubbles with ambient pressure.

## Engine ownership and readability

- Keep each engine and its adapter in `UniformCPU`, `UniformWebGPU`, `AdaptiveCPU` or `AdaptiveWebGPU`. Engines may use `Shared` helpers, but must not include or link another engine. Keep backend selection in `FluidGrid` and legacy reference code in `LegacyWebGL`.
- Share actual common geometry, types and infrastructure. Preserve separate algorithms, precision handling and execution schedules where they differ; matching file shape is not a reason to introduce a framework or unused hooks.
- Follow the Bitloop development skill and local `.clang-format`. Use Allman braces, readable names without member prefixes/suffixes, normally one declaration or operation per line, and blank lines between meaningful sections. Simple accessors may stay compact.
- Use short `//` comments to explain assumptions, stages and non-obvious numerical choices. Optimize for minimal, fast, expressive code rather than the fewest lines.
- Format embedded WGSL as readable source: spaced expressions, one statement per line, indented blocks and separated functions. C++ formatting alone does not format raw shader strings. Preserve arithmetic and reduction order during cleanup.
- Nest embedded WGSL within its surrounding C++ indentation. For a shader passed as a function argument, put `R"wgsl(` on its own continuation line, indent the shader one additional level, and align `)wgsl"` with the opening marker. For named shader initializers, the opening marker may stay beside the declaration; indent the shader one level beyond the declaration and align the closing marker with the declaration. Use four spaces per level and preserve relative indentation inside complete shaders and concatenated fragments. Review this layout after clang-format, which does not handle WGSL indentation.

### Logic groups and conditional braces

- Put a blank line before a standalone `//` comment that introduces a new section after other code. Keep consecutive comment lines together. A comment at the beginning of a block does not need an extra leading blank line, and trailing inline comments stay attached to their statements.
- Separate distinct logic groups with a blank line even when they share a general topic. In ImGui code, a control and its setup, change handler, tooltip and disabled scope form one group; the next independent control starts a new group.
- Omit braces for a simple unwrapped condition with one single-line body. Use Allman braces if the condition/header wraps across lines, or if its body spans multiple lines. A lambda callback or nested control statement counts as a multiline body even when it is technically one C++ statement. Apply the same readability rule to loops and `else` bodies; preserve normal `else if` chains.
- Do not treat clang-format as the whole style check: it cannot express the conditional brace rule or identify semantic UI groups. Review those explicitly after formatting.
