# FluidSim project instructions

## Learning and defensive checks

This is a learning project. Keep implementations, tutorials, and example code focused on the algorithm; speculative defensive checks make the underlying ideas harder to understand.

- Add guards for realistic failure modes, such as a zero timestep before division, numerical solver breakdown, or failure to converge.
- Treat project-controlled densities, `cell_size`, `width`, `height`, and `divergence_tolerance` as valid by default. Do not add non-finite or non-positive checks for them unless the actual input source or computation makes invalid values a plausible concern.
- A mathematical precondition alone does not justify runtime validation. Briefly state an assumption when useful instead of adding checks for almost impossible cases.
- Do not automatically replace unnecessary runtime guards with assertions, exceptions, or other validation boilerplate.
- When a non-obvious guard is warranted, explain the concrete risk with a short comment.
