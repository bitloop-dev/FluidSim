# FluidSim project instructions

## Working memory

Read [CURRENT_WORK.md](CURRENT_WORK.md) when starting or resuming work, plus relevant [DEVELOPMENT.md](DEVELOPMENT.md) sections for implementation/checks. Read [WORK_STACK.md](WORK_STACK.md) at scope changes and parent handoffs; reports and archives only for specific questions.

- Keep one current brief: goal, parent/return link, status/authorized scope, essential findings/risks, next action and validation. Update in place after meaningful progress; distinguish proposals from agreed work. No session log.
- Before a substantive detour, save the parent's remaining objective, decisions, blockers and resume point in the stack, nearest parent first. Replace the brief with the child task. Routine subtasks and documentation maintenance need no stack entry.
- On completion or abandonment, record a compact outcome/evidence link in the parent. Read and refresh that parent, then make its next unresolved question current. Continue authorized work; proposals alone do not authorize implementation. If nothing remains, say so.
- Aim for one or two screens per brief and a short entry per parent. Keep decision-relevant facts; move detailed/superseded history to `.codex/runs/` or `.codex/plans/archive/`. Remove solved/stale stack entries. Keep enduring rules here and technical contracts in `DEVELOPMENT.md`.

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
- Discuss consequential architectural changes before adopting them unless already authorized. Keep changes and dependencies inside FluidSim; do not change framework or registry baselines to accommodate project work.
- For adaptive-grid work, inspect actual resolution overlays at representative frames and trace what requests/retains refinement. Confirm that layouts respond to evolving scene detail before claiming an adaptive-performance improvement; timings and aggregate material errors alone are insufficient.
- Keep experiment setup, frame timing and rendering in the scene, and the ordered physics substep in the engine. Stay strictly 2D; a Mandelbrot slope experiment means an in-plane force field, not a shallow-water model.
- Make physical modelling choices explicit. Automatic optimisation may choose how to execute a selected model, but must not silently replace simulated gas or trapped bubbles with ambient pressure.

## Fidelity and performance

- Use the user's agreed visual/physical target. Small numerical differences need not reject a visually acceptable result; local error estimates are not accumulated-error guarantees. Record conservation, convergence and volume drift separately, and never conceal failures with post-update clipping or renormalization.
- Compare matched initial conditions, physical scale, timestep, damping, quality settings and simulated duration. Include selection and regrid/restoration in total simulation cost; identify rendering/diagnostic exclusions. Keep current-method controls distinct from the previous optimized uniform baseline. Repeat noisy comparisons; do not select only the fastest result.
- Preserve requested refinement and the selected quality policy; do not loosen thresholds, suppress requests or force fine patches merely to manufacture a speedup or an attractive overlay. Use uniform-fine and wholly-coarse controls when assessing adaptive benefit. Choose a behavioral horizon appropriate to the change; smoke checks alone cannot validate evolving physics.
- Before changing AMR, read [its numerical contract](DEVELOPMENT.md#amr-numerical-contract). Keep one adaptive engine, real coarse/fine state, conservative transfers and observation-only rendering/preview. Remove temporary probes/overrides after retaining evidence; preserve user settings and unrelated work.
