# Current work: source style consistency

A project-wide style sweep now follows the user's explicit grouping and brace preferences.
Standalone section comments and independent UI controls have blank separators. Wrapped
conditions/headers and multiline bodies have braces; simple unwrapped cases retain their
compact form. Production code, tests and embedded shaders were reviewed. Exact token,
preprocessor and declaration-scope checks preserve logic. See
[style sweep plan](.codex/plans/20261101_style_sweep.md) and
[evidence](.codex/runs/run0158_20261101_style_sweep/README.md).

Embedded WGSL now also follows its surrounding C++ indentation, with shader arguments
on their own continuation lines and aligned raw-string delimiters. All 175 blocks in
25 source files retain the same shader lines apart from leading whitespace.

The latest pass applies the purpose-based grouping guide throughout all 134 source
files, with changes in 115. Guards, helpers and processing stages have breathing room;
paired calculations and related setup stay grouped. Abbreviated decimal literals use
complete spellings with unchanged values and type suffixes. Exact token/directive and
shader-layout comparisons pass, as does the final native all-target Release build.
See [grouping evidence](.codex/runs/run0161_20261101_purpose_grouping/README.md).

## Completed engine cleanup

The CPU/WebGPU x uniform/adaptive cleanup is complete. Each backend has its own
folder and core library; shared adaptive and WebGPU helpers own no engine. The facade
alone selects a backend. Legacy WebGL is isolated and no longer linked by the app.

Shared cell observations use spans/native patches. Removed stale AMR conditionals,
the redundant adaptive CPU runtime object, an unused GPU accessor alias, and unreachable
uniform-GPU reduction modes. Viscosity shaders are separate from solver orchestration.
C++ and WGSL are expanded for readability, with local formatting/style guidance.
Native Release builds pass. All 23 existing CTest suites and the full refinement regression
passed across staged gates; a stale SplashCubes palette/placement fixture was corrected
to the scene settings already present before cleanup.

No missing features, numerical models, defaults, checkpoint versions, solver tolerances
or update schedules were added or changed. Existing CPU/GPU arithmetic variants remain
separate where their implementations differ. [Plan](.codex/plans/20261101_backend_cleanup.md)
and [validation](.codex/runs/run0157_20261101_backend_cleanup/README.md).

## Retained physics decisions

HoneyTap uses explicit ceiling inflow with two balanced ceiling vents; ordinary material
edits never imply boundary flow. Zero inlet speed closes the openings. Gas remains fully
simulated, and any material reaching a vent can leave. Viscosity uses the physical driven
residual without a work-count cutoff or relaxed convergence tolerance.
[Boundary implementation evidence](.codex/runs/run0156_20261101_boundary_flows/README.md).

Retain impermeable free-slip walls and the 90-degree capillary contact model. Visible
droplets may attach. The uniform WebGPU fragment fallback fixes reproduced numerical
pinning, not all wall attachment; it remains conservative but can spread unresolved
fragments. No repulsion, separating-cell model or persistent contact state is pending.
[Fragment evidence](.codex/runs/run0153_20261101_fragment_transport/README.md),
[earlier cleanup](.codex/runs/run0155_20261101_wall_cleanup/README.md).
