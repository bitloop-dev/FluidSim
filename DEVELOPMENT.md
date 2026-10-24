# FluidSim development guide

The uniform CPU/GPU liquid engines implement geometric material transport and consistent mass/momentum transport. The native adaptive engine uses one project-owned 2D hierarchy and composite projection for gas and optional liquid phases. Liquid geometry, component masses and momentum share transfers through successive X/Y passes. Uniform CPU and GPU pressure default to multigrid-preconditioned CG; AMR uses the source-derived composite MAC projection. Optional surface tension remains a later stage. `VISION.md` holds the longer-term ambitions and maths; this file is the short map of the working code and possible next steps.

All engines use the component system. For the uniform liquid engines, configure the element catalogue and liquid/gas mixtures, then upload component masses alongside `liquid_fraction`, the total liquid volume fraction. Density, momentum and displayed colours come from those masses. Engine geometry uses liquid/gas terminology; Water, Red, Air and Cloud are scene ingredients. Several miscible ingredients share each phase's geometry; adding an ingredient does not add another interface reconstruction or pressure solve.

The scene catalogue contains blue Water and Red liquid, plus Air and a cyan Cloud gas ingredient. The scene's `red_share` selects the liquid blend. Air and Cloud share density 0.1. The initial atmosphere uses three Perlin scales to divide its unit gas volume between Air and Cloud. These colours are transported normally so they trace motion. Scene walls are closed, including the top. The adaptive engine advances actual coarse/fine level cells; it does not retain hidden fine material samples.

## Where things belong

| File | Responsibility |
| --- | --- |
| `FluidSim/AmrGas.h/.cpp` | Native fluid hierarchy, conservative phase/ingredient/momentum transport, source-derived MAC projection and regrid |
| `FluidSim/FluidInterfaceGeometry.h` | Shared uniform/AMR MYC normal and PLIC geometric area formulas |
| `FluidSim/AmrField2D.h/.cpp`, `AmrTransfer2D.h/.cpp` | 2D patch arrays, clustering, cached ghost/face exchange and conservative transfers |
| `FluidSim/AmrPressure2D.h/.cpp` | Composite operator, GMRES+MLMG and standalone MLMG/BiCGStab |
| `FluidSim/AmrGasRuntime.h/.cpp` | Native worker configuration; no external initialization |
| `FluidSim/FluidCellView.h` | Non-owning views of actual committed cells for display and preview |
| `FluidSim/FluidCoarseningPreview.h/.cpp` | Merge/refinement error estimates, stable requests and source-derived patch clustering; caller applies layout |
| `FluidSim/FluidGrid.h` | Engine data and public operations, in namespace `fluid` |
| `FluidSim/FluidGrid.cpp` | Advection, gravity, material transport and pressure solver |
| `FluidSim/FluidCpuExecution.h` | Worker count for independent CPU cell/row updates |
| `FluidSim/PressureMultigrid.h/.cpp` | CPU pressure hierarchy, fixed symmetric V-cycle and small direct coarse solve |
| `FluidSim/FluidGridGPU.h/.cpp` | Separate float GPU implementation of the same physics, using fragment-shader passes |
| `FluidSim/PressureMultigridGPU.h/.cpp` | GPU pressure hierarchy with an owned CPU coarse solver; GPU-only tiny Cholesky remains selectable |
| `FluidSim/FluidGridGPUInternal.h` | Private numeric textures, shaders, pass runner and graphics-state preservation shared by those two GPU components |
| `FluidSim/FluidGridTypes.h` | Common state layout and projection result; CPU state uses double, GPU state uses float |
| `FluidSim/FluidGridEdit.h` | Shared staged material replacements, validation and sparse commit ordering |
| `FluidSim/FluidGridDisplay.h/.cpp` | Render-thread colour, resolution and preview textures; actual-cell AMR colours |
| `FluidSim/FluidAmbientMaskDisplay.h/.cpp` | Render-thread palette adapter for the CPU ambient mask |
| `FluidSim/FluidAmbientMask.h/.cpp` | CPU per-cell compatibility with a supplied ambient state, history, buffer and reservoir connectivity |
| `FluidSim/FluidSim.h/.cpp` | This experiment's initial pool, frame timing, failure reporting, camera, rendering and captures |
| `tests/FluidGridTests.cpp` | Small engine regression set, independent of scenes and drawing |
| `tests/FluidAmbientTests.cpp` | CPU gas-damping and prescribed-pressure boundary checks |
| `tests/FluidAmbientMaskTests.cpp` | Ambient-mask composition, pressure, orientation, connectivity, history and unchanged-physics checks |
| `tests/FluidGridGPUTests.cpp` | GPU/CPU comparisons and GPU regressions in a hidden graphics context |

`FluidSimEngine` is a local CMake library used by both the scene and tests. It depends on Bitloop's vector type and thread pool, but has no Scene, camera, OpenGL or UI code. Another experiment can include `FluidGrid.h` and link that target. Packaging it as a separate repository is a later decision.

A new `FluidGrid(width, height, cell_size)` starts with zero velocity and zero liquid fraction. Configure its mixtures and populate component masses before advancing physics. The scene creates an initial `State` in `initializePool()` and passes it to `upload()`; that sloped pool is an experiment, not an engine default. Existing CPU public arrays and individual operations remain available for learning. Dimensions and cell size describe a fixed grid; changing them requires constructing another grid.

CPU density, pressure coefficients and the multigrid hierarchy are now cached. After directly editing `liquid_mass`, `gas_mass` or derived density/pressure-coefficient arrays, call `fluid.invalidateCaches()` before the next physics operation. It marks derived data stale; rebuilding happens when needed. `upload()` and engine material transport do this automatically. Direct velocity edits need no cache notification because CPU velocity statistics are recomputed, but transport still requires projected velocity. Mixture membership changes continue to require `upload()`. GPU textures remain engine-owned. Its `upload()` already invalidates its caches; direct CPU-array access is not part of the shared interface.

## Current CPU experiment and uniform engines

Native scenes use the one `AmrGas` hierarchy, now with optional distinct liquid ingredients. Scenarios 2/3 retain closed circulating Air/Cloud, no gravity, optional damping, and automatically selected fine patches over a coarse background. Scenario 2 retains the Perlin cloud. Scenario 3 (`MOVING_CLOUD_DETAIL`) starts one compact cloud with a one-world-unit tanh edge; its centre is (.3W,.5H), radius .06 min(W,H), and circulation speed W/128. Its damping defaults to zero so the cloud travels beyond its original footprint during the 32-second comparison. Selection uses the existing tolerances and actual fields; no patch follows a prescribed trajectory. Old regions should merge as the cloud leaves. Scenarios 0/1 now use geometric liquid transport in that hierarchy, gravity 100 and closed walls. The user explicitly removed the old open atmospheric top. Automatic selection protects all liquid and its predicted halo at 1x1; only eligible gas merges. The initial liquid validation is CRASHING_WAVE at 512², maximum 4, capped at 100 frames. The GPU engine remains independently available and tested; GPU AMR is not implemented. Web builds use the uniform CPU fallback until the extracted adaptive CPU path is explicitly enabled and validated there. The scene owns exactly one physical engine, not an unused full-resolution copy alongside AMR.

The scene starts at full resolution so initial material detail is assessed before any merge. At intervals of at least 0.32 simulated seconds, `FluidCoarseningPreview::proposeLayout()` turns the current error decisions into buffered fine patches. Requests are quantized to tiles of max(16, parent cell width) finest cells and clustered by the source-derived 2D Berger–Rigoutsos algorithm at 70% tag efficiency. Every requested cell remains covered. If fine coverage reaches 75%, a single fine grid avoids allocating more cells than uniform. Quiet regions merge after the existing 0.24-second dwell; new resolved detail requests refinement on the next assessment. The selected maximum is 2x2, 4x4, 8x8, 16x16 or 32x32, with adjacent 2:1 levels. The default remains 4x4; deeper levels are options, not a promised speedup. At each candidate width, selection assesses actual material/flow detail and its own quiet history. Each child patch is nested inside its parent with two parent cells of padding for ghost/interpolation stencils. If clustered coverage at a finer size reaches 75%, that size becomes the full-domain base and all coarser levels are dropped; 1x1 coverage at 75% selects uniform fine. There are no gravity, horizontal-cut or material-name assumptions.

A full coarse base remains stored beneath fine patches. The sidebar reports both active and allocated cells. **Adaptive grid** restarts the same scene when toggled, preserving other settings: on enables automatic fine/coarse selection; off runs the same native engine uniformly fine and skips adaptive assessment/decision overlays. **Maximum cell size** selects 2x2, 4x4, 8x8, 16x16 or 32x32 and restarts the scene. With adaptivity off, **Uniform coarse control** instead starts the entire scene at that size, conservatively averaging the same fine initial material. Actual outlines are blue (1x1), green (2x2), amber (4x4), magenta (8x8), cyan (16x16), and red (32x32). Both mode controls restart from the original state so previously discarded detail does not contaminate comparisons. They are disabled in web builds, where adaptivity is unavailable. Hiding the decision overlay alone never changes physics. The check interval and existing error tolerances remain editable, and restart preserves them. Timing includes remap/restoration in physics and reports selection separately. Cadence bounds assessment frequency, not the propagation speed of pressure; newly introduced sources cannot be predicted from old state.

The scene owns catalogue/initialization, timing, layout schedule, UI and rendering. `sceneProcess()` advances physics once per scene tick. `viewportProcess()` does not advance it. Restart preserves controls and rebuilds the current initial state. `AmrGasRuntime` configures eight OpenMP workers by default; an explicit `OMP_NUM_THREADS` overrides that default. Display and preview read non-owning patch views without creating another physical state; scene scheduling must keep those reads separate from physics mutations.

`FluidCoarseningPreview` reads actual committed cells. Native gas fine groups measure material loss to the same bounded linear plane used for transport, display and refinement interpolation. Velocity compares cell momentum/density and independent MAC faces with the actual bounded regrid reconstruction, rather than requiring small variation around a constant mean. Smooth reconstructible motion can merge; unresolved child residuals, neighbour curvature and distinct adjoining fine-face samples can still request refinement. Reflecting walls and the shared momentum limiter can flatten slopes and retain detail. Ingredient fractions, relative density and pressure-gradient detail remain independent criteria. These are local representation indicators, not bounds on accumulated transport error or an oracle for discarded detail. Uniform liquid previews retain their flat-mean indicator because their transport model is unchanged.

A swept feature envelope looks ahead by at least the check interval plus one scene frame, with an additional coarse-cell spatial buffer before clustering. AMR pressure is a correction impulse divided by the last physical substep; geometric restoration has no physical pressure sample. Physical gas walls and coarse/fine joins are assessed using available neighbours. Uniform liquid previews keep their protected liquid halo and unassessed outer edge.

Only current detail/refinement indicators seed that spatial buffer. A quiet fine cell waiting for its dwell retains its own tile without requesting padding around it. This distinction prevents a newly refined buffer from repeatedly growing new buffers as its history resets. Quiet dwell, feature prediction, clustering and thresholds are otherwise unchanged.

Green means settled merge candidate, yellow retain/wait, purple liquid/buffer, grey unavailable stencil, red resolved refinement detail and orange approaching detail. Transparent fills preserve the clouds; blue/green/amber/magenta/cyan/red outlines show the **actual** 1x1/2x2/4x4/8x8/16x16/32x32 cells. The fill is the last selection assessment, which can precede the displayed remap. Quiet history survives in unchanged regions; an exact shared byte map of actual sizes per 2x2 footprint resets only affected groups, without a packed-signature size limit. Changed tolerances, look-ahead, state replacement or rewind reset history. Observation owns labels/history/owner indices, never finer physical samples. In automatic gas mode it runs at the selection cadence even if hidden; in the uniform fallback it remains an optional read-only preview.

The uniform engines share the existing 3x3 MYC liquid reconstruction and split material/momentum transport. Exactly absent phases skip unnecessary work; traces are never thresholded away. Their public `State`, `Edit`, `upload`, `download`, `step`, `advect`, `applyGravity`, `project` and sampling APIs remain unchanged. AMR has its own explicit phase-mixture API; it is not a `FluidGrid` facade. Mixture membership/order must be configured before upload. Uniform state masses are mass per cell; AMR converts them once to internal mass densities. GPU operations still require their owning GL context.

CRASHING_WAVE uses scene-specific detail tolerances 0.4 velocity, 0.4 pressure-gradient, 0.04 relative density and 0.04 ingredient fraction. Other scenarios retain 0.1/0.1/0.02/0.02. The relaxed wave setting accepts modest cloud smoothing while preserving finest liquid; pressure tolerance and projection cadence are unchanged. Its 512²/100-frame comparison is a useful starting point, not a long-run accuracy guarantee or recovery of the older uniform baseline: [quality/performance sweep](.codex/runs/run0092_20261024_044518/findings.md).

## Editing material at runtime

`beginEdit()` creates a move-only, reusable session over the existing grid. Draw any shape by selecting individual cells; staging does not read or copy the live grid:

```cpp
auto edit = fluid.beginEdit();
edit.reserve(256);
for (int y = center_y - radius; y <= center_y + radius; ++y)
for (int x = center_x - radius; x <= center_x + radius; ++x)
{
    if ((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) <= radius * radius)
        edit.setCell(x, y, fluid::Phase::Liquid, water_element);
}
edit.commit(); // CPU; GPU commits must run on the owning render context.
```

The scene chooses positions, shapes and timing. Keep the chosen cells inside the grid. The grid must outlive its sessions, and `upload()` must have initialized the configured mixtures before a nonempty commit. Stage on one thread at a time and commit between physics/drawing operations; GPU commit needs the owning GL context. Staging, `reserve()`, `size()`, `empty()` and `clear()` need no GL context. Sessions provide staged writes, not reads of existing GPU cells.

`setCell(x, y, Phase::Liquid/Gas, element)` appends a 16-byte command on the supported native build. With reserved capacity it only appends that record; component lookup and mass calculation happen at commit. Each selected cell becomes entirely the chosen ingredient: its liquid fraction becomes one or zero, its selected component mass becomes phase density times cell volume, and all other liquid/gas component masses become zero. The element must already belong to the selected mixture; if it appears more than once there, the first matching component is selected. Elements present in both phases use the density of the selected phase.

The last queued write to a cell wins. Commit validates the whole batch before material changes: out-of-grid coordinates throw `std::out_of_range`, and a missing phase ingredient throws `std::invalid_argument`. Rejected commands remain queued for inspection/counting or `clear()`. Successful commit clears the commands and retains staging capacity for reuse. Empty commits do nothing; destroying or clearing a session discards uncommitted writes. A session must not be used concurrently with a mutation of its grid or mixture configuration.

Commit preserves current face velocities and pressure prediction history, so replacement material inherits the local flow and the existing divergence-free transport condition. The next physics operation rebuilds stale density/pressure data. This is an explicit material source/sink: replacing gas with water removes the gas mass in those cells; it does not compress or relocate that gas.

Commit sorts only when edits are out of raster order and resolves duplicates. CPU commit writes only the selected cells, using the existing thread pool for large batches once each cell has a unique final replacement. GPU commit packs contiguous row spans; equal adjacent spans merge into rectangles without covering unedited gaps. It uploads only the selected fractions and packed ingredient masses, reusing a staging buffer and preserving graphics state; there is no readback or full-grid upload. With the scene's four ingredients, 100 distinct edited cells transfer 2,000 bytes of texture payload. Widely scattered edits still require more upload calls than compact shapes.

## GPU component execution

- **Storage:** consecutive liquid ingredients followed by gas ingredients occupy lanes of RGBA32F textures, four per group. Each group has current mass, next mass and frozen correction textures: about 12 bytes per cell per ingredient, rounded to groups of four. These buffers stay on-device.
- **Shared transport:** one geometric liquid-volume flux serves every ingredient. Summed phase masses give the total mass flux for momentum; gas mass is stored separately from liquid mass to avoid subtracting nearly equal numbers at large density contrasts.
- **Fused updates:** X transport captures the component correction weights for both X/Y passes. When all ingredients fit one group, the same draw also writes the new phase density and liquid fraction. Larger mixtures use one transport draw per group, with density/colour sums processing up to sixteen ingredients per draw.
- **Reuse:** density remains valid until masses change. Pressure keeps its multigrid solver, device-side reductions and small convergence/CFL readbacks. Private shader programs retain unchanged scalar uniforms between draws.
- **Float geometry:** face intersections are bounded by the outgoing strip and available donor phase volumes. This prevents a rounded sliver from exporting more mass than exists; component masses are not clipped or renormalized afterwards.

The CPU implementation remains the readable algorithm reference. The GPU executes the same X/Y split transport and pressure model with float arithmetic; long chaotic trajectories need not match pixel-for-pixel.

The shared operations require no base class or virtual dispatch. Direct CPU vectors, `sampleVelocity`, `addVortex` and CPU working-data inspection remain CPU-specific learning tools. Pass an explicit divergence tolerance for comparisons: the engines retain their different precision-appropriate defaults.

Unused dye buffers, `advectDye`, particles and the unused `FluidCell` placeholder have been removed. `sampleVelocity` and `addVortex` remain useful experiment operations.

## AMR numerical contract

Preserve the coupled algorithm below when optimizing it. Chosen physical cell sizes and the multigrid hierarchy used to accelerate pressure are distinct; retain useful intermediate solver levels. [Current selection and observation rules](#current-cpu-experiment-and-uniform-engines) cover quiet-history padding, committed-state reads and scene ownership.

`AmrGas` uses project-owned regular 2D patches, cached boundary exchange, conservative restriction/refinement and the source-derived composite pressure solver below. The application builds and runs without AMReX; only the separate `reference/` project links it. Gas-only ingredient transport reconstructs a bounded linear plane and traces it back half a step in both directions before forming each ingredient flux. Momentum uses the summed mass flux and existing donor cell velocity; it remains first order so this step does not also replace bulk-flow advection. A common mixture limiter preserves summed ingredient volume. Refinement interpolation shares that limiter with momentum at differing intrinsic densities, preserving constant velocity; equal-density momentum has its own limiter, independent of passive labels. `FluidLinearReconstruction.h` defines the spatial plane; `AmrTransfer2D` applies it to the coarse/fine stencil. New fine averages and rendered coarse footprints use that same plane, without persistent fine samples. Two ghost layers provide the slopes of face donors; the slope field is derived transport scratch, not another physical state. The full pressure-coupled integrator is not claimed to be second order in time.

Fine face velocities remain independent. Fine ingredient and momentum fluxes are averaged onto the coarse face before divergence updates, so both levels exchange the same mass. On a single resolution, `stableTimeStep()` limits each donor's summed outgoing face velocities to a Courant number of .45, using that level's actual cell width. This avoids combining distant horizontal/vertical extrema and protects bounded linear donors up to twice their mean. Mixed-level hierarchies retain the previous `.45 / max_level((max_abs_u + max_abs_v) / dx)` bound: twice-reconstructed coarse/fine donors and opposing fine-face flows require separate treatment before using a local join bound. Pressure retains the upstream composite operator and averaged MAC correction to cell momentum. [Local-CFL validation and limits](.codex/runs/run0095_20261024_144010/findings.md).

`AmrPressure2D` owns both pressure paths. Differing ingredient densities use extracted GMRES+MLMG; equal-density mixtures use standalone MLMG with normalized BiCGStab at the bottom and upstream's smoothing fallback. The equation, warm pressure impulse, boundary conditions, 200 gas / 2,000 liquid iteration ceilings and composite face corrections are preserved. GMRES uses absolute L2 tolerance equal to the requested maximum-divergence tolerance, a stricter stopping target; standalone MLMG checks the maximum residual. Final actual active-cell divergence remains authoritative. The per-layout adapter copies current coefficients/RHS/warm pressure to retained solver arrays and corrected faces back; these costs remain inside projection/regrid timings. Production has no backend selector. The upstream engine is isolated in `reference/amrex/`, namespace `fluid::reference`.

Each `step(dt)` holds the previously projected face velocities fixed while CFL-subcycling conservative material/momentum transport. Ghost fill, flux synchronization and restriction still occur within each transport substep. The stability bound is computed once because the advecting faces stay fixed. After transport, apply damping over the full `dt`, rebuild the predictor with the full force impulse and updated density, and project once. The caller's `dt` therefore controls force/pressure temporal resolution (the wave uses .016 seconds); transport subdivisions alone do not resolve rapid force changes. Regrid restoration remains a separate required projection. This temporal split follows the old uniform engine's scheduling and does not claim identical trajectories to the previous every-substep projection. [Cadence validation and visual comparison](.codex/runs/run0093_20261024_135203/findings.md).

Projection measures actual predictor divergence first: if it already meets the **same requested tolerance**, pressure correction is zero and expensive solver preparation can be skipped. Otherwise the projected-minus-predicted face difference supplies the momentum correction without calculating the pressure flux twice. Momentum advection remains first order; the material reconstruction above does not establish high-order accuracy for the complete integrator. Clouds/vorticity can still diffuse. At loose scene tolerance (`1e-3`), small local volume-closure errors accumulate despite global ingredient conservation; tight tests and loose cost controls are reported separately.

Each level owns component-major patch arrays with two conserved ghost layers, avoiding a separate full-state copy. Cached `AmrBoundary2D` recipes read current same-level valid cells or interpolate current coarse state, with even material/tangential and odd normal-momentum reflection at walls. They write ghosts only. `AmrFaceSync2D` synchronizes duplicate faces, and cached `AmrTransfer2D` intersections average fine cells/faces conservatively. Regrid rebuilds these geometry plans; uploads/direct valid-state edits require no invalidation. The upstream comparison poisons old ghosts and checks refill, walls/corners, equal/unequal densities, multiple ingredients and layout changes.

`regrid()` transfers native conserved fields. Fine overlap is copied exactly, merging averages down, and new fine cells use bounded conservative interpolation from the real coarse state. No fine download/upload roundtrip occurs. After a changed layout, call `project(tolerance)` before transport. An identical normalized layout is a no-op. The engine supports arbitrary gas ingredients/densities, closed slip walls and no subcycling between levels. The gas-only constructor rejects liquid at upload; the two-phase overload accepts liquid and gas mixtures, storing their ingredient lanes in that order. Automatic 1x1/2x2/4x4/8x8/16x16/32x32 layout selection is implemented; automatic liquid coarsening and sizes beyond 32x32 still need their own agreed steps.

When liquid ingredients are configured, transport uses the uniform engine's MYC/PLIC geometry with directional splitting and frozen Weymouth-Yue ingredient/momentum corrections. `FluidInterfaceGeometry.h` shares the geometric formulas between both engines. Each phase's ingredients move with its geometric volume flux; momentum uses their combined mass flux. Fine fluxes are averaged down before each directional update. Corrections cancel for exactly divergence-free faces; with finite solver tolerance, record residual mass-budget error separately from near-roundoff volume closure. The smaller phase is reconstructed directly so a tiny gas sliver is not obtained by subtracting nearly equal liquid/total fluxes. No post-update mass clipping or normalization is used. The interface/phase volumes are derived from conserved ingredient masses; no fine physical copy is retained. Gas damping affects gas-only cells; pressure couples all materials. The AMR cell-momentum discretization remains distinct from the older uniform engine's staggered momentum, so matched AMR fine controls and older-engine visual controls answer different questions.
The [source map and numerical contract](reference/engine/README.md) identify the retained upstream algorithms and license. The normal application manifest/build has no AMReX dependency, include, linkage or initialization. `reference/CMakeLists.txt` separately builds pinned AMReX/Hydro and the frozen upstream engine. Native Windows uses LLVM OpenMP; the complete CPU library also builds without OpenMP or Bitloop through `reference/engine/CMakeLists.txt`. Web adaptive execution and GPU AMR remain unvalidated.

### CPU extraction validation

`FluidExtractionTests` in the optional reference project compares clustering, patch splitting, poisoned-ghost refill, regrid, transport and both pressure branches through six levels. On identical evolving inputs, the tested transfers/transport match exactly and projected fields agree within 3e-14 relative error. `FluidPressure2DTests` retains fourteen composite-operator/cycle/solve/flux oracle cases, including density contrast and GMRES restarts. Production `FluidRefinementTests` and `FluidCoarseningPreviewTests` use only our engine. `liquid-short` caps the mixed-ingredient selection fixture at 96 physical steps.

[Complete CPU extraction evidence](.codex/runs/run0100_20261024_cpu_extraction/findings.md): current cloud-free 512² CRASHING_WAVE, max32, two 60-frame pairs. Total including setup/copies/selection/regrid averages 20.73 -> 13.06 s (37% lower); transport 28% and complete physical-step projection 48% lower. Material and actual-grid captures inspected at 1/20/40/60; extracted liquid stays fine and surfaces are close. Layout histories and tiny droplets differ with accumulated roundoff, including between upstream repeats. This comparison establishes short-run extraction parity/cost, not a win over the older optimized uniform engine or a GPU gain.

### Scenario captures and validation

`scenario_index` selects cosine waves (0), crashing wave (1), Perlin gas (`LOCAL_COARSE_PATCH`, 2), or localized moving gas (`MOVING_CLOUD_DETAIL`, 3). Each scenario owns `capture_i0`, exclusive `capture_i1`, `frame_time` and capture stride. Headless `--snapshots` records both `grid` (transparent preview and actual cell outlines) and `material` (raw colour) through the interval. Gas also captures committed layout changes. Use the framework reconstruction tool for compact trace frames. Colour expansion is render output, not fine physics storage.

`FluidRefinementTests liquid` checks exact sharp-slab translation across a three-level hierarchy, preservation of sub-roundoff gas slivers without clipping, and evolving multiple liquid/gas ingredients with gravity at 1000–3000:1 density contrast. The subcycled-step fixture checks 16 physical steps with several transport subdivisions each on uniform/three-level layouts, full-step pressure interval, finite/near-positive material, conservation, volume closure and zero-time invariance. It also checks automatic nesting and finest liquid. These short checks do not establish long-run wave stability.

`FluidRefinementTests cfl` checks separated velocity peaks in divergence-free vortices, stillness, speed scaling, all four outgoing faces, internal FAB boundaries, and the preserved mixed-level bound despite cancelling fine-face velocities. Twenty-four transport updates at the computed limit check gas/liquid positivity, volume and ingredient conservation on single-resolution 1x1 and 4x4 grids. It also runs the evolving multi-substep liquid fixture. The full refinement suite includes these checks.

Pressure reference comparisons now live in the separate `FluidExtractionTests` and `FluidPressure2DTests` executables; the normal test build does not depend on AMReX. Historical GMRES integration evidence remains in [run0091](.codex/runs/run0091_20261024_042630/findings.md).

`FluidRefinementTests depth` checks 16/32 selection, active intermediate levels, whole-parent patch alignment, nesting, finest look-ahead requests, quiet release, direct refinement bridges, conservative evolution and history reset after a change in the first quadrant of a large group. The complete selection suite includes these checks. Trace tile dimensions are per level, matching its alignment.

`FluidRefinementTests` covers nested 1x1/2x2/4x4/8x8/16x16/32x32 selection and release, direct refinement from 8/16/32 to 1, conservative evolution through up to six levels, look-ahead from a fine feature into 4x4/8x8 regions, quiet merging, diagonal material protection, coarse-feature refinement, stationary layout settling, affine material reconstruction across FAB/level joins and expected reflecting-wall loss, aligned-edge stability, and 24-second automatic trajectories against a uniform control. Its optional `512` / `1024` argument measures the actual three-scale Perlin initial condition with full selection/remap cost; `reverse` swaps execution order and an optional third argument overrides ingredient tolerance for diagnosis (`512 forward 0.03`).

Mixed-layout regressions require uniform quiet fine patches to retain their dwell, then disappear without spreading, while real detail still gets a buffer. Representation tests compare the material indicator with actual restriction followed by bounded coarse-plane reconstruction at widths 2, 4, 8, 16 and 32 for constant, affine, smooth and alternating fields at equal and 10:1 densities. The indicator covers measured immediate loss, is exact for constant/alternating fixtures, and retains the expected affine loss at reflecting corners. Display and refinement must reconstruct the same material. These checks do not bound accumulated transport error.

`FluidRefinementTests velocity` compares the velocity estimate with actual adjacent-level restriction/refinement at widths 2, 4, 8, 16 and 32, including differing densities, limited affine/curved motion, checkerboard momentum and independent face fluctuations. A projected smooth-circulation control separately measures reconstructed velocity and coarse-face flux differences over one assessment horizon. Passing an instantaneous merge threshold does not promise that same error bound after evolution; retain long scene comparisons with uniform-fine and wholly-coarse controls. [Velocity validation and scene evidence](.codex/runs/run0087_20261024_014849/findings.md).

For selection diagnosis, pass an optional `FluidCoarseningPreview::Trace*` to both `update(AmrGas,...)` and the following `proposeLayout(...)`. It records each scale's actual error components, source label before look-ahead, raw/stabilized decisions, quiet age, requested footprints and tile masks after padding/nesting, plus the fallback base width (zero when none is needed). Observe before committing the proposal; a post-regrid estimate would answer a different question. No diagnostic storage or extra error estimates are needed with the default null pointer. The observation-parity regression checks identical decisions/history/layouts through evolution and remapping. [Reason-map investigation](.codex/runs/run0086_20261024_012844/findings.md) distinguishes full-tolerance failures, the stricter merge band, dwell and patch expansion.

The optional upstream-oracle `FluidAmrGasTests mode=check n_cell=64` retains the historical checks for analytic smooth-transport convergence, conservative remap endpoints/no-op/exact overlap, force equilibrium, CFL-limit conservative transport, and thirty-two 24-second evolving trajectories with moving three-level layouts, seven ingredients at a 100:1 density contrast, independent three-ingredient fractions and repeated remaps. Raw refinement also checks constant velocity across density changes and equal-density label independence. The horizon covers three initial domain-crossing times. `mode=bench layout=costs n_cell=1024 steps=625 world=256 tolerance=0.001 damping=0.20000000298023224` includes same-kernel uniform, sparse fixed/moving patches, and the old low-coarse-coverage controls. Pin eight workers/affinity consistently for paired measurements. Timings include complete stepping and regrid, but exclude initialization, observation and drawing; scene timings report those separately. See [CURRENT_WORK.md](CURRENT_WORK.md) for the active question and [WORK_STACK.md](WORK_STACK.md) for parent goals, accepted outcomes and remaining limits.

### Prescribed-pressure uniform boundary

CPU wave scenarios keep the tested `top_ambient` API with zero pressure and scene gas fractions. Inflow supplies that composition and zero donor velocity; outflow carries interior ingredients/momentum. Exchange counters track the open-system budget. Adaptive gas is closed and does not substitute atmosphere with a reservoir. These are explicit physical models.

## One physics substep

`FluidGrid::step()` accepts an optional fourth argument, `FluidGrid::StepTimings*`, reporting transport, forces and complete pressure-projection milliseconds. A null pointer performs no clock reads. Dev-mode scene traces include stage timings alongside complete physics time.

Read `FluidGrid::step()` first:

```cpp
advect(dt);
dampGasVelocity(dt); // CPU, optional; zero rate returns immediately
applyGravity(dt);
return project(dt, max_iterations, divergence_tolerance);
```

1. **Transport material and momentum:** reconstruct interfaces, calculate liquid-volume transfers, transport component masses and carry both momentum components through X, then Y. After each material pass, recover velocity by dividing momentum by the new face density. Repeat with smaller transport steps when needed to limit outflow.
2. **Apply forces:** the CPU optionally damps gas-only velocities, then gravity adds acceleration to vertical faces, including the optional open face. Damping before gravity preserves the hydrostatic background. Both materials receive the same gravitational acceleration; density affects their pressure response.
3. **Project pressure:** rebuild density coefficients from the current component masses, apply the selected closed/open boundaries, solve for pressure and correct velocity until cell divergence meets the requested tolerance.
Transport uses the previous substep's projected velocity. The scene starts with zero velocity, which already satisfies incompressibility. An experiment that directly changes velocities must project them before transport. The final projection handles the transported state and gravity, preparing the next substep.

`sceneProcess()` uses the selected scenario's `frame_time` and divides it into `ceil(frame_time * 60)` equal steps: three for the cosine scenario, five for the crashing wave. It calls `fluid.step(step, 2000, 1e-3)` for the uniform CPU wave path; AMR gas uses its own ordered step at the same substep interval. The CPU regression reference retains its double-precision `1e-8` checks; the scene uses the same explicit float-compatible target for comparisons. The engine advances exactly the time passed to it; it does not know about frames or recording. Older measurements below used different timesteps; compare timing only with matched timestep and simulated duration.

On failed final projection, `step()` returns an unconverged result after transport and gravity have already run. It does **not** roll back material, velocity or pressure. The scene pauses and reports the failure rather than continuing with that state. A zero timestep leaves state unchanged.

## Material and momentum transport map

| Function | Question it answers |
| --- | --- |
| `liquidArea` | How much liquid lies inside this local rectangle, given a dividing line? |
| `interfaceNormal` | Which way does the line face, estimated from neighbouring fractions? |
| `interfaceOffset` | Where must that line sit to enclose this cell's liquid fraction? |
| `rebuildInterfaces` | What are the current normal and offset for each mixed cell? |
| `liquidTransferX/Y` | How much liquid lies in the outgoing strip beside this face during this substep? Returns a signed fraction of cell volume, already accounting for time |
| `rebuildTransfersX/Y` | Cache liquid-volume transfers through each face for this pass |
| `componentMassTransfer` | How much of one ingredient's mass follows its phase's transferred volume? |
| `rebuildComponentTransfersX/Y` | Cache one ingredient's transfers and accumulate total mass transfers for momentum |
| `freezeComponentCorrections` | Select the phase and fix ingredient/momentum split corrections in small contiguous blocks before the X/Y pair |
| `advectComponents` | Update each ingredient using its transfers and the frozen correction |
| `rebuildDensity` | What cell density follows from total component mass divided by cell volume? |
| `rebuildMomentum` | What momentum density belongs to each face velocity? Uses the average density of the two neighbouring cells |
| `momentumTransfer` | How much momentum crosses a boundary? Mass transfer times donor velocity |
| `advectMomentumX/Y` | Carry both momentum components in this direction using the cached mass transfers and matching split corrections |
| `recoverVelocity` | Divide momentum by current face density and write `next_u/next_v` |
| `advect` | Coordinate material and momentum through successive X/Y passes and CFL substeps |

Local geometry uses a unit square per cell; no local-coordinate fields are stored. `liquid_fraction` tracks combined liquid geometry; `liquid_mass` and `gas_mass` track ingredient masses. Their occupied volumes stay consistent through the shared transfers. Interface arrays are rebuilt working data. `split_weights` are numerical correction flags fixed across both passes, not rounded material fractions. Pure gas/liquid cells bypass interface-area calculations. The shared geometric boundary is still needed even though several miscible ingredients can occupy either side.

During the entire `advect(dt)` call, `u/v` remain fixed for material transport and momentum corrections. `next_u/next_v` hold the evolving donor velocities; momentum is updated in place because no loop reads a neighbour's momentum. After each fraction swap, density and working velocity are rebuilt. The velocity buffers swap only after all transport substeps finish. Frozen correction weights are chosen separately for each X/Y pair; their combined contributions vanish for exactly divergence-free transport velocity.

The current method always processes X before Y and uses first-order donor velocities for momentum. It preserves sharper interfaces and couples mass/momentum transport, but numerical damping remains. It does not establish smooth droplets, realistic splashes or complete liquid behaviour.

In both engines, `interfaceNormal()` reads the eight surrounding fractions once. Three-cell column/row totals estimate the surface slope; a weighted Youngs gradient provides an alternative near corners and thin features. MYC selects the dominant height direction and compares the two slope estimates. `interfaceOffset()` still positions the chosen line to match the original liquid volume. This changes the reconstructed angle, without filtering fractions or adding a force. The GPU performs this work inside its existing mixed-cell reconstruction pass, with no additional textures or passes. For the underlying method, see [Basilisk's 2D MYC implementation](https://basilisk.fr/src/myc2d.h).

The shallow-line regression checks both coordinate axes, both travel directions, opposite slopes and both phase orientations. Its worst cell-fraction error after translating five cells fell from `0.122386` to `2.18e-11`. At frame 60 of the actual 512x512 CPU scene, the same broad-wave detrending over columns 80–300 measured ripple RMS falling from `0.494` to `0.000837` cells. This is a local surface-quality measurement, not a general claim of exact fluid dynamics.

On the Ryzen 9 9950X3D with eight workers, median startup physics time for 60 output frames (360 steps) changed from 21.64 s to 22.58 s, about 4% slower, with some run-to-run variation. Rebuilding interfaces on the identical saved frame-60 fraction field changed from 261 to 347 microseconds per call. The CPU regression suite passes, including its 20-second pool/component cases; a separate 128x128 run passed 40 simulated seconds with the scene's `1e-3` divergence target. Retained measurements and snapshots: [CPU MYC experiment](.codex/runs/run0023_20261019_022518/validation.json).

The GPU port passes the same forty shallow-line cases with a maximum cell-fraction error of `3.58e-7`, and restores all existing CPU/GPU comparisons without loosening their tolerances. Its native-grid frame-60 capture shows the same smooth surface. On the RX 7700 XT, three 512x512 four-ingredient startup runs measured median physics time changing from 2.534 to 2.547 seconds (about 0.5%, within run-to-run variation). A separate 512x512, cell-size-4 mixed-density pool passed 40 simulated seconds with checks after every step: maximum divergence `0.000999451`, relative ingredient-mass drift `3.20e-7`, and local phase-volume mismatch `0.000660` cells, below the existing `0.002` float budget. These are startup throughput and bounded-duration stability results. Retained evidence: [GPU MYC validation](.codex/runs/run0024_20261019_024053/validation.json).

The Stage 1 check covered 20.05 simulated seconds on the current 128x128 scene at its experimental 100,000:1 density ratio, with every physics substep checked. Relative water-volume drift was below `6e-13`; fraction bounds differed from `[0,1]` by less than `1e-14`; all projections met `1e-8` divergence. At 10 seconds, cells containing 1–99% water fell from 5,065 in the earlier run to 282. This validates this checkpoint, not arbitrary future configurations. Local evidence: [run0011 validation](.codex/runs/run0011_20261016_172158/validation.json).

The Stage 2 check passed all seven regression cases and advanced the current 128x128 sloped scene for 40 simulated seconds (4,800 physics steps) at 1000:1 density contrast. Every projection met `1e-8` divergence, relative water-volume drift stayed below `5.4e-12`, and fraction-bound errors stayed below `1e-14`; all wall velocities remained zero. Discrete kinetic plus gravitational energy peaked 0.082% above its initial value during the initial transient and ended at 49.9% of that value; this first-order method is not an exact energy-conservation scheme. Local evidence: [run0012 validation](.codex/runs/run0012_20261017_141309/validation.json).

## What the existing optimisations do

These preserve the current pressure equation and the outer PCG algorithm. Multigrid changes the preconditioner; the other optimisations change the amount of work or its arrangement. Floating-point rounding and stopping iterations can affect the trajectory within the requested accuracy.

| Optimisation | Plain meaning | Where to look |
| --- | --- | --- |
| Reused vectors | Keep allocated working memory between solves | `project` and private buffers |
| Explicit cache invalidation | Reuse density and pressure coefficients until component masses change; direct CPU edits call `invalidateCaches()` | `upload`, `advectComponents`, `rebuildDensity` |
| Momentum/donor initialization | Copy initial donor velocity while building momentum, avoiding a density rebuild and multiply/divide round trip | `rebuildMomentum` |
| Pressure prediction | Use the last two pressures to guess the next one; rebuild the residual against the current system | Start of `solvePressure` |
| Lazy multigrid setup | Rebuild a stale hierarchy only after prediction leaves a residual that needs solving | `solvePressure` |
| Cached inverse diagonal | Compute each cell's diagonal reciprocal when density changes rather than divide every CG iteration | `rebuildPressureCoefficients` |
| Multigrid V-cycle | Smooth local errors, solve broad errors on smaller grids, interpolate corrections back and smooth again | `PressureMultigrid::cycle` |
| Density-aware coarse weights | Combine pressure conductances in series and parallel instead of averaging away density barriers | `PressureMultigrid::coarsen` |
| Reused coarse factorization | Factor the smallest system when rebuilding a stale hierarchy; reuse it in every V-cycle | `factorCoarsest`, `solveCoarsest` |
| Zero halo | Zero wall weights and a padded border remove boundary branches from multigrid stencils | `PressureMultigrid::Level` |
| Interior/boundary split | Interior cells always have four neighbours; process them without boundary checks, then handle walls separately | `applyPressureOperator` |
| Row pointers | Locate each row once, then walk along it instead of reconstructing flattened indices | Interior operator loop |
| Fused updates and statistics | Update pressure/residual and measure the residual in the same pass | `updatePressureAndResidual` |
| Fused stencil/dot product | Accumulate CG's denominator while applying the pressure operator | `applyPressureOperator` |
| Fused RHS/true-residual statistics | Accumulate compatibility or convergence statistics while forming those fields | `project`, `rebuildPressureResidual` |
| Fused preconditioner reduction | Remove the correction's mean while accumulating the dot product CG needs | `applyPreconditioner` |
| Four accumulators | Let the CPU work on several independent sums instead of waiting on one running sum | Residual/preconditioner loops |
| Size-aware pressure parallelism | Keep small/coarse grids serial; batch pressure updates, reductions, setup and copies as well as stencils; merge reductions in batch order | `forBatches`, `reduceBatches`, multigrid `forRows` |
| Row-based transport | Traverse complete rows to avoid per-face coordinate division; workers read fixed inputs and write disjoint entries | Transport and interface loops |
| Batched component corrections | Choose phase and its reciprocal once per cell; traverse ingredient arrays in small blocks within one worker dispatch | `freezeComponentCorrections` |
| Equal frame substeps | Avoid tiny remainder timesteps that make pressure prediction less useful | `sceneProcess` |

`applyPressureOperator` defines the equation; `applyPreconditioner` helps CG solve it faster. Mean removal before and after preconditioning preserves symmetry on mean-zero pressures. Both engines expose `PressurePreconditioner::{Multigrid, Diagonal}` and a `pressure_preconditioner` member. Set `grid.pressure_preconditioner = decltype(grid)::PressurePreconditioner::Diagonal` to compare with the original method without changing the scene's backend declaration.

Read `PressureMultigrid::cycle()` for the sequence. Each level groups roughly 2x2 cells until at most 64 remain. Four red/black Gauss-Seidel sweeps reduce local error; `restrictResidual` passes the remaining error down. The smallest system fixes one pressure, solves by Cholesky and removes the arbitrary constant. `prolongCorrection` interpolates the correction back, followed by four sweeps in the reverse colour order. Restriction is exactly the transpose of prolongation. These choices make each application a fixed linear symmetric positive preconditioner, as ordinary PCG requires. Coarse conductances approximate the fine system; the fine-grid equation and its final convergence checks remain authoritative.

The hierarchy's dimensions, interpolation maps and buffers are reused. Density-dependent conductances and the small factorization rebuild when masses have changed and a solve actually needs the preconditioner. Independent cell/row updates use `detail::updateWorkers`: below 32,768 work items stay serial, medium batches get roughly 8,192 items per worker, and larger batches retain the eight-worker cap, bounded by the shared pool. This prevents a 32K block from losing all parallelism when a uniform grid is split. Reductions retain their original 65,536 threshold and batch order; changing their grouping can change floating-point rounding and stopping iterations. Multigrid retains its original four workers on medium levels and eight above about one million cells, with small/coarse levels serial. Red/black smoothing retains a barrier between colours and four symmetric pre/post sweeps. The fine pressure equation, true-residual verification and final divergence checks are unchanged.

True-residual verification and the final divergence check are correctness checks, not optional performance overhead. A small internal CG residual alone does not prove that the final velocity meets the requested divergence tolerance.

The CPU execution pass was measured on the Ryzen 9 9950X3D, MSVC Release, with a fixed eight-worker pool. Three sequential, alternating before/after runs used identical sloped pools, `dt=0.004`, density ratio 1000:1 and divergence tolerance `1e-3`. Times include the complete physics step and exclude initialization, rendering and state inspection. The 128/256/512 grids cover the first simulated second (250 steps); full HD covers 0.2 second (50 steps).

| Grid | Ingredients | Before median | After median | Less CPU time |
| --- | --- | --- | --- | --- |
| 128x128 | 4 | 0.666 s | 0.598 s | 10.2% |
| 256x256 | 4 | 1.865 s | 1.627 s | 12.8% |
| 512x512 | 4 | 5.788 s | 4.607 s | 20.4% |
| 512x512 | 18 | 9.640 s | 7.392 s | 23.3% |
| 1920x1080 | 4 | 10.553 s | 8.574 s | 18.8% |

Pressure iteration totals matched before and after in every case. These are startup measurements, not developed-wave or rendered frame rates. The correction weights use small contiguous ingredient blocks; medium transport batches retain eight workers because reducing them to four regressed the eighteen-ingredient case. No component limit, phase model, interface algorithm or accuracy target changed. Retained measurements: [CPU execution validation](.codex/runs/run0025_20261019_131749/validation.json).

**Current CPU stability limitation:** the additional 512x512, cell-size-1 run did not pass its planned 40-second validation. Both the original and optimized engines develop component masses inconsistent with phase geometry at the 12.82-second inspection. The original then fails projection at 12.972 seconds; the optimized run still meets divergence through 20 seconds, but its worst local phase-volume disagreement reaches 1.27 cell volumes, so that state is not physically valid. The run was stopped. Small-grid 20-second regressions still pass at their existing tighter tolerance. The subsequent diagnosis traced this pre-optimization defect to roundoff allowing a nearly full liquid cell to export more gas than it contains. CPU X/Y liquid intersections now use the GPU's donor-phase and outgoing-strip volume bounds; the total volume uses the same arithmetic as component transport. Component masses and pressure tolerances are unchanged. Tests and the long-run replay were not rerun for this correction at the user's request, so sustained stability remains unverified. Diagnosis: [CPU transfer failure](.codex/runs/run0026_20261019_144524/validation.json).


On the Ryzen 9 9950X3D, matched-input sloped pressure benchmarks at `1e-3`, including coefficient/hierarchy rebuild, fell from 683 ms to 36 ms at 512x512 and from 28.68 s to 0.524 s at 1920x1080. These compare the original serial diagonal solver with the tuned multigrid solver; they are projection timings, not complete frames. At the previous wave-stopping time (14.77 s), copies of the same preprojection state required 2,062 diagonal iterations versus 22 multigrid iterations, taking 1.12 s versus 70 ms. The comparison allowed 8,000 iterations so the diagonal reference could finish; the scene retains its 2,000 limit.

The 512x512 sloped scene then passed 40 simulated seconds / 4,800 committed substeps with the existing `1e-3` tolerance and 2,000 limit: maximum 33 iterations, maximum actual divergence `0.00099967`, relative water-volume drift below `1.6e-7`, finite fields and closed walls. Fraction-bound error stayed below `1.3e-16`; energy peaked 0.038% above its initial value and ended at 68.7%. Volume error plateaued and repeated wave travel continued between 20 and 40 seconds. This instrumented run took about 351 seconds for 800 output frames (~2.3 frames/s); isolated pressure speedups do not make the whole engine scale identically. The 120-step single-thread and parallel runs produced identical authoritative-state fingerprints. Local evidence and native grid captures: [run0016 validation](.codex/runs/run0016_20261017_165714/validation.json).

## GPU implementation

`FluidSimEngineGPU` is a separate local library. CPU physics remains the reference for learning and experiments. The GPU port was validated on the 128x128 sloped pool, cell size 4, densities 0.1/100 and gravity 100. Both air and water are simulated; no physical model or boundary condition is selected automatically.

Read `FluidGridGPU::step()` and then the similarly named `Impl` methods. Each numeric array is a nearest-filtered float texture; fragment passes write new textures and swap them with the old inputs. A shader never reads the texture it is currently writing. The small private pass runner handles numeric textures; the framework's `ShaderSurface` is still used only to colour the result for display.

| GPU detail | Purpose |
| --- | --- |
| `rebuildInterfaces`, `rebuildTransfersX/Y`, `advectMomentumX/Y`, `recoverVelocity` | Preserve the CPU's geometric material and consistent momentum transport, with both components carried in each direction |
| Scaled normals and factored triangle areas | Evaluate the same interface geometry without tiny float products becoming `0/0` |
| Packed pressure coefficients | Build left/right/up/down weights together in one RGBA texture pass; use them for both the pressure operator and velocity correction |
| `PressureMultigridGPU::cycle` | Smooth local errors, restrict the residual, solve on smaller grids and prolong the correction; diagonal preconditioning remains selectable |
| Fused red/black sweeps | On levels up to 65,536 cells, recompute the first colour's new neighbours while updating the second colour, replacing two draws with one |
| Polynomial smoothing | On larger levels, fixed Chebyshev-weighted Jacobi uses fewer texture samples; reverse the same weights after the coarse correction to preserve symmetry |
| Tiny GPU Cholesky | Factor at most 15 unknowns once per projection and reuse that factor in every V-cycle, without copying it to the CPU |
| CPU coarse pressure | Replace small GPU multigrid passes with a CPU correction below the configured cutoff; reuse its workspace and per-projection factorization |
| 4x4 reduction passes | Combine cell values into one GPU texel for sums, dot products and maximum divergence, including odd/thin grids |
| GPU PCG coefficients | Keep dot products, alpha, beta and the active iteration count on-device; check short solves directly and retain predicted batches for longer solves |
| Pressure corrections | Apply the predicted pressure first, solve for the remaining divergence from zero, then apply the correction separately so small updates survive float rounding |
| Bounded refinement | If the measured final divergence is still too large, solve its remaining error again within the same total iteration budget; only the actual velocity check establishes convergence |
| Shared velocity statistics | Measure actual divergence and the next CFL outflow limit together, starting directly at the first reduction level; reuse outflow until velocity changes |
| Momentum/velocity MRT | Initialize momentum and donor velocity together, saving the initial density rebuild and two recovery draws |
| Texture-owned framebuffers | Keep a framebuffer attached to each single-output texture across swaps, avoiding repeated attachment changes |
| Native scalar downloads | Explicit inspections read RED/FLOAT directly; WebGL2 retains the portable RGBA/FLOAT path |
| Face-wise split cancellation | Pair each phase/component flux with its frozen split correction at the same face, reducing float cancellation |

Read `PressureMultigridGPU::cycle()` alongside the CPU version. Both use density-aware series/parallel coarse conductances, bilinear interpolation and its exact transpose for restriction, with mean removal before and after preconditioning. GPU levels use five fixed pre/post sweeps rather than four; large levels target the upper spectrum of the normalized operator with a fixed polynomial. By default, the hierarchy hands levels of at most 1,024 cells to its owned CPU multigrid solver. With `cpu_pressure_cells = 0`, GPU coarsening continues until at most 16 cells remain for the fragment-local direct solve, rather than the CPU's 64. These are execution/preconditioning choices, not a change to the fine pressure equation or physical model. The maps and buffers persist; coefficients and the coarse factor rebuild once per projection. The GPU result is checked against actual velocity divergence.

CG updates freeze once their residual reaches the internal target, so submitting a predicted batch does not advance the solution past convergence. Only active updates count towards `ProjectionResult::iterations` and the total refinement budget. A negative device-side dot product marks solver breakdown. Short GPU-only solves (at most eight iterations in the previous solve) now check each iteration and skip the unused final V-cycle; longer solves retain predicted batches. Readback counts therefore depend on the solve and refinements. No full-grid readback occurs in normal stepping or drawing.

Construction itself does not allocate GL resources. `upload(State)` allocates and initializes the grid on its owning GL context; empty input arrays mean zero. `download()` explicitly copies fractions, component masses, face velocities and pressure back for inspection. It is not part of normal stepping or drawing. Upload initializes pressure history rather than restoring a complete checkpoint. Grid dimensions are fixed after upload.

When GPU is selected, the scene calls initialization and physics through `renderThreadQueue().invokeBlocking()` and destroys resources through that queue before scene deletion. The display adapter uses `colorTexture()` for catalogue colours in both the display shader and the existing `scene-0:grid` snapshot route. `liquidFractionTexture()` remains available for inspecting geometry. No full-grid CPU copy/upload occurs each GPU frame. The CPU still controls substeps and PCG batches and reads small reduction results.

Native code targets Bitloop's OpenGL 3.2 context; the shader variant for WebGL2 requires `EXT_color_buffer_float`. Native headless runs must include `--snapshots` to create a graphics context, even when no images are requested. The GPU harness creates its own hidden context. Browser execution has not yet been validated.

The component port passed both regression suites, CPU/GPU comparisons through eighteen ingredients, and 40 simulated seconds of mixed-density pools: four ingredients at 128x128 and 512x512, plus eight at 128x128. The actual 256x256 Water/Red/Air/Cloud scene completed 800 frames with eight native-grid snapshots. Changing only the scene's engine declaration compiled and ran on both backends.

In the 512x512 component run, each ingredient's global mass drift stayed below `1.23e-7` relative, and actual divergence stayed below `0.001`. Float accumulation produced a worst local disagreement between mass-derived phase volume and geometric fraction of `0.001265` of one cell (about 0.13%). This is a measured precision limit, not a loss of 0.13% of the total material. No mass clipping or normalization hides it; arbitrary-duration accuracy is not established.

Current component startup timings on the RX 7700 XT, MSVC Release, are medians of three runs. Each output frame contains six physics steps; setup, rendering and full-state inspections are excluded:

| Grid / startup horizon | Ingredients | Output frames/s |
| --- | --- | --- |
| 512x512 / first 3 simulated seconds | 2 | 24.56 |
| 512x512 / first 3 simulated seconds | 4 | 23.57 |
| 512x512 / first 3 simulated seconds | 8 | 23.09 |
| 512x512 / first 3 simulated seconds | 16 | 25.15 |
| 1920x1080 / first simulated second | 4 | 10.59 |
| 1920x1080 / first simulated second | 16 | 9.71 |

Small differences at 512x512 should not be read as more ingredients making the solver faster. Pressure work dominates these cases. Developed waves cost more: the instrumented 512x512 four-ingredient run spent 130.32 seconds in physics for 800 output frames (about 6.14 frames/s), with diagnostic downloads between steps. Full-HD developed-wave throughput has not been measured. Retained commands, accuracy results, rejected optimizations and captures: [component-port validation](.codex/runs/run0021_20261019_003948/validation.json).

The following timings and health results are historical two-material baselines, before component masses were added.

The original diagonal GPU port passed a 128x128, 40-second run on the RX 7700 XT at `1e-3`: maximum divergence `0.000993`, relative water-volume drift below `3.27e-5` (0.00327%), finite fields and zero normal wall velocities. Energy peaked about 0.082% above its initial value and ended at 49.65%, close to the CPU reference's 49.91%; individual droplets need not follow identical float/double trajectories. Full-step CPU comparisons and the retained logs are under [run0013](.codex/runs/run0013_20261017_151739/validation.json).

That first port was not a speedup at 128x128: a separate 360-step timing measured GPU `1e-3` at 6.23 output frames/s versus CPU `1e-3` at 19.67, excluding setup/rendering/full-state readbacks. The multigrid/pass optimizations below supersede those historical GPU timings; the float tolerance change must not be counted as a backend speedup.

Matched-input native GPU projection benchmarks, including coefficient/hierarchy rebuild, now measure:

| Sloped mixed-fluid grid | Original diagonal GPU | Multigrid GPU |
| --- | --- | --- |
| 128x128 | 49.68 ms | 7.50 ms |
| 512x512 | 284.29 ms | 14.26 ms |
| 1920x1080 | 3,404.59 ms | 39.38 ms |

These are medians of three projections at the same `1e-3` target on the RX 7700 XT. At 512x512, the sloped case fell from 38,341 draws / 243 tiny readbacks to 1,840 / 7. A saved preprojection wave state near the original stopping time required 2,376 submitted diagonal iterations versus 27 active multigrid iterations, taking 433 ms versus 54 ms; both uploads reset pressure history identically, and the comparison allowed 8,000 iterations so the diagonal reference could finish. The scene retains 2,000. Original counts round up to batches of eight; new counts report active updates.

Full-step startup timings improved from 8.12 to 36.64 output frames/s at 128x128 (first three simulated seconds), and from 2.31 to 28.46 at 512x512 (first second). Each output frame contains six physics steps. Startup is cheaper than a developed wave; these figures exclude initialization, rendering and full-state inspection. The larger-grid projection timing does not establish full-HD wave throughput or long-run stability.

The final 512x512 GPU run passed 40 simulated seconds / 4,800 committed substeps with unchanged physics, `1e-3` tolerance and 2,000 iteration limit. Maximum active iterations were 40, maximum actual divergence `0.00099945`, relative water-volume drift below `7.5e-8` (0.0000075%), and fraction-bound error below `1.2e-7`; all fields remained finite and walls closed. Energy peaked 0.041% above its initial value and ended at 68.4%. Instrumented physics took 93.58 seconds for 800 output frames (~8.55 frames/s); a separate run without full-state inspection took 89.15 seconds (~8.97 frames/s). Repeated wave travel continued through the 20-to-40-second confirmation window. A separate 128x128 40-second health run also passed, averaging 27.35 output frames/s with volume drift below `8.1e-8`.

Observer-on/off replay was bit-identical through 1,770 ticks / 14.75 simulated seconds. The developed-wave 40-second fingerprints differed; long-run bitwise reproducibility is not established. The full-state health checks, final divergence and bounded physical statistics are the acceptance criteria for these longer float trajectories.

The first long validation caught float cancellation pushing a full cell above `1.00002` after 25 seconds. Pairing the existing split correction with its water flux at each face fixed the cause without clipping fractions or relaxing limits. An exact pure-water circulation regression protects this rearrangement. Retained timing, health and native-grid evidence: [run0017 validation](.codex/runs/run0017_20261017_172952/validation.json).

## GPU coarse-pressure optimization

`FluidGridGPU` retains material transport, forces, the fine pressure equation, CG vector updates and rendering on the GPU. Its private `PressureMultigridGPU` owns a small CPU `PressureMultigrid` workspace for a coarse grid covering the **whole domain**; its correction is returned to the GPU hierarchy. Both gas and liquid remain simulated. CPU/GPU stages alternate; there is no spatial partition or load balancer.

Select `FluidGridGPU fluid;` in the scene and link `FluidSimEngineGPU`. Coarse execution is an internal GPU-engine optimization, with no separate fluid engine or display adapter. Construction requires no GL context; upload, physics, edits, display and destruction follow the GPU engine's owning-context rule. Its `Scalar`/`State` remain float.

Set `FluidGridGPU::cpu_pressure_cells` before `upload()` to choose the first coarse level with at most that many cells. The default is 1,024; the GPU hierarchy bottoms out at 16 cells even for smaller positive cutoffs. Zero keeps the entire hierarchy on the GPU. The setting takes effect on upload, including reupload after a mode change. This is a fixed hierarchy cutoff, not a target percentage of execution time. `FluidGrid` and CPU multigrid mathematics are unchanged.

The fine-grid pressure operator, geometric transport and final actual-divergence check remain authoritative. GPU smoothing, restriction and interpolation continue above the cutoff; the CPU applies its fixed symmetric V-cycle below it. Coarse coefficients are copied and factored lazily once a projection needs a correction, then reused throughout that solve. Different preconditioning and float/double rounding can change iterations and chaotic trajectories within the requested accuracy.

The last GPU restriction pass packs the current global residual maximum/failure flag with the coarse RHS. This lets the CPU handoff detect convergence without separate per-iteration scalar round trips, stop the V-cycle's unused return passes and preserve the active-iteration budget. With a 32x32 cutoff, ordinary cycles transfer 16 KiB of packed RHS/statistics and 4 KiB of correction; a stale matrix additionally reads 16 KiB of coefficients once per projection. The normal path never downloads the full fluid state. Native scalar reads use RED/FLOAT; WebGL retains RGBA/FLOAT but has not been run.

The existing coarse-pressure checks now belong to `FluidSimGPUTests`; there is no separate hybrid test target. Projection, iteration caps, mode changes, edits, display, graphics-state preservation and multigrid algebra exercise CPU-coarse and GPU-only execution.

Retained experiment commands are `FluidSimGPUTests --coarse-compare 512 1200`, `--coarse-benchmark 512 0` and `--coarse-pool-steps 4800 512`. Comparison uses an identical saved 10-second GPU-only pool state, five alternating pairs, five warmup steps and 48 timed steps at `dt=0.004`, `1e-3` divergence, with GPU completion included. Setup, shared history replay, rendering and explicit state inspection are excluded. A history of zero measures startup. The pool command checks 40 simulated seconds at `dt=1/120`, using the existing budgets of `1e-3` actual divergence, `1e-4` relative ingredient-mass drift and `0.002` cell phase-volume disagreement.

Before the ownership cleanup, five alternating pairs on the Ryzen 9 9950X3D / RX 7700 XT, MSVC Release, measured these full-step medians for the four-ingredient test pool (`cell_size=4`, `dt=0.004`, tolerance `1e-3`). The GPU-only reference includes the short-solve scheduling improvement; these are the **additional** gains from CPU coarse execution:

| Grid / initial state | GPU-only ms/step | CPU coarse ms/step | More steps/second |
| --- | ---: | ---: | ---: |
| 512x512 startup | 4.717 | 4.037 | 16.8% |
| 512x512 after 10 simulated seconds | 10.403 | 8.564 | 21.5% |
| 1024x1024 startup | 7.180 | 6.837 | 5.0% |
| 1024x1024 after 10 simulated seconds | 16.357 | 13.940 | 17.3% |

These are historical synthetic-pool timings; the user's scene measured about 89 seconds GPU-only versus 86.5 seconds with CPU coarse execution, roughly 2.9% higher throughput. Before cleanup, the 512-square CPU-coarse pool passed 4,800 steps / 40 seconds with maximum actual divergence `0.00025177`, relative ingredient-mass drift `5.60e-8`, worst phase-volume disagreement `2.05e-5` cells and closed walls. The former CPU/GPU/hybrid suites passed then. The cleanup preserves the pressure algorithms and cutoff; runtime tests, behavior horizons and benchmarks are not rerun at the user's request. Retained timing ranges, rejected schedules and behavior evidence: [coarse-pressure experiment validation](.codex/runs/run0038_20261020_140000/validation.json).

## Regression baseline

Enable `FLUIDSIM_BUILD_TESTS=ON` when configuring, build `FluidSimTests`, then run:

```text
ctest --test-dir <build-directory> -R "^FluidGridTests$" --output-on-failure
```

On Windows, configure/build from a Visual Studio developer shell. The test executable needs the same vcpkg runtime DLL directory on `PATH` as the application (`<build-directory>/vcpkg_installed/x64-windows/bin`).

With the same test option enabled, build `FluidSimGPUTests` and run both suites with `ctest --test-dir <build-directory> -R "^FluidGrid(GPU)?Tests$" --output-on-failure`. GPU cases cover hydrostatics, odd/thin projection, CPU comparisons, all four diagonal directions, nonuniform momentum transport, tiny interface remnants, sharp slabs, exact pure-material preservation, zero-time/failure handling and graphics-state restoration. GPU multigrid algebra tests cover odd/thin/high-contrast grids through 513x257, including the larger-grid polynomial smoother. Additional cases protect 512x512 default-budget convergence, the one-iteration cap and exact cached-versus-refreshed CFL parity. Common-API tests exercise upload/download, zero/reset behavior and actual display pixels for both types, including an empty catalogue. Component comparisons cover liquid-only, gas-only, 4, 7 and 18 ingredients, empty fields, differing densities, an element in both phases, and populated ingredients beyond the first sixteen. A 512x512 regression exercises the formerly failing vanishing-sliver case. Both engines' legacy fixtures explicitly populate component masses through `tests/FluidTestSetup.h`.

Edit-session regressions cover staged/cancelled/moved/reused sessions, empty commits, initialization requirements, arbitrary sparse masks, duplicate-write ordering, row-boundary gaps, liquid/gas replacement through nineteen ingredients, invalid-batch atomicity, cache refresh against a fresh upload, edited display colours and GL state restoration. A 257x129 full-grid edit also covers threaded CPU commits and merged GPU uploads. A two-second, 240-step source/sink case on each backend commits twelve batches during pool motion; every step checks projection, source-adjusted component mass, fraction bounds and mass-derived phase volumes. Existing twenty-second CPU pool/component regressions continue to cover the unchanged transport/pressure algorithms.

`FluidSimGPUTests --edit-benchmark` measures reserved staging and GPU-complete commits for 1, 10 and 100 separate 6x6 drops on the 1024x1024 four-ingredient grid, plus a full-state download/upload reference. These are isolated edit costs with explicit GPU synchronization, excluding physics and drawing.

`FluidSimGPUTests --pool-steps 4800 1e-3 512 4` runs a 512x512, four-ingredient mixed-density pool for 40 simulated seconds. It downloads each committed step to check finite fields, fractions, each ingredient's mass, mass-derived phase volumes, closed walls, actual divergence and energy. The last two arguments default to 128 and 2. Short comparisons allow `5e-6` local phase-volume error; long float runs have an explicit quality budget of `0.002` of one cell (0.2%), while global ingredient-mass drift must remain below `1e-4`. This separates accumulated local roundoff from global conservation; it does not normalize masses to hide disagreement.

`--component-benchmark 120 1920 1080 16` times one simulated second with sixteen ingredients sharing the original phase densities. It excludes initialization, rendering and full-state downloads and counts six physics steps per output frame. `--benchmark-steps 360` retains the small CPU/GPU comparison, reporting CPU `1e-8` and matched `1e-3` tolerances separately. Startup timings are not developed-wave performance.

| Implemented case | What it protects |
| --- | --- |
| Affine velocity sampling | Staggered offsets, interpolation, world origin and cell size |
| Filled tank under gravity | Hydrostatic pressure balances gravity; walls stay closed |
| A transported material patch | Conserved volume, expected centroid motion and CFL subcycling |
| A translated straight material slab | Whole-cell translation preserves its sharp interfaces and expected cell values |
| A translated shallow straight interface | CPU and GPU MYC preserve analytic cell fractions across X/Y motion, positive/negative slopes and reversed phases, within their precision tolerances |
| Diagonal motion across a density contrast | Both velocity components remain uniform around a 1000:1 material patch, for all four diagonal directions and multiple transport substeps |
| Sloshing closed pool for 20 simulated seconds | Repeated convergence, fraction bounds, conserved volume, finite velocities and closed walls at 1000:1 density ratio |
| Miscible components for 20 simulated seconds | Three populated liquid ingredients, two gas ingredients, differing densities and one element present in both phases; conserved component masses, agreement of mass-derived volumes with phase geometry, and an absent component staying absent |
| Ingredient labels preserve bulk motion | Two ingredients versus nine sharing the same phase densities; spatially varying labels, absent ingredients, open/closed exchange, aggregate mass and velocity remain consistent over 80 steps |
| Pure phases accept new material | Liquid/gas reactivation after a pure-phase step, transported traces down to 1e-14, and gas inflow/liquid outflow budgets from an initially full-liquid open tank |
| Multigrid algebra and projection agreement | Linear, symmetric, positive, history-independent correction; constant nullspace; odd/thin and parallel grids; agreement with diagonal PCG at `1e-8` |
| Zero timestep / failed projection | No zero-time advance; failed final projection is reported so the caller can stop |

These checks do not establish realistic splashes or bubble physics; the slab case protects sharpness in a simple prescribed flow. Native grid captures verify the current experiment's interface quality, with retained evidence under `.codex/runs`. Earlier extraction comparisons preserve the cleanup baseline; they do not require the new transport method to reproduce the old, diffusive trajectory.

## Small future milestones

Choose **one** with the user; the list is not a commitment to implement everything. Each new feature should add one clear experiment or focused regression before another feature is added.

| Possible milestone | Small demonstration / acceptance case |
| --- | --- |
| General 2D acceleration | Uniform gravity reproduces today's result; a smooth custom field moves fluid in-plane. Mandelbrot depth can generate that field; it adds no water-depth variable |
| Colour labels | Two colours travel with the same fluid properties. Colour alone does not require a new physical element or pressure solve |
| Material-dependent surface tension | A droplet tends towards a round interface when enabled; zero tension preserves the no-tension case |
| Omit eligible ambient cells | Use the per-cell mask with irregular boundaries, preserving the fine-cell resolution and accounting for material/momentum exchange; compare against the fully simulated reference |
| Trapped gas pockets | A sealed bubble resists collapse through a pocket-pressure/volume model or resolved gas; connection to the exterior changes its treatment |

Keep the current engine concrete until a second working case needs an abstraction. Start generalising forces where they are applied, material properties where coefficients are built, and exterior conditions where pressure boundaries are imposed. Avoid a feature registry, automatic physics selection or a collection of unused switches now.

Execution choices may eventually become automatic from grid size and profiling. Whether air is a real simulated fluid, an ambient-pressure exterior, or a trapped pocket remains a modelling choice: those cases have different physical behaviour.
