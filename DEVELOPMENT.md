# FluidSim development guide

Stage 1 geometric material transport and Stage 2 consistent mass/momentum transport are implemented. Liquid geometry, component masses and momentum share transfers through successive X/Y passes. CPU and GPU pressure default to multigrid-preconditioned CG. Optional surface tension remains a later stage. `VISION.md` holds the longer-term ambitions and maths; this file is the short map of the working code and possible next steps.

Both engines use the component system: configure the element catalogue and liquid/gas mixtures, then upload component masses alongside `liquid_fraction`, the total liquid volume fraction. Density, momentum and displayed colours come from those masses. Engine geometry uses liquid/gas terminology; Water, Red, Air and Cloud are scene ingredients. Several miscible ingredients share each phase's geometry; adding an ingredient does not add another interface reconstruction or pressure solve.

The scene initializes blue Water and Red liquid, plus Air and a cyan Cloud gas ingredient. Air and Cloud share density 0.1; three fixed Perlin scales divide the initial gas volume between them. Noise is evaluated only in `initializePool()`, then ordinary component transport carries the clouds. This visualizes atmospheric flow without adding condensation or evaporation physics.

## Where things belong

| File | Responsibility |
| --- | --- |
| `FluidSim/FluidGrid.h` | Engine data and public operations, in namespace `fluid` |
| `FluidSim/FluidGrid.cpp` | Advection, gravity, material transport and pressure solver |
| `FluidSim/PressureMultigrid.h/.cpp` | CPU pressure hierarchy, fixed symmetric V-cycle and small direct coarse solve |
| `FluidSim/FluidGridGPU.h/.cpp` | Separate float GPU implementation of the same physics, using fragment-shader passes |
| `FluidSim/PressureMultigridGPU.h/.cpp` | GPU pressure hierarchy with an owned CPU coarse solver; GPU-only tiny Cholesky remains selectable |
| `FluidSim/FluidGridGPUInternal.h` | Private numeric textures, shaders, pass runner and graphics-state preservation shared by those two GPU components |
| `FluidSim/FluidGridTypes.h` | Common state layout and projection result; CPU state uses double, GPU state uses float |
| `FluidSim/FluidGridEdit.h` | Shared staged material replacements, validation and sparse commit ordering |
| `FluidSim/FluidGridDisplay.h/.cpp` | Render-thread texture adapter for either engine |
| `FluidSim/FluidSim.h/.cpp` | This experiment's initial pool, frame timing, failure reporting, camera, rendering and captures |
| `tests/FluidGridTests.cpp` | Small engine regression set, independent of scenes and drawing |
| `tests/FluidGridGPUTests.cpp` | GPU/CPU comparisons and GPU regressions in a hidden graphics context |

`FluidSimEngine` is a local CMake library used by both the scene and tests. It depends on Bitloop's vector type and thread pool, but has no Scene, camera, OpenGL or UI code. Another experiment can include `FluidGrid.h` and link that target. Packaging it as a separate repository is a later decision.

A new `FluidGrid(width, height, cell_size)` starts with zero velocity and zero liquid fraction. Configure its mixtures and populate component masses before advancing physics. The scene creates an initial `State` in `initializePool()` and passes it to `upload()`; that sloped pool is an experiment, not an engine default. Existing CPU public arrays and individual operations remain available for learning. Dimensions and cell size describe a fixed grid; changing them requires constructing another grid.

CPU density, pressure coefficients and the multigrid hierarchy are now cached. After directly editing `liquid_mass`, `gas_mass` or derived density/pressure-coefficient arrays, call `fluid.invalidateCaches()` before the next physics operation. It marks derived data stale; rebuilding happens when needed. `upload()` and engine material transport do this automatically. Direct velocity edits need no cache notification because CPU velocity statistics are recomputed, but transport still requires projected velocity. Mixture membership changes continue to require `upload()`. GPU mass/velocity textures are private and its `upload()` already invalidates its caches; direct CPU-array access is not part of the shared interface.

## Switching CPU and GPU

Select the scene backend with `FluidGrid fluid;` or `FluidGridGPU fluid;`. Both accept the same catalogue, mixture definitions and per-cell component masses, and default to multigrid preconditioning.

Both engines use the same 3x3 mixed Youngs-centered (MYC) normal estimate. The GPU scales the chosen normal before its float geometry calculations; scaling leaves the reconstructed interface unchanged. CPU/GPU transport comparisons pass with their existing precision tolerances.

Both engines provide `Scalar`, `State`, `ProjectionResult`, `Edit`, `beginEdit`, `upload`, `download`, `step`, `advect`, `applyGravity`, `project`, `maxDivergence`, `uIndex`, `vIndex` and `destroy`. `decltype(fluid)::State` automatically uses double or float fields. Empty uploaded fields mean zero, and upload resets pressure prediction history; it is initialization rather than complete checkpoint restoration. CPU `destroy()` has no device resources to release; its vectors remain owned by its destructor.

`withFluidContext()` uses the engine's `requires_render_thread` constant: CPU initialization/physics run on the simulation worker, while GPU work runs through the render queue. `FluidGridDisplay::texture(fluid)` uploads CPU catalogue colours into a reusable RGBA float texture, or builds them directly from GPU masses. Drawing and the native-grid capture route use the same adapter. GPU stepping and drawing perform no full-grid download; `download()` explicitly reads every state field, including component masses.

Configure mixture membership/order before `upload()`, and upload again when changing it. State masses are actual mass per cell (unit depth, cell volume = `cell_size * cell_size`) on either backend. An empty mass field initializes to zero; the engine never invents an Air ingredient. The same element can appear in both mixtures with different component densities.

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
withFluidContext([&] { edit.commit(); });
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

## One physics substep

Read `FluidGrid::step()` first:

```cpp
advect(dt);
applyGravity(dt);
return project(dt, max_iterations, divergence_tolerance);
```

1. **Transport material and momentum:** reconstruct interfaces, calculate liquid-volume transfers, transport component masses and carry both momentum components through X, then Y. After each material pass, recover velocity by dividing momentum by the new face density. Repeat with smaller transport steps when needed to limit outflow.
2. **Apply gravity:** add acceleration to interior vertical faces. Both materials receive the same acceleration; density affects their pressure response.
3. **Project pressure:** rebuild density coefficients from the current component masses, close the tank walls, solve for pressure and correct velocity until cell divergence meets the requested tolerance.
Transport uses the previous substep's projected velocity. The scene starts with zero velocity, which already satisfies incompressibility. An experiment that directly changes velocities must project them before transport. The final projection handles the transported state and gravity, preparing the next substep.

`viewportProcess()` currently fixes each frame at 0.02 simulated seconds and divides it into five equal steps of 0.004 second. It calls `fluid.step(step, 2000, 1e-3)` for either selected backend. The CPU regression reference retains its double-precision `1e-8` checks; the scene uses the same explicit float-compatible target for comparisons. The engine advances exactly the time passed to it; it does not know about frames or recording. Older measurements below used six 1/120-second steps per frame; compare timing only with matched timestep and simulated duration.

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
| Equal frame substeps | Avoid tiny remainder timesteps that make pressure prediction less useful | `viewportProcess` |

`applyPressureOperator` defines the equation; `applyPreconditioner` helps CG solve it faster. Mean removal before and after preconditioning preserves symmetry on mean-zero pressures. Both engines expose `PressurePreconditioner::{Multigrid, Diagonal}` and a `pressure_preconditioner` member. Set `grid.pressure_preconditioner = decltype(grid)::PressurePreconditioner::Diagonal` to compare with the original method without changing the scene's backend declaration.

Read `PressureMultigrid::cycle()` for the sequence. Each level groups roughly 2x2 cells until at most 64 remain. Four red/black Gauss-Seidel sweeps reduce local error; `restrictResidual` passes the remaining error down. The smallest system fixes one pressure, solves by Cholesky and removes the arbitrary constant. `prolongCorrection` interpolates the correction back, followed by four sweeps in the reverse colour order. Restriction is exactly the transpose of prolongation. These choices make each application a fixed linear symmetric positive preconditioner, as ordinary PCG requires. Coarse conductances approximate the fine system; the fine-grid equation and its final convergence checks remain authoritative.

The hierarchy's dimensions, interpolation maps and buffers are reused. Density-dependent conductances and the small factorization rebuild when masses have changed and a solve actually needs the preconditioner. Engine batches below 65,536 entries stay serial; larger transport and pressure-vector loops use up to eight workers, bounded by the shared pool's configured worker count. Multigrid retains four workers on medium levels and eight above about one million cells, with small/coarse levels serial. Red/black smoothing retains a barrier between colours and four symmetric pre/post sweeps. The fine pressure equation, true-residual verification and final divergence checks are unchanged; reduction grouping can change floating-point rounding and stopping iterations.

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
| Prescribed ambient exterior | Open liquid surface sees a chosen exterior pressure; resolved-gas mode remains a separate explicit choice |
| Trapped gas pockets | A sealed bubble resists collapse through a pocket-pressure/volume model or resolved gas; connection to the exterior changes its treatment |

Keep the current engine concrete until a second working case needs an abstraction. Start generalising forces where they are applied, material properties where coefficients are built, and exterior conditions where pressure boundaries are imposed. Avoid a feature registry, automatic physics selection or a collection of unused switches now.

Execution choices may eventually become automatic from grid size and profiling. Whether air is a real simulated fluid, an ambient-pressure exterior, or a trapped pocket remains a modelling choice: those cases have different physical behaviour.
