# FluidSim development guide

Stage 1 geometric material transport and Stage 2 consistent mass/momentum transport are implemented. Water and momentum share transfers through successive X/Y passes. CPU and GPU pressure default to multigrid-preconditioned CG. Optional surface tension remains a later stage. `VISION.md` holds the longer-term ambitions and maths; this file is the short map of the working code and possible next steps.

## Where things belong

| File | Responsibility |
| --- | --- |
| `FluidSim/FluidGrid.h` | Engine data and public operations, in namespace `fluid` |
| `FluidSim/FluidGrid.cpp` | Advection, gravity, material transport and pressure solver |
| `FluidSim/PressureMultigrid.h/.cpp` | CPU pressure hierarchy, fixed symmetric V-cycle and small direct coarse solve |
| `FluidSim/FluidGridGPU.h/.cpp` | Separate float GPU implementation of the same physics, using fragment-shader passes |
| `FluidSim/PressureMultigridGPU.h/.cpp` | GPU pressure hierarchy, fixed symmetric V-cycle and device-side tiny Cholesky solve |
| `FluidSim/FluidGridGPUInternal.h` | Private numeric textures, shaders, pass runner and graphics-state preservation shared by those two GPU components |
| `FluidSim/FluidGridTypes.h` | Common state layout and projection result; CPU state uses double, GPU state uses float |
| `FluidSim/FluidGridDisplay.h/.cpp` | Render-thread texture adapter for either engine |
| `FluidSim/FluidSim.h/.cpp` | This experiment's initial pool, frame timing, failure reporting, camera, rendering and captures |
| `tests/FluidGridTests.cpp` | Small engine regression set, independent of scenes and drawing |
| `tests/FluidGridGPUTests.cpp` | GPU/CPU comparisons and GPU regressions in a hidden graphics context |

`FluidSimEngine` is a local CMake library used by both the scene and tests. It depends on Bitloop's vector type and thread pool, but has no Scene, camera, OpenGL or UI code. Another experiment can include `FluidGrid.h` and link that target. Packaging it as a separate repository is a later decision.

A new `FluidGrid(width, height, cell_size)` starts with zero velocity and zero water fraction. The scene creates an initial `State` in `initializePool()` and passes it to `upload()`; that sloped pool is an experiment, not an engine default. Existing CPU public arrays and individual operations remain available for learning. Dimensions and cell size describe a fixed grid; changing them requires constructing another grid.

## Switching CPU and GPU

Change only the member declaration in `FluidSim.h` between `FluidGrid fluid;` and `FluidGridGPU fluid;`, then rebuild. Both headers are included. Grid dimensions belong to the scene; either backend uses the same cell size, initialization, gravity, frame substeps and explicit `1e-3` tolerance. Both engines default to multigrid preconditioning.

Both engines provide `Scalar`, `State`, `ProjectionResult`, `upload`, `download`, `step`, `advect`, `applyGravity`, `project`, `maxDivergence`, `uIndex`, `vIndex` and `destroy`. `decltype(fluid)::State` automatically uses double or float fields. Empty uploaded fields mean zero, and upload resets pressure prediction history; it is initialization rather than complete checkpoint restoration. CPU `destroy()` has no device resources to release; its vectors remain owned by its destructor.

`withFluidContext()` uses the engine's `requires_render_thread` constant: CPU initialization/physics run on the simulation worker, while GPU work runs through the render queue. `FluidGridDisplay::texture(fluid)` uploads only the CPU water fractions into a reusable float texture, or returns the GPU engine's fraction texture directly. Drawing and the native-grid capture route use the same adapter. GPU drawing performs no full-grid download.

The common API supports the current experiment without a base class or virtual dispatch. Direct CPU vectors, `sampleVelocity`, `addVortex` and CPU working-data inspection remain CPU-specific learning tools. Pass an explicit divergence tolerance for comparisons: the engines retain their different precision-appropriate defaults.

Unused dye buffers, `advectDye`, particles and the unused `FluidCell` placeholder have been removed. `sampleVelocity` and `addVortex` remain useful experiment operations.

## One physics substep

Read `FluidGrid::step()` first:

```cpp
advect(dt);
applyGravity(dt);
return project(dt, max_iterations, divergence_tolerance);
```

1. **Transport material and momentum:** reconstruct interfaces, calculate water/mass transfers and carry both momentum components through X, then Y. After each material pass, recover velocity by dividing momentum by the new face density. Repeat with smaller transport steps when needed to limit outflow.
2. **Apply gravity:** add acceleration to interior vertical faces. Both materials receive the same acceleration; density affects their pressure response.
3. **Project pressure:** rebuild density coefficients from the current material fractions, close the tank walls, solve for pressure and correct velocity until cell divergence meets the requested tolerance.
Transport uses the previous substep's projected velocity. The scene starts with zero velocity, which already satisfies incompressibility. An experiment that directly changes velocities must project them before transport. The final projection handles the transported state and gravity, preparing the next substep.

`viewportProcess()` fixes each frame at 0.05 simulated seconds and divides it into six equal steps of 1/120 second. It calls `fluid.step(step, 2000, 1e-3)` for either selected backend. The CPU regression reference retains its double-precision `1e-8` checks; the scene uses the same explicit float-compatible target for comparisons. The engine advances exactly the time passed to it; it does not know about frames or recording.

On failed final projection, `step()` returns an unconverged result after transport and gravity have already run. It does **not** roll back material, velocity or pressure. The scene pauses and reports the failure rather than continuing with that state. A zero timestep leaves state unchanged.

## Material and momentum transport map

| Function | Question it answers |
| --- | --- |
| `waterArea` | How much water lies inside this local rectangle, given a dividing line? |
| `interfaceNormal` | Which way does the line face, estimated from neighbouring fractions? |
| `interfaceOffset` | Where must that line sit to enclose this cell's water fraction? |
| `rebuildInterfaces` | What are the current normal and offset for each mixed cell? |
| `waterTransferX/Y` | How much water lies in the outgoing strip beside this face during this substep? Returns a signed amount, already accounting for time |
| `massTransfer` | How much mass is carried by the water and air portions of a transfer? |
| `rebuildTransfersX/Y` | Store the water and mass amounts crossing each face for this pass |
| `rebuildDensity` | What cell density follows from the current fractions? |
| `rebuildMomentum` | What momentum density belongs to each face velocity? Uses the average density of the two neighbouring cells |
| `momentumTransfer` | How much momentum crosses a boundary? Mass transfer times donor velocity |
| `advectMomentumX/Y` | Carry both momentum components in this direction using the cached mass transfers and matching split corrections |
| `recoverVelocity` | Divide momentum by current face density and write `next_u/next_v` |
| `advect` | Coordinate material and momentum through successive X/Y passes and CFL substeps |

Local geometry uses a unit square per cell; no local-coordinate fields are stored. `water_fraction` is authoritative material state. Interface arrays are rebuilt working data. `split_weights` are numerical correction flags fixed across both passes, not rounded material fractions. Pure air/water cells bypass interface-area calculations.

During the entire `advect(dt)` call, `u/v` remain fixed for material transport and momentum corrections. `next_u/next_v` hold the evolving donor velocities; momentum is updated in place because no loop reads a neighbour's momentum. After each fraction swap, density and working velocity are rebuilt. The velocity buffers swap only after all transport substeps finish. Frozen correction weights are chosen separately for each X/Y pair; their combined contributions vanish for exactly divergence-free transport velocity.

The current method always processes X before Y, estimates normals with four neighbours and uses first-order donor velocities for momentum. It preserves sharper interfaces and couples mass/momentum transport, but numerical damping remains. It does not establish smooth droplets, realistic splashes or complete liquid behaviour.

The Stage 1 check covered 20.05 simulated seconds on the current 128x128 scene at its experimental 100,000:1 density ratio, with every physics substep checked. Relative water-volume drift was below `6e-13`; fraction bounds differed from `[0,1]` by less than `1e-14`; all projections met `1e-8` divergence. At 10 seconds, cells containing 1–99% water fell from 5,065 in the earlier run to 282. This validates this checkpoint, not arbitrary future configurations. Local evidence: [run0011 validation](.codex/runs/run0011_20261016_172158/validation.json).

The Stage 2 check passed all seven regression cases and advanced the current 128x128 sloped scene for 40 simulated seconds (4,800 physics steps) at 1000:1 density contrast. Every projection met `1e-8` divergence, relative water-volume drift stayed below `5.4e-12`, and fraction-bound errors stayed below `1e-14`; all wall velocities remained zero. Discrete kinetic plus gravitational energy peaked 0.082% above its initial value during the initial transient and ended at 49.9% of that value; this first-order method is not an exact energy-conservation scheme. Local evidence: [run0012 validation](.codex/runs/run0012_20261017_141309/validation.json).

## What the existing optimisations do

These preserve the current pressure equation and the outer PCG algorithm. Multigrid changes the preconditioner; the other optimisations change the amount of work or its arrangement. Floating-point rounding and stopping iterations can affect the trajectory within the requested accuracy.

| Optimisation | Plain meaning | Where to look |
| --- | --- | --- |
| Reused vectors | Keep allocated working memory between solves | `project` and private buffers |
| Pressure prediction | Use the last two pressures to guess the next one; rebuild the residual against the current system | Start of `solvePressure` |
| Cached inverse diagonal | Compute each cell's diagonal reciprocal once per projection rather than divide every CG iteration | `rebuildPressureCoefficients` |
| Multigrid V-cycle | Smooth local errors, solve broad errors on smaller grids, interpolate corrections back and smooth again | `PressureMultigrid::cycle` |
| Density-aware coarse weights | Combine pressure conductances in series and parallel instead of averaging away density barriers | `PressureMultigrid::coarsen` |
| Reused coarse factorization | Factor the smallest system once per projection; reuse it in every V-cycle | `factorCoarsest`, `solveCoarsest` |
| Zero halo | Zero wall weights and a padded border remove boundary branches from multigrid stencils | `PressureMultigrid::Level` |
| Interior/boundary split | Interior cells always have four neighbours; process them without boundary checks, then handle walls separately | `applyPressureOperator` |
| Row pointers | Locate each row once, then walk along it instead of reconstructing flattened indices | Interior operator loop |
| Fused updates and statistics | Update pressure/residual and measure the residual in the same pass | `updatePressureAndResidual` |
| Fused preconditioner reduction | Remove the correction's mean while accumulating the dot product CG needs | `applyPreconditioner` |
| Four accumulators | Let the CPU work on several independent sums instead of waiting on one running sum | Residual/preconditioner loops |
| Size-aware pressure parallelism | Keep small/coarse grids serial; batch independent rows on large grids, with a barrier between red/black colours | `applyPressureOperator`, multigrid `forRows` and `smooth` |
| Parallel advection/transport | Workers read fixed input arrays and write disjoint entries; in-place momentum updates read only their own old momentum | `Thread::forEachBatch` calls |
| Equal frame substeps | Avoid tiny remainder timesteps that make pressure prediction less useful | `viewportProcess` |

`applyPressureOperator` defines the equation; `applyPreconditioner` helps CG solve it faster. Mean removal before and after preconditioning preserves symmetry on mean-zero pressures. Both engines expose `PressurePreconditioner::{Multigrid, Diagonal}` and a `pressure_preconditioner` member. Set `grid.pressure_preconditioner = decltype(grid)::PressurePreconditioner::Diagonal` to compare with the original method without changing the scene's backend declaration.

Read `PressureMultigrid::cycle()` for the sequence. Each level groups roughly 2x2 cells until at most 64 remain. Four red/black Gauss-Seidel sweeps reduce local error; `restrictResidual` passes the remaining error down. The smallest system fixes one pressure, solves by Cholesky and removes the arbitrary constant. `prolongCorrection` interpolates the correction back, followed by four sweeps in the reverse colour order. Restriction is exactly the transpose of prolongation. These choices make each application a fixed linear symmetric positive preconditioner, as ordinary PCG requires. Coarse conductances approximate the fine system; the fine-grid equation and its final convergence checks remain authoritative.

The hierarchy's dimensions, interpolation maps and buffers are reused. Density-dependent conductances and the small factorization rebuild once per projection. Large levels use four workers, rising to eight from one million cells, bounded by the shared pool's configured worker count. Smoothing and transfer passes dispatch from 65,536 cells; the fine PCG stencil dispatches from 262,144. These thresholds and four smoothing sweeps were selected by measurement rather than a change to the physical model.

True-residual verification and the final divergence check are correctness checks, not optional performance overhead. A small internal CG residual alone does not prove that the final velocity meets the requested divergence tolerance.

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
| 4x4 reduction passes | Combine cell values into one GPU texel for sums, dot products and maximum divergence, including odd/thin grids |
| GPU PCG coefficients | Keep dot products, alpha, beta and the active iteration count on-device; predict the next CPU check from the previous solve, then grow missed batches |
| Pressure corrections | Apply the predicted pressure first, solve for the remaining divergence from zero, then apply the correction separately so small updates survive float rounding |
| Bounded refinement | If the measured final divergence is still too large, solve its remaining error again within the same total iteration budget; only the actual velocity check establishes convergence |
| Shared velocity statistics | Measure actual divergence and the next CFL outflow limit together, starting directly at the first reduction level; reuse outflow until velocity changes |
| Momentum/velocity MRT | Initialize momentum and donor velocity together, saving the initial density rebuild and two recovery draws |
| Texture-owned framebuffers | Keep a framebuffer attached to each single-output texture across swaps, avoiding repeated attachment changes |
| Native scalar downloads | Explicit inspections read RED/FLOAT directly; WebGL2 retains the portable RGBA/FLOAT path |
| Face-wise split cancellation | Evaluate the same material equation with the water flux and split correction paired per face, avoiding accumulated float cancellation in pure water |

Read `PressureMultigridGPU::cycle()` alongside the CPU version. Both use density-aware series/parallel coarse conductances, bilinear interpolation and its exact transpose for restriction, with mean removal before and after preconditioning. The GPU groups cells until at most 16 remain, rather than the CPU's 64, to keep the fragment-local direct solve small. It uses five fixed pre/post sweeps rather than four; large levels target the upper spectrum of the normalized operator with a fixed polynomial. These are execution/preconditioning choices, not a change to the fine pressure equation or physical model. The maps and buffers persist; coefficients and the coarse factor rebuild once per projection. The GPU result is checked against actual velocity divergence.

CG updates freeze once their residual reaches the internal target, so submitting a predicted batch does not advance the solution past convergence. Only active updates count towards `ProjectionResult::iterations` and the total refinement budget. A negative device-side dot product marks solver breakdown. Each ordinary full step now typically needs about three tiny readbacks; difficult solves and refinements need more. No full-grid readback occurs in normal stepping or drawing.

Construction itself does not allocate GL resources. `upload(State)` allocates and initializes the grid on its owning GL context; empty input arrays mean zero. `download()` explicitly copies fractions, face velocities and pressure back for inspection. It is not part of normal stepping or drawing. Upload initializes pressure history rather than restoring a complete checkpoint. Grid dimensions are fixed after upload.

When GPU is selected, the scene calls initialization and physics through `renderThreadQueue().invokeBlocking()` and destroys resources through that queue before scene deletion. The display adapter uses `waterTexture()` to supply the simulation fraction texture directly to the display shader and the existing `scene-0:grid` snapshot route. No full-grid CPU copy/upload occurs each GPU frame. The CPU still controls substeps and PCG batches and reads small reduction results.

Native code targets Bitloop's OpenGL 3.2 context; the shader variant for WebGL2 requires `EXT_color_buffer_float`. Native headless runs must include `--snapshots` to create a graphics context, even when no images are requested. The GPU harness creates its own hidden context. Browser execution has not yet been validated.

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

## Regression baseline

Enable `FLUIDSIM_BUILD_TESTS=ON` when configuring, build `FluidSimTests`, then run:

```text
ctest --test-dir <build-directory> -R "^FluidGridTests$" --output-on-failure
```

On Windows, configure/build from a Visual Studio developer shell. The test executable needs the same vcpkg runtime DLL directory on `PATH` as the application (`<build-directory>/vcpkg_installed/x64-windows/bin`).

With the same test option enabled, build `FluidSimGPUTests` and run both suites with `ctest --test-dir <build-directory> -R "^FluidGrid(GPU)?Tests$" --output-on-failure`. GPU cases cover hydrostatics, odd/thin projection, CPU comparisons, all four diagonal directions, nonuniform momentum transport, tiny interface remnants, sharp slabs, exact pure-material preservation, zero-time/failure handling and graphics-state restoration. GPU multigrid algebra tests cover odd/thin/high-contrast grids through 513x257, including the larger-grid polynomial smoother. Additional cases protect 512x512 default-budget convergence, the one-iteration cap and exact cached-versus-refreshed CFL parity. Common-API tests exercise upload/download, zero/reset behavior and actual display pixels for both types; the CPU suite also checks exact trajectory equivalence between direct array initialization and upload.

`FluidSimGPUTests --pool-steps 4800 1e-3` runs the 128x128 scene for 40 simulated seconds, downloading each committed step to check finite fields, fraction bounds, water volume, wall velocities, actual divergence and energy. `--benchmark-steps 360` times the same first three simulated seconds on CPU and GPU without those full-state downloads; initialization, drawing and capture are excluded. It reports the CPU's original `1e-8` target and a matched `1e-3` comparison separately.

| Implemented case | What it protects |
| --- | --- |
| Affine velocity sampling | Staggered offsets, interpolation, world origin and cell size |
| Filled tank under gravity | Hydrostatic pressure balances gravity; walls stay closed |
| A transported material patch | Conserved volume, expected centroid motion and CFL subcycling |
| A translated straight material slab | Whole-cell translation preserves its sharp interfaces and expected cell values |
| Diagonal motion across a density contrast | Both velocity components remain uniform around a 1000:1 material patch, for all four diagonal directions and multiple transport substeps |
| Sloshing closed pool for 20 simulated seconds | Repeated convergence, fraction bounds, conserved volume, finite velocities and closed walls at 1000:1 density ratio |
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
