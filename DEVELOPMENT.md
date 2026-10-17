# FluidSim development guide

Stage 1 geometric material transport and Stage 2 consistent mass/momentum transport are implemented. Water and momentum now share transfers through successive X/Y passes; the pressure solver is unchanged. Optional surface tension remains a later stage. `VISION.md` holds the longer-term ambitions and maths; this file is the short map of the working code and possible next steps.

## Where things belong

| File | Responsibility |
| --- | --- |
| `FluidSim/FluidGrid.h` | Engine data and public operations, in namespace `fluid` |
| `FluidSim/FluidGrid.cpp` | Advection, gravity, material transport and pressure solver |
| `FluidSim/FluidSim.h/.cpp` | This experiment's initial pool, frame timing, failure reporting, camera, rendering and captures |
| `tests/FluidGridTests.cpp` | Small engine regression set, independent of scenes and drawing |

`FluidSimEngine` is a local CMake library used by both the scene and tests. It depends on Bitloop's vector type and thread pool, but has no Scene, camera, OpenGL or UI code. Another experiment can include `FluidGrid.h` and link that target. Packaging it as a separate repository is a later decision.

A new `FluidGrid(width, height, cell_size)` starts with zero velocity and zero water fraction. The scene fills `water_fraction` in `initializePool()`; that sloped pool is an experiment, not an engine default. Existing public arrays and individual operations remain available for learning. Dimensions and cell size describe a fixed grid; changing them requires constructing another grid.

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

`viewportProcess()` currently fixes each frame at 0.05 simulated seconds, divides that interval into equal steps of at most 1/120 second, and calls `fluid.step(step, 2000, 1e-8)`. The engine advances exactly the time passed to it; it does not know about frames or recording. Those scene solver settings are preserved, not universal recommendations.

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

All of these preserve the current pressure equation and diagonal-preconditioned CG method. They change the amount of work or its arrangement; floating-point summation order can affect rounding.

| Optimisation | Plain meaning | Where to look |
| --- | --- | --- |
| Reused vectors | Keep allocated working memory between solves | `project` and private buffers |
| Pressure prediction | Use the last two pressures to guess the next one; rebuild the residual against the current system | Start of `solvePressure` |
| Cached inverse diagonal | Compute each cell's diagonal reciprocal once per projection rather than divide every CG iteration | `rebuildPressureCoefficients` |
| Interior/boundary split | Interior cells always have four neighbours; process them without boundary checks, then handle walls separately | `applyPressureOperator` |
| Row pointers | Locate each row once, then walk along it instead of reconstructing flattened indices | Interior operator loop |
| Fused updates and statistics | Update pressure/residual and measure the residual in the same pass | `updatePressureAndResidual` |
| Fused preconditioner reduction | Remove the correction's mean while accumulating the dot product CG needs | `applyPreconditioner` |
| Four accumulators | Let the CPU work on several independent sums instead of waiting on one running sum | Residual/preconditioner loops |
| Serial pressure stencil | Avoid worker-dispatch overhead measured on the small grid; this choice needs profiling again at larger sizes | `applyPressureOperator` |
| Parallel advection/transport | Workers read fixed input arrays and write disjoint entries; in-place momentum updates read only their own old momentum | `Thread::forEachBatch` calls |
| Equal frame substeps | Avoid tiny remainder timesteps that make pressure prediction less useful | `viewportProcess` |

`applyPressureOperator` defines the equation; `applyPreconditioner` helps CG solve it faster. Its current mean removal before and after diagonal scaling preserves the symmetry CG relies on. A future multigrid preconditioner would replace that operation while keeping its mathematical contract. It is a separate learning milestone.

True-residual verification and the final divergence check are correctness checks, not optional performance overhead. A small internal CG residual alone does not prove that the final velocity meets the requested divergence tolerance.

## Regression baseline

Enable `FLUIDSIM_BUILD_TESTS=ON` when configuring, build `FluidSimTests`, then run:

```text
ctest --test-dir <build-directory> -R "^FluidGridTests$" --output-on-failure
```

On Windows, configure/build from a Visual Studio developer shell. The test executable needs the same vcpkg runtime DLL directory on `PATH` as the application (`<build-directory>/vcpkg_installed/x64-windows/bin`).

| Implemented case | What it protects |
| --- | --- |
| Affine velocity sampling | Staggered offsets, interpolation, world origin and cell size |
| Filled tank under gravity | Hydrostatic pressure balances gravity; walls stay closed |
| A transported material patch | Conserved volume, expected centroid motion and CFL subcycling |
| A translated straight material slab | Whole-cell translation preserves its sharp interfaces and expected cell values |
| Diagonal motion across a density contrast | Both velocity components remain uniform around a 1000:1 material patch, for all four diagonal directions and multiple transport substeps |
| Sloshing closed pool for 20 simulated seconds | Repeated convergence, fraction bounds, conserved volume, finite velocities and closed walls at 1000:1 density ratio |
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
| Stronger pressure preconditioning | Same physical tests and divergence target, fewer seconds spent solving |
| GPU stage | CPU/GPU agreement within stated tolerances, with the same model and meaningful timing comparison |

Keep the current engine concrete until a second working case needs an abstraction. Start generalising forces where they are applied, material properties where coefficients are built, and exterior conditions where pressure boundaries are imposed. Avoid a feature registry, automatic physics selection or a collection of unused switches now.

Execution choices may eventually become automatic from grid size and profiling. Whether air is a real simulated fluid, an ambient-pressure exterior, or a trapped pocket remains a modelling choice: those cases have different physical behaviour.
