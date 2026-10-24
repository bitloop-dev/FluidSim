# Source-derived 2D CPU fluid hierarchy

`FluidSim/AmrGas.{h,cpp}` is the production engine. Its fields, transfers,
clustering and both pressure paths are project-owned code. The application has
no AMReX headers, linkage or initialization. The frozen upstream implementation
lives only in `reference/amrex/`, namespace `fluid::reference`, and is linked only
by the separate reference project. It always uses upstream pressure.

The extraction preserves the existing closed-wall, square-cell, 2:1 algorithm;
it does not implement MPI, other dimensions, embedded boundaries or GPU execution.
The catalogue and state API remain standard C++. Native CPU workers use optional
OpenMP; `AmrGasRuntime` now only configures that worker count.

Build the complete CPU library without AMReX, Bitloop or OpenMP:

```sh
cmake -S reference/engine -B build/cpu-extraction-no-amrex -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON
cmake --build build/cpu-extraction-no-amrex
```

## Source map and validation

Pinned AMReX: `015c14b65715b277fbc54706ebe7455b196d2254` (26.10).
Retain [its copyright/license](../pressure2d/AMReX-LICENSE) when redistributing
source or binaries derived from these algorithms.

| Extracted operation | Upstream contract |
|---|---|
| `AmrField2D`, `splitAmrBoxes` | Component-major FAB arrays; `BoxList::maxSize` preserves factors of two and balances pieces |
| `AmrBoundary2D` | `FillPatchSingleLevel/TwoLevels`, same-level valid ownership before coarse interpolation, even/odd reflecting physical boundaries |
| `AmrTransfer2D`, `AmrFaceSync2D` | `average_down`, `average_down_faces`, `OverrideSync`: conservative valid-cell/face transfers and lowest-index face ownership |
| `refineAmrField`, `regrid` | `InterpFromCoarseLevel` / `FillPatchTwoLevels`, exact old fine overlap and temporary successive 2:1 bridges; existing project bounded plane retained |
| `clusterAmrTags` | `Cluster::FindCut/new_chop`, `ClusterList::new_chop`: holes, histogram curvature, balanced cut, alternate direction and 70% efficiency |
| Both pressure paths | [Operator, cycle, flux and solver source map](../pressure2d/README.md) |

`FluidExtractionTests` compares both engines on identical evolving inputs through
uniform, three-level, uniform-32, six-level and direct-refined endpoints. It checks
exact conservative remaps, poisoned-ghost refill, both transports and equal/unequal
density projection, including seven gas ingredients and liquid at 1000:1 contrast.
One hundred tagged masks and 21 patch-size fixtures match upstream clustering and
box splitting exactly. The independent pressure tests still check the composite
operator, preconditioning cycle, solve and corrected flux.

[Matched 512² wave evidence](../../.codex/runs/run0100_20261024_cpu_extraction/findings.md):
two 60-frame pairs with current scene settings, setup/copy/selection/regrid costs
included, material and committed-layout images inspected. This is a short CPU
extraction check; the broader adaptive-versus-old-uniform and GPU goals remain open.

## State and API

Construct with finest width/height/cell size, catalogue, gas mixture and a
`GasLayout`. `baseRatio` selects the full-domain cell width in finest units (1, 2,
4, 8, 16 or 32). `refinedPatches` contains the nonempty patch lists for successive 2:1 finer
levels, ordered coarse to fine. A level's width is `baseRatio >> level`; all
`GasPatch` rectangles use finest-cell coordinates. Empty lists are omitted.
Examples: `{8,{}}` is uniform 8x8; `{8,{coarse,middle,fine}}` is a real 8/4/2/1 hierarchy.

The final optional constructor argument remains `maxGridSize` (64). Production
has no runtime backend selector or route into the reference implementation.

The two-phase constructor additionally takes a liquid mixture before the gas
mixture. Conserved lanes are liquid ingredients, gas ingredients, then X/Y
momentum. `materialMixture()` describes all ingredient lanes in that order;
`gasMixture()` retains the gas catalogue order. Automatic wave layouts protect
liquid and its predicted buffer at finest resolution; coarse liquid interfaces
are not the current optimization target.

Prescribed layouts are project-controlled: same-level rectangles are disjoint and
aligned to their parent; parent valid cells cover each child's ghost and slope
stencil. For the two fine ghost layers, cover `coarsen(grow(fine valid,2),2)` grown
by one parent cell, clipped at physical walls. Automatic selection leaves two
parent cells of padding around each child patch at every join (four finest cells
around 1x1 patches, eight around 2x2 patches). Closed slip walls and
Cartesian square cells are supported; no subcycling between levels is used.

`upload(state, tolerance)` converts the supplied finest-cell masses and staggered
velocities once, restricts mass and cell momentum, and establishes a projected
face field. Input gas masses are actual cell masses, as in `FluidGrid`; internal
quantities are densities per area. Together the configured phase ingredients
occupy the complete cell volume. No fine material copy remains in the engine.
Empty velocity fields mean rest. Supply each phase's mass lanes in its mixture
order. The gas-only overload continues to reject nonzero liquid input.

`step(dt, divergenceTolerance, damping, acceleration)` advances the entire fluid
model and returns `GasStepResult`. Every level advances by the same CFL transport
substep, using fixed previously projected faces throughout the physical `dt`.
After transport, apply damping and acceleration over that full interval and
project once. The caller's `dt` controls force/pressure temporal resolution;
transport subdivisions alone do not resolve rapid force changes.
`project`, `transport`, `stableTimeStep`, `maxDivergence` and `totals` expose the
algorithm for focused tests. Direct `transport(dt)` callers must honor
`stableTimeStep()` and supply the previously projected face field. A failed
projection is reported; the caller must stop normal stepping after failure.
Numerical solver breakdown or failure to converge throws instead of aborting the application. `upload` also throws if its final actual
divergence check fails. Callers must catch solver exceptions and report failure;
steps are not transactional rollbacks.
Differing-density mixtures use the extracted GMRES+MLMG composite operator;
equal-density mixtures use extracted standalone MLMG with a BiCGStab bottom solve.
Gas-only solves retain a 200-iteration ceiling; liquid uses 2,000. GMRES uses
the requested divergence tolerance as its absolute L2 target, a stricter bound
than maximum divergence. The final actual-divergence check is unchanged.
Preconditioner failures still throw; outer convergence is checked on the
corrected faces. `GasStepResult.projection` describes the one final physical-step
projection, while `substeps` counts only transport subdivisions.

`regrid(layout)` transfers conserved fields directly between real hierarchies.
It returns false for the same prescribed rectangles (independent of their order),
preserving every field and allocation. A true return requires `project(tolerance)`
before the next transport/step. Keeping these operations separate lets focused
tests measure raw conservative momentum before closed-wall pressure impulses.
No finest upload/download state is generated or retained during regrid.

The old fine level is restricted first. `copyAmrOverlap` preserves same-resolution
valid values exactly, `AmrTransfer2D` averages merged state, and `refineAmrField`
reconstructs new fine values using the shared bounded plane. Same-resolution old
fine overlap then overrides interpolation. This follows incflo's `RemakeLevel` / `MakeNewLevelFromCoarse`
contract. Uniform fine, uniform coarse and mixed endpoints all use the same
transfer. Changing an 8x8 base directly to uniform 1x1 uses temporary 4x4 and 2x2
interpolation bridges, not retained fine state. Target levels at or above the old
base width restrict the original old base, preserving its finer averages rather
than interpolating from a newly coarsened parent. Pressure history and projector
storage are rebuilt with the topology.

`levels()` exposes actual project-owned level data for an actual-cell renderer.
It does not synthesize a fine field.
For each level, `conserved` components `[0, ingredientCount)` are ingredient mass
densities, followed by X/Y momentum densities. `velocity[0/1]` are independent
normal MAC faces; `active` excludes fine-covered coarse cells from composite
budgets. Coarse covered cells remain stored, evolved and restricted. A hierarchy
with a 2x2 base and 25% fine coverage has 0.4375 N² active cells and 0.5 N² allocated level cells.
Scratch/solver storage is additional and follows those level boxes.
`conserved` also owns two ghost layers, refreshed before stencil reads. There is
no separate full-state ghost workspace. Regrid transfers valid cells and refreshes the parent ghosts needed for the next
refinement level; its overlap rules are unchanged.

## Ordered timestep

The numbered algorithm below remains the gas-only path. With liquid ingredients,
the transport stage uses the shared `FluidInterfaceGeometry.h` MYC/PLIC geometry
and X/Y directional phase transport. Ingredients follow their phase's geometric
volume flux; momentum follows the same combined ingredient mass flux. Frozen
Weymouth-Yue corrections preserve phase-volume closure and cancel for exactly
divergence-free faces. With finite projection residual, separately report the
resulting small mass-budget error. Average fine ingredient/momentum fluxes onto
coarse faces before each directional update, then restrict covered state.

The smaller phase is reconstructed directly from its ingredient volume, avoiding
cancellation of a tiny gas flux against nearly equal total/liquid fluxes. The
geometry bounds outgoing roundoff by donor availability; it does not clip or
renormalize committed masses. Phase geometry and split weights are scratch.
Pure gas scenes retain their bounded-linear transport below. Wave damping affects
gas-only cell momentum, leaving liquid/interface cells coupled through pressure.
All native scene walls are closed; the earlier prescribed top was removed with
the user's authorization for this liquid experiment.

Repeat steps 1–3 at the computed CFL limit with the projected faces held fixed;
steps 4–6 then run once for the full physical `dt`. The shared face field makes
the CFL bound constant during transport, so it is computed once per `step`.

1. Fill two layers of ghosts in place from cached address recipes. Same-level
   valid data takes precedence; remaining ghosts use the current coarse bounded
   plane. Reflect physical-wall positions and reverse normal momentum. Owned
   cells remain untouched, and no full-state copy is made. Normal
   momentum is odd across a wall, other fields even. Spatial reconstruction uses
   common monotonized-central slope factors and a corner bound per mixture.
   Differing-density momentum shares its ingredient limiter; equal-density momentum
   has a separate group so passive labels cannot alter velocity. New fine averages,
   coarse-cell display and material-error estimates use this same spatial plane.
2. Reconstruct slopes for valid cells and their immediate face donors. Trace each
   upwind face location back half a step along the normal projected face velocity
   and transverse donor velocity, bounded to the donor cell. Form conservative
   ingredient fluxes from that plane. The summed mass flux carries momentum at
   the existing donor cell velocity, retaining first-order bulk-flow advection.
   Ingredient linear identities survive the shared limiter. This is an unsplit
   piecewise-linear transport improvement; the existing pressure-coupled integrator
   is not claimed to be second order in time. An analytic prescribed-flow test
   separately checks convergence of the transport operator.
3. Average **all** fine ingredient and momentum fluxes down before any cell
   update. Apply the divergence of these shared fluxes to both levels, then
   restrict covered coarse conserved fields. This is the no-subcycling form of
   the synchronization in incflo's `compute_advection_term`.
4. Apply the explicitly supplied exponential momentum damping. Predict face
   normal velocities by averaging cell momentum divided by cell density, and
   add the selected acceleration impulse on non-wall faces. Physical normal
   wall velocities stay zero.
5. Build `beta_face = 2/(rho_left + rho_right)` and average predicted fine faces
   down, matching the upstream composite predictor. Measure actual divergence;
   when it already meets the requested tolerance, use zero pressure correction
   and skip solver preparation. Pressure impulse is zeroed for that step.
   Otherwise solve with the extracted composite
   `AmrPressure2D` operator. Its warm-start field `pressure` is a **pressure
   impulse**, not pressure: `D(beta G phi)=D(U*)`, `U=U*-beta G phi`.
   Relative tolerance is zero. Differing-density mixtures use
   GMRES with one multigrid preconditioning cycle; equal-density mixtures use
   standalone MLMG with BiCGStab at the bottom. Absolute tolerance is the requested
   divergence residual (L2 for GMRES, maximum norm for standalone MLMG).
   The result additionally checks actual active-cell divergence.
6. Save the averaged predictor before the solve and subtract it from the final
   projected faces. `AmrPressure2D::fluxes` prepares coarse/fine
   pressure boundary data, then returns the correction to add to the saved
   predictor and average down. Both solver paths share this operation.
   Their difference reuses the correction without a second gradient pass.
   Add its cell average to cell momentum, multiplied by that
   cell's current density. Average the acceleration faces into the cell using
   the same operation. Averaging the two together preserves stationary
   constant-density force equilibrium at closed walls. Restrict covered coarse
   momentum again. When no solve is required, apply only the selected force
   (or return immediately if it is zero). The independently projected fine normal faces remain the
   advecting field for the next physical step.

The last operation follows incflo's approximate cell-centred MAC projection:
**add the averaged pressure correction to cell velocity**, rather than replacing
cell velocity with averaged total face velocity. Cell velocity and MAC face
velocity are distinct discretizations. Only the MAC field is discretely
divergence-free. The former staggered half-cell kinetic norm is not the invariant
of this method. Closed walls can exchange momentum through pressure; pure
transport separately conserves total composite momentum and each ingredient.

Use one shared slope limiter for the ingredient group: independent component
limiting can break the local volume sum. Equal-intrinsic-density momentum has a
separate limiter so passive labels cannot flatten flow. Spatial planes are bounded
at their corners; the half-step face sample stays inside that donor. On a single
resolution, the timestep limits each cell's actual outgoing face sum to .45:
`dt * (max(uR,0) + max(-uL,0) + max(vT,0) + max(-vB,0)) / dx <= .45`.
Nonnegative bounded planes are at most twice their cell mean, so that outgoing
bound protects positive donor averages without combining distant X/Y extrema.
It also retains the geometric phase scheme's conservative outflow margin. The
cell width is the actual level width, including a wholly coarse grid.

Mixed-level hierarchies still use the previous global per-level velocity bound.
Signed coarse-face averages can hide opposing fine-face flows. At a 2:1 join,
coarse interpolation followed by fine-face reconstruction is bounded by four times
the coarse average. The finest-level Courant bound of 0.45 halves the coarse
Courant sum, protecting its outgoing update too. This assumes the caller obeys
`stableTimeStep()`; it is not post-update clipping or mass renormalization.
The local-bound change does not claim a less restrictive bound at these joins.

## Source correspondence

The incflo study checkout is commit
`c7d6f7edcfa1919e254910acc2d11f9e7e26b47c`. The relevant code is:

- `src/convection/incflo_compute_advection_term.cpp`: average fine conservative
  fluxes down before their divergence updates (around lines848–860).
- `src/projection/incflo_apply_cc_projection.cpp`: face beta from density,
  composite MAC projection, `getFluxes`, face-to-cell correction and adding that
  correction to cell velocity. This kernel uses its documented arithmetic
  cell-to-face averaging alternative, not its MOL face extrapolator. The kernel
  reuses the projected-minus-predicted face difference for that same correction.
- AMReX-Hydro `Projections/hydro_MacProjector.cpp`, `project_doit`: average the
  predictor down, solve, get pressure flux, add it to faces and average down.
  Averaging the saved predictor too is necessary for the difference to equal
  the averaged pressure correction at a coarse/fine boundary.
- AMReX `Src/LinearSolvers/MLMG/AMReX_GMRES_MLMG.H`: upstream outer GMRES
  with MLMG preconditioning. Hydro's public `getFluxes` retains the composite
  coarse/fine pressure-boundary preparation for this alternative solver.
- `src/incflo_regrid.cpp`: build a new level from coarse interpolation; remake
  an existing level with coarse/fine FillPatch; reset pressure and rebuild the
  MAC projector when topology changes.
- AMReX `Src/AmrCore/AMReX_MFInterp_2D_C.H`,
  `mf_cell_cons_lin_interp_llslope`: common component slope limiter.
- `../README.md`: full projection contract, exact source pins and build command.

## Historical upstream validation and remaining boundary

The separate upstream-oracle `FluidAmrGasTests mode=check n_cell=64` runs closed force equilibrium,
isolated conservative transport at the CFL limit, then 24 simulated seconds of
undamped evolving cloud flow on uniform, one-patch, four-patch and wall-touching
fine-patch layouts. Both
two equal-density ingredients and seven ingredients with a sharp 100:1 density
contrast are exercised. Every committed step scans actual active cells for
finite/nonnegative ingredients, conservation and volume closure; it checks
divergence and interface flux matching. Integrated absolute cloud flux across
patch boundaries verifies repeated transport opportunities. A material-change
metric verifies that clouds actually evolve. The horizon is three initial
domain crossing times (world16, speed2), with 1500 steps at dt=.016.

The retained Stage 2 report is in
`../../.codex/runs/run0072_20261023_151735/stage2-validation.md`.
It separates these behavioral checks from 10.32-second damped cost controls at
world256. Those controls include 20 warmup plus625 measured steps, covering two
five-second damping times. They are not a domain-crossing gate.

Stage 3a is recorded in
`../../.codex/runs/run0072_20261023_151735/stage3a-validation.md`.
The default check also covers fully coarse and four moving-layout schedules,
with raw mass/momentum checks before every pressure restoration. A focused gate
cycles 21 transfers through uniform/coarse/mixed endpoints, verifies exact fine
overlap values and tests unchanged regrids with reversed rectangle order.
Moving trajectories each contain 11 changes during the same 24-second horizon.
The interface metric excludes internal boundaries between adjacent fine patches.

`mode=bench layout=costs` adds matched old 37.5%-coarse one/two/four layouts,
their changing schedule, the old scene's changing coarse footprints, and sparse
25%-fine fixed/moving controls. `reverse=1` reverses case order for paired timing.
The scene-footprint control uses the common cosine cloud field, not scene Perlin
noise or rendering. At 1024 the 75%-coarse moving control costs 4.64–4.68 ms versus
9.57–10.76 ms for the same kernel's uniform control; the matched 37.5%-coarse four
layout costs 16.51–16.67 ms and remains slower than new uniform. The gain depends
on coverage; the full coarse base is allocated and advanced under fine patches.

At this historical foundation checkpoint, automatic layout choice, additional
levels and liquid/GPU AMR were later work. Automatic levels and the initial liquid
port are now implemented; GPU AMR remains future work. Foundation measurements are recorded in
`../../.codex/runs/run0072_20261023_151735/final-validation.md`.
First-order transport diffuses clouds and vorticity at finite resolution;
the low-resolution 24-second run visibly exposes that numerical effect in its
energy metrics. No extra physical damping is added. The same transport and
projection algorithm is used for the uniform/adaptive cost comparison.

## Production observation and runtime

`AmrGasRuntime` configures the simulation worker; no AMReX runtime is initialized.
Eight OpenMP workers are the default; an explicit `OMP_NUM_THREADS` is respected.
Native liquid scenes now use the same AMR hierarchy with geometric phase transport;
Emscripten builds retain the existing uniform CPU engine.

`FluidCellView` reads current valid native patch arrays and active masks directly.
It caches only the owner of each 2x2 finest group. Display expansion stores colour
pixels, and preview history stores requests; neither creates fine physics state.
Cell velocities come from committed conserved momentum, with independent MAC faces
used for local detail and feature motion. The stored `pressure` is a correction impulse:
preview divides it by `pressureInterval()` for pressure-gradient units. An initial
or geometric dt=0 restoration has no physical pressure sample. It is the last
correction pressure, not accumulated thermodynamic pressure.

The scene selects its fine patches automatically from the material/flow indicators.
Prescribed central/two/four/moving layouts remain reference controls only.
The benchmark retains `matched-*`, `changing` and `scene-changing` for the older
low-coarse-area controls, in addition to `sparse-changing`; use `mode=benchmark
layout=costs`. `production-changing` matches the scene's exact schedule; it is
also included in the full behavioral gate. Those old layouts can be slower than
the same-kernel uniform path.

Final review added independent three-ingredient tests: raw refinement preserves
volume and uniform velocity for equal and100:1 density; equal-density colour
patterns preserve bulk flow; two additional24-second changing-layout cases use
independent fraction fields. Earlier seven-ingredient cases derive fractions
from a single cloud scalar and alone do not establish this property.
