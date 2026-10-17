# VISION — Adaptive 2D Fluid Simulation in bitloop

**Status:** 15 October 2026

**Purpose:** Self-contained handover for a new ChatGPT conversation or Codex session. Read [DEVELOPMENT.md](DEVELOPMENT.md) for the current file map, optimisation guide, regression cases and incremental milestones.

**Project:** A C++/bitloop fluid-simulation learning project that may ultimately become a substantial, interactive, adaptive fluid-world simulation.

> **Most important principle:** Build the simplest working, understandable CPU implementation *now*, while preserving a credible route toward the ambitious world described below. Avoid premature abstraction and optimisation, but also avoid foundations we already know will have to be discarded.

---

## 1. Long-term vision

Build an interactive **2D fluid world** that can contain moving, swirling and turbulent fluids, with user-specified velocities and forces. Eventually, the *same spatial fluid system* should be capable of representing both **water and atmosphere**, including their interface, and of interacting with the rest of a world.

Desired physical and visual phenomena include:

- Currents, circulation, vortices, interacting eddies and turbulence-like detail.
- Gravity, pressure, buoyancy and eventually viscosity.
- Solid boundaries and internal obstacles, terrain, containers and pools.
- Water surfaces that can slosh, form gravity waves, crash against walls, reflect back and forth, and eventually break and splash.
- Air above the water as a **real lower-density fluid**, not necessarily empty space or a mere visual background.
- Eventually, water/air interaction, surface tension, bubbles, droplets and more convincing free surfaces.
- Arbitrary camera **panning and zooming**, potentially over an enormous or apparently infinite world, without needing a full-resolution fluid grid everywhere.
- Interactive forces and controls: stirring with the mouse, adding impulses, vortices, sources of motion, etc.

An even broader *exploratory* aspiration mentioned in earlier discussions was a unified approach to phenomena such as smoke, steam, temperature, different viscosities, sediment/sand/quicksand and perhaps plasma. **Do not treat all these as already solvable by this one incompressible solver.** Solids/granular material, phase transitions, plasma and truly compressible flow may require additional models or coupled solvers. The current implementation is a **two-phase, variable-density, 2D incompressible fluid**. New material or exterior models remain separate learning milestones.

### What “apparently infinite” means

The user proposed progressively larger fluid cells outside the visible region, all still representing real coarse-scale fluid state. Panning into those regions would split/coarsen cells as appropriate, reconstructing plausible fine detail that wasn't previously simulated. Zooming in would similarly activate finer scales, so the grid density in **screen pixels** remains roughly constant while its **world-space** resolution increases.

The goal is **seamless perceived continuity**, not a claim to reproduce the uniquely correct microscopic history of an uncomputed infinite fluid. Coarse cells cannot retain every pair of small opposing vortices; fine detail reconstructed from them must necessarily be plausible rather than exact.

Two kinds of detail may ultimately coexist:

1. **Simulated detail:** actually evolved by fluid equations while active, including its interaction with other regions.
2. **Reconstructed detail:** created deterministically when refinement uncovers scales that were not actively simulated; constrained by the larger-scale flow and blended in smoothly.

Cache recently visited fine-scale regions to preserve local history across zoom/pan reversals. For old uncached regions, reconstruction can be deterministic and visually plausible, but not physically identical to a fully simulated world.

---

## 2. The decisions already made

### A. Start with incompressibility

The discussion briefly considered a fully compressible, unified fluid. The user explicitly chose to **stick with incompressible for now while learning**. Incompressibility is not water-only: low-speed air/smoke can also be modelled incompressibly, including variable-density two-phase air/water systems.

### B. Prefer a **square staggered MAC grid**

A triangular lattice, general node/vertex mesh and particle-based solver were considered. Current recommendation is a **Cartesian Marker-and-Cell (MAC) grid**:

- It is easier to index, interpolate, debug and build in plain C++.
- It is especially natural for calculating discrete divergence and applying pressure corrections.
- Every shared cell face has **one velocity value**, not separate values for each adjacent cell.
- A square naturally subdivides into four children, offering a relatively direct path to quadtree/tiled adaptive mesh refinement.
- More complex triangular/unstructured meshes are possible but do not inherently give a better beginner pressure solver; they complicate connectivity and coarse/fine transitions.
- The fluid state lives on the grid. Unused visual tracer particles have been removed.

For a `width × height` grid of cells:

- `u`: horizontal face velocities, `(width + 1) × height` scalars.
- `v`: vertical face velocities, `width × (height + 1)` scalars.
- `pressure`: one value at each cell centre, `width × height` scalars.
- `water_fraction`: one conserved volume fraction at each cell centre.
- Future `solid`/obstacle representation: cells and/or solid face fractions, ultimately with correct obstacle boundary conditions.

**Example:** A 64×64 grid has 65×64 horizontal face velocities and 64×65 vertical face velocities. The right face of cell `(x,y)` is literally the same `u` entry as the left face of cell `(x+1,y)`.

MAC positions (in world units, with origin at grid corner and `h = cell_size`):

```cpp
// u(x,y): (origin.x + x*h,       origin.y + (y+0.5)*h)
// v(x,y): (origin.x + (x+0.5)*h, origin.y + y*h)
// p(x,y): (origin.x + (x+0.5)*h, origin.y + (y+0.5)*h)
```

Positive Y convention must remain consistent across bitloop, pressure calculations, gravity and visuals. In the current MAC representation, increasing `y` moves from one grid row to the next; calling a face “top” vs “bottom” depends on the chosen world coordinate convention.

### C. World-space architecture independent of the camera

Keep camera/view logic outside the fundamental fluid mathematics. Velocity sampling should accept a **world-space position** and produce a world-space velocity. `origin` and `cell_size` must remain meaningful even if the grid later becomes tiled or adaptive.

Conceptual operations (general force injection does not exist yet):

```cpp
DVec2 sampleVelocity(DVec2 worldPosition) const;
void addForce(DVec2 worldPosition, DVec2 force, double radius);
void addVortex(DVec2 center, double radius, double strength);
ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6);
```

Keep rendering and experiment setup outside the fluid mathematics. The engine currently remains a concrete fixed grid; do not add adaptive interfaces before they are needed.

### D. Learn on the CPU, move selected stages to the GPU later

Use understandable `std::vector` arrays, ordinary loops and simple numerical methods now, even if inefficient. The GPU is a **later optimisation/backend**, not a prerequisite for continuing.

Candidate GPU-heavy stages: face-velocity advection, divergence, pressure solve, pressure-gradient correction, optional large particle populations. CPU likely manages world/tiles/refinement, input, camera and obstacle edits initially.

bitloop can target native OpenGL and Emscripten/WebGL2, so note that WebGL2 **does not support general compute shaders**. Native GL 4.3+ compute, WebGPU compute, or WebGL2 fragment-shader ping-pong are possible later, with different portability trade-offs. CPU double-precision world coordinates with local float-based GPU tiles may be useful for extreme zoom.

---

## 3. Current implementation

The engine lives in `FluidSim/FluidGrid.h/.cpp`, namespace `fluid`, with no Scene, camera or drawing dependency. The scene in `FluidSim/FluidSim.h/.cpp` owns the initial condition, viewport timing, rendering and failure reporting. [DEVELOPMENT.md](DEVELOPMENT.md) explains the boundary and existing optimisations in more detail.

The scene currently uses a 128×128 grid with `cell_size = 4.0`, spanning 512×512 world units. It starts with a sloped, rippled pool. Current density values are experimental: `air_density = 0.01`, `water_density = 100.0`. They are not prescribed physical constants. A new engine grid itself starts at rest with zero water fraction.

`viewportProcess()` caps catch-up at 0.05 seconds and divides that interval into equal steps no longer than 1/120 second. Each calls `fluid.step(step, 2000, 1e-8)`:

1. Advect staggered velocity using semi-Lagrangian backtracing and bilinear interpolation.
2. Apply vertical gravity to interior faces.
3. Rebuild material density/face coefficients, impose closed outer walls, solve pressure with matrix-free diagonal-preconditioned CG, and correct velocity.
4. Transport water fraction conservatively using that projected velocity, with internal CFL substeps as needed.

Material transport is skipped if projection fails; the scene pauses and reports the failure. There is no automatic rollback of velocity or pressure. The default engine solver budget/tolerance remain 200 and 1e-6; the scene explicitly chooses stricter values.

PCG reuses its buffers, predicts from the previous two pressures, caches the inverse diagonal and fuses several vector passes/reductions. The pressure operator uses a row-based interior loop and separate boundary handling. `applyPreconditioner` remains the future replacement point for a multigrid cycle. Ordinary PCG needs a fixed symmetric positive preconditioner on the mean-zero pressure subspace; changing boundaries or adding disconnected regions requires revisiting the operator and constant-mode handling together.

The scene displays water fraction through its shader surface and retains a native grid-size `grid` snapshot route. The project UI creates separate scenes for its viewports; it does not yet share one fluid world across views. Dye, particles, `advectDye` and the unused `FluidCell` placeholder have been removed. `sampleVelocity` and `addVortex` remain available for experiments.

## 4. Current limits and regression coverage

The earlier diagonal bilinear-interpolation bug is fixed. Project-owned tests now cover affine velocity sampling, hydrostatic equilibrium, conservative transport, repeated pool motion and failure behaviour. See [DEVELOPMENT.md](DEVELOPMENT.md) for how to run them and the limits of what they establish.

- Semi-Lagrangian velocity advection is numerically diffusive, so small vortices lose detail.
- Donor-cell material transport conserves water but smears the interface. A fraction between zero and one represents a cell average; it does not by itself implement a physical intermediate phase.
- There is no surface tension, viscosity, internal obstacle model, prescribed exterior pressure, trapped-pocket pressure model, temperature or phase transition yet.
- Both air and water currently participate in the whole closed-box incompressible solve. This is an explicit current model, not a universal atmosphere model.
- Finite solver tolerance leaves small divergence; large density ratios and grids can make diagonal PCG expensive or exhaust the iteration budget.
- Grid edges are currently closed walls. Future tile boundaries must exchange coherent neighbour state rather than become artificial walls.
- Revisit timestep ownership before sharing a fluid world between viewports. Keep simulation timing independent of rendering when introducing offline recording controls.

---

## 5. Essential mathematics (with notation explained)

**Teaching requirement:** Whenever introducing or using a mathematical equation, **always provide an adjacent legend defining every symbol**, including Greek letters, operators, subscripts, vector notation, units and any calculus ideas. The user is comfortable with algebra but wants to learn higher-level mathematics without unexplained notation. Explain intuition first, then equation, then how it maps to C++.

### A. Semi-Lagrangian advection

```text
previous_position = current_position - velocity_at_current_position * dt
```

In mathematical notation:

$$
\mathbf{x}_{\mathrm{prev}}=\mathbf{x}-\mathbf{v}(\mathbf{x})\Delta t
$$

**Legend:** **bold** `x` is a 2D position vector; `x_prev` is the backward-traced position; **bold** `v(x)` is the 2D velocity sampled at position `x` (world units per second); `Δ` is Greek delta, indicating a difference; `Δt` is the timestep in seconds. Subtracting velocity × time traces where fluid arriving here came from.

### B. Incompressibility / divergence

For square MAC cells:

$$
d=\frac{u_R-u_L+v_T-v_B}{h}
$$

**Legend:** `d` is cell divergence (per second); `u_R`, `u_L` are horizontal velocities on the right/left faces; `v_T`, `v_B` are vertical velocities on the two Y-normal faces (top/bottom naming assumes Y points up); `h` is square-cell width in world units. The target `d = 0` means no net local volume expansion or compression.

### C. Pressure projection (constant density normalised to 1)

$$
\nabla^2p=\frac{d}{\Delta t}
\qquad
\mathbf{v}_{\mathrm{new}}=\mathbf{v}_{\mathrm{old}}-\Delta t\,\nabla p
$$

**Legend:** `p` is pressure in normalised units; `∇` (“nabla”/“del”) is the spatial derivative operator; `∇p` is the pressure gradient (2D vector of pressure-change rates); `∇²p` is the Laplacian (sum of spatial second derivatives); `d` is divergence; `Δt` is elapsed time; **bold** `v_old` and `v_new` are velocity vectors before/after correction. For physical density `ρ`, the full constant-density form is `∇²p = (ρ/Δt)d` and the correction is `v_new = v_old − (Δt/ρ)∇p`.

This constant-density example explains the projection idea. The current solver extends it with shared inverse-density face weights. Normal-velocity zero at solid boundaries supplies the wall condition. The closed-wall, Neumann-boundary pressure equation has an arbitrary additive constant; only pressure differences matter. PCG explicitly removes this constant mode.

### D. Vortex initialisation

$$
\mathbf{v}(x,y)=(-dy,dx)\,\omega\,e^{-(dx^2+dy^2)/(2R^2)}
$$

**Legend:** `v(x,y)` is velocity at world coordinate `(x,y)`; `dx = x − center.x`, `dy = y − center.y` are offsets from the vortex centre; `ω` (omega) is vortex strength, approximately inverse seconds for this formula; `e` is Euler's number (~2.71828); `R` is the falloff-radius parameter; superscript `²` means squared. The vector `(-dy, dx)` is a 90° rotation of the offset, producing circular flow.

### E. Two-phase density

$$
\rho=\alpha\rho_w+(1-\alpha)\rho_a
$$

**Legend:** `ρ` (rho) is local density; `α` (alpha) is water volume fraction in `[0,1]`; `ρ_w` is water density; `ρ_a` is air density, using the same density units. `α=1` is all water, `α=0` all air, and intermediate values describe cells containing the interface. This cell-averaged relation is implemented in `rebuildPressureCoefficients`; the simulation's configured densities are experimental values.

### F. Future divergence-free procedural detail

A 2D **stream function** can generate fine-scale circulation:

$$
\mathbf{v}=
\left(\frac{\partial\psi}{\partial y},
-\frac{\partial\psi}{\partial x}\right),
\qquad \nabla\cdot\mathbf{v}=0
$$

**Legend:** `ψ` (“psi”) is a scalar-valued stream function of `(x,y)`; `∂` denotes a *partial derivative* (rate of change with respect to one coordinate, keeping the other constant); the two components form a 2D velocity; `∇·v` is divergence (the sum of the horizontal derivative of horizontal velocity and vertical derivative of vertical velocity). For a smooth stream function, that divergence is identically zero in the continuous mathematics. Numerically discretised fields and coarse/fine boundaries still require care.

Use this later for spatially/temporally coherent refinement detail that respects large-scale flow, rather than simply adding uncorrelated random vectors at newly created cells. A procedural stream function alone does not replace two-way fluid coupling or a pressure solver.

---

## 6. Planned physical development, starting from the current solver

### Near term — one understandable milestone at a time

The engine extraction and regression baseline are complete. Use [DEVELOPMENT.md](DEVELOPMENT.md) to choose one next experiment with the user; do not implement this list as one package.

- A general **strictly 2D force field**, with ordinary gravity as the baseline. For the Mandelbrot experiment, a smoothed depth slope determines in-plane acceleration. Do not introduce shallow-water equations or an evolving water-depth variable unless explicitly requested.
- Colour-only material labels for seeing motion; identical physical properties do not require separate pressure systems or distinct physical elements.
- Sharper conservative interface transport, followed by material-dependent surface tension as a separate milestone. The no-tension case must remain available.
- Explicitly different exterior models: fully resolved gas versus a specified ambient pressure. Enclosed bubbles need resolved gas or their own pressure/volume constraint; they must not silently become ambient exterior.
- Interior obstacles, viscosity and more materials only when an agreed experiment needs them.
- Stronger pressure preconditioning and selected GPU stages after profiling, preserving the model and regression cases.

Choose physical behaviour explicitly. Automatic optimisation may choose an execution strategy for that behaviour; it must not silently change the physical model. Keep new options limited to implemented cases, with a small demonstration and regression check for each.

### Long term — adaptive world and zoom

12. **Tiles and local grids with world-coordinate origins**, maintaining the same underlying MAC/pressure ideas.
13. **Hierarchical resolution:** quadtree-style subdivision into four square child regions; coarser state maintained outside the region of interest. Refine not just for camera position/zoom, but also near interfaces, strong vortices and obstacles.
14. **Resolution switching:** refinement (*prolongation*) and coarsening (*restriction*) that preserve meaningful coarse flow, with hysteresis so camera zoom near a threshold doesn't constantly split/merge cells.
15. **Conservative coupling:** coarse/fine face fluxes must match; pressure projection is nonlocal, so independent pressure solves on each resolution ring may give incorrect interface behaviour. Investigate multilevel/adaptive projection rather than merely interpolating cell velocities.
16. **Procedural fine-scale reconstruction:** stream-function or curl-noise-like divergence-free detail with consistent world-coordinate seeds, time-coherent evolution and bounded added energy. Cache previously simulated fine state where feasible. Coarse information alone cannot reconstruct exact discarded vortex history.
17. **Arbitrarily wide exploration:** keep a finite active computational budget; use coarse grids and ultimately procedural outer conditions beyond finite stored regions. The result is a **plausible approximation** of an enormous world, not an exact infinite Navier–Stokes solution.
18. **GPU solver backend:** parallel advection/projection/particles for active tiles, subject to native OpenGL vs WebGL2 vs future WebGPU constraints. Pressure iteration strategy may change (Jacobi, parallel red-black SOR, multigrid, etc.); keep algorithms and rendering decoupled.

---

## 7. Important design and physics constraints to preserve

- **Grid boundaries and physical obstacles are different concepts.** The current grid's solid outer walls are merely a learning-stage boundary condition. Internal tiles of a larger world need coherent neighbour data, not artificial walls at every tile edge.
- **Velocity is not density.** Velocity lives on faces; water fraction lives at cell centres and is transported as conserved material. Display colours or optional future tracers do not define fluid mass.
- **A fluid interface is not guaranteed by different densities alone.** A naive bilinearly advected density field tends to smear. A credible pool requires explicit interface tracking/conservation and an appropriate variable-density pressure solve.
- **Incompressible doesn't mean equal density.** Water and air can both be approximately incompressible at ordinary flow speeds while having different densities. Truly compressible air (pressure/temperature-dependent density and acoustic effects) is a different extension.
- **Waves aren't generated by applying gravity to a fully filled constant-density box alone.** Interface displacement, pressure imbalance and inertia generate gravity waves and reflections.
- **No need for every water molecule.** This is a continuum fluid solver; individual molecules are not simulated.
- **Adaptive resolution is harder than procedural rendering LOD.** The coarse representation must genuinely communicate motion/flux/pressure with fine cells if it is to count as a coupled fluid simulation.
- **Camera motion must not change the fluid's world-space identity.** Detail refinement may depend on the camera, but the velocity field's physical coordinates and stable procedural features should not move with it.
- **Maintain clear physical units.** Current `u`/`v` are world units per second, `cell_size` is world units, and `dt` is seconds; the current pressure correction uses the configured material densities.
- **Do not claim visual realism equals numerical accuracy.** It is legitimate to accept approximations for an interactive visual simulation, but document what is approximated.

---

## 8. Developer collaboration preferences

- The user is a competent C++ developer who knows bitloop well, but is learning computational fluid dynamics and advanced mathematical notation.
- Teach incrementally: explain **why** one operation is necessary, then introduce small, directly integrable C++ changes. Avoid large unexplained rewrites or dumping a production-grade solver at once.
- **Always include a legend for every mathematical equation**: define all symbols, subscripts, operators (such as ∇, ∇², ∂, Σ), units and physical meaning. Do not assume academic calculus notation is familiar. The user is interested in learning it properly.
- Respect their desire to **keep computation CPU-based while learning**. Optimisation/GPU changes come later unless a bottleneck prevents useful progress.
- Match current code conventions where possible (`DVec2`, `FluidGrid`, `cell_size`, `uIndex`, `vIndex`, `next_u`, `next_v`, etc.). Don't unnecessarily introduce an unrelated vector library or framework.
- For code comments: comment only non-obvious intent; use neutral voice; start lowercase unless capitalization is needed; for single-line `//` comments omit a final period; avoid trailing underscores in identifiers.
- Prefer diagnosing the user's **actual code** over speculating about their implementation. Keep an explicit distinction between *existing behaviour*, *bugs to fix*, *next tasks*, and *long-term aspirations*.
- Keep cleanup, new physics and optimisation in separate understandable changes. Explain the purpose and mathematical contract of optimised code without requiring the user to master every low-level detail. Do not replace the rendering architecture just to teach fluid maths.

---

## 9. Recommended next action for a new session

Read `AGENTS.md` and [DEVELOPMENT.md](DEVELOPMENT.md), then the relevant engine or scene file. Check the actual implementation before relying on older conversation examples.

The immediate objective is to keep the user involved and able to adapt the engine to their own experiments. Explain one agreed change, implement it locally, and demonstrate its effect with the existing regression set plus a focused new case when needed. Do not begin a GPU rewrite, general feature framework or additional physics merely because it appears in this vision.

If the user asks to pursue the adaptive/infinite camera idea later, handle conservation and multilevel pressure coupling explicitly rather than merely subdividing rendered samples.
