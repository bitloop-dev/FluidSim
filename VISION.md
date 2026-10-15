# VISION — Adaptive 2D Fluid Simulation in bitloop

**Status:** 11 October 2026  
**Purpose:** Self-contained handover for a new ChatGPT conversation or Codex session. Read this alongside the current `FluidSim.h` and `FluidSim.cpp`.  
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

An even broader *exploratory* aspiration mentioned in earlier discussions was a unified approach to phenomena such as smoke, steam, temperature, different viscosities, sediment/sand/quicksand and perhaps plasma. **Do not treat all these as already solvable by this one incompressible solver.** Solids/granular material, phase transitions, plasma and truly compressible flow may require additional models or coupled solvers. The immediate committed goal is a **2D incompressible fluid**, extended later to **two-phase incompressible water/air**.

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
- Marker particles are currently **visual tracers** rather than the fluid's primary mass/pressure representation.

For a `width × height` grid of cells:

- `u`: horizontal face velocities, `(width + 1) × height` scalars.
- `v`: vertical face velocities, `width × (height + 1)` scalars.
- `pressure`: one value at each cell centre, `width × height` scalars.
- Future `water_fraction`: one volume fraction at each cell centre.
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

Proposed persistent conceptual interface (not all methods exist yet):

```cpp
DVec2 sampleVelocity(DVec2 worldPosition) const;
void addForce(DVec2 worldPosition, DVec2 force, double radius);
void addVortex(DVec2 center, double radius, double strength);
void step(double dt);
```

Keep renderer/tracers insulated from internal grid layout wherever practical.

### D. Learn on the CPU, move selected stages to the GPU later

Use understandable `std::vector` arrays, ordinary loops and simple numerical methods now, even if inefficient. The GPU is a **later optimisation/backend**, not a prerequisite for continuing.

Candidate GPU-heavy stages: face-velocity advection, divergence, pressure solve, pressure-gradient correction, optional large particle populations. CPU likely manages world/tiles/refinement, input, camera and obstacle edits initially.

bitloop can target native OpenGL and Emscripten/WebGL2, so note that WebGL2 **does not support general compute shaders**. Native GL 4.3+ compute, WebGPU compute, or WebGL2 fragment-shader ping-pong are possible later, with different portability trade-offs. CPU double-precision world coordinates with local float-based GPU tiles may be useful for extreme zoom.

---

## 3. Exact state of the uploaded code

**Relevant existing files:** `FluidSim.h`, `FluidSim.cpp`. This summary describes those supplied source files, **not hypothetical improvements already applied**.

### `FluidSim.h`

Defines:

- `FluidCell { float pressure; float water_fraction; bool solid; }` — currently a **placeholder only**; it is **not used** in the working solver.
- `sampleScalar(...)` — bilinear scalar interpolation with boundary clamping; see **bug below**.
- `FluidGrid` with:
  - `width`, `height`, `cell_size`, `DVec2 origin`;
  - `std::vector<double> u, v, next_u, next_v`;
  - constructor taking `(int w, int h, double cell_size)`;
  - `uIndex(x,y)` and `vIndex(x,y)`;
  - `sampleVelocity(DVec2 pos)`;
  - `advect(double dt)`;
  - `addVortex(DVec2 center, double radius, double strength)`;
  - `project(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6)`, returning `ProjectionResult`;
  - `maxDivergence()`.
- `Particle` with `DVec2 pos`.
- `FluidSim_Scene` derived from `Scene<FluidSim_Scene>`; camera, navigator, fluid and particle storage.
- `FluidSim_Project` with a viewport count control.

**Actual configured simulation:**

```cpp
int grid_w = 128;
int grid_h = 128;
double cell_size = 4.0;
```

This gives a **512 × 512 world-unit** region. (Earlier tutorial examples mentioned 64×64 and 1,000 particles, but that is **not** the current source.)

### `FluidSim.cpp`

In `sceneStart()`:

- Creates a regular `(128 + 1) × (128 + 1)` particle lattice: **16,641 tracer particles**.
- Adds **five random vortices** with random world-space centres, radii and positive/negative strengths, once at start.

In `viewportProcess(ctx, dt)`:

```cpp
fluid.advect(dt);
double before = fluid.maxDivergence();
fluid.project(dt);
double after = fluid.maxDivergence();
blPrint() << "divergance before: " << before << "  after: " << after;

for (auto& particle : particles) {
    DVec2 velocity = fluid.sampleVelocity(particle.pos);
    particle.pos += velocity * dt;
}
```

`viewportDraw()` draws each particle as a small white ellipse via bitloop and draws the world axis, using the camera transform. The sidebar exposes camera controls. The navigator supports direct camera panning. The project UI supports 1–8 viewports, currently by creating a `FluidSim_Scene` for each viewport (i.e. separate scenes in this source, not an established single shared fluid world).

### Current numerical methods

**Velocity advection:** first-order semi-Lagrangian backward tracing, sampling the old staggered velocity field via bilinear interpolation, writing into `next_u`/`next_v`, then swapping buffers. It is easy to understand and typically stable, but numerically diffusive, so vortices gradually lose sharpness.

**Vortex initialisation:** defines a tangential velocity around the centre with Gaussian falloff, and adds the relevant component at each MAC face. It is applied during initialisation, not continuously injected every frame.

**Pressure projection:**

1. Set *outer* normal face velocities to zero (closed, solid walls).
2. Compute negative, scaled divergence as the pressure equation's right-hand side.
3. Solve with **single-threaded, matrix-free preconditioned conjugate gradient (PCG)** and diagonal preconditioning. Pressure, its previous value, and five working buffers are reused across calls. The last two pressures predict the initial guess; the residual is always rebuilt using the current density coefficients and velocities.
4. Remove the constant pressure mode by working in the mean-zero subspace. The right-hand side is made compatible with the closed walls; diagonal preconditioning projects both its input and output into that subspace.
5. Stop at the requested maximum-divergence tolerance, or the iteration cap. Recompute the true equation residual before accepting iterative convergence.
6. Correct interior `u` and `v` using the same inverse-density face weights as the pressure operator, then measure actual final divergence.

`ProjectionResult` reports `iterations`, `max_divergence` (inverse seconds), and `converged`. The default 200 is a maximum work budget, not a convergence guarantee; callers can increase it or choose a different absolute divergence tolerance. Invalid parameters return an unconverged result with infinite reported divergence. There is no internal obstacle handling yet, and only the outer rectangular computational domain is currently treated as solid.

`applyPressureOperator` sums centre-minus-neighbour pressure differences weighted by shared inverse-density face coefficients. The inverse diagonal is cached once per projection. `applyPreconditioner` accepts the residual mean and returns the residual/correction dot product, combining that reduction with mean removal. It remains the replacement point for a future multigrid cycle. Ordinary PCG requires a fixed symmetric positive preconditioner on the mean-zero subspace; future multigrid smoothing and transfers must preserve this property. Obstacles and adaptive interfaces will require revising the operator and preconditioner consistently. Disconnected closed fluid regions would each need their own constant-mode handling.

The pressure stencil runs serially because small thread-pool batches were slower on the current grid. PCG combines pressure/residual updates with residual statistics and uses standard reductions that allow reassociation. Scene updates divide each capped frame interval into equal substeps, avoiding nearly zero remainder steps that disrupt pressure prediction. The requested divergence tolerance and true-residual verification are retained.

**Diagnostics:** `maxDivergence()` scans every cell for the maximum absolute finite-difference divergence.

### Observed outcomes

The user reports the simulation is producing **nice visual results**, although it is somewhat slow. They supplied long diagnostics showing substantial divergence reductions: commonly pre-projection ~15–25 and post-projection declining from ~4 toward ~0.5 as the simulation evolved. This demonstrates that projection is having a useful effect, **not** that incompressibility has converged fully or that every numerical issue is solved. Earlier suggestions included trying more pressure iterations, using SOR and measuring pressure residuals.

The previous 128×128×200 SOR setup performed **3,276,800 pressure-cell visits per simulation step**, not counting advection, interpolation, tracers or rendering. PCG has several array passes per iteration, so compare elapsed time at the same final divergence rather than comparing iteration counts.

---

## 4. Known code issues / short-term clean-up

These are **not yet fixed in the supplied files** unless the user subsequently modifies them.

### Critical interpolation bug — fix first

In `FluidSim.h`, `sampleScalar(...)` currently calculates:

```cpp
double a = std::lerp(at(x0, y0), at(x1, y0), tx);
double b = std::lerp(at(x0, y0), at(x1, y1), tx); // incorrect lower-row interpolation
```

The second sample should interpolate across **the row at `y1`**, not diagonally:

```cpp
double a = std::lerp(at(x0, y0), at(x1, y0), tx);
double b = std::lerp(at(x0, y1), at(x1, y1), tx);
return std::lerp(a, b, ty);
```

This matters to *all* velocity sampling, particle movement and advection. Recheck simulation behaviour and divergence after fixing it.

### Other immediate improvements

- `project()` reuses pressure and PCG working buffers, predicts pressure from the previous two solves, and resets that history when the pool is initialized or no pressure correction is needed.
- Logging divergence with `blPrint()` **every frame** may noticeably impair performance. Log less often or expose counters in ImGui.
- Divergence is measured **before** `project()` enforces solid outer-wall velocities, but *after* projection with those walls already applied. For a fair diagnostic, explicitly enforce the same boundary conditions **before** the “before” measurement. Keep clear which errors are due to advection, wall correction or projection.
- Benchmark separately: `advect`, `project`, particle updates, and `viewportDraw`. Only then optimise what matters.
- `project()` now uses diagonal PCG with a divergence tolerance and iteration cap. Diagonal preconditioning is a baseline; multigrid preconditioning remains the intended stronger accelerator.
- `sceneStart()` loops use `y <= grid_w` and `x <= grid_h` rather than `y <= grid_h` and `x <= grid_w`. Both dimensions happen to be 128, so there is no effect now; fix before supporting nonsquare grids.
- Random vortex radius is drawn from a range that includes zero. Clamp to a small positive minimum or guard `radius > 0` to prevent division by zero if that edge case occurs.
- Tracers have no wall-collision or boundary-wrapping logic; clamped velocity sampling outside the grid is **not** a physical boundary condition for particles.
- The simulation currently advances in `viewportProcess()`, while `sceneProcess()` requests redraws. Revisit timestep/update ownership if integrating multiple views, sharing a fluid world, or ensuring a single simulation step independently of rendering. The current project creates separate scenes per viewport; do not assume they all reference one shared simulation.
- Use a fixed simulation timestep/substeps when appropriate, rather than assuming the render-frame `dt` is always suitable. This will become more important as forces, gravity, free surfaces and level-of-detail transitions are introduced.
- `FluidCell` (pressure/water fraction/solid) exists as an unused placeholder; don't mistake it for an implemented multiphase model.

### Numerical limitations already understood

- Semi-Lagrangian advection introduces numerical diffusion and does **not** by itself enforce incompressibility.
- A finite-iteration Poisson solve leaves residual divergence.
- Current sample clamping and solid outer walls are temporary choices; in an adaptive/infinite world, a grid tile edge does *not* automatically represent a physical wall.
- A perfectly steady uniform velocity field should not visibly evolve merely from advection. The first vortex field has spatial variation, so it is useful for testing.
- Adding arbitrary velocities/forces may create divergence, which pressure projection should subsequently remove as well as practical numerical tolerances permit.

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

The current matrix-free PCG solver approximates this pressure equation using the same adjacent-cell differences. Normal-velocity zero at solid boundaries supplies the wall condition. The all-solid, Neumann-boundary pressure equation has an arbitrary additive constant; only pressure differences matter. PCG explicitly removes this constant mode.

### D. Vortex initialisation

$$
\mathbf{v}(x,y)=(-dy,dx)\,\omega\,e^{-(dx^2+dy^2)/(2R^2)}
$$

**Legend:** `v(x,y)` is velocity at world coordinate `(x,y)`; `dx = x − center.x`, `dy = y − center.y` are offsets from the vortex centre; `ω` (omega) is vortex strength, approximately inverse seconds for this formula; `e` is Euler's number (~2.71828); `R` is the falloff-radius parameter; superscript `²` means squared. The vector `(-dy, dx)` is a 90° rotation of the offset, producing circular flow.

### E. Future two-phase density

$$
\rho=\alpha\rho_w+(1-\alpha)\rho_a
$$

**Legend:** `ρ` (rho) is local density; `α` (alpha) is water volume fraction in `[0,1]`; `ρ_w` is water density (~1000 kg/m³); `ρ_a` is air density (~1.2 kg/m³). `α=1` is all water, `α=0` all air, and intermediate values describe cells containing the interface. This is the *future* model, not currently implemented.

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

### Near term — continue learning, CPU only

1. **Fix and validate the bilinear sampler** (see bug above). Verify particles move smoothly and divergence behaviour changes as expected.
2. **Make profiling/diagnostics inexpensive and reliable**: pressure buffers are now reused; enforce boundaries before comparison; measure advection, projection, particles and rendering independently; test PCG tolerances, iteration limits, convergence and timestep stability.
3. **Interactive force injection:** mouse-drag forces in world space, adding impulses to MAC faces using a distance-based kernel; project afterwards; separate velocity impulses from sustained acceleration (units/timestep matter).
4. **Visual debugging:** toggleable velocity arrows, pressure/ divergence heatmaps, solid-boundary display, and particles. Keep this optional and simple.
5. **Introduce interior solid obstacles:** initially axis-aligned cell masks with no-penetration face velocities; modify divergence/pressure-neighbour handling appropriately. Later use fractional/cut faces for smoother obstacles and moving boundaries.
6. **Introduce gravity** and clear force/integration conventions. In a completely filled, constant-density closed incompressible domain, uniform gravity alone won't create water waves: hydrostatic pressure can balance it. A **moving water/air interface** is necessary for sloshing/waves.

### Medium term — pool, atmosphere and waves

7. **Two-phase incompressible, one-fluid formulation:** a shared velocity field and shared pressure solve, but an advected water fraction `alpha` marking water vs air. Both phases can be treated as incompressible while having very different densities and viscosities.
8. **Variable-density pressure projection:** pressure acceleration depends on `1 / rho` at faces. The present constant-density Poisson solver will **need revision**; the MAC grid remains useful. Air/water density ratios can make the numerical solver significantly more challenging.
9. **Sharp, mass-conserving interface tracking:** investigate Volume of Fluid (VOF), ideally geometric/interface-aware advection, so the surface does not blur into a grey density gradient. This is substantial additional work beyond advecting a scalar with bilinear sampling.
10. **Gravity waves and sloshing:** add a pool, perturb the interface, let gravity/pressure/inertia generate propagating waves; support reflection off walls, then progressively tackle breaking waves, droplets, surface tension and air entrainment.
11. **Further physics as needed:** viscosity, buoyancy/temperature, possibly surface tension and external rigid-body interactions. These are separate milestones, not preconditions for the first moving water surface.

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
- **Velocity is not density.** Rendering 16,641 tracers doesn't make them fluid mass, and their spacing does not automatically measure density. A future water fraction is separately advected as conserved material.
- **A fluid interface is not guaranteed by different densities alone.** A naive bilinearly advected density field tends to smear. A credible pool requires explicit interface tracking/conservation and an appropriate variable-density pressure solve.
- **Incompressible doesn't mean equal density.** Water and air can both be approximately incompressible at ordinary flow speeds while having different densities. Truly compressible air (pressure/temperature-dependent density and acoustic effects) is a different extension.
- **Waves aren't generated by applying gravity to a fully filled constant-density box alone.** Interface displacement, pressure imbalance and inertia generate gravity waves and reflections.
- **No need for every water molecule.** This is a continuum fluid solver. The particles currently shown are markers for the computed velocity field.
- **Adaptive resolution is harder than procedural rendering LOD.** The coarse representation must genuinely communicate motion/flux/pressure with fine cells if it is to count as a coupled fluid simulation.
- **Camera motion must not change the fluid's world-space identity.** Detail refinement may depend on the camera, but the velocity field's physical coordinates and stable procedural features should not move with it.
- **Maintain clear physical units.** Current `u`/`v` are world units per second, `cell_size` is world units, and `dt` is seconds; initial pressure code normalises density to 1.
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
- The existing bitloop simulation renders with `ctx->fillEllipse`, `ctx->transform(camera)` and `ctx->worldHudMode()`, with ImGui camera controls. Do not replace the rendering architecture just to teach fluid maths.

---

## 9. Recommended next action for a new session

**Start by reading `VISION.md`, `FluidSim.h` and `FluidSim.cpp`.**

The immediate follow-up to the last conversation is **not** a GPU rewrite. The user has a functioning, visually pleasing, somewhat slow CPU MAC-grid fluid. A good continuation is:

1. Fix `sampleScalar`'s diagonal bilinear interpolation bug.
2. Make a few low-risk CPU cleanliness/performance improvements and benchmark, if the user wants.
3. Implement **interactive mouse-driven force injection** into `u`/`v` and observe the resulting flow after projection.
4. Then progress incrementally to obstacles and, later, actual water/air free surfaces.

If the user instead asks to pursue the adaptive/infinite camera idea, work from the world-space MAC grid towards tiled refinement, but explicitly handle conservation and multilevel pressure coupling rather than just subdividing cell samples.

**Suggested short handover prompt:**

> I have a working 2D MAC-grid incompressible fluid simulator in my C++ bitloop framework. Read `VISION.md` and the attached `FluidSim.h`/`.cpp` for the current architecture, identified bugs, learning preferences and long-term plans. Continue incrementally from the *actual code*, explaining the physics and any maths with a full symbol legend. We want interactive forces next, then obstacles, two-phase water/air and waves; much later, camera-dependent adaptive zoom/pan and GPU acceleration.
