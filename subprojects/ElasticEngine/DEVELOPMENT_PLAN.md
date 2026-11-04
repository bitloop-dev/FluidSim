# ElasticEngine development plan

Status: selectable impact and cup-drop experiments with simultaneous spring/contact response, continuous collision checks, configurable minimum node spacing, stable node slots and tensile bond fracture. Updated 2026-11-04.

## Immediate objective

Build a simple, independent 2D spring network: nodes, bonds with resting lengths, and a mechanical update that tries to maintain those distances. Keep the spring physics compatible with later FluidGrid coupling.

There is no area-conservation requirement for this standalone experiment. Do not add fluid simulation, trapped-volume constraints or a balloon-specific pressure approximation to make a spring loop behave like a filled balloon before it contains simulated fluid.

This plan records direction. Implement one agreed increment at a time; the later integration notes are compatibility requirements, not permission to build those systems now.

## Project context

- Work inside `FluidSim/subprojects/ElasticEngine`.
- `ElasticEngine/ElasticTypes.h` holds node/bond data, `ElasticEngine.h` exposes the engine and scene, `ElasticSprings.cpp` applies spring/damping response, and `ElasticCollisions.h/.cpp` owns collision search and response. `ElasticSum.h` provides compensated accumulation for cancelling responses. `ElasticEngine.cpp` orders the physics steps and implements scene controls; `ElasticScenarios.cpp` creates the experimental geometry.
- The parent already includes `bitloop_add_dependency(FluidSim subprojects/ElasticEngine)`. Use the existing parent build/launch context. No AGENTS.md, Codex or skill-workflow changes are required.
- Keep scene setup and controls in the scene, and spring mechanics in the `ElasticEngine` core. Drawing currently delegates to `ElasticEngine::draw`; there is no separate `ElasticNetwork` API.
- Keep local tests and diagnostic evidence in this subproject. Preserve standalone project support without claiming it was independently built during this planning task.
- The former grid-elasticity experiment remains on `fluid-elasticity`. FluidSim's baseline is `062a320` plus pressure precision/stagnation fix `7b93aa7`; this experiment does not depend on the old elastic solver or its ingredient restrictions.

Read [FluidSim's development guide](../../DEVELOPMENT.md) when integration begins. Existing project instructions continue to apply.

## Small starting model

A node stores world-space position, velocity, an active flag and a fixed flag. Fixed nodes are anchored supports with zero motion; they are distinct from the intended massless, fluid-carried markers. Each material bond stores its two endpoint IDs, rest length, stiffness, axial damping and tensile breaking strain. Damping responds to relative velocity along a surviving bond, preserving shared drift. `addBoundary(a, b)` explicitly marks a collision-only segment in the same packed collection; its spring, damping and fracture parameters are ignored. Fixed endpoints do not implicitly turn a material bond into a permanent boundary.

Store each physical bond once in a flat collection. Evaluate its response once and apply equal/opposite velocity changes to its endpoints. There are no per-node adjacency lists yet.

Material bonds persist until explicitly removed, their tensile breaking strain is exceeded, or their stiffness becomes zero. Infinity (the API default) disables tensile breaking, but does not override zero-stiffness release. A positive-duration step removes zero-stiffness material bonds before collision search, so no segment barrier, axial damping or restoring force survives. Nodes and their current velocities remain; minimum node clearance remains independent. There is no rebonding: raising stiffness cannot restore released bonds or their old shape memory. A resting length around 10 world units is a useful scene setting, not a grid-alignment rule. Shape memory is local to the bonds' rest data; nodes have no automatic target world positions.

### Current storage and update contract

- `NodeId` is a 32-bit index into contiguous node slots. Insertion may reallocate the vector, but surviving node IDs do not change. Retain IDs rather than pointers or references across insertion.
- `removeNode(id)` removes its incident bonds, marks its slot inactive and puts the ID on a free list. Subsequent insertion reuses vacant slots. An ID expires when its node is removed; an old ID must not be used after reuse.
- Bonds and explicit boundaries remain packed. `removeBond(index)` replaces the removed entry with the last entry and pops the vector. Automatic fracture and zero-stiffness release compact survivors in their existing order. Indices can change after node removal, bond removal, release or fracture; they are not persistent handles.
- Supply live endpoint IDs and edit topology between steps. Node deletion scans the bond list once; bond deletion is constant-time. Stepping skips inactive node slots but still scans the allocated slots. Slots are retained for reuse, not compacted.
- Defaults preserve the practice scene: rest length `10.0`, stiffness multiplier `1.0`, damping `0.99`. Stiffness multiplies normalised extension `(distance - rest_length) / rest_length`; it is not yet a physical force-per-length parameter.
- A positive-duration `step(dt, acceleration)` first releases zero-stiffness material bonds, applies acceleration to mobile nodes, checks and adjusts velocities for contact, moves nodes, then removes overstretched material bonds and applies the surviving spring/damping response. Explicit boundaries participate only in contact. Each response pass reads a shared state and accumulates equal/opposite impulses before changing velocities; an early bond cannot change the damping seen by a later bond. A failed contact solve restores motion but retains the requested zero-stiffness release; `step(0)` leaves all state unchanged.
- Strong damping uses `ceil(dt * maximum sum of incident damping)` inexpensive velocity-only passes. This bounds the damping modes without repeating positions or collision searches. Elastic kicks are accumulated separately and applied after damping, preserving the original isolated-bond update for damping at most one. Nonnegative damping is assumed. This bound controls damping, not arbitrary explicit spring stiffness.
- Compensated sums retain cancelling contributions whose roundoff otherwise seeds large changes at nearly coincident endpoint contacts. Centered contact interpolation and symmetric arithmetic also avoid a preference for either bond endpoint. These are numerical measures, with no scene-specific mirroring or symmetry constraint.
- `step()` still defaults to one playground time unit. Fractional `dt` scales acceleration, motion, spring impulses, axial damping and continuous collision search consistently. `advance(dt, acceleration)` subdivides using the maximum incident `stiffness / rest_length` sum, retaining the existing explicit integrator. The scene advances one playground unit per output update; these units are not calibrated seconds or physical material moduli.

Optional native regression tests are enabled with `ELASTICENGINE_BUILD_TESTS=ON`. Build `ElasticEngineTests` in the parent Codex build and run `ctest --test-dir build/msvc-release-codex -R '^ElasticEngineTests$' --output-on-failure` from FluidSim. They cover vector growth, repeated slot reuse, incident-bond deletion, packed bond removal, restoring direction, relative damping, 4,096 steps of triangle relaxation/shared drift and 1,024 steps of detached-node motion. This does not establish stability for arbitrary stiffness or network connectivity.

Open chains, loops and networks are all valid. Triangles are optional braces. A chain has no bending stiffness unless we add such a term, and an unbraced quadrilateral can shear while preserving its edge lengths. These are expected properties of distance springs, not bugs to hide with shape restoration.

The standalone playground currently uses equal unit test inertia so that nodes can move before FluidGrid is coupled. This is not a material mass stored on each node. The intended coupled membrane remains massless and has zero displaced area; fluid motion supplies inertia through the coupled solve. This includes fluid around open fibres, not only fluid trapped inside closed loops. Do not duplicate enclosed fluid mass on the nodes, or confuse a pinned node with a massless immersed boundary.

The useful separation is small: calculate spring response, then advance node motion. Keeping those operations separate lets a later fluid coupling supply the motion/force balance. There is no need for a general solver framework or external-engine plugin API now.

## First experiments

1. One spring: pull its endpoints apart or together and release it. Check restoring direction, equilibrium, symmetry and the effect of damping.
2. A short chain: demonstrate connected motion and optional explicit attachments.
3. A closed loop and a small braced patch: demonstrate local elasticity and the different flexibility produced by connectivity.

Use a fixed physics timestep and simple readable integration first, within its demonstrated stable stiffness range. Check oscillation, passive damping, finite state and sensitivity to a smaller timestep. Do not promise arbitrarily stiff springs at a large timestep; revisit integration only when measurements justify it.

A standalone loop is allowed to change area or flatten. There is no trapped incompressible fluid in this experiment yet. Measure area later as a fluid-coupling diagnostic, not as a spring-engine constraint.

### Tensile fracture

Each bond's `breaking_strain` is the maximum permitted `(length - rest_length) / rest_length`. It breaks only when that value is exceeded. `0.10` allows 10% extension; infinity keeps the bond unbreakable. The nonnegative threshold is independent of stiffness and damping. Compression alone does not break a bond, and triangles/angles play no role in this rule.

Failure is evaluated at the end of the collision-checked move, using the distance already needed for the spring response. All decisions see the same positions. Surviving bonds and their cached directions are compacted together before the damping passes; failed bonds contribute neither spring impulses nor damping weights. Deletion applies no extra velocity kick. Stored elastic energy of a broken bond is treated as dissipated fracture energy. This is step-boundary failure, not an exact within-step rupture time.

Node slots, velocities and small fragments are retained. A single node and a two-node segment remain supported by the standalone collision engine. Deciding whether a future massless, unconnected structural node has useful payload is a separate cleanup policy, not a fracture law. No connected-component tracking or minimum-fragment-size rule is required here.

`ElasticEngine_Scene::Config::breaking_strain` controls the large ball and all directions of the cup lattice, and currently defaults to the user-selected `0.02`; the projectile remains unbreakable. The two-ball defaults of stiffness `1.0`, damping `0.99` and projectile speed `2.0` are unchanged. The initially considered 30% limit survives this impact, whose unbreakable peak extension is about 18.2%.

Regression coverage includes exact threshold boundaries, compression, zero stiffness, simultaneous failures, survivor compaction, and 1,024-step isolated/two-node fragment motion. The breakable impact and mirrored/reordered/endpoints-reversed variants run for 2,048 steps alongside the old unbreakable cases. A compensated moving-frame sum in collision search/response preserves order independence after fragmentation. See `.codex/runs/run0006_20261104_153145/README.md` for results.

### Current collision model

Every active node is tested against nearby nonincident bonds, including bonds in the same network. A triangle's vertex can therefore collide with its opposite edge and resist inversion. Contact does not create bonds. Detached nodes also collide; their search entries are degenerate segments. The initial network should be nonintersecting: this is crossing prevention, not an arbitrary-overlap untangler.

`setMinimumNodeDistance(distance)` adds a swept, frictionless centre-to-centre clearance between all active nodes, including bonded neighbours and detached fragments. The raw engine defaults to zero additional clearance; both playground scenarios use `Config::minimum_node_distance = 8.0` against a spring rest length of 10. Start with separated geometry. Small spacing overlaps recover gradually; increasing clearance into a densely overlapping or intersecting network is not an arbitrary untangling operation. It is independent of stiffness, so setting stiffness to zero does not disable node contact. It prevents crowding without integrating an extra stiff repulsion force.

With positive clearance, every node receives a point entry in the existing spatial search. Each node pair is checked once per discovery pass, and those pairs replace the bond endpoint-cap checks. Bond-interior crossing tests remain necessary: a long or stretched bond can block a node even when its endpoints are distant. Minimum node spacing bounds local point density, not the number of swept candidates, long bonds or iterations required by a packed contact network.

The broad phase stores swept segment bounds in 10-unit spatial cells using a flat chained hash table. Bucket heads and entries retain capacity; rebuilding inserts entries directly instead of sorting the whole table on every discovery pass. Queries check the full cell key and deduplicate segments shared by several buckets, so hash collisions cannot merge physical cells or lose candidates. Contacts are still sorted into their existing canonical order before response. A translating search frame removes the mean nodal velocity, so shared drift does not inflate the swept bounds. Long/stretched bonds occupy every cell of their swept bounding box. This remains a uniform-cell search over free-moving nodes, with no body/surface classification. Large relative travel or crowded buckets can still increase work.

This search-only optimisation reduced the measured 1,024-node cup update averages from 6.96 to 1.96 ms at stiffness 30, and from 9.72 to 2.47 ms at 100, over the existing 2,048-update tests. Reported non-timing regression results match the baseline. That run changed only search: its contact response, tolerances and 0/0.1 convergence limit were unchanged, and its solver acceleration trials were discarded. Practical tolerance and overlap recovery were added subsequently (see below). See `.codex/runs/run0009_20261104_contact_efficiency/README.md` for measurements and limits.

The narrow phase solves moving point/segment collinearity over the full step, including both quadratic roots, and handles endpoint/collinear contacts with swept endpoint tests. Node pairs use the configured clearance (at least the numerical gap). Bond interiors retain the `0.001` world-unit numerical contact gap, independent of rendered node radius and line thickness. This is node-to-node spacing, not thick segments or circular nodes against walls: the cup floor still stops node centres at its numerical gap. Resting contact prefers the closest interior feature over an endpoint cap, and interior normals come directly from the edge to avoid subtracting nearly equal positions.

Response uses **speculative, frictionless, inelastic velocity constraints** with equal unit mobile-node inertia and immobile fixed supports. Impulses distribute to bond endpoints using the contact interpolation weights. Simultaneous corrections use a conservative bound on the coupled contact matrix; a backtracked step length improves slow convergence while respecting a kinetic-energy bound. Only nodes involved in contact participate in these iterations. Equivalent endpoint constraints are merged before response, so several incident bonds do not duplicate a point contact. Internal corrections preserve total linear momentum. Ordinary collision response is dissipative; small overlap recovery deliberately supplies separation work, accounted for separately in the step-length energy bound. This is numerical stabilization, not elastic spring energy. Fixed supports absorb reaction momentum; their contacts use the stationary world frame for the energy bound. Springs supply rebound. This intentionally approximates impact timing rather than advancing the whole network to each exact collision time; it does not exactly conserve angular momentum.

Node spacing accepts a projected-motion residual of `0.00025 * clearance` per substep and recovers 20% of existing overlap per substep. At clearance 8, the residual is 0.002 world units and sustained overlap is bounded by 0.01 world units (0.125% of clearance) for initially separated geometry. Inner iterations aim for half the acceptance tolerance, leaving margin for the final swept check. Error is measured in distance and converted to speed with the actual substep duration.

Bond interiors and numerical endpoint caps use a separate 0.0001 world-unit tolerance, smaller than their 0.001 safety gap. Recovery targets that gap in one step. The existing moving-segment quadratic crossing tests and final trajectory rechecks still reject wall/membrane crossings; no unchecked position corrections are applied. An approaching trajectory that stays on its original side can temporarily end closer than the target gap. This does not make nodes into circles against walls.

The solver retains discovered constraints within the step and rechecks corrected trajectories. It allows at most 64 discovery passes with up to 256 simultaneous correction iterations each, exiting early when resolved. If it cannot find an accepted move, it restores the original velocities and reports nonconvergence before positions advance. `getCollisionStats()` exposes passes, narrow-phase candidate tests and discovered contacts for profiling. No friction, surface culling or FluidGrid coupling is included; tensile fracture is a separate stage after accepted motion.

The native regression suite now covers fast moving/stretched/rotating bonds, collinear endpoints, unequal parallel bonds, isolated nodes, opposite-edge self-contact, removal/reuse and a 1,200-node separated-patch search. The production ball scene runs for 2,048 steps, plus 1,024-step rotated/drifting and four-times-faster impact variants. Checks include finite state, linear momentum, triangle orientation throughout each committed linear trajectory, and nonadjacent bond intersections after every update. The symmetry regression also runs the actual scene for 2,048 steps with reversed node/bond storage, reflected geometry and reversed bond endpoints. A 32-spoke damping test checks energy, shared drift and traversal independence. Measurements and approximation limits are recorded in `.codex/runs/run0005_20261104_024939/README.md`; the earlier collision baseline is in `run0003_20261104_021929`.

## Compatibility with FluidGrid

The intended division is:

| Part | Responsibility |
| --- | --- |
| ElasticEngine | Nodes, bonds, elastic response and eventually contact |
| Standalone playground | Test inertia, external inputs, controls and rendering |
| FluidGrid | Fluid mass, composition, momentum, pressure and material transport |
| Later coupling | Make boundary motion, fluid forces and permitted fluid transfers consistent |

ElasticEngine should not depend on a concrete FluidGrid backend. The fluid backend will need its own boundary-aware pressure/transport work when integration is agreed. Initially target one uniform backend rather than requiring every CPU/GPU/adaptive implementation at once.

### Where area conservation comes from

For a closed impermeable boundary moving with an incompressible fluid, the enclosed area should remain constant in 2D, within numerical accuracy. Springs may change the shape; the fluid pressure supplies the response that prevents net compression of the enclosed fluid. That is the intended source of area conservation. ElasticEngine does not need a second area-preservation rule.

The fixed Eulerian grid cells always have the same geometric area. The pressure solve constrains fluid fluxes; it does not automatically recognise an arbitrary polygon made of nodes. We must teach pressure and transport about the moving boundary so fluid cannot cross it relative to its motion. The spring loop cannot be advanced independently and assumed to inherit incompressibility afterwards.

Merely interpolating grid velocity onto nodes can still produce enclosure-volume errors in a discrete method. [Bao et al., divergence-free immersed-boundary coupling](https://arxiv.org/abs/1701.07169) documents this issue and an approach to improving it. This is background for the later numerical choice, not a commitment to implement that particular method.

Pressure should account for the spring response, and the network should respond to the fluid. A compatible coupling may change how nodes are advanced or how spring forces are solved at a timestep. Keeping spring response separate now preserves that option; it does not make the eventual coupling free or guarantee that every standalone integrator can be reused unchanged.

### Boundaries and fluid state

A bond is a mechanical connection. Whether that segment also blocks fluid is a separate physical choice, to be made when coupling is introduced. Internal braces need not trap separate fluid compartments. An open barrier blocks crossing locally, but fluid can travel around its ends. A closed barrier can contain fluid.

Keep the full FluidGrid active, including fluid on both sides of a zero-thickness membrane. Existing substances and their fluid properties remain in FluidGrid, as should future chemical/thermal fields. There is no second copy of the fluid contents or chemistry on the spring network.

Initially use the existing incompressible model for both liquid and simulated gas. Compressible gas, permeability, chemical changes and active muscle contraction are later experiments. Tensile rupture and explicit bond removal are available in the standalone engine; making rupture open fluid connections consistently remains integration work.

The coupling must eventually distinguish fluid on opposite sides of a segment, handle pressure jumps and moving-boundary fluxes, and specify supported thin-gap/subcell resolution. These are integration questions to resolve with a small prototype, not extra data structures or algorithms required for the first spring scene.

## Later integration sequence

Once the simple spring mechanics are useful, choose one integration step at a time:

1. A stationary boundary in FluidGrid: verify that fluid cannot cross it, including when it is oblique to the grid.
2. A boundary undergoing prescribed area-preserving motion: verify conservative transfers as it sweeps across cells.
3. A spring loop coupled to fluid: verify deformation, pressure feedback, enclosed area, contact and density-dependent motion.

Check both enclosed area and cross-boundary material exchange. Equal leakage in both directions can preserve area while failing containment. Include repeated motion/contact and conservation checks over a physically meaningful time, not just a plausible screenshot.

Measure complete coupled frame cost, including global pressure and transport. The old high-stiffness performance problem is not proven solved by making the spring network inexpensive. With no network, existing FluidSim scenarios should retain their physics and avoid additional boundary work.

Longer-term fibres and muscles can be built from the same nodes and bonds, for example by actively changing preferred lengths. Those ideas do not require organ-specific definitions, rigid-body promotion, adaptive grids or general material conversion in this experiment.

## Standalone scenarios

`ElasticScenarios.cpp` contains scene setup, separate from the engine's ordered physics steps. The **Experiment** panel selects a scenario, adjusts stiffness live from 0 to 100 (logarithmic slider), adjusts gravity, and restarts the selected scenario while retaining its controls. Selecting a different scenario restores its defaults and camera framing.

- **Two Balls** preserves the 151-node target and 31-node projectile, their 10-unit equilateral springs, default stiffness 1, axial damping 0.99 and rightward projectile speed 2. The target uses the configured tensile threshold (currently 0.02); the projectile remains unbreakable under tensile fracture but releases at zero stiffness. Gravity defaults to zero.
- **Cup Drop** has exactly four fixed nodes at `(-220, -180)`, `(-220, 200)`, `(220, 200)`, `(220, -180)` and three explicit boundary segments. Positive Y points downward. A staggered 32-by-32 lattice (1,024 mobile nodes, 2,945 equal-rest-length springs) begins above the floor, using rest length 10 and row spacing `10 * sqrt(3) / 2`. Lattice defaults are stiffness 30, damping 0.99 and gravity 0.02. All horizontal and diagonal material bonds use the configured breaking strain. The reference stiffness regression explicitly supplies infinity to keep its lattice intact.

Zero stiffness releases material bonds on the next positive-duration physics step. Their spring forces, axial damping and segment collisions all disappear. Explicit boundaries and minimum node spacing remain. Raising stiffness changes only surviving material; use Restart to rebuild the lattice. Positive stiffness still uses the existing spring/damping/fracture laws, with no new near-zero cutoff or relaxation rule. There is still no fluid pressure, displaced fluid volume or area constraint; free collision particles are not a liquid simulation. Packed node contacts remain a demanding benchmark. Do not infer coupled-fluid behaviour or performance from them.

Packed contact remains the main performance limit. Practical tolerances and overlap recovery now complete the default 400-update scene: about 37 ms per update overall, with the crowded final 100 averaging 84 ms. The comparable pre-failure window (updates 251–282) improved from 325 ms to 71 ms. The former solver stopped during update 283. The new run's maximum measured node overlap was 0.00877 world units; containment, surviving-triangle orientation, finite state and fixed anchors passed. These are single-machine CPU physics timings, not a realtime guarantee. See `.codex/runs/run0012_20261104_202231/README.md` for the full comparison and longer-run limits. The default scene also passed 800 updates; a zero-stiffness variant passed checks through 832 before its optional longer run was stopped. Dense settling remains slow (up to about 164/326 ms respectively in those confirmation windows).

The scene still stops physics and displays an error if its contact solve cannot produce a checked trajectory. No fallback accepts an unchecked move. Changing stiffness clears the stop but cannot rebuild released structure; restarting rebuilds the selected scene. The normal regression suite includes the actual 400-update scene and unbreakable 30/100 stiffness drops for 2,048 updates, plus overlap recovery against fixed and moving membranes. `ElasticEngineTests --current-scene` runs the focused default-scene gate (an optional following integer overrides 400 updates); `--material-cup` extends stiffness 0/30 with breaking strain 0.02 to 2,048 updates. Earlier pre-recovery failures at updates 167/283 are retained in run0010 as historical evidence.

Node motion in this standalone driver still uses unit test inertia. The eventual fluid-attached model derives node motion from the coupled fluid velocity and transfers structural/contact forces to FluidGrid; it does not duplicate fluid momentum on nodes. A cached node velocity would then be derived data.

Storage/removal, collision handling, minimum node spacing, tensile rupture, anchored supports and selectable drop/impact tests are implemented. Node clearance is structural geometry; it does not assign material mass or displaced fluid volume to nodes. Fluid pressure, area conservation, thick-segment contact and the detailed FluidGrid coupling remain separate future steps.
