# Material physics roadmap

**Direction recorded:** 2 November 2026.

This document preserves the goal and proposed progression for emergent materials and artificial life. It is a discussion roadmap, not authorization to implement every milestone. Agree one conceptual step at a time. [DEVELOPMENT.md](DEVELOPMENT.md) remains the source for implemented behaviour, engine ownership and regression coverage; [VISION.md](VISION.md) covers the wider fluid world.

## Goal

Build a performant, strictly 2D world in which evolving organisms can contain hard supports, elastic tissue, liquids and gas. Teeth, bones, skin, muscles, blood vessels, hearts and lungs should arise from material arrangement and local physical behaviour, rather than requiring predefined organ types.

A starting body may be supplied. Heritable growth and control should be able to change its material distribution, structure and activity: larger supports, stronger contractile tissue, different cavities, additional teeth, and new functional arrangements.

The physics must support those possibilities, but physics alone does not provide heredity, development or reproduction. Define useful local laws and their energetic/material costs; let evolution discover structures that exploit them. Molecular simulation is not required for this goal.

## Direction and current boundary

Continue from the existing grid toward a continuum material solver. A spatial grid can represent fluid and solid mechanics; the material laws and transported state determine the behaviour. Investigate rigid-motion constraints or a coupled rigid-body representation when measured cost or error justifies them.

Current FluidSim provides conservative material and momentum transport, variable-density incompressible liquid/gas pressure, optional surface tension, implicit material viscosity on uniform WebGPU, and a material-flux logarithmic-strain elastic-solid experiment on uniform CPU and WebGPU. The elastic block recovers and oscillates, but numerical damping and accumulating interface/history error still need improvement before the first milestone is accepted for long-lived bodies. Solid contact, yielding, reactions, heat transport, phase exchange and a general miscibility model remain unimplemented. See [DEVELOPMENT.md](DEVELOPMENT.md) for measured limits.

The existing liquid/gas geometry and miscible ingredient fields do not already encode separate solid surfaces or mechanical bonding. Adding elasticity is a new physical capability, not merely another viscosity coefficient.

Gas is currently incompressible. The experimental pressure-recovery mode is not a thermodynamic gas equation of state. Compressible cavities need a separate pressure/volume model; simulated gas and trapped pockets must never silently become ambient pressure for performance.

## Material behaviour has several independent dimensions

| Property or state | Physical role |
| --- | --- |
| Viscosity | Resists the rate of deformation and dissipates motion |
| Elastic stiffness and deformation memory | Generate restoring stress from deformation relative to a relaxed configuration |
| Stress relaxation | Controls how quickly some or all elastic memory fades |
| Yielding and plastic deformation | Allow permanent changes of relaxed shape under sufficient stress |
| Bonding, damage and fracture | Determine which material remains attached and how it separates |
| Directional structure and active stress | Support fibre-like response and energy-consuming contraction |
| Compressibility | Relates pressure to changes in occupied volume |
| Mixing affinity, diffusion and permeability | Determine preferred mixtures and the rates/routes of material exchange |
| Thermal and chemical state | Influence properties, reactions, phase exchange and energy accounting |

A material whose entire elastic response relaxes eventually flows. A lasting shape-retaining solid needs a persistent elastic contribution. Yielding is distinct from stress relaxation. These behaviours should not be collapsed into one "solidness" value.

Keep material identity, local state/history and derived properties conceptually separate. Composition alone does not determine every property or whether material is bonded. Introduce concrete state only when the selected experiment requires it; this is not a request for a generic material framework or unused fields.

## Proposed mechanical progression

The current implementation step is **one shape-retaining elastic solid on the existing grid**. Its CPU experiment and optimized WebGPU port exist; improve and validate its history accuracy before proceeding. Temperature, chemistry and rigid-body extraction are not prerequisites. Bring contact and stiff inclusions forward before detailed thermal or chemical work.

| Step | Small experiment | Evidence needed before extending it |
| --- | --- | --- |
| 1. Persistent elasticity | Squeeze and release one soft solid; translate and rotate it through fluid | Elastic recovery, correct rotation, fluid response, controlled energy behaviour and no accumulating shape/history drift; zero elastic contribution preserves the existing fluid case |
| 2. Contact and attachment | Two pieces of the same material touch and separate; a strip folds against itself | No unintended welding or interpenetration; an explicit distinction between contact and bonding; appropriate sliding behaviour |
| 3. Relaxation and yielding | Compare a shape-retaining gel, a slowly flowing material and putty that keeps a dent | Persistent elasticity, relaxation and permanent deformation remain distinguishable and behave correctly over their relevant timescales |
| 4. Stiff inclusions | Repeatedly load a stiff bar embedded in soft material, then test material enclosing a liquid cavity | Load transfer, attachment and containment remain sound; measure deformation error and cost as stiffness increases |
| 5. A small functional body | Contractile tissue drives fluid through a cavity around stiff supports | Repeated actuation and material exchange have consistent mechanical and energy consequences; add gas compression when the experiment requires it |

Each row is a direction, not one bundled change. Contact, self-contact, relaxation, yielding and actuation may each need separate implementation/validation steps. Damage and fracture are later extensions when a selected experiment needs breakage.

The original reference map was replaced after gravity impact exposed incompatible geometry/history transport. CPU and WebGPU now transport local logarithmic deformation through the existing solid-volume fluxes, with an objective, stress-work-compatible kinematic update and elastic-wave subdivision. This stores no initial outline or target position. It retains the finite-strain Neo-Hookean response and fully simulated gas.

The gravity=1 soft-block case now passes floor impact and repeated rebound through 60s, with timestep and resolution comparisons. Numerical history mixing still dissipates energy, so this accepts that impact case without closing the first milestone's long-term shape-memory requirement. Improve transport accuracy before adding another material feature. See [implementation and limits](DEVELOPMENT.md#uniform-cpu-and-webgpu-elasticity-experiment) and [validation evidence](.codex/runs/run0167_20261102_material_history/README.md).

## Difficulties to expose early

**Preserving material history.** Motion across the grid must not erase the relaxed shape, create stress under pure rotation or accumulate unintended creep. Long repeated motion matters more than a short attractive wobble. Growth, plasticity and new bonds will eventually require well-defined updates to the relaxed configuration.

**Contact versus continuity.** Identical substances can belong to distinct touching pieces. Skin can contact itself without bonding. A bone can be attached to surrounding tissue. Some representation of mechanical continuity and separate surfaces is necessary, even without organ labels. Spatial adjacency alone is insufficient. A single shared velocity cannot represent arbitrary independent sliding motions within an unresolved cell.

**Stiffness and integration.** Explicit elasticity introduces an elastic-wave timestep restriction. Implicit integration can avoid that stability restriction, but does not make large steps automatically accurate or cheap. Coupling with pressure, contact and material transport must be evaluated. A stable result that loses useful elasticity through numerical damping is not sufficient.

**Resolution and topology.** Walls, vessels, gaps, tooth tips and folded tissue must be resolved or have an explicit smaller-scale model. A rigid-body solver does not fix leakage through an unresolved membrane. Refinement/coarsening must preserve meaningful material history and physical connections; visual detail alone cannot restore lost topology.

## Decision point for very stiff material

Compare these options using the stiff-inclusion and contact experiments:

| Representation | Benefit | Main trade-off |
| --- | --- | --- |
| Finite-stiffness elasticity | Retains bending, vibration and elastic energy | Increasing stiffness can increase timestep or solve cost |
| Rigidity constraints on the grid | Enforces rigid motion within the continuum representation | More involved coupled solve; contact and geometry remain necessary |
| Rigid bodies coupled to the grid | A bonded rigid region has three motion degrees of freedom in 2D: two translations and one rotation | Requires consistent interface coupling, attachment, changing geometry and conversion |

Rigidity is an approximation that suppresses deformation. Use it when the omitted deformation is unimportant at the chosen scale, not merely because a material coefficient crosses an arbitrary threshold. A nearly rigid region can still require finite-stiffness treatment if its bending, stored energy, failure or actuation matters.

Automatically discovering a suitable bonded region is compatible with emergence. A solver identity need not denote a bone or any other organ. Preserve internal composition, thermal state and structure even if its motion is represented as a rigid body.

Switching must preserve mass and linear/angular momentum, account for stored elastic energy, and restore a coherent velocity/deformation state when material softens or breaks. Avoid repeated switching near a criterion. Do not merge contacting pieces merely because their cells are adjacent. "Seamless" must be demonstrated through repeated transitions, not assumed from smooth property interpolation.

A separate rigid-body engine is therefore an option, not a predetermined destination or an automatic consequence of adding solids.

## Connecting mechanics to living materials

After the mechanical foundation is credible, add the interactions needed by the next body experiment:

- Directional, energy-consuming actuation for muscle-like tissue. Preferred-shape changes can generate motion without an organ-specific controller in the physics solver.
- Cohesion, permeability and selective transport for skins, vessels and membranes. Mechanical containment and chemical exchange are separate concerns.
- A suitable gas pressure/volume model for compressible cavities. Keep gas mass and its interaction with tissue explicit.
- Heat transport and energy accounting, initially changing mechanical properties. Actual freezing/melting requires phase-change energy, including latent heat; boiling also changes occupied volume substantially.
- Mixing preferences and transport rates. Partial miscibility is not simply slow diffusion, and mechanical attachment is not chemical miscibility.
- Small reaction networks that consume and produce material, modify structure and supply or consume energy. Elemental formulas alone do not define reaction kinetics or mechanical behaviour.

These are interacting branches, not a requirement to finish realistic chemistry before attempting organisms. A small material vocabulary with meaningful costs and feedback can support an early evolvable body. Keep the order responsive to demonstrated needs.

## Performance and acceptance

The expensive additions are likely to be transported history, interface/contact representation and stiff coupled solves, rather than evaluating a few local property relationships. Avoid assuming one full-domain state allocation per substance or discovered body will scale; storage strategy must follow the actual representation and measurements.

Benchmark at a stated resolution, timestep, material contrast and number of interacting pieces. Compare equivalent physical behaviour and accuracy. Record complete-step cost, additional state and solver work; do not infer a universal speedup from an isolated kernel. Keep disabled features cheap and preserve backend ownership.

Choose validation horizons from the phenomenon: repeated rotations, sustained loading, many collisions, actuation cycles, slow relaxation and eventual representation transitions. Check conservation, unintended creep, energy behaviour, leakage and topology as appropriate. A short run establishes startup, not long-running organism viability. Slowness alone is not a numerical failure.

The target is convincing, useful 2D mechanics at an explicit scale and budget. Published methods establish possible routes, not real-time performance for this engine or literal three-dimensional biological realism.

## Research starting points

- [Reference Map Technique for Incompressible Fluid-Structure Interaction](https://arxiv.org/abs/1810.03015): finite-deformation solids and actuated structures on a fixed fluid/solid grid.
- [Eulerian Solid-Fluid Coupling](https://www.tkim.graphics/EULERSF/): implicit integration and frictional solid contact coupled to incompressible fluid.
- [An incompressible Eulerian method for fluid-structure interaction with mixed soft and rigid solids](https://arxiv.org/abs/2111.09942): grid-based rigidity constraints and rigid components within deformable structures.

Update this roadmap after an agreed milestone or a measured limitation changes the direction. Record implemented behaviour in DEVELOPMENT.md and keep temporary execution plans and evidence under the project's existing .codex structure.
