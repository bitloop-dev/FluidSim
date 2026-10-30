# FluidSim development guide

Stage 1 geometric material transport and Stage 2 consistent mass/momentum transport are implemented. Liquid geometry, component masses and momentum share transfers through successive X/Y passes. Uniform CPU and legacy WebGL pressure default to multigrid-preconditioned CG; adaptive pressure uses the composite solver described below. Uniform CPU and WebGPU also support optional constant liquid-gas surface tension. `VISION.md` holds the longer-term ambitions and maths; this file is the short map of the working code and possible next steps.

The engines use the component system: configure the element catalogue and liquid/gas mixtures, then upload component masses alongside `liquid_fraction`, the total liquid volume fraction. Density, momentum and displayed colours come from those masses. Engine geometry uses liquid/gas terminology; Water, Red, Air and Cloud are scene ingredients. Several miscible ingredients share each phase's geometry; adding an ingredient does not add another interface reconstruction or pressure solve.

The CrashingWave scenario initializes blue Water and Red liquid, plus Air and a cyan Cloud gas ingredient. Air and Cloud share density 0.1; three fixed Perlin scales divide the initial gas volume between them. Noise is evaluated only in `CrashingWave::initialFluidState()`, then ordinary component transport carries the clouds. This visualizes atmospheric flow without adding condensation or evaporation physics.

SplashCubes uses the same Perlin gas distribution with black Air and white Cloud. Two initially stationary liquid squares sit side by side around the domain centre: red on the left, blue on the right, each side 30% of the shorter domain dimension with a 6% gap. Exact rectangle/cell overlap supplies partial edge cells. Water/Red retain the existing colours and densities 10/30, gas density remains 0.1, and the default 1024x720 grid, gravity 500, surface tension 20000 and timing policy match CrashingWave. The closed bottom wall supplies the ground; ordinary fluid transport/projection produces the fall, splash and miscible-liquid mixing.

## Where things belong

| File | Responsibility |
| --- | --- |
| `FluidSim/FluidGridBase.h` | Abstract common simulation contract and backend-independent edit sessions |
| `FluidSim/FluidGrid.h/.cpp` | Concrete public API, selection and forwarding to one owned backend |
| `FluidSim/FluidGridCPU.h/.cpp` | Original uniform CPU engine, native arrays/helpers, unchanged physics; derived catalogue colours |
| `FluidSim/FluidGridWebGPU.h/.cpp` | Uniform WebGPU backend implementing FluidGridBase; explicit host precision conversion and native step/edit/colour forwarding |
| `FluidSim/UniformWebGPU/` | Imported native uniform float engine, GPU multigrid, shaders, compute/readback and retained pressure reduction |
| `FluidSim/FluidGridAdaptiveWebGPU.h/.cpp` | FluidGridBase adapter for native adaptive GPU state, policy cadence, explicit observations and sparse material edits |
| `FluidSim/AdaptiveWebGPU/` | Imported adaptive GPU transport, composite pressure, reconstruction shaders and GPU selection; separate explicit edit kernel |
| `FluidSim/Adaptive/FluidAdaptiveRaster.h` | Temporary finest-raster observation shared by CPU/GPU native hierarchies; no persistent physical state |
| `FluidSim/FluidGridAdaptiveCPU.h/.cpp` | FluidGridBase backend over one native AMR hierarchy, common edits/observation and automatic selection cadence |
| `FluidSim/FluidAdaptiveSettings.h` | Scene-supplied quality, quiet dwell, check cadence and prediction horizon |
| `FluidSim/Adaptive/` | Imported native AMR transport, conservative transfers, composite pressure, selection and bounded observation helpers |
| `FluidSim/PressureMultigrid.h/.cpp` | Unchanged CPU pressure hierarchy, symmetric V-cycle and small direct coarse solve |
| `FluidSim/FluidGridWebGL.h/.cpp` | Legacy float OpenGL engine, retained separately for reference/tests; not yet selectable through FluidGrid |
| `FluidSim/PressureMultigridGPU.h/.cpp` | Legacy GPU pressure hierarchy and CPU coarse solver |
| `FluidSim/FluidGridWebGLInternal.h` | Legacy private textures, shaders, pass runner and graphics-state preservation |
| `FluidSim/FluidGridTypes.h` | Shared host state template and projection result; public interface uses double |
| `FluidSim/FluidCheckpoint.h` | Backend-owned checkpoint metadata/native bytes; independent of file scheduling and UI |
| `FluidSim/FluidCheckpointIO.h/.cpp` | Generic checkpoint file storage and optional lossless Brotli; independent of backend serialization |
| `FluidSim/FluidSnapshot.h/.cpp` | Versioned WebP XMP envelope, Brotli-1/base64 checkpoint packet and native RGBA conversion |
| `FluidSim/FluidGridEdit.h` | Staged replacements, validation and sparse commit ordering |
| `FluidSim/FluidGridDisplay.h/.cpp` | Render-thread adapter consuming common derived colours, plus legacy GPU texture overload |
| `FluidSim/FluidSim.h/.cpp` | Project controls and scene/viewport declarations |
| `FluidSim/FluidScene.h/.inl` | Shared scene state, lifecycle, timing, failure reporting, camera, rendering and captures |
| `FluidSim/FluidSceneUI.inl` | Shared buffered sidebar controls, instantiated for each concrete scenario |
| `FluidSim/scenarios/CrashingWave.h/.cpp` | Wave defaults, material catalogue, initial state and scenario snapshot data |
| `FluidSim/scenarios/SplashCubes.h/.cpp` | Centred falling squares, black/white atmosphere and scenario snapshot data |
| `FluidSim/scenarios/PerlinAtmosphere.h` | Shared initial three-scale gas pattern; no ongoing noise forcing |
| `tests/FluidGridTests.cpp` | Existing direct FluidGridCPU numerical regressions |
| `tests/FluidGridAPITests.cpp` | Common configuration, exact CPU/wrapper/base state and step parity, edits, colours and failure forwarding |
| `tests/FluidGridWebGLTests.cpp` | Retained legacy GPU/CPU and display comparisons in a hidden graphics context |

`FluidSimEngine` is the project-local library shared by the scene and CPU/API tests. It depends on Bitloop's vector type and thread pool, with no Scene, camera, OpenGL or UI code. The existing CPU numerical implementation is renamed to `FluidGridCPU`; its function bodies are unchanged. Palette colour calculation moved from the display adapter to a read-only CPU operation, with its original arithmetic and traversal. Packaging the engine separately remains a later decision.

## Scene structure

`FluidScene<Derived>` inherits `bl::Scene<Derived>` so Bitloop constructs the concrete scenario's UI and buffers both inherited and scenario-specific members. Each instance owns its grid, settings, display resources, camera and clocks. The project's **Scenario** selector chooses Splash Cubes (the default) or Crashing Wave; the viewport-count control creates independent instances of the selected scenario on project start. Existing instance IDs remain `scene-0`, `scene-1`, etc. Each scenario's `Config::settings` holds its own initial values.

Define a scenario in `FluidSim/scenarios/`, derive from `FluidScene<MyScenario>`, supply a stable `scenario_id`, and give it a display `name()`. Its `Config` derives from the base launch config and supplies `FluidSceneSettings`; the scenario constructor passes these to the base and configures its material catalogue. `CrashingWave::Config::settings` is the place to change its initial resolution, gravity and tension. Passing a different config to `create<CrashingWave>(id, config)` creates another independently configured instance.

Implement `initialFluidState()` to supply material and velocity arrays. The shared startup/restart path uploads them, observes display colours, resets clocks/statistics and reports failures. Startup, stepping and common cleanup are final lifecycle hooks. Optional `beforeFluidFrame(dt)` runs on the fluid owner once per simulated scene frame, before the common substeps; `dt` is the selected simulation frame time. The engine still owns each ordered physical substep. Scene-specific resource cleanup can use Bitloop's `sceneStop()`.

Physics advances in `sceneProcess()`, once per project frame, including scenes without a viewport. `viewportProcess()` does not advance physics. Mounting a second view of a scene therefore does not double its simulation speed. Instances still share Bitloop's project scheduler and frame budget.

The inherited UI is sufficient when a scenario needs only common controls. To add controls, derive its nested `UI` from `Base::UI`, inherit the constructor, and call the common sidebar:

```cpp
struct UI : Base::UI
{
    using Base::UI::UI;
    void sidebar() override
    {
        Base::UI::sidebar();
        // Scenario controls use bl_scoped / bl_schedule as usual.
    }
};
```

`drawScenario(viewport)` adds world-space drawing after the fluid surface. `viewportDraw()` and `renderFluidSurface()` can also be overridden; the latter is shared by viewport and native-image rendering. Native captures omit viewport annotations by default. Override `captureFluidImage()` for a custom capture image while keeping the shared checkpoint envelope. Capture scheduling remains project-owned.

Scene snapshot payload version **2** records `scenario` and a separate `scenario_state`. Both scenarios store their ingredient IDs there. Override `captureScenarioState()`, `validateScenarioState()` and `restoreScenarioState()` for additional persistent state. Validation runs against the replacement grid before the live simulation changes; restore only commits already validated values. Wrong-scenario images and old version-1 scene payloads are rejected explicitly; the backend checkpoint format is unchanged.

`FluidSceneTests` checks four independent wave instances against direct engine stepping, zero/two-view scheduling, isolated restart, concrete buffered UI construction/edits, shared-sidebar extension, owner-scheduled actions, and custom image/state capture. `FluidSnapshotSceneTests` covers invalid/wrong-scenario payload rejection and the existing native rendering checks. The extraction also retained exact native checkpoint bytes and raw images after a matched 200-frame/4-second production wave run.

`FluidSplashCubesTests` checks exact initial liquid volumes, separated/centred squares, zero velocities, original liquid colours and matching Perlin gas proportions on rectangular, odd and portrait grids. Its optional `<output-directory> <frames>` arguments run the production WebGPU scene with per-frame finite-state, fraction, mass and phase-consistency checks and selected shaded images; the evolving check requires floor contact and mixed-liquid cells.

## Liquid appearance

`writeColorPixels(pixels, true)` puts liquid coverage in alpha while retaining the original RGB and the existing single GPU colour readback. The default call remains opaque RGBA. Uniform engines use their geometric fraction; adaptive engines accumulate the same bounded reconstructed material volumes used by their colour output. This is derived observation, not a physical-state download or a change to transport.

`FluidGridDisplay::proximityTexture` smooths continuous gas coverage with forward/backward exponential filters along both axes. Every representable gas fraction contributes; there is no50% detection cutoff. Filtering linear coverage before boosting sensitivity keeps a bubble's contribution stable when it splits across cells, and blends nearby pockets without nearest-seed region boundaries. Walls add no gas. The scene GLSL exposes **Gas proximity range** (4 cells) and **Minimum brightness** (.6), with extra sensitivity near tiny pockets after smoothing. Final alpha stays one; partial cells blend the effect by liquid coverage. This is an artistic 2D cue, with no performance claim.

The native scene test includes `FluidAppearanceTests.h`: four-backend coverage/RGB/state, actual GLSL falloff/bubble/gas/opacity, tiny positive gas, continuity across50%, subcell movement and full-tank resize checks. An optional saved WebP argument renders the same checkpoint with and without shading and verifies exact shaded capture/XMP round-trip. [Original coverage work](.codex/runs/run0025_20261029_liquid_depth/README.md), [capture/proximity follow-up](.codex/runs/run0026_20261029_snapshot_bubbles/README.md).

## Per-frame statistics

The scene's **Frame statistics** collapsible table reports total measured scene work, render CPU submission, fluid processing, display colour/readback and remaining scene work in milliseconds per frame. Total is scene update plus the latest viewport render call; UI, pacing/presentation and GPU drawing completion are excluded. The render value stays on the GUI owner instead of passing through the simulation shadow buffer.

`FluidGridBase::step(dt, iterations, tolerance, StepTimings*)` optionally fills transport, forces, projection, selection and regrid/restoration costs. The scene clears and sums these across every physics substep in the frame. Child rows are included in the fluid total. Adaptive force/predictor work belongs to projection; selection/regrid remains in its backend. The native adaptive transport timer includes damping (the current common path passes zero damping).

Detailed stage timings are enabled initially and can be disabled without restarting. GPU measurements wait for completed stage work, including host dispatch/readback costs; these are wall-clock stage costs, not isolated kernel timestamps. These additional completion waits can increase total cost. A null timing pointer preserves the previous execution path. With profiling off, colour/readback may include completion of queued GPU tail work; total scene-update time still covers completed display output. Disabled stage rows show `--`. Existing quality, iteration limits, timestep, physical state and backend selection are unchanged.

## Uniform projection performance and pressure quality

The scene's **Pressure Quality** number input sets the maximum cell divergence per second directly (the scene member selects the initial value, currently `0.1`). Press Enter or leave the field to commit a positive finite value; partial edits stay in the UI, and invalid completed values revert to the value from before editing. Larger tolerances trade pressure accuracy for speed. It applies on the next physical step without restarting. The common/native engine API defaults and the 2000-iteration scene cap are unchanged. Every backend receives the same selected target, so comparisons must use matching quality. The previously tested `1e-3`, `1e-2` and `1e-1` remain useful reference values.

**Restart Simulation** schedules a scene-owned reset on the simulation owner. It reuses the existing upload/initialization path, resetting material, velocities/pressure history, elapsed clocks, failure state and frame statistics. It preserves pressure tolerance, profiling, camera and backend configuration, reuses uniform WebGPU pipelines, and requests a fresh display even while paused.

Resolution uses two integer input fields. Edits remain staged until **Restart Simulation** commits them, retaining the existing32..4096range. Device/adaptive changes restart at the current resolution without applying pending resolution edits.

Uniform WebGPU's existing shared-memory multigrid tail accepts power-of-two widths with odd/truncated heights. Only X indexing uses a bit shift; stored coefficients and geometric transfer maps handle Y. Existing four-level/1360-cell limits, five pre/post sweeps, bottom solve, PCG and actual-divergence acceptance remain. For 1024x720, the tail32x23/16x12/8x6/4x3 contains988 cells and reduces each preconditioner from105 to66 dispatches.

The shared bottom solve now uses the generic solver's accumulated validity flag rather than an early return inside the factor loop. On the current native compiler the previous control flow produced corrections differing by up to about35% in thin-grid tests; the equivalent flag form restores generic/fused agreement to float roundoff. Focused tests compare nonconstant/constant RHSs, refreshed coefficients, odd/thin heights, the actual1024x720 hierarchy and a non-power-of-two width fallback. This changes execution/control flow without replacing physical state or coarsening the simulation. Floating-point trajectories can still differ.

See [.codex/runs/run0011_20261028_022921/README.md](.codex/runs/run0011_20261028_022921/README.md) for retained source, numerical checks, evolving comparisons and quality limits.

## Uniform WebGPU transport fusion

The first packed component group also updates the next liquid fraction during each X/Y transport sweep. `UniformWebGPU/FluidTransportFusionKernels.h` reuses the existing component transport helper and writes the same geometric split update in a64-thread32x2 workgroup. All later groups keep reading the old fraction; `advectComponents()` swaps the fraction only after every group finishes. X component corrections and geometric split weights remain frozen for Y. This removes two full-grid dispatches per transport substep without changing component packing, phase geometry, density reconstruction or the pressure solve.

Fusion applies to every nonempty catalogue, including more than four components. The empty-catalogue path retains the standalone fraction update. Each fused shader uses the existing portable eight-storage-buffer limit; there is no extra persistent field or public tuning setting. Density accumulation during transport was tested separately and in combination, then rejected because extra accumulator traffic/dependencies outweighed the removed density passes as component counts grew.

Direct fused/separate shader checks cover both axes, pure/mixed fractions, frozen weights and odd/thin dimensions. Native CPU comparisons include populated catalogues through32 components. The600-frame production confirmation saves1.27% in the four-component developed wave; the32-component result is effectively unchanged. Matched crashing-wave comparisons, startup cost and replay limits are retained in [the transport experiment](.codex/runs/run0020_20261029_transport_fusion/README.md).

## Uniform surface tension

`setSurfaceTension(sigma)` / `getSurfaceTension()` select a constant liquid-gas coefficient in scene units. Zero is the engine default and preserves the original numerical path; the scene supplies its own initial coefficient below. Both CPU/DISABLED and GPU/DISABLED support nonzero values and the scene control. Adaptive backends reject them, including pre-upload backend changes that would discard an enabled coefficient. No ingredient-to-ingredient tension, viscosity, bubble deletion or ambient-pressure replacement is introduced.

The scene exposes **Surface Tension** beside Pressure Quality, committing nonnegative finite values on Enter/blur through the simulation owner. Set initial physics values through each scenario's `Config::settings` in `scenarios/CrashingWave.h` or `scenarios/SplashCubes.h`: `surface_tension`, `gravity_y` (positive down), `projection_tolerance`, grid dimensions and timestep policy. The base constructor forwards gravity and tension to the selected engine. Engine member defaults are library defaults, not scene controls: backend selection preserves the current backend's settings, which can override a newly constructed GPU backend's defaults. The CLI launch-config parser currently exposes checkpoint storage and image resume only. Restart retains live settings; checkpoint/image restore intentionally restores saved gravity and tension. Rebuild/relaunch after editing source defaults; Restart alone does not reload C++ source.

The retained crashing-wave benchmark used1024x720, dx1, gravity100, liquid densities100/300, gas density.1 and two .01 steps per .02 frame. A useful trial coefficient was10000; pressure tolerance.1 balanced weak tension more accurately than the then-selected tolerance1. These are the recorded benchmark settings, independent of later scenario edits.

`UniformWebGPU/FluidSurfaceTensionKernels.h` computes fresh geometry after transport: bounded nine-cell columns, oriented height-function curvature, a mixed-height quadratic fit, then neighbouring valid curvature or a local PLIC-centroid fit. Pure0/1 face transitions are included. Even reflection at closed walls imposes a neutral90-degree contact angle; wall-normal velocities remain zero. Standard height curvature is bounded by1/dx and fitted curvature by2/dx because subcell radii cannot be resolved. These limits affect estimated curvature, never transported fractions or masses. Degenerate fits remain invalid; faces with neither valid side receive zero capillary force. This is a resolution limit, not a guarantee of preserving arbitrarily tiny droplets/bubbles.

`FluidSurfaceTensionCPU.h` implements the same geometry in double, reusing the existing MYC normal/PLIC helpers. Its geometry sampler treats fractions within32 double epsilons of0/1 as pure: CPU transport can leave nominally pure cells a few rounding units from an endpoint, creating false height endpoints and spurious currents. This affects only geometry sampling, never stored fractions or component masses. Float/double roundoff can still choose different fallbacks in degenerate fits.

On WebGPU, the capillary impulse is fused into the two existing pressure warm-start velocity passes, once per complete physical step:

`u_face += dt * a_face / dx * (sigma * kappa_face * (c_right-c_left) - (p_right-p_left))`

`a_face` is the existing2/(rho_left+rho_right). Curvature averages valid adjacent values. Matching the pressure gradient and coefficient permits a static capillary pressure jump to balance without an extra pressure solve. PCG, multigrid, actual-divergence acceptance and iteration caps per solve are unchanged. Standalone native `project()` remains pure projection; capillarity is part of `step()`.

CPU adds the capillary term with the same face coefficients after acceleration and before its existing total-pressure solve. Its pressure implementation remains unchanged; it does not adopt the GPU's correction/warm-start execution. Both uniform engines apply an active acceleration field on every complete capillary subdivision.

For positive tension, each requested step is split into equal **complete** transport/damping/gravity/curvature/projection steps using

`dt_cap = 0.5 * sqrt((rho_liquid_min + rho_gas_min) * dx^3 / (2*pi*sigma))`.

Configured component density minima provide a conservative phase-density bound without GPU reductions. Internal transport CFL subdivisions still apply independently; they freeze velocity and cannot resolve capillary forcing alone. In the historical density100/300 benchmark, sigma10000 gave dt_cap≈.01996, so its .01 steps needed no extra splits. A failed projection stops further complete steps and reports failure, without concealing it by clipping/renormalization. `ProjectionResult` sums iterations and reports actual `physical_steps` and uniform-GPU `transport_steps`; divergence is the final attempted solve's result. Other backends leave transport_steps0 (unreported). Scene statistics sum these counters across the frame.

Enabled capillarity adds three geometry sweeps per complete step and no extra solve or readback. Lazy geometry scratch is32 bytes/cell on GPU (22.5MiB at1024x720) and64 bytes/cell on CPU (45MiB). The GPU predictor uses six storage buffers, within WebGPU's eight-buffer baseline. Scratch/pipelines are retained when switched off, but zero skips all geometry and capillary-force work; a never-enabled CPU grid allocates no geometry scratch. Geometry is rebuilt rather than serialized. Forces timing includes geometry and the CPU impulse; GPU projection timing includes the fused impulse/pressure work. Detailed GPU profiling still adds its existing completion waits.

References are [Popinet's generalized curvature implementation](https://basilisk.fr/src/curvature.h), [balanced interfacial forces](https://basilisk.fr/src/iforce.h), and [explicit capillary stability](https://basilisk.fr/src/tension.h). These are project-owned uniform2D WGSL/C++ implementations, not ports of the Basilisk solver or its adaptive ghost-cell machinery.

`FluidSurfaceTensionTests` covers planar/pure states, static drop/bubble/wall Laplace balance, restoring deformation, coherent motion, exact timed/manual full-substep equivalence, positive-tension checkpoint continuation and legacy/unsupported handling. `FluidSurfaceTensionWave` (tests+DEV build) runs the actual crashing-wave initializer/update, with matched evolving timing, state-health CSVs and pictures. At sigma10000/tolerance.001, radius8 static pressure errors are<.8% with peak currents<.01; tolerance1 gives≈22% pressure error on that fixture, while.1 passes. Do not silently tighten the user's pressure setting. Six clean1000-frame wave runs establish stability and visual improvement but no reliable wall-clock speedup. [Measurements, reproduction and limits](.codex/runs/run0016_20261028_surface_tension/README.md).

CPU coverage includes the same static/moving/deforming fixtures, direct geometry comparison, copied-grid scratch rebinding, field composition, failure accounting and exact zero-tension equivalence to the original granular CPU operations. Checkpoint continuation remains GPU-only. `FluidSurfaceTensionWave --pair <output-prefix>` compares both uniform engines for200 frames/4 seconds on a reduced96x72 version of the actual scene, with current densities/gravity, explicitly enabled tension20000 and matched scheduling. Broad motion and conservation agree closely, while local differences grow; this is not a universal pointwise-equivalence or performance claim. [CPU port evidence and limits](.codex/runs/run0023_20261029_cpu_surface_tension/README.md).

## Stored acceleration fields

`FluidField` owns cell-centred 2D data independently of the engine. Its immutable layout specifies width, height, cell size, origin and one to four float channels. Samples are row-major with interleaved channels. The class assigns no physical meaning to those channels. Copying the handle shares live storage; rebinding a local handle does not redirect an engine that already retained the old field.

- `FluidField(layout)` creates zeroed CPU storage. `values()` gives CPU access; `generateFluidField(field, callback)` evaluates a CPU function at each world-space cell centre. The callback returns four floats; only the layout's channels are stored.
- `FluidField(layout, context)` creates zeroed WebGPU storage. `FluidFieldWebGPUGenerator(context, shader)` compiles an independent producer; `generate(field, parameters)` updates its destination. Define `fn fieldValue(position: vec2f) -> vec4f`; `fieldParameter(index)` reads up to12 vec4 uniforms. The pipeline is retained across parameter updates. The generator accepts only GPU fields sharing its context.
- `cpu.upload(context)` makes a new GPU field; `gpu.download()` makes a new CPU field. `write(values)` updates existing storage, explicitly uploading when it is on the GPU; `read()` returns a CPU copy, explicitly blocking for GPU readback. Transfers produce independent copies with no automatic synchronization back to their source. GPU `values()` access is rejected.
- `FluidWebGPUContext` shares the device and ordered command stream. Create it independently and pass it to a uniform GPU grid with `setWebGPUContext(context)` before upload, or use the grid's lazily created `getWebGPUContext()`. Select the backend first. All operations and final resource releases stay on the same simulation owner thread; sharing does not make concurrent access safe. No raw GPU handles escape the public API.

Both uniform engines consume a field through `setAccelerationField(field)`. They require XY channels and matching width/height/cell size/origin. CPU requires CPU storage; WebGPU requires GPU storage on its context. Binding performs no upload, generation or readback. Adaptive engines still explicitly reject acceleration binding; those ports and resampling are separate future work. The current field follows the cell indices after binding; changing world geometry requires a newly generated/rebound matching field.

An active acceleration field replaces `gravity_y`; values use simulation distance/time² units. The engine retains the shared field, so subsequent writes/generation affect following steps without rebinding. It persists through ordinary grid upload/restart. `clearAccelerationField()` releases that binding and restores uniform gravity without additional passes. The caller's field remains usable after clear or engine destruction. The scene lazily creates its XY buffer (8 bytes/cell,5.625MiB at1024x720) and retains it while disabled, but does no generation, upload or readback then. Damping, capillarity and pressure keep their existing stages.

Each complete physics substep averages adjacent cell accelerations onto U/V faces and adds `dt * acceleration` before pressure projection. CPU performs this calculation in double, WebGPU in float; both consume the field's float samples. Closed-wall normal faces remain zero. The same stored field applies to every CPU or WebGPU capillary subdivision; transport-only CFL subdivisions retain their existing behavior. Native `applyGravity()` remains the legacy uniform-only granular operation. Field generation is explicit, so a scene controls when its formula/parameters change; pressure still determines the resulting incompressible motion.

The example lives in `FluidSim/MouseAttraction.h`, separate from engine kernels, with equivalent CPU and WGSL formulas. **Mouse attraction** is available on both uniform backends. The target follows camera-transformed pointer coordinates; leaving the viewport keeps the last target. Disable it to return to `gravity_y`. Initial enable/strength/positive softening radius live in `FluidScene.h`; a scenario can set these inherited members in its constructor. There is no launch-config override. The scene owns its field and producer, generates once per frame, and retains its target/parameters in supported snapshots. CPU uses its callback directly, without a WebGPU context.

For example, after grid initialization:

```cpp
// Create these once, on the simulation owner.
const auto context = fluid.getWebGPUContext();
fluid::FluidField acceleration({fluid.getWidth(), fluid.getHeight(),
    fluid.getCellSize(), fluid.getOrigin(), 2}, context);
fluid::FluidFieldWebGPUGenerator generator(context, mouse_attraction_shader);
fluid.setAccelerationField(acceleration);

// Update as often as the producer's inputs change.
const std::array<float, 4> parameters[] = {
    {float(mouse_world.x), float(mouse_world.y), 500.0f, 20.0f}
};
generator.generate(acceleration, parameters);
// fluid.step(...) consumes the same buffer automatically.
```

For a CPU producer, create the same layout without a context, fill `values()` or call `generateFluidField`, and bind directly to a CPU engine. For a GPU consumer, explicitly upload and bind the returned GPU field; later CPU changes require `gpu_field.write(cpu_field.values())`. A GPU-produced field can similarly be explicitly downloaded for CPU consumption. CPU evaluation uses double world positions; WGSL uses float positions, so arbitrary formulas need not agree bit for bit.

The primary parity target is GPU versus CPU within the same grid method: uniform CPU/WebGPU first, and adaptive CPU/WebGPU when those features are ported. Uniform versus adaptive agreement is a separate, harder numerical comparison. Match selected physics, initial data, timestep and solver quality; allow precision and convergence differences without silently substituting a different physical model. The CPU field follow-up uses the user's requested short40-step/0.4-second check, not a long-run equivalence claim: [evidence](.codex/runs/run0022_20261029_cpu_field/README.md).

The unchanged native checkpoint format3 stores the active field's values, not executable shader source. Restore creates independent GPU storage, leaving previously bound external fields untouched. A restored mouse controller builds its producer on the replacement engine's context and regenerates from saved settings on the next frame. Formats1/2 still clear acceleration. Generation remains included in total scene fluid time; the detailed Forces row measures application inside the physics steps. Generation submits commands without a mandatory completion readback.

`FluidAccelerationTests` checks independent CPU/GPU scalar/vector generation, transfer copies, shared storage/lifetime, later updates, geometry/device rejection, uniform-gravity equivalence, clearing, capillary steps, exact checkpoint continuation and legacy clearing. Scene snapshot tests cover controller settings, disabling a restored field, camera-transformed hover movement and retaining the target outside the viewport. `FluidSurfaceTensionWave` accepts optional `fixed` or `moving` attraction modes for evolving checks using the actual scene. [Refactor evidence](.codex/runs/run0021_20261029_fluid_field/README.md); [original force-model evidence](.codex/runs/run0017_20261028_acceleration_field/README.md).

## Frame checkpoints

`FluidGridBase::saveCheckpoint(FluidCheckpoint&)` and `loadCheckpoint(const FluidCheckpoint&)` are optional owner-thread operations, forwarded by `FluidGrid`. A checkpoint contains backend-owned versioned JSON metadata and an opaque byte payload. Unsupported backends return false. Uniform WebGPU is implemented first; CPU uniform and both adaptive backends remain unsupported. Loading requires the selected backend and fixed geometry to match; it does not switch engines or convert their states. A successful load initializes the wrapper just like upload. Malformed metadata, incompatible format/geometry or the wrong payload size throws before applying state.

Uniform WebGPU format3 stores native float32 fraction, U/V faces, current and previous pressure, then four-lane packed component-mass groups in mixture order. When `acceleration_field` is true, interleaved cell XY acceleration follows the component groups. Metadata includes catalogue/mixtures/densities, origin, gravity, damping, surface tension, preconditioner, previous iteration count, pressure tolerance and cached CFL/outflow validity. Formats1/2 are still accepted and clear acceleration; format1 also implies zero surface tension. Without a field the binary layout is unchanged. The scene stores clocks, failure state, pressure quality, profiling choice, mouse-controller settings and scene element IDs; frame/index and scheduling metadata belong to Bitloop. Derived density, momentum, colour, curvature, multigrid coefficients and solver workspaces are rebuilt. Ordinary download/upload is still initialization and does not preserve this history. Capture/restore occurs between completed frames, never inside a physics substep; no checkpoint readback occurs on unscheduled frames.

Files use Bitloop's existing checkpoint chunks, available with `BITLOOP_DEV_MODE=ON`. Saving is uncompressed by default. For example, from this project directory in PowerShell:

```powershell
$app = Join-Path $PWD 'build/ninja-release-codex/app/app.exe'
Start-Process $app -Wait -ArgumentList '--headless --project FluidSim --frames 101 --dt 0.02 --trace .codex/checkpoint-capture --checkpoint-at 100 --checkpoint-dir .codex/checkpoints'
Start-Process $app -Wait -ArgumentList '--headless --project FluidSim --frames 400 --dt 0.02 --trace .codex/checkpoint-resume --restore-checkpoint .codex/checkpoints/frame_00000100'
```

Frame indices are **zero-based and post-update**: frame100 has completed101 scene updates (2.02 simulated seconds at .02/frame). Use99 for exactly100 updates. `--checkpoint-at` accepts any index or comma-separated indices; `--checkpoint-every-frames N` schedules periodic captures. A resumed run starts at the following absolute frame, and `--frames` specifies additional frames. Restore requires matching scene selection/config hash (framework), backend/geometry (engine) and timestep/iteration policy (scene). Saved pressure quality is restored even if the current source default differs. Camera, renderer resources and frame timing samples are display-only and are not saved.

The current1024x720 four-ingredient payload is26,549,056 bytes; the complete frame100 directory is26,551,012 bytes (**25.32 MiB**). This is lossless native float storage, including pressure history, without scratch buffers. Fresh-process resume from100 through500 matches uninterrupted fields/history/settings/clocks byte-for-byte over400 further frames /8 simulated seconds, reaching10.02s total. Focused tests cover warm/initial state, multiple component groups, edits, diagonal/nonzero-damping settings and rejected files. Verified on the current native build/GPU; cross-device/compiler numerical replay is not guaranteed. [Checkpoint evidence and retained example](.codex/runs/run0013_20261028_141024/README.md).

### Resumable WebP snapshots

The scene's `captureFrame(const CapturePreset&, CaptureFrame&)` hook supplies the same shaded native image to snapshots and video. `captureFluidImage` performs render-owner readback; only image requests collect and serialize a checkpoint. Video fixes native input dimensions on its first frame, remains independent of camera/viewport size, and uses the existing recording controls. Changing grid dimensions during a clip fails capture and requires a new recording. H.264/H.265 retain the encoder's even-sized output conversion.

The normal **Snapshot** button saves the shaded fluid surface at native dimensions, independent of the camera, viewport size, HUD and preset resolution. It works while running or paused. The simulation owner captures exact native state between updates, synchronously renders/reads the same ShaderSurface on the render owner, and passes both through Bitloop's ordinary WebP encoder and save path. A viewport's scissor clip cannot crop the capture. Appearance controls are saved alongside scene settings; old snapshots without these controls retain the current values. Snapshot quality/lossless settings affect the visible image only; embedded native state remains exact. The `scene-0:grid` trace route uses this same shaded capture path.

Drop the saved WebP onto a FluidSim viewport to resume that scene. The scene decodes and validates a replacement engine before replacing the current simulation, restores saved geometry/backend/materials/clocks/pressure quality, rebuilds display colours and fits the camera to the saved grid. An invalid or unsupported image displays an error in the sidebar and retains the current state. Uniform WebGPU is the currently supported resumable backend; other engines report unsupported through the generic checkpoint hooks. Normal file checkpoints retain their existing format and scene policy.

`FluidSimSnapshot/1:<uncompressed bytes>:<base64>` carries a Brotli-quality-1 packet in Bitloop's existing WebP XMP save string. The packet contains a JSON header (geometry, scene settings, native engine metadata), a NUL separator and the engine's opaque native bytes. Base64 makes it safe for the existing XML envelope. Compression/base64 and GPU readback occur only for requested captures. `captureSceneState`/`restoreSceneState` are shared with development file checkpoints; the engine does not know about images or files.

For headless validation, `--snapshot-route scene-0:grid` uses the same native capture hook, including its XMP payload. Scene launch config `{"resume_image":"C:/path/snapshot.webp"}` starts from an image. Unlike `--restore-checkpoint`, this is a new driver session: local trace indices start at0 while the saved scene clocks are restored. There is no timeline/index-seeking feature yet.

The current1024x720 frame100 image is18,940,804 bytes (18.06MiB), including its viewable image and exact state. Native image/state equality to the original file checkpoint and400 more frames of exact continuation through original frame500 pass. Running/paused toolbar captures, byte-identical snapshots across camera zoom and atomic rejection/geometry recreation also pass. [Evidence and limits](.codex/runs/run0015_20261028_164139/README.md).

### Optional Brotli storage

Set scene launch config `checkpoint_brotli_quality` to an integer0..11 to compress the native binary payload losslessly. `-1` (the default) saves the existing uncompressed format. `FluidCheckpointIO` owns this policy: engines continue supplying their ordinary opaque bytes, and small JSON metadata stays uncompressed. The `fluid-state` chunk manifest records `compression=brotli`, `uncompressed_bytes` and `quality`; reading detects the codec automatically, including existing files with no compression tag. Unknown codecs, missing size metadata, failed/truncated streams and decoded-size mismatches throw before applying fluid state. Brotli is an explicit project dependency already present through Bitloop; no text/base64 expansion, quantization or numerical changes are involved.

For example, enable quality1 for the selected scene:

```powershell
'{"checkpoint_brotli_quality":1}' | Set-Content .codex/checkpoint-brotli.json
Start-Process $app -Wait -WindowStyle Hidden -ArgumentList '--headless --project FluidSim --scene scene-0 --scene-config .codex/checkpoint-brotli.json --frames 101 --dt 0.02 --trace .codex/brotli-capture --checkpoint-at 100 --checkpoint-dir .codex/checkpoints-brotli'
Start-Process $app -Wait -WindowStyle Hidden -ArgumentList '--headless --project FluidSim --scene scene-0 --scene-config .codex/checkpoint-brotli.json --frames 400 --dt 0.02 --trace .codex/brotli-resume --restore-checkpoint .codex/checkpoints-brotli/frame_00000100'
```

Pass the same scene config on restore because Bitloop verifies its launch-config hash. The decoder uses the stored codec, while the launch quality selects how subsequent checkpoints are written. Compression runs synchronously on scheduled saves and decoding on restore; ordinary frames do no additional work. [Repeated frame100 size/codec/file-cost measurements and resume evidence](.codex/runs/run0014_20261028_150111/README.md).

On the current frame100 state, quality1 reduces25.32MiB to13.43MiB (47.0% smaller). Three alternating-order native Release measurements on the Ryzen9950X3D give median encode/decode58/70ms and complete file-layer save/load68/81ms, versus14/21ms uncompressed. These exclude GPU state capture/upload and durable disk flush; reads use the OS cache. Quality0 saves14.00MiB in44ms; quality6 reaches11.35MiB in681ms; quality11 reaches9.82MiB but takes28.45s to save. Quality1 is a useful space/time compromise; default remains off. Actual quality1 capture100 and fresh-process restore through500 reproduce the original uninterrupted native state/history/scene settings exactly, including501 matching semantic frames across capture/resume. Original uncompressed restore also still works.

## Shared API and backend selection

`FluidGridBase` has pure virtual operations and a virtual destructor. The concrete public `FluidGrid` wrapper, `FluidGridCPU`, `FluidGridAdaptiveCPU`, `FluidGridWebGPU` and `FluidGridAdaptiveWebGPU` implement it. `FluidGrid` owns one `unique_ptr<FluidGridBase>` and forwards operations; it has no duplicate physical state or CPU-array escape hatch. The default is `FluidGridCPU`. CPU adaptive is now available through `FluidGridAdaptiveCPU`, which owns one imported native `FluidGridAdaptive` hierarchy. That core lives under `Adaptive/`; it is an implementation/learning tool, while the Base-compliant backend is the public simulation boundary. Uniform WebGPU is now available through `FluidGridWebGPU`, owning the imported float `UniformWebGPU/FluidGridGPU` core including its retained pressure reduction. Adaptive WebGPU is available through `FluidGridAdaptiveWebGPU`, owning the native GPU hierarchy and GPU refinement assessment.

```cpp
fluid::FluidGrid grid(width, height, cell_size);
grid.setDevice(fluid::FluidDevice::CPU);
grid.setAdaptiveMode(fluid::AdaptiveMode::DISABLED);
grid.configureMaterials(elements, liquid_mixture, gas_mixture);
grid.setGravityY(100.0);
grid.upload(initial_state);
auto result = grid.step(dt, 2000, 1e-3);
```

Select CPU and either DISABLED (original uniform CPU) or MAX_2/MAX_4/MAX_8/MAX_16/MAX_32 (adaptive CPU, maximum physical cell width relative to the finest grid). The finest domain dimensions are assumed divisible by the selected maximum scale. Choose mode before upload; setting the current mode is a no-op. Changing mode after upload throws `std::logic_error` before mutation: explicitly destroy, select and upload to restart. Pre-initialization selection preserves catalogue, mixtures, gravity and origin. GPU with DISABLED selects uniform WebGPU. GPU with MAX_2/MAX_4/MAX_8/MAX_16/MAX_32 selects adaptive WebGPU. Changing device after upload requires the same explicit restart as changing adaptive mode. Selection belongs on the wrapper, not the simulation base; there is no implicit fallback or live state migration.

The common interface exposes dimensions/cell size/origin, material configuration and read-only catalogue/mixture access, vertical gravity, explicit upload/download, `step()` and `maxDivergence()`, staged edits, derived colour output and lifecycle/thread requirements. `getWidth()`, `getHeight()`, `getCellSize()` and `getOrigin()` replace direct geometry-field access. Fixed dimensions/cell size are selected at construction. `configureMaterials()` supplies the whole catalogue and both mixtures; configure their membership/order before upload and upload again after changing it.

Common `Scalar`/`State` use double, and `ProjectionResult` retains the existing structure. Native backend precision remains an implementation choice for later migrations. Upload is initialization: empty fields mean zero and pressure prediction history resets; it is not complete checkpoint restoration. Masses remain actual mass per cell, unit depth and area `cell_size * cell_size`. The engine never invents an Air ingredient. Shared step defaults retain the original CPU defaults (200 iterations, 1e-6); the scene passes 2000 iterations and its selected pressure quality.

A new grid starts with zero velocity/liquid fraction; populate configured component masses before physics. The scene's sloped/clouded pool remains scene-owned. Mode changes require the explicit destroy/select/upload sequence described above.

The CPU engine's public arrays, `invalidateCaches()`, sampling, vortex helper and working-data inspection remain on `FluidGridCPU` for learning and direct numerical tests. After directly changing component masses/derived density or coefficient arrays, invalidate its caches before the next physics operation. Upload/transport do this automatically. Direct velocity edits need no cache invalidation but must be projected before transport. The public wrapper exposes operations rather than these storage details.

The scene queries `requiresRenderThread()` at runtime before executing engine operations. All four current backends return false (no GL-context requirement). Rendering remains on the render owner. The scene asks `writeColorPixels(display_pixels)` on the simulation/device owner after initialization and once after a frame's physical steps. Rendering uses `FluidGridDisplay::texture(width,height,rgba)`, so GL never invokes WebGPU commands from another thread. This derived image is not a physical-state copy. The Base-taking display overload remains available when observation and rendering share the correct owner; neither path performs a full physical-state download or casts to a CPU backend. Physics and observation must remain externally scheduled apart, as before. CPU `destroy()` retains its previous no-op behavior; the object can be reinitialized by upload, with owned memory released by destruction.

The legacy OpenGL GPU class still has float State and its original owning-context rule/API. It remains covered by its regression suite but does not inherit the shared base until its own migration. No WebGPU implementation has been substituted for it.

## Uniform WebGPU API

```cpp
fluid::FluidGrid grid(width, height, cell_size);
grid.setDevice(fluid::FluidDevice::GPU);
grid.setAdaptiveMode(fluid::AdaptiveMode::DISABLED);
grid.configureMaterials(elements, liquid_mixture, gas_mixture);
grid.upload(initial_state);
auto result = grid.step(dt, 2000, 1e-3);
```

The current scene selects uniform WebGPU. Its `fluid_device` member selects CPU/GPU before initialization; `adaptive_mode` selects the physical hierarchy. Uniform WebGPU is the GPU/DISABLED combination, and GPU/MAX selects adaptive WebGPU. Device/mode selection preserves geometry, catalogue, mixtures, gravity and origin before upload. Selecting the current value is a no-op; a real change after upload throws before mutation. Explicitly destroy, select and upload to restart.

The public backend implements the same double host State/Edit and step-focused contract. Native fields/shaders remain float; conversion occurs only on explicit upload/download. No conversion/full-state readback is inserted into stepping or edits. Shared step defaults remain 200/1e-6 for every implementation; the scene passes 2000 iterations and its selected pressure quality. Caller limits/tolerance reach the native solver unchanged. `step(0)` reports existing divergence without advancing; the native granular API retains its previous zero-time result.

Device creation is lazy at upload. Keep upload, stepping, editing, observation/readback and destruction on one owning thread; `requiresRenderThread()==false` means no GL context is required, not that the live device may migrate between threads. Main uses its simulation owner for these operations and hands derived RGBA to the GL renderer. Common colour output reuses caller vector storage and the existing mapped readback staging; it does not download masses, pressure or velocities. Source edit batches retain sparse writes and validate atomically before mutation. Device/readback errors propagate; the scene records failure and stops, keeping rendering away from an unavailable image.

At migration, native PCG, multigrid, projection/transport/force ordering, stopping rules and numerical kernels were imported unchanged. This includes the development worktree's retained two-pass bounded pressure reduction (run0129); divergence/CFL reductions stay on their original path. No new reduction/solver optimization was introduced during migration. The native-core bridge edits added a reusable colour output method, reused vector storage in the existing readback body, and granted the common backend access to the existing sparse edit commit. Subsequent multigrid execution changes are described below.

FluidSim requires the Bitloop WebGPU module in every configuration. Its root CMake enforces BITLOOP_WITH_WEBGPU/ASYNC_WAIT before project/provider discovery, including existing caches containing OFF; presets need no extra flags or cache deletion. The underlying framework component remains optional for other projects. Build enables the existing optional Bitloop WebGPU module and uses the project-local Dawn overlay in `ports/dawn`, copied unchanged from development `tooling/ports/dawn`, with its vcpkg license. The root manifest requests the same Dawn features. The overlay supplies the already-used dxc packaging and cache patches; registry baselines/framework sources are unchanged. CMake deploys the shader compiler/runtime DLLs with app/tests. Browser harness/flags are retained, but this migration's validation is native Windows.

`FluidUniformWebGPUTests` runs the imported native regression suite (only uniform CPU rename/header paths adapted). `FluidWebGPUAPITests` covers common configuration, unsupported combinations in both selection orders, double/float state parity, zero time, iteration failure, edits/atomic rejection, reusable/empty palette output, restart and worker-to-GL pixel equality. `FluidWebGPUEvolvingParity` compares direct native and common wrapper over the first 200 512-square four-ingredient wave frames, using main's .02-second frame / two .01-second steps. Projection results match every step; full committed fields match exactly at frames 1/40/80/120/160/200. Ingredient drift and local phase-volume error are reported separately. This is integration parity, not a new performance A/B or the previous .08-second Perlin-cloud benchmark. [Source proof, snapshots and logs](.codex/runs/run0004_20261027_192920/README.md).

### Uniform WebGPU multigrid execution

The native solver passes its existing residual stopping tolerance into `PressureMultigridGPU::apply`. Converged or failed residual statistics select a cheap zero-correction path through the remaining queued V-cycles. Ordinary dispatches and ping-pong copies remain, preserving the authoritative pressure/residual, PCG iteration/failure accounting and final actual-divergence check. No per-iteration CPU readback or indirect-dispatch machinery is added. A negative tolerance applies the full cycle for direct preconditioner use and algebra tests.

`PressureSmoothingKernels.h` shares first-color red/black results in a16x16workgroup with a one-cell halo. On polynomial levels, it executes two original sweeps using a two-cell halo and11,200bytes of shared storage. Each five-sweep pre/post sequence is2+2+1, retaining all original weights and reversing them after the coarse correction. Wall samples, RHS mean removal, transfers, hierarchy and the selected pressure target remain unchanged. Both new kernels dispatch full workgroups so every lane reaches the barriers; partial output tiles are bounded after synchronization. Generic global sweeps remain private regression controls.

Native regression tests compare72 generic/shared V-cycles, including odd fine grids, coefficient refreshes and density contrasts; stopping/restart cases verify zero corrections and exact restoration of active cycles. Matched600-frame crashing-wave comparisons preserve native fields/history, with separate conservation and convergence checks. [Implementation, repeated timings and limits](.codex/runs/run0019_20261028_mg_execution/README.md). This is automatic execution of the same pressure model, with no new scene setting or quality tradeoff.

## Adaptive WebGPU API

```cpp
fluid::FluidGrid grid(width, height, cell_size);
grid.setDevice(fluid::FluidDevice::GPU);
grid.setAdaptiveMode(fluid::AdaptiveMode::MAX_4);
fluid::FluidAdaptiveSettings quality;
quality.prediction_horizon = quality.check_interval + frame_time;
grid.setAdaptiveSettings(quality);
grid.configureMaterials(elements, liquid_mixture, gas_mixture);
grid.upload(initial_state);
auto result = grid.step(dt, 2000, 1e-3);
```

All four backend combinations are available through the same step-focused Base: CPU/DISABLED is uniform CPU; CPU/MAX is adaptive CPU; GPU/DISABLED is uniform WebGPU; GPU/MAX is adaptive WebGPU. Device/mode order before initialization is interchangeable. Live changes still require explicit destroy/select/upload, and selecting current values is a no-op. The scene default remains CPU uniform.

The GPU adaptive backend follows the already-agreed CPU adaptive scheduling contract: assessment starts at simulated time zero, then on a physical-step boundary no earlier than `check_interval`. The backend runs existing GPU error assessment/quiet history/look-ahead and downloads only compact request tiles for existing host clustering. Regrid transfers real native state and restores projected faces before transport. Thresholds, quiet dwell, buffers, nesting and finest-liquid policy are unchanged. Native combined force/projection and zero damping remain unchanged. `automatic_refinement=false` freezes the current hierarchy; starting OFF/upload begins wholly fine. Quality updates reset assessment history and schedule a fresh check. Rendering/download never drive refinement.

Native material/momentum/face state remains in float GPU buffers; pressure retains its compensated high/low planes. The explicit caller iteration ceiling reaches ordinary/restoration projection unchanged. Native-core default calls still use their original 200 gas/2000 liquid ceilings, and upload retains its native 1e-4 initial projection tolerance. Shared step defaults remain 200/1e-6; the scene supplies 2000 iterations and its selected pressure quality explicitly. The existing native nonconvergence exception now carries its iteration count; the common adapter returns an unsuccessful ProjectionResult with actual native divergence so the scene can stop. No solver tolerance is silently relaxed.

`writeColorPixels` uses the existing native GPU material reconstruction/render shader and reusable mapped/caller storage. Explicit `download()` temporarily reads native-resolution arrays, combines pressure high/low in double, and uses the same bounded material/raster observer as adaptive CPU. Ingredient densities become per-finest-cell masses; pressure impulse becomes physical pressure using the last projection interval. MAC faces are observed/interpolated, not reprojected on a fine grid. No CPU engine or pressure solve is created for readback, no data feeds back into physics, and no fine physical raster persists. Native `maxDivergence` remains authoritative. Keep device commands/readback/destruction on the simulation owner; the existing scene RGBA handoff works for this fourth backend too.

Common edits validate/deduplicate the entire batch before mutation. Geometry-only requests refine coarse selected footprints with the existing clustering/nesting, retaining existing resolved regions, and restore projected faces before replacement. A separate explicit source/sink kernel replaces selected finest ingredients, preserves cell velocity by rescaling momentum with density, then invokes existing conservative restriction for covered parents. MAC faces and pressure history stay unchanged after replacement; normal transport/projection uses the original kernels. Commands stay sparse; no full material/velocity download or upload occurs during commit. Tests cover non-unit cell area, coarse-gas liquid sources, duplicate ordering, atomic rejection, moving-cell velocity inheritance, pressure/face history and subsequent gravity evolution.

Production source is under `AdaptiveWebGPU/` with existing shared project-owned mesh/transfer/topology/compute. The upstream AMReX reference project, oracle targets, checkout and historical `fluid-amrex` overlay are absent. The existing [provenance/license](FluidSim/Adaptive/provenance/README.md) covers adapted algorithms; this is library independence, not a claim that those algorithms were written from scratch. No framework, registry baseline or dependency package changes were needed for this migration.

Native `AmrGasWebGPUTests`, `FluidCoarseningWebGPUTests` and `AmrPressureWebGPUTests` are imported regression tests against the local CPU core, without upstream AMReX. `FluidAdaptiveWebGPUAPITests` exercises common integration, all MAX modes and 200 evolving 96-square gas frames against a direct native/policy control. Projection results/layouts agree each step; valid conserved fields, compensated pressure and faces agree at frames1/32/100/200. Actual active-cell overlays were inspected; 13 checks produce two layout changes and retain finest detail while quiet regions release. The layout settles inside its feature buffer, so this does not establish moving-feature behavior beyond it or a new adaptive speedup. All16 native CTest entries and the Release app pass. [Exact source proof, snapshots, logs and limits](.codex/runs/run0006_20261027_202107/README.md).

## CPU adaptive API and scheduling

```cpp
fluid::FluidGrid grid(width, height, cell_size);
grid.setAdaptiveMode(fluid::AdaptiveMode::MAX_4);
fluid::FluidAdaptiveSettings quality;
quality.check_interval = 0.32;
quality.prediction_horizon = quality.check_interval + frame_time;
grid.setAdaptiveSettings(quality);
grid.configureMaterials(elements, liquid_mixture, gas_mixture);
grid.upload(initial_state);
auto result = grid.step(dt, 2000, 1e-3);
```

The shared API is step-focused. Uniform `advect`, `applyGravity` and `project` stay on `FluidGridCPU`; they are no longer methods of the wrapper/base. Adaptive transport and combined acceleration/projection stay native. Splitting them into extra common stages would add work and complicate the force impulse. Each common adaptive step calls the existing ordered native step with unchanged damping (zero here), solver equations and transport. The explicit caller iteration ceiling applies to both normal and regrid-restoration projection. Default native-core calls retain the original 200 gas / 2000 liquid ceilings. Native nonconvergence still throws, with a typed failure carrying iterations; the public backend translates it into an unsuccessful ProjectionResult with actual native face divergence, so the scene stops as before. Initialization/edit-restoration errors still throw.

The scene supplies quality/cadence; the adaptive backend owns simulated-time scheduling, assessment, conservative regrid and required restoration before transport. The existing tolerances (velocity/pressure gradient 0.1, relative density/ingredient 0.02), quiet band 0.5 and quiet dwell 0.24 seconds are preserved. Checks start at time zero and then occur no earlier than the check interval, on a physics-step boundary. A zero-time step performs no selection. Supply a prediction horizon covering the check interval plus the scene frame/observation interval (main uses 0.32 + 0.02 seconds); do not substitute a shorter substep horizon. `setAdaptiveSettings()` can update quality/cadence between steps; it resets assessment history and schedules a fresh check. An identical settings value is a no-op. There are no selection scans between scheduled checks and no full-state readbacks in stepping.

Upload begins fully fine. `automatic_refinement=false` freezes the current hierarchy; starting OFF therefore gives the adaptive method on a uniform finest grid. DISABLED selects the separate original uniform method. Turning OFF after coarsening retains that coarsening until policy is re-enabled or the grid is reinitialized. Automatic selection protects liquid at finest resolution, as in the source engine; this import does not add automatic liquid coarsening or replace simulated gas with ambient pressure.

Colour output uses the same bounded native material plane as the imported display, with reusable caller RGBA storage. Explicit `download()` constructs a temporary finest raster of reconstructed ingredient masses and liquid volume, sampled native MAC faces and physical pressure (native pressure impulse / last projection interval; zero for a geometric restoration). At a level join it observes the finer face; it is not a divergence-preserving fine velocity solve or a complete checkpoint. `maxDivergence()` and the step result measure actual native active-cell faces. Observation does not advance/refine/reproject the hierarchy and no finest physical copy persists.

Common material edits validate the entire batch first. If a requested finest cell is coarse, retain existing resolved footprints and use the existing request clustering/nesting to refine locally, conservatively transfer and restore projected faces, then replace only selected finest-cell materials. Cell momentum inherits local velocity; covered parents receive the conservative change. Existing projected MAC faces remain fixed after replacement. This explicit source/sink and refinement cost belongs to commit, not staging. Invalid batches do not regrid or write material. Tests cover duplicate ordering, atomic rejection, edited mass budgets and subsequent evolution.

### AMR numerical contract

Keep one native hierarchy with real physical coarse/fine state, two refreshed conserved ghost layers, conservative restriction/interpolation and synchronized coarse/fine fluxes. Never use a saved fine physical raster to repair coarsening. Gas material uses bounded linear reconstruction and half-step tracing; momentum uses mass flux and a limited velocity residual. Mixed liquid/gas uses native MYC/PLIC directional transport and frozen phase corrections. No clipping or renormalization is added. Single-resolution CFL uses the existing local outgoing donor bound; mixed layouts retain the conservative global bound. All transport subdivisions hold projected faces fixed, followed by one full-step force/projection. Regrid restoration is separate.

Pressure is the imported composite AMReX-derived 2D operator: GMRES+MLMG for differing densities, standalone MLMG/BiCGStab for globally equal densities. Tolerance and actual active-cell divergence remain authoritative. Native pressure/transfer/reconstruction sources are unchanged in this import apart from forwarding the explicit iteration ceiling and typing the pre-existing convergence exception. Pressure source provenance and retained license are in [Adaptive/provenance](FluidSim/Adaptive/provenance/README.md). Direct deep restriction still requires its footprint to fit a source FAB; the retained block width 64 covers supported maximum 32. Refinement policy preserves requested detail, quiet history, nesting and finest liquid; rendering is observation-only.

Build `FluidAdaptiveAPITests`, `FluidCoarseningPreviewTests` and `FluidRefinementTests` with FLUIDSIM_BUILD_TESTS. The default CTest set includes short reconstruction, CFL, mixed-liquid and selection/depth cases alongside the existing uniform/API/WebGL suites. Adaptive API tests compare native fields/layouts through 200 evolving 96-square gas frames, gas/mixed-density liquid step parity, OFF, zero time, caller iteration limits, reconstructed observations, sparse edits, all MAX modes and lifecycle. These are integration/regression checks, not a new 512-square performance comparison or long-run wave-fidelity claim. [Import evidence](.codex/runs/run0003_20261027_164329/README.md).

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

The shared public contract is enforced by FluidGridBase. Direct CPU vectors, `sampleVelocity`, `addVortex` and working-data inspection remain FluidGridCPU-specific learning tools. Pass explicit divergence tolerances when comparing the common CPU API with the retained legacy GPU class.

Unused dye buffers, `advectDye`, particles and the unused `FluidCell` placeholder have been removed. `sampleVelocity` and `addVortex` remain useful experiment operations.

## One physics substep

Read `FluidGridCPU::step()` first; the public wrapper delegates to it:

```cpp
advect(dt);
applyGravity(dt);
return project(dt, max_iterations, divergence_tolerance);
```

1. **Transport material and momentum:** reconstruct interfaces, calculate liquid-volume transfers, transport component masses and carry both momentum components through X, then Y. After each material pass, recover velocity by dividing momentum by the new face density. Repeat with smaller transport steps when needed to limit outflow.
2. **Apply gravity:** add acceleration to interior vertical faces. Both materials receive the same acceleration; density affects their pressure response.
3. **Project pressure:** rebuild density coefficients from the current component masses, close the tank walls, solve for pressure and correct velocity until cell divergence meets the requested tolerance.
Transport uses the previous substep's projected velocity. The scene starts with zero velocity, which already satisfies incompressibility. An experiment that directly changes velocities must project them before transport. The final projection handles the transported state and gravity, preparing the next substep.

`viewportProcess()` advances 0.02 simulated seconds per frame with `ceil(frame_time * 40)` subdivision, currently one step of 0.02 second. It passes 2000 iterations and the selected pressure-quality tolerance to `fluid.step()` for every backend. The CPU regression reference retains its double-precision `1e-8` checks; backend comparisons must use the same explicit scene quality. The engine advances exactly the time passed to it; it does not know about frames or recording. Older measurements below used six 1/120-second steps per frame; compare timing only with matched timestep and simulated duration.

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

`FluidSimEngineWebGL` is a separate local library. CPU physics remains the reference for learning and experiments. The GPU port was validated on the 128x128 sloped pool, cell size 4, densities 0.1/100 and gravity 100. Both air and water are simulated; no physical model or boundary condition is selected automatically.

Read `FluidGridWebGL::step()` and then the similarly named `Impl` methods. Each numeric array is a nearest-filtered float texture; fragment passes write new textures and swap them with the old inputs. A shader never reads the texture it is currently writing. The small private pass runner handles numeric textures; the framework's `ShaderSurface` is still used only to colour the result for display.

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

`FluidGridWebGL` retains material transport, forces, the fine pressure equation, CG vector updates and rendering on the GPU. Its private `PressureMultigridGPU` owns a small CPU `PressureMultigrid` workspace for a coarse grid covering the **whole domain**; its correction is returned to the GPU hierarchy. Both gas and liquid remain simulated. CPU/GPU stages alternate; there is no spatial partition or load balancer.

The retained legacy `FluidGridWebGL` is constructed directly and links `FluidSimEngineWebGL`; it is not yet selectable through the public FluidGrid wrapper. Coarse execution is an internal GPU-engine optimization, with no separate fluid engine or display adapter. Construction requires no GL context; upload, physics, edits, display and destruction follow the GPU engine's owning-context rule. Its `Scalar`/`State` remain float.

Set `FluidGridWebGL::cpu_pressure_cells` before `upload()` to choose the first coarse level with at most that many cells. The default is 1,024; the GPU hierarchy bottoms out at 16 cells even for smaller positive cutoffs. Zero keeps the entire hierarchy on the GPU. The setting takes effect on upload, including reupload after a mode change. This is a fixed hierarchy cutoff, not a target percentage of execution time. `FluidGridCPU` and CPU multigrid mathematics are unchanged.

The fine-grid pressure operator, geometric transport and final actual-divergence check remain authoritative. GPU smoothing, restriction and interpolation continue above the cutoff; the CPU applies its fixed symmetric V-cycle below it. Coarse coefficients are copied and factored lazily once a projection needs a correction, then reused throughout that solve. Different preconditioning and float/double rounding can change iterations and chaotic trajectories within the requested accuracy.

The last GPU restriction pass packs the current global residual maximum/failure flag with the coarse RHS. This lets the CPU handoff detect convergence without separate per-iteration scalar round trips, stop the V-cycle's unused return passes and preserve the active-iteration budget. With a 32x32 cutoff, ordinary cycles transfer 16 KiB of packed RHS/statistics and 4 KiB of correction; a stale matrix additionally reads 16 KiB of coefficients once per projection. The normal path never downloads the full fluid state. Native scalar reads use RED/FLOAT; WebGL retains RGBA/FLOAT but has not been run.

The existing coarse-pressure checks now belong to `FluidSimWebGLTests`; there is no separate hybrid test target. Projection, iteration caps, mode changes, edits, display, graphics-state preservation and multigrid algebra exercise CPU-coarse and GPU-only execution.

Retained experiment commands are `FluidSimWebGLTests --coarse-compare 512 1200`, `--coarse-benchmark 512 0` and `--coarse-pool-steps 4800 512`. Comparison uses an identical saved 10-second GPU-only pool state, five alternating pairs, five warmup steps and 48 timed steps at `dt=0.004`, `1e-3` divergence, with GPU completion included. Setup, shared history replay, rendering and explicit state inspection are excluded. A history of zero measures startup. The pool command checks 40 simulated seconds at `dt=1/120`, using the existing budgets of `1e-3` actual divergence, `1e-4` relative ingredient-mass drift and `0.002` cell phase-volume disagreement.

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

With the same test option enabled, build `FluidSimWebGLTests` and run both suites with `ctest --test-dir <build-directory> -R "^FluidGrid(GPU)?Tests$" --output-on-failure`. GPU cases cover hydrostatics, odd/thin projection, CPU comparisons, all four diagonal directions, nonuniform momentum transport, tiny interface remnants, sharp slabs, exact pure-material preservation, zero-time/failure handling and graphics-state restoration. GPU multigrid algebra tests cover odd/thin/high-contrast grids through 513x257, including the larger-grid polynomial smoother. Additional cases protect 512x512 default-budget convergence, the one-iteration cap and exact cached-versus-refreshed CFL parity. Common-API tests exercise upload/download, zero/reset behavior and actual display pixels for both types, including an empty catalogue. Component comparisons cover liquid-only, gas-only, 4, 7 and 18 ingredients, empty fields, differing densities, an element in both phases, and populated ingredients beyond the first sixteen. A 512x512 regression exercises the formerly failing vanishing-sliver case. Both engines' legacy fixtures explicitly populate component masses through `tests/FluidTestSetup.h`.

Edit-session regressions cover staged/cancelled/moved/reused sessions, empty commits, initialization requirements, arbitrary sparse masks, duplicate-write ordering, row-boundary gaps, liquid/gas replacement through nineteen ingredients, invalid-batch atomicity, cache refresh against a fresh upload, edited display colours and GL state restoration. A 257x129 full-grid edit also covers threaded CPU commits and merged GPU uploads. A two-second, 240-step source/sink case on each backend commits twelve batches during pool motion; every step checks projection, source-adjusted component mass, fraction bounds and mass-derived phase volumes. Existing twenty-second CPU pool/component regressions continue to cover the unchanged transport/pressure algorithms.

`FluidSimWebGLTests --edit-benchmark` measures reserved staging and GPU-complete commits for 1, 10 and 100 separate 6x6 drops on the 1024x1024 four-ingredient grid, plus a full-state download/upload reference. These are isolated edit costs with explicit GPU synchronization, excluding physics and drawing.

`FluidSimWebGLTests --pool-steps 4800 1e-3 512 4` runs a 512x512, four-ingredient mixed-density pool for 40 simulated seconds. It downloads each committed step to check finite fields, fractions, each ingredient's mass, mass-derived phase volumes, closed walls, actual divergence and energy. The last two arguments default to 128 and 2. Short comparisons allow `5e-6` local phase-volume error; long float runs have an explicit quality budget of `0.002` of one cell (0.2%), while global ingredient-mass drift must remain below `1e-4`. This separates accumulated local roundoff from global conservation; it does not normalize masses to hide disagreement.

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
| Another field experiment | Uniform WebGPU now supports stored 2D acceleration. For example, Mandelbrot depth could generate an in-plane field; it adds no water-depth variable |
| Colour labels | Two colours travel with the same fluid properties. Colour alone does not require a new physical element or pressure solve |
| Material-dependent surface tension | A droplet tends towards a round interface when enabled; zero tension preserves the no-tension case |
| Prescribed ambient exterior | Open liquid surface sees a chosen exterior pressure; resolved-gas mode remains a separate explicit choice |
| Trapped gas pockets | A sealed bubble resists collapse through a pocket-pressure/volume model or resolved gas; connection to the exterior changes its treatment |

Keep the current engine concrete until a second working case needs an abstraction. Start generalising forces where they are applied, material properties where coefficients are built, and exterior conditions where pressure boundaries are imposed. Avoid a feature registry, automatic physics selection or a collection of unused switches now.

Execution choices may eventually become automatic from grid size and profiling. Whether air is a real simulated fluid, an ambient-pressure exterior, or a trapped pocket remains a modelling choice: those cases have different physical behaviour.
