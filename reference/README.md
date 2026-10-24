# Executable AMReX MAC reference

This separate CPU-only project is the **only** build which uses AMReX/Hydro.
The normal application and its regression tests use the extracted production
engine, with no AMReX dependency. `amrex/AmrGas.*` preserves the upstream engine
as an oracle, in `fluid::reference`; it always uses upstream pressure.

- `FluidExtractionTests` compares production storage/transfers, clustering,
  transport and both pressure paths with the upstream engine.
- `FluidPressure2DTests` compares the independent composite operator and solver.
- `FluidAmrGasTests` retains the older upstream-engine regressions.
- `FluidAmrexReference` isolates Hydro's MAC projection, as documented below.

[Full extraction contract](engine/README.md) and [60-frame wave comparison](../.codex/runs/run0100_20261024_cpu_extraction/findings.md).

## Reproduce on this Windows host

```powershell
./reference/build-reference.ps1
$env:OMP_NUM_THREADS = '8'
$env:OMP_DYNAMIC = 'FALSE'
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]65535
./build/amrex-reference-clang-codex/FluidExtractionTests.exe
./build/amrex-reference-clang-codex/FluidPressure2DTests.exe
./build/amrex-reference-clang-codex/FluidAmrexReference.exe n_cell=64 repeats=1
./build/amrex-reference-clang-codex/FluidAmrexReference.exe n_cell=512 repeats=7 tolerance=1e-9
./build/amrex-reference-clang-codex/FluidAmrexReference.exe n_cell=1024 repeats=7 tolerance=1e-9
```

The build script discovers Visual Studio with `vswhere` and initializes its
developer environment in the same shell as CMake. It uses the installed
`C:/Program Files/LLVM/bin/clang-cl.exe` (22.1.8), the MSVC ABI/CRT and LLVM OpenMP.
Native MSVC 14.51 rejects upstream class/namespace `omp threadprivate`, including
with `/openmp:llvm`. No numerical source was patched to work around that.

The one retained upstream patch, `amrex-windows-library-copy.patch`, changes the
legacy library-name alias from a symlink to a copy, since this Windows account
cannot create symlinks. The actual library target is unchanged. The script
applies this patch idempotently and copies the LLVM OpenMP runtime beside the
executable. Source checkouts and build outputs are under ignored `build/`.

| Dependency | Pin |
|---|---|
| AMReX, tag 26.10 | `015c14b65715b277fbc54706ebe7455b196d2254` |
| AMReX-Hydro, development on 2026-10-20 | `5829a90a6cd521b378ebb7bce1390937beb49506` |

Configuration: two dimensions, double precision, OpenMP enabled; MPI, Fortran,
embedded boundaries, particles and GPU disabled. The code constructs an actual
`Hydro::MacProjector`/`MLABecLaplacian` and executes the upstream multilevel solve.
The same code runs uniform, one central fine patch, and four disconnected fine
patches. The latter two have identical fine area. Each uses a complete N/2 coarse
level and 25% fine coverage: 0.4375 N² active cells, 0.5 N² allocated level cells.
Covered coarse cells are explicitly counted and are excluded only from composite
diagnostics, not hidden from memory/work counts. There is no full fine level.

Optional command arguments are `layout=uniform|patch|four`,
`case=curl|hydro_x|hydro_y|curl_gradient|variable_gradient|quadratic_gradient`,
`max_grid_size=64`, `repeats=3`, `tolerance=1e-10` and `n_cell=128`.
The aligned fixtures use N divisible by 8. The CTest entry runs all 18 cases at64.

## Numerical contract

1. Store one cell-centred pressure per level and separate X/Y face velocities.
   A coarse/fine face has two independent fine normal velocities. Same-level
   overlapping nodal-face copies must agree. Fine patches align to coarse cells.
2. Restrict face velocity by area average before taking divergence. In 2D,
   `U_coarse = (U_fine_0 + U_fine_1)/2`. Compute each active cell divergence from
   its actual face fluxes divided by its cell width; ignore covered coarse cells
   when reporting composite norms and budgets.
3. With positive face coefficient beta, solve
   `D(beta G phi) = D(U*)`, then `U = U* - beta G phi`. This reference exercises
   both beta=1 and beta=1+0.25x+0.125y. In a pressure solve beta can be `dt/rho_face`
   and phi pressure, or `1/rho_face` with a pressure impulse; use one convention
   consistently for forcing, tolerance and velocity correction.
4. Set predicted normal velocities on closed physical walls to zero. Homogeneous
   Neumann pressure-correction boundary data then preserve those walls. Do not
   add a separate wall pressure-gradient velocity impulse. The all-Neumann
   pressure is defined up to a constant; only gradients matter.
5. The fine-side pressure ghost value comes from coarse pressure boundary data
   and fine interior pressure. Coarse boundary data are interpolated in the
   tangential direction with a sliding polynomial of up to degree two. Available
   neighbour masks select centred/one-sided/lower-order stencils near corners.
   The default boundary `maxorder=3` means three interpolation points, not a
   third-order global pressure operator.
6. For a regular 2:1 low fine boundary at x=0, let B be the coarse pressure
   interpolated tangentially to that subface at x=-h, P0 the fine pressure at
   x=h/2, and P1 the next fine pressure at x=3h/2. The normal ghost pressure is
   `Pg = (8/15) B + (2/3) P0 - (1/5) P1`; the fine face gradient is
   `(P0-Pg)/h`. Reverse orientation at a high boundary. With both tangential
   coarse neighbours present, B at transverse offset s=±1/4 is
   `0.5*s*(s-1)*Cminus + (1-s*s)*C0 + 0.5*s*(s+1)*Cplus`.
7. This gradient is part of a **composite multilevel operator**. Its coarse
   residual receives reflux corrections replacing coarse pressure flux with
   summed fine flux at coarse/fine boundaries. The solve also exchanges coarse
   boundary pressure and multigrid corrections. Pressure flux evaluation must
   use those same filled ghosts. Use the projector/MLMG flux API; calling the
   bare linear operator with stale coarse boundary data is incorrect.
8. Correct each fine face independently with that pressure flux, then average
   corrected fine faces down. A coarse neighbour sees the sum of fine fluxes;
   fine cells retain their transverse normal-velocity detail.

The custom graph from run0071 used a different coupling: tangential interpolation
plus its explicit transpose corrections on other velocity samples, tied to old
lumped half-cell momentum weights. AMReX instead uses level pressure ghosts,
independent fine face fluxes, and reflux in the composite divergence equation.
It does not establish that the old graph's Euclidean SPD/kinetic-energy norm
applies to this operator. Do not transplant only B or Pg into the old PCG graph
and call it the same method. A native implementation would need parity tests for
the entire composite operator/flux correction and an appropriate multilevel
solver, then a separately justified conservative transport/momentum scheme.

## Exact source map

- [Hydro MacProjector](https://github.com/AMReX-Fluids/AMReX-Hydro/blob/5829a90a6cd521b378ebb7bce1390937beb49506/Projections/hydro_MacProjector.cpp): `project_doit` at lines332–469, average-down at587–605; zero initial guess overload at300, reusable pressure overload at314; options at506 onward. Its absolute tolerance is the divergence residual for the variable-beta constructor used here.
- [Upstream MAC example](https://github.com/AMReX-Fluids/AMReX-Hydro/blob/5829a90a6cd521b378ebb7bce1390937beb49506/Tests/MAC_Projection_EB/main.cpp): setup pattern; this reference omits its EB geometry and supplies a true partial-coverage hierarchy.
- [Tangential boundary interpolation](https://github.com/AMReX-Codes/amrex/blob/015c14b65715b277fbc54706ebe7455b196d2254/Src/Boundary/AMReX_InterpBndryData_2D_K.H): `interpbndrydata_x_o3` at23 and `interpbndrydata_y_o3` at80.
- [Normal ghost interpolation](https://github.com/AMReX-Codes/amrex/blob/015c14b65715b277fbc54706ebe7455b196d2254/Src/LinearSolvers/MLMG/AMReX_MLLinOp_K.H): Dirichlet ghost polynomial at47–68. Physical Neumann copies the interior pressure at26–36.
- [Boundary location](https://github.com/AMReX-Codes/amrex/blob/015c14b65715b277fbc54706ebe7455b196d2254/Src/LinearSolvers/MLMG/AMReX_MLMGBndry.H): internal boundary distance is coarse half-width, lines197–206.
- [Composite operator](https://github.com/AMReX-Codes/amrex/blob/015c14b65715b277fbc54706ebe7455b196d2254/Src/LinearSolvers/MLMG/AMReX_MLCellLinOp.H): `updateSolBC` at952; `prepareForFluxes` at1586; `reflux` at1621; `compFlux` at1690.
- [Face pressure flux](https://github.com/AMReX-Codes/amrex/blob/015c14b65715b277fbc54706ebe7455b196d2254/Src/LinearSolvers/MLMG/AMReX_MLABecLap_2D_K.H): `mlabeclap_flux_x` at69, equal to `-beta*(Pright-Pleft)/h` after ghost filling.
- [Projection documentation](https://amrex-fluids.github.io/amrex-hydro/docs_html/Projections.html).

## Evidence and limits

See `../.codex/runs/run0072_20261023_151735/validation.md` and its CSV files.
The curl comes from nodal streamfunction differences, so its divergence cancels
and coarse face averages telescope exactly. Hydrostatic X/Y and variable-beta
affine cases test equilibrium without constructing input with the solver's own
gradient API. The quadratic case also tests the normal interpolation order.

CSV timing covers the whole projector call (average-down, RHS, solve, velocity
correction, final average-down). `setup_ms` includes arrays, initialization and
projector construction. `cold_project_ms` additionally exposes lazy MLMG setup.
Repeated calls reuse solver allocations but start from zero pressure; restoring
the identical input and diagnostic scans are outside that timed region. These
are deliberately difficult, cold-pressure projection cases, not typical warm
time steps. Pressure history should use the explicit `project(phi, rtol, atol)`
overload in a time-stepping engine.

There is no transport, remap, classification or rendering in this stage. No
behavioral horizon beyond the deterministic single-projection invariants is
claimed. Subsequent production validation must cover cumulative ingredient and
momentum conservation, repeated patch crossings/remaps, visible clouds, and
complete-step performance. Same-timestep conservative transport needs shared
fine fluxes averaged down before coarse divergence updates, plus consistent
covered-cell restriction and ghost filling; pressure projection alone supplies
none of that machinery.
