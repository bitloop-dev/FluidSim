# Extracted 2D pressure reference

`FluidSim/AmrPressure2D.h/.cpp` owns both regular Cartesian 2D pressure paths:
GMRES+MLMG for differing ingredient densities and standalone MLMG/BiCGStab for
equal densities. It uses contiguous double arrays and cached patch-boundary
recipes with optional OpenMP. The production engine has no AMReX dependency;
only the separate reference executables link upstream.

## Source and retained mathematics

Derived from AMReX commit `015c14b65715b277fbc54706ebe7455b196d2254` (26.10), with the copyright and redistribution terms in [AMReX-LICENSE](AMReX-LICENSE). The matching local source is `build/amrex-reference-deps/amrex`; the dependency remains the executable oracle. Source and binary redistribution of the extracted code must retain the notices required by that license.

| Project operation | Pinned upstream source |
|---|---|
| `applyGrid`, `smooth`, pressure flux | `Src/LinearSolvers/MLMG/AMReX_MLABecLap_2D_K.H`: `mlabeclap_adotx`, `abec_gsrb`, `mlabeclap_flux_*` |
| Normal boundary stencil | `AMReX_MLLinOp_K.H`: `mllinop_apply_bc_*`; quadratic Dirichlet reconstruction at the coarse cell centre, even reflection at closed walls |
| Tangential coarse boundary values | `Src/Boundary/AMReX_InterpBndryData_2D_K.H`: `interpbndrydata_*_o3`, including the one-sided/constant fallback when neighbors are covered |
| `reflux`, restriction and active-cell norms | `AMReX_MLCellLinOp.H`: `reflux`, `avgDownResAmr`, `averageDownAndSync`, `dotProductPrecond`, `norm2Precond` |
| `cycleBase`, `precondition` | `AMReX_MLMG.H`: `mgVcycle`, `oneIter`, `computeResWithCrseSolFineCor`, `computeResWithCrseCorFineCor`, `interpCorrection`, `makeSolvable` |
| AMR correction assignment | `AMReX_MLMG_2D_K.H`: `mlmg_lin_cc_interp_r2` (9/16, 3/16, 3/16, 1/16); base MG correction instead uses the upstream piecewise-constant additive interpolation |
| `solveMultigrid`, `solveBottom` | `AMReX_MLMG.H`: `solve/oneIter/actualBottomSolve/postCG`; `AMReX_MLCGSolver.H`: `solve_bicgstab`, diagonal normalization, 1e-4 relative target, 200 bottom iterations and eight-sweep fallback |
| `solve` | `Src/LinearSolvers/AMReX_GMRES.H` and `AMReX_GMRES_MLMG.H`: right preconditioning, restart length 30, two classical Gram-Schmidt passes, Givens rotations and a warm-start correction |

The operator is `-D(beta G)`, with face-centred beta. Fine faces remain independent; their combined flux replaces the coarse-face flux in the composite operator and returned correction. Coarse cells covered by fine state are excluded from the Krylov norm. The Krylov norm deliberately follows AMReX's **unweighted active-cell** sum; the closed-domain solvability correction uses restriction to the full base grid. Fine AMR levels use two red/black smoothing sweeps at each pass. Base multigrid uses two pre/post sweeps and the eight-sweep bottom smoother selected by upstream `GMRESMLMG`.

The specialization assumes a full rectangular base domain, square cells, closed Neumann walls, disjoint even-aligned patches and proper nesting through successive 2:1 levels. These are controlled layout inputs, not conditions to rediscover in every kernel. It removes 1D/3D, MPI, embedded boundaries, overset masks, metric coordinates, virtual operators and GPU backend machinery. Numerical breakdown and nonconvergence are still reported. Base solver levels use contiguous coarse storage; physical patches are retained. Geometry recipes are built once per solver instance, coefficients are updated separately, and interior stencil loops do not search for neighbors.

Standalone MLMG repeats the composite correction cycle until the active-cell
maximum residual meets tolerance. Its bottom solve uses upstream's normalized
BiCGStab, closed-domain mean removal, numerical-breakdown checks and smoothing
fallback. GMRES retains its eight-sweep bottom smoother. The engine checks actual
corrected-face divergence afterward in both paths. The extraction tests compare
equal-density projection too, rather than routing those cases through GMRES.
Float arithmetic, new boundary models and GPU execution remain unvalidated.

## Build and checks

The normal application and its regression targets do not find AMReX. Build `reference/` for `FluidPressure2DTests` and `FluidExtractionTests`. The small standalone project can also verify the pressure core without AMReX or the framework:

```sh
cmake -S reference/pressure2d -B build/pressure2d-standalone -DPRESSURE2D_REFERENCE_TESTS=OFF -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON
cmake --build build/pressure2d-standalone
```

For upstream comparisons, leave `PRESSURE2D_REFERENCE_TESTS=ON` and set `CMAKE_PREFIX_PATH` to the installed pinned AMReX package. Use the native compiler environment and Release configuration for timings.

- `FluidPressure2DTests`: fourteen short upstream comparisons, covering uniform, isolated, wall-touching, fragmented, three-level, six-level and small ragged-patch layouts at 1:1 and 1000:1 density. Compare the operator, a complete preconditioning cycle and corrected face velocities. The rough-density case exercises 79 iterations and multiple GMRES restarts. Warm solutions, zero forcing and the iteration cap are checked too.
- `FluidPressure2DTests 512 tiled`: the first four layouts, split into matching maximum-64-cell patches for both implementations, with four alternating timing pairs per density. Fixed state, double precision and absolute L2 tolerance `1e-5` match; correctness checks use `1e-8`.
- `FluidPressure2DTests 512`: the same test without splitting large boxes; useful for diagnosing patch scheduling, not the preferred production comparison.

The harness prepares and checks coefficients before timing. Timed solves include their Krylov allocations and internal work, but exclude hierarchy construction, coefficient setup, transport, selection, regrid, rendering and any future engine-adapter copies. Report these as **fixed-layout pressure costs**, never whole-frame or automatic-adaptivity gains. The constructor time is reported separately; no matched upstream setup-cost claim is made.

Initial results and limitations: [extraction evidence](../../.codex/runs/run0098_20261024_pressure2d/findings.md).

## Integrated liquid comparison

[512² CRASHING_WAVE integration evidence](../../.codex/runs/run0099_20261024_pressure_integration/findings.md): repeated 60-frame pairs, identical current scene settings, initial setup plus copying/coefficient preparation/transport/selection/regrid/restoration included. Total time 28.28 -> 21.10 s (25.4% lower); complete physical-step projection 14.52 -> 7.39 s (49.1% lower). Liquid and committed resolution captures were visually inspected. The engine adapter agrees with upstream to 1.5e-13 on identical evolving inputs through layout changes. These results do not establish long-time equivalence, a gain against the old uniform engine or GPU performance.

The complete CPU extraction, including subsequent transfer/transport gains, is documented in [run0100](../../.codex/runs/run0100_20261024_cpu_extraction/findings.md). Earlier integration figures above are historical and used a different timing session.
