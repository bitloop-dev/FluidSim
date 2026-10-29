# Current work: backend parity follow-up complete

**Goal / authorized scope:** uniform CPU surface tension is implemented and validated. CPU with sigma20000 now starts successfully. Work is in `FluidSim_main` on `main`; sibling `FluidSim` is untouched. Current source defaults (CPU,256x256,sigma0), densities and unrelated WIP are preserved; tests explicitly enable tension. Headless only; no commit.

**Parent / return:** [four-backend parity audit](WORK_STACK.md#four-backend-parity-audit), [feature audit and remaining proposals](.codex/plans/20261029_four_backend_parity_audit.md). User prioritizes GPU/CPU fidelity within each grid method over uniform/adaptive agreement.

**Completed:** uniform CPU ports the existing GPU three-stage generalized-height curvature, fallbacks, neutral walls and balanced capillary force to double. Its existing total-pressure solver remains unchanged. Both uniform engines use the same complete-step stability bound and support the current independent acceleration field. Zero skips capillary geometry/forces and allocation on never-enabled grids; after disabling it, native CPU operations/state match the original exactly. Adaptive still rejects nonzero tension.

**Essential finding / limitation:** double transport can leave nominally pure fractions a few rounding units from0/1, creating false curvature endpoints. Geometry-only sampling now classifies those as pure without changing stored fractions/masses. Degenerate fit branches can still differ in float/double, and local trajectory differences grow over time. No claim of universal pointwise identity or speedup.

**Validation:** Release app, full CPU numerical suite and four focused suites pass. Static drop/bubble/wall pressure jumps agree closely with GPU, CPU errors<0.83% and peak currents<0.01; moving/deforming cases pass. The corrected matched200-frame/4s96x72 crashing-wave comparison explicitly enables20000 and passes convergence, finite/bounds/walls and separate conservation/volume checks; saved broad motion agrees closely. Actual256x256 CPU sigma20000 first frame passes. Initial paired run inherited sigma0 and is retained only as a control; initial1024x720 startup description was incorrect. [Evidence, reproduction and limits](.codex/runs/run0023_20261029_cpu_surface_tension/README.md).

**Next unresolved decision:** select any next parity work: adaptive fields/tension, work counters, CPU/adaptive persistence or measured optimization trials. These remain proposals. No authorized implementation remains; do not begin another port without selection.
