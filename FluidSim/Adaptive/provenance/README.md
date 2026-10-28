# Adaptive import provenance

Imported from the actual working files in sibling `FluidSim`, branch `fluid-adaptive-cells`, on 2026-10-27. The working tree includes uncommitted numerical work; a Git commit alone is not the source snapshot. Per-file hashes, integration differences and regression evidence are retained in `.codex/runs/run0003_20261027_164329/` from the project root.

The project-owned native AMR hierarchy, transfer, reconstruction and pressure core are under `FluidSim/Adaptive/`. Shared main catalogue/mixture/host-state types were preserved. Observer/selection helpers and their tests only rename the old uniform class to FluidGridCPU, plus a forward declaration in the preview header. Core changes only expose an optional caller iteration limit (native defaults preserved) and type the existing convergence exception; no pressure equations or kernels changed.

The extracted pressure algorithms derive from AMReX commit `015c14b65715b277fbc54706ebe7455b196d2254` (26.10). Retain [AMReX-LICENSE](AMReX-LICENSE) for source/binary redistribution. [Original source map](SOURCE_MAP.md) is copied unchanged from development `reference/pressure2d/README.md`; its paths/results describe that source checkout, not a migrated reference build in this main tree. Main builds without an AMReX dependency, using the existing project-owned core and compiler OpenMP runtime.

## Adaptive WebGPU migration

The actual native GPU implementation and GPU refinement policy were imported on 2026-10-27 under `../AdaptiveWebGPU/` from the same development working tree. Pressure/transport/reconstruction/selection kernels are unchanged; native bridge edits expose explicit iteration ceilings, type the existing failure, and reuse colour output storage. A separate explicit edit kernel supplies the established common source/sink semantics; it is outside the ordinary physics step. Source snapshots, inverse source proof and all16 regression results are in `.codex/runs/run0006_20261027_202107/` from the project root. No upstream reference project, oracle target, checkout/package or fluid-amrex overlay was imported. Preserve the existing adapted-code attribution and license.
