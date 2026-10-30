# Frozen GPU pressure comparison

Replay the same captured hierarchy, face coefficients, RHS and warm pressure in
our WebGPU solver and the pinned native AMReX GPU solver. This is a standalone
reference utility; AMReX and HIP are not application dependencies.

The main Desktop test build provides `FluidPressureWebGPUReference` when
`FLUIDSIM_BUILD_TESTS` is enabled:

```text
FluidPressureWebGPUReference case.fpc 3
FluidPressureWebGPUReference case.fpc 3 portable
```

Configure this directory separately with the pinned sources in
`build/amrex-reference-deps`, an appropriate HIP compiler, `AMReX_AMD_ARCH` and
`AMReX_PRECISION=SINGLE|DOUBLE`. It produces:

```text
FluidPressureNativeReference case.fpc 3
FluidPressureNativeReference case.fpc 3 synchronized
```

Native defaults to AMReX's single-stream/no-implicit-sync region. Each timed
batch explicitly waits for completion. Minimal AMReX initialization omits RNG,
signal handlers and asynchronous I/O; pressure algorithms are unchanged.

Both utilities warm the cycle/solve before reported samples. They report setup
and upload separately, batch eight preconditioner applications, then repeat full
solves from the original warm pressure. A shared double CPU oracle evaluates
the final active-cell L2 and maximum residual, independently of each solver's
internal stopping estimate. Cycle error compares the same normalized initial
residual against the extracted CPU preconditioner. Float AMReX may stop while
missing the true L2 target because its pressure has no compensated low plane;
retain that result instead of weakening the check.

`issue_ms` measures calls before final completion; native calls can themselves
wait. `drain_ms` includes final synchronization and, for WebGPU, command
submission/readback. They are not separate pure CPU and GPU execution times.

The `.fpc` v1 file stores float RHS/coefficients losslessly as doubles,
and reconstructs warm-pressure values from their high/low planes into doubles. It is a trusted local x64 test format.
Capture hooks are temporary diagnostics, not a production serialization API.

[Captured cases, native Windows toolchain details and measurements](../../.codex/runs/run0117_20261025_223233/README.md).
