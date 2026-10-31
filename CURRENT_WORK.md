# Current work: combined uniform WebGPU timestep limits

The scene now avoids nested rounding of its minimum step count and the engine's capillary count. Default CrashingWave uses five complete 4 ms steps per .02-second frame instead of six 3.333 ms steps, satisfying both limits. Every step still runs transport, forces and pressure. Preset values and physics algorithms are unchanged; CPU/adaptive scene scheduling is unchanged.

Native Release, 32 simulated seconds: mean frame time including synchronized surface rendering fell from 77.28 to 72.51 ms (6.2%). Fluid work fell about 8%. A separate short profile pair attributes the saving to less pressure work, partly offset by increased transport subdivisions. Larger steps change the trajectory, so the gain varies with wave phase.

Both long runs completed without solver failure and stayed within existing sampled mass, volume, fraction and divergence budgets. Native colour fields were inspected at 8 and 32 seconds. The final desktop app builds; no test suite was run. Temporary probe source and build integration are removed after archiving. [Evidence, reproduction and limits](.codex/runs/run0142_20261031_140109/README.md).

## Previous work: temporal pressure recovery implemented

**Scope:** the user authorized implementing the proposed multi-frame volume recovery experiment on uniform WebGPU. Existing materials and accurate defaults are preserved. This adds actual conservative compression and volume feedback, beyond the earlier tolerance experiment.

**Control:** `Pressure recovery frames`, 0-10. Zero is the original accurate solver/transport; positive values target gradual restoration over simulated output frames. Crossing zero restarts on release; positive changes are live. The scene sends a duration to the engine. Native/scene checkpoints preserve the model and extra-update cadence; older files load as accurate mode.

**Implementation:** relaxed transport conserves component mass, derives occupied rest volume and normalized phase geometry, and feeds volume error into pressure. An accurate seed is followed by two local sweeps and three bounded global PCG updates per physical substep, with one extra update at four opportunities per recovery horizon. Retained pressure is damped. The original sparse-global proposal became unstable at frame 46 and was rejected; moving density jumps need the minimum global budget.

**Measured:** 32-second normal-profile runs, including synchronized native shading. CrashingWave averages 78.1 / 64.4 / 62.7 ms per frame at 0 / 5 / 10. SplashCubes averages 70.4 / 58.8 / 59.0 ms. Roughly 16-20% lower frame time overall; pressure itself is about halved in the developed wave. No reduced grid, gravity, surface tension, gas model or rendering quality.

**Validation:** all 23 native suites pass; focused recovery/scene tests pass after the final integration edits. Both zero-mode 400-frame runs match the pre-implementation executable's work counts and 10 sampled hashes exactly. All 0/1/2/5/10 settings completed eight seconds in both scenes; 0/5/10 completed 32 seconds. Component conservation/finite-state checks pass. Isolated 5/10-frame disturbances lose about 88-89% of RMS volume error over the requested horizon. UI transitions, exact small-fixture checkpoint continuation, invalid checkpoint rejection, edits, hydrostatics and capillarity are covered.

**Limits:** experimental, with increased local compression and capillary currents. Worst sampled RMS occupancy error is about .30%, but individual interface cells reach about 45% error. Relaxed full-resolution cubes are not bit-for-bit repeatable in every run; repeated timing and health measurements agree, but the exact cause of late state-hash differences was not established. No long-term artificial-life or browser-performance claim. Native Release app is built in `build/msvc-release-codex/app/app.exe`.

[Measurements and reproduction](.codex/runs/run0139_20261030_224858/README.md). [Revised implementation plan](.codex/plans/20261030_temporal_pressure_relaxation.md). Future optimization and new physics remain separate work.
