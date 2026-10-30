# Current work: uniform WebGPU pressure-tolerance sweep measured

**Goal / authorized scope:** try relaxing pressure accuracy on uniform WebGPU before designing multi-frame volume recovery. Measure .1/.25/.5/1 in native Release CrashingWave; preserve materials, algorithms and existing uncommitted work.

**Measurements:** 1024x720, dt .02, gravity 500, sigma 20000, current .1/.1 atmosphere. Initial profiled frames 201–400: total frame 95.34/97.17/92.00/91.63 ms respectively. At 1.0 pressure falls 17.3%, but extra transport work limits total saving to 3.9%. A repeated .1/1.0 comparison gives 98.72/92.71 ms (6.1%). A separate normal-profile 32-second pair shows 9.8–12.0% lower total frame time in developed windows. All timing includes synchronized native shading, excludes app UI/presentation and diagnostic state scans.

**Health / limits:** .1 and 1.0 each complete 1,600 frames / 32 seconds / 9,600 physical substeps. All projections converge; sampled complete states pass existing finite/conservation/fraction/wall/phase-volume gates. Worst local phase disagreement is .08514% versus .09076% of a cell (budget .2%). Mass drift stays below .00457% (budget .01%). Normal/profile first-400-frame work counts and all shared sampled hashes match. This is a bounded wave experiment, not long-run evolution or a 5–10-frame volume-restoration guarantee. Existing static-capillarity accuracy remains a reason not to silently raise the default.

**Outcome:** no production code/default changes; the existing Pressure Quality control can try 1.0. Temporary probe/build hook removed after archiving. [Evidence, CSVs and reproduction](.codex/runs/run0138_20261030_213556/README.md). The earlier chemistry refactor overhead comparison remains in [run0137](.codex/runs/run0137_20261030_172434/README.md); the user accepted that the recalled regression was not reproduced.

**Next scope:** discuss whether the modest accuracy/speed trade-off warrants changing a default or investigating a separate integration strategy. Explicit volume-error recovery and reducing the six capillary-driven complete projections per frame are unimplemented modelling proposals, not authorized follow-on work.
