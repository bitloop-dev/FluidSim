# Current work: native resumable WebP snapshots

**Status: complete on main.** All four authorized pieces are implemented: generic native-size Snapshot hook, coherent owner-thread image/state capture, exact Brotli-1/base64 checkpoint embedded as XMP, image-based resume. Framework changes were explicitly approved; no timeline or other backend work was performed.

**Outcome:** Snapshot saves the full native fluid surface and exact state; drop the image onto a FluidSim viewport to resume. Existing worker/GUI queues and viewport fallback are reused. Shared scene helpers serve images and file checkpoints. Replacement grid is validated before releasing old state; geometry/backend/settings/clocks restore, malformed images leave the current state intact. Existing file-drop events avoid an additional file-dialog path.

**Validation:** actual1024x720 frame100 WebP (18.06MiB) -> fresh resume through original500 matches native fields/history/scene metadata exactly over400 further frames/8s. Running/paused toolbar, camera-independent bytes, packet/WebP tests, queued drop/geometry/backend/atomic rejection,18 framework assertions and normal/development Release builds pass. Numerical sources/current tolerance1 and .02 frame/two .01 substeps unchanged. [Evidence/limits/source delta](.codex/runs/run0015_20261028_164139/README.md). Browser/package-release gates untested; physical cross-window drag unavailable to automation, queued event path tested.

**Parent / next:** [four-backend API](WORK_STACK.md#four-backend-main-api) and follow-ups complete. No unresolved authorized implementation, commit or merge remains. Await the user's next request; timelines remain discussion only.
