# Current work: native video capture complete

**Goal / authorized scope:** user approved the shared native capture hook and FluidSim video integration, requesting a minimal implementation and one short H.264 test. Complete. Preserve other WIP, physics/settings and the preceding snapshot repair. Parent [Main scene controls](WORK_STACK.md#main-scene-controls) refreshed; its other proposals remain parked.

**Implementation:** CaptureFrame/captureFrame replace the snapshot-only API, with shared worker preparation, first-frame source/dimension selection, pending recording FPS and cancellation, fixed-size native frames, and the existing encoder backpressure. Native recording leaves the preview independent. FluidSim_main reuses shaded image readback and serializes checkpoint/XMP only for snapshots. FFmpeg packed-row/odd-height copying and its previously ineffective frame limit are corrected.

**Validation:** final FluidSim Release app and framework test target build. One 12-frame H.264 source/encoder test passes 65 assertions; all frames decode at 66x48/12 fps from odd 67x49 input, with correct orientation and advancing frame markers. The same test first exposed the frame-limit bug and was rerun after correction. Manifest hashes and diff whitespace checked. [Evidence/clip](../../bitloop/.codex/runs/run0013_20261030_001853/README.md). No interactive FluidSim GPU/browser/H.265/full regression checks, as requested.

**Next action:** user will check live behavior. Rebuilt app: `../FluidSim_main/build/ninja-release-codex/app/app.exe` from the development worktree, or `build/ninja-release-codex/app/app.exe` from main. Existing running instances remain unchanged. No implementation remains for this request; other proposals need separate authorization.
