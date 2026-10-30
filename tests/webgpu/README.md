# Browser numerical tests

Build the `FluidSimGPUTests` target with the Emscripten preset. Run its existing C++ selectors in Chrome or Edge:

```powershell
python tests/webgpu/run.py --browser "C:/Program Files/Google/Chrome/Application/chrome.exe" --build build/wasm32-release-codex --output .codex/runs/<run>/browser-smoke.json -- --smoke
```

Other focused selectors are `--projection`, `--transport`, `--multigrid`, and `--pool-steps 48 0.001 64 4`. Use `--timeout <seconds>` before `--` for a longer case. The runner serves COOP/COEP headers, supplies arguments through `Module`, and records C++ output and the browser's actual exit status. No external Python packages or special WebGPU browser flags are required.

Use `--executable FluidPressureWebGPUTests -- smoke` for the adaptive pressure test, or `-- all` for its complete fixed-layout numerical suite. `--executable` selects any separately built numerical test using the same `main.cpp` exit wrapper; the default remains `FluidSimGPUTests`.

`main.cpp` is a browser-only wrapper around the unchanged numerical test entry point. It explicitly exits after that function and its local destructors return: Emdawn's queued device-loss promises can otherwise keep Emscripten's proxied application thread alive after a completed test. This harness does not change production device teardown or validate full-app scene-switch behavior.
