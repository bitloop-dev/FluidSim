> Historical optional reference packaging. The production manifest no longer depends on this port; the normal application does not register an overlay or find AMReX. Use `reference/` for current oracle tests.

# Pinned native CPU dependency

This FluidSim-owned vcpkg overlay builds AMReX 26.10 and AMReX-Hydro from the
exact commits used by `reference/README.md`: 2D, double precision, OpenMP, no MPI,
Fortran, particles, embedded boundaries or GPU backend. Both upstream projects
use the BSD 3-Clause license. Their copyrights are installed by the port.

The Windows patch changes diagnostic/backtrace thread storage to C++
`thread_local`, gives two OpenMP copy loops scalar bounds for MSVC, and replaces
legacy library symlinks with copies. Pressure, interpolation and transport
arithmetic are unchanged. MSVC uses `/openmp:llvm`; consumers must use that same
OpenMP runtime. The package is currently qualified on native Windows CPU builds.

The standalone FluidSim CMake registers this overlay before manifest discovery.
When composing FluidSim into another project with the current Bitloop manifest
merger, supply `-DVCPKG_OVERLAY_PORTS=<absolute FluidSim path>/ports` at configure
time: that merger does not propagate project overlay paths. Registry baselines
are unchanged. The native AMR dependency is excluded from Emscripten manifests.

Run0072 records the package installation, independent installed-package consumer
build and all 18 executable projection checks. The separate Clang reference
remains useful for checking the upstream numerical contract independently of
the Windows portability patch.
