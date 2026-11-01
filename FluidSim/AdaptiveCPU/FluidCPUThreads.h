#pragma once
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fluid
{
// Set the existing worker default only when the process has no explicit override.
inline void configureAdaptiveThreads()
{
#ifdef _OPENMP
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(8);
#endif
}
}
