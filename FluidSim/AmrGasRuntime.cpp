#include "AmrGasRuntime.h"
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace fluid {
AmrGasRuntime::AmrGasRuntime() {
#ifdef _OPENMP
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(8);
    workers=omp_get_max_threads();
#else
    workers=1;
#endif
}
AmrGasRuntime::~AmrGasRuntime() = default;
}
