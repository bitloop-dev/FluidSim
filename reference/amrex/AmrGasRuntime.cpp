#include "AmrGasRuntime.h"
#include <AMReX.H>
#include <mutex>
#include <cstdlib>
#include <omp.h>

namespace fluid::reference {
namespace { std::mutex runtimeMutex; int users = 0; bool owned = false; }
AmrGasRuntime::AmrGasRuntime() {
    std::lock_guard lock(runtimeMutex);
    if (users == 0) {
        owned = !amrex::Initialized();
        if (owned) {
            amrex::Init_minimal();
            amrex::system::throw_exception = true;
        }
    }
    ++users;
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(8);
    workers=omp_get_max_threads();
}
AmrGasRuntime::~AmrGasRuntime() {
    std::lock_guard lock(runtimeMutex);
    if (--users == 0 && owned) amrex::Finalize_minimal();
}
}
