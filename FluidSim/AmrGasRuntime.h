#pragma once

namespace fluid {
// Configure the simulation worker's CPU thread count; no external runtime.
class AmrGasRuntime {
public:
    AmrGasRuntime();
    ~AmrGasRuntime();
    int workerCount() const { return workers; }
    AmrGasRuntime(const AmrGasRuntime&) = delete;
    AmrGasRuntime& operator=(const AmrGasRuntime&) = delete;
private:
    int workers=8;
};
}
