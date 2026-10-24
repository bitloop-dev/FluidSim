#pragma once

namespace fluid::reference {
// Scene-owned lease: engines are destroyed before their lease is released.
// Bitloop constructs/processes/destroys scenes on its simulation worker.
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
