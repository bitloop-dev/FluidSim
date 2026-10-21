#pragma once

namespace fluid::test
{
void coarsePressureRegressions();
void coarsePressureBenchmark(int size, int history, bool paired);
void coarsePressurePool(int steps, int size);
}
