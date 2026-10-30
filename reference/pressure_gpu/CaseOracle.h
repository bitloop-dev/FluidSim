#pragma once
#include "PressureCase.h"
#include "AmrPressure2D.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace fluid::reference {
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }

struct CaseOracle
{
    PressureCase problem;
    AmrPressure2D cpu;
    AmrPressureTopology2D topology;
    PressureCase::Field cycleRhs, expectedCycle;

    explicit CaseOracle(PressureCase input) : problem(std::move(input)), cpu(problem.levels), topology(problem.levels)
    {
        cpu.setCoefficients(problem.beta);
        auto initial = problem.pressure;
        cycleRhs = cpu.makeField(); cpu.apply(cycleRhs, initial);
        for (int l = 0; l < int(cycleRhs.size()); ++l)
            for (int i = 0; i < int(cycleRhs[l].size()); ++i) cycleRhs[l][i] = problem.rhs[l][i] - cycleRhs[l][i];
        const double norm = cpu.norm(cycleRhs);
        for (auto& level : cycleRhs) for (auto& value : level) value /= norm;
        expectedCycle = cpu.makeField(); cpu.precondition(expectedCycle, cycleRhs);
    }

    void reportCycle(const PressureCase::Field& values) const
    {
        double error = 0, scale = 0;
        for (int l = 0; l < int(values.size()); ++l)
            for (const auto& row : topology.grids[l].rows)
                for (int i = row.begin; i < row.end; ++i)
                {
                    if (!std::isfinite(values[l][i])) throw std::runtime_error("Nonfinite cycle output");
                    error = std::max(error, std::abs(values[l][i] - expectedCycle[l][i]));
                    scale = std::max(scale, std::abs(expectedCycle[l][i]));
                }
        std::cout << "cycle_relative_error=" << error / std::max(1., scale) << '\n';
    }

    void reportResidual(PressureCase::Field values)
    {
        auto residual = cpu.makeField(); cpu.apply(residual, values);
        double maximum = 0;
        for (int l = 0; l < int(residual.size()); ++l)
            for (const auto& row : topology.grids[l].rows)
                for (int i = row.begin; i < row.end; ++i)
                {
                    residual[l][i] -= problem.rhs[l][i];
                    if (topology.grids[l].active[i]) maximum = std::max(maximum, std::abs(residual[l][i]));
                }
        const double norm = cpu.norm(residual);
        std::cout << "true_residual_l2=" << norm << ",true_residual_max=" << maximum << ",target_met=" << (norm <= problem.tolerance) << '\n';
    }
};
} // namespace fluid::reference
