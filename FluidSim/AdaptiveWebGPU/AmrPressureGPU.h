#pragma once
#include "../Adaptive/AmrPressureTopology2D.h"
#include "../UniformWebGPU/FluidWebGPUCompute.h"
#include <memory>

namespace fluid
{

// The source-derived composite pressure algorithm, with GPU-resident float fields.
class AmrPressureGPU
{
public:
    using Box = AmrPressureTopology2D::Box;
    using Level = AmrPressureTopology2D::Level;
    using Patch = AmrPressureTopology2D::Patch;
    using Field = std::vector<webgpu::Field>;
    using Faces = std::array<Field, 2>;
    struct Result { int iterations; double residual; bool converged; };

    AmrPressureGPU(webgpu::Compute& compute, const std::vector<Level>& levels);
    ~AmrPressureGPU();
    AmrPressureGPU(const AmrPressureGPU&) = delete;
    AmrPressureGPU& operator=(const AmrPressureGPU&) = delete;
    // Compensated pressure keeps the original high plane and a low plane at
    // width*height. Coefficients, RHS and Krylov/preconditioner fields stay float.
    Field makeField(bool compensated = false) const;
    const std::vector<Patch>& patches(int level) const;
    const AmrPressureTopology2D& topology() const;
    void setCoefficients(const Faces& coefficients);
    void apply(Field& output, Field& pressure);
    void fluxes(Faces& output, Field& pressure);
    void precondition(Field& output, const Field& rhs, bool bottomKrylov = false);
    Result solve(Field& pressure, const Field& rhs, double tolerance, int max_iterations = 2000);
    Result solveMultigrid(Field& pressure, const Field& rhs, double tolerance, int max_iterations = 200);
    double norm(const Field& field);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace fluid
