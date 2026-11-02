#pragma once
#include "ElasticityGPU.h"
#include "PressureMultigridGPU.h"
#include "UniformPressureReduction.h"

namespace fluid::webgpu
{

// Coupled velocity/pressure MINRES with a determinant-preserving elastic update.
// Geometry is frozen during the solve; the caller transports the shared log history.
class ImplicitElasticityGPU
{
public:
    struct Result
    {
        int iterations = 0;
        double divergence = 0;
        double relative_residual = 0;
        bool converged = false;
        bool precision_limited = false;
    };

    ImplicitElasticityGPU(Compute& compute, int width, int height, float cell_size, float modulus);
    Result solve(double dt, double tolerance, int max_iterations, ElasticityGPU& elasticity, const Field& fraction,
        const Field& density, Field& u, Field& v, Field& pressure);

private:
    friend struct ImplicitElasticityGPUTestAccess;
    Compute& compute;
    Parameters parameters;
    PressureMultigridGPU multigrid;
    UniformPressureReduction reduction;
    Field material;
    Field corners;
    Field diagonal;
    Field rounding;
    Field coefficients;
    Field stress;
    Field corner_stress;
    Field solution;
    Field compensation;
    Field r1;
    Field r2;
    Field next_r;
    Field z;
    Field lanczos;
    Field product;
    Field w1;
    Field w2;
    Field next_w;
    Field pressure_rhs;
    Field terms;
    Field stats;
    Field alpha;
    Field beta;
    Field control;
    Field next_control;
    Field next_history;
    Kernel prepare;
    Kernel corner;
    Kernel build_diagonal;
    Kernel weights;
    Kernel strain;
    Kernel gather;
    Kernel matrix;
    Kernel initial;
    Kernel pressure_extract;
    Kernel precondition;
    Kernel normalize;
    Kernel recurrence;
    Kernel scalar_update;
    Kernel update;
    Kernel reduction_terms;
    Kernel residual;
    Kernel commit;
    Kernel clear;

    void apply(const Field& source, Field& destination);
    void trueResidual(Field& destination, const Field& u, const Field& v);
    void preconditionResidual();
    void reduce(const Field& a, const Field& b, Field& output, int mode);
};
}
