#pragma once
#include "Shared/WebGPU/FluidWebGPUCompute.h"
#include "FluidGridTypes.h"

namespace fluid::webgpu
{

// Logarithmic strain transported by material volume; no reference outline or exterior history.
class ElasticityGPU
{
public:
    ElasticityGPU(Compute& compute, int width, int height, float cell_size, float shear_modulus);
    void upload(const FluidGridState<float>& state);

    void setShearModulus(float value)
    {
        parameters.value[3] = value;
        stress_valid = false;
        statistics_valid = false;
    }

    // Convert the packed displacement restored from a version-9 checkpoint.
    void importReference(const Field& fraction);
    void download(FluidGridState<float>& state);
    void deform(double dt, const Field& fraction, const Field& u, const Field& v);
    void advect(double dt, bool horizontal, const Field& velocity, const Field& fraction, const Field& transfers,
        const Field* split_weights = nullptr);
    void rebuildStress(const Field& fraction);
    double stepLimit(const Field& density);
    void applyForces(double dt, Field& u, Field& v, Field& next_u, Field& next_v, const Field& density);

    // Packed (half difference, off-diagonal, half trace) of log(B).
    Field history;
    Field normal;
    Field shear;
    bool stress_valid = false;
    bool statistics_valid = false;

private:
    Compute& compute;
    Parameters parameters;
    Field next_history;
    Field correction;
    Field incoming;
    Field partials;
    Field statistics;
    Kernel deformation;
    Kernel transport_x;
    Kernel transport_y;
    Kernel stress;
    Kernel forces;
    Kernel reduce;
    Kernel finish;
    double cached_limit = 0;
};
}
