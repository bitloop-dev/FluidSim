#pragma once
#include "FluidGridTypes.h"
#include <array>
#include <span>
#include <vector>

namespace fluid
{

// One continuous incompressible elastic material, carried by the condensed-phase fluxes.
struct ElasticityCPU
{
    int width;
    int height;
    double cell_size;
    double shear_modulus;

    // Half diagonal difference, off-diagonal and half trace of log(B), per material volume.
    std::vector<std::array<double, 3>> history;

    // Normal deviatoric stress at cells (yy = -xx), shear stress at MAC corners.
    std::vector<double> normal_stress;
    std::vector<double> shear_stress;

    ElasticityCPU(int width, int height, double cell_size, double shear_modulus);
    void upload(const FluidGridState<double>& state);
    void download(FluidGridState<double>& state) const;
    void deform(double dt, std::span<const double> fraction, std::span<const double> u, std::span<const double> v);
    void advect(double dt, bool horizontal, std::span<const double> velocity, std::span<const double> fraction,
        std::span<const double> transfers, std::span<const unsigned char> split_weights);
    void rebuildStress(std::span<const double> fraction);
    double stepLimit(std::span<const double> inverse_density_u, std::span<const double> inverse_density_v) const;
    void applyForces(double dt, std::span<double> u, std::span<double> v, std::span<const double> inverse_density_u,
        std::span<const double> inverse_density_v) const;

private:
    std::vector<std::array<double, 3>> next_history;
    std::vector<std::array<double, 3>> source_history;
    std::vector<std::array<double, 3>> incoming_history;
    std::vector<double> original_share;
    std::vector<double> cell_shear;
    std::vector<double> cell_stiffness;
    std::vector<double> corner_stiffness;

    double coverage(int x, int y, std::span<const double> fraction) const;
};

}
