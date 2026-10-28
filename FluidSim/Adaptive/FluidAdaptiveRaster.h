#pragma once
#include "../FluidGridBase.h"
#include "FluidCellView.h"

namespace fluid
{
// Explicit finest raster observation shared by native CPU/GPU hierarchies.
inline FluidGridBase::State adaptiveRaster(const FluidCellView& view)
{
    const int width = view.width, height = view.height;
    const double cell_size = view.cellSize;
    FluidGridBase::State state;
    const size_t cells = size_t(width) * height;
    const int nl = view.liquidComponents, nc = view.components();
    state.liquid_fraction.resize(cells); state.pressure.resize(cells);
    state.liquid_mass.assign(nl, std::vector<double>(cells));
    state.gas_mass.assign(nc - nl, std::vector<double>(cells));
    state.u.resize(size_t(width + 1) * height); state.v.resize(size_t(height + 1) * width);
    FluidCellView::MaterialPlane plane;
    for (const auto& p : view.patches)
    for (int y = p.y; y < p.y + p.height; y += p.scale)
    for (int x = p.x; x < p.x + p.width; x += p.scale)
    {
        if (!p.active(x / p.scale, y / p.scale)) continue;
        if (p.scale > 1) view.reconstruct(plane, x, y, p.scale);
        for (int dy = 0; dy < p.scale; ++dy)
        for (int dx = 0; dx < p.scale; ++dx)
        {
            const int i = (y + dy) * width + x + dx;
            double fraction = 0.;
            for (int k = 0; k < nc; ++k)
            {
                const double q = p.scale == 1 ? p.q(x, y, k) :
                    plane.value(k, (dx + .5) / p.scale - .5, (dy + .5) / p.scale - .5);
                (k < nl ? state.liquid_mass[k] : state.gas_mass[k - nl])[i] = q * cell_size * cell_size;
                if (k < nl) fraction += q / view.mixture->components[k].density;
            }
            state.liquid_fraction[i] = fraction;
            state.pressure[i] = view.pressureInterval > 0. ? p.pressure(x / p.scale, y / p.scale) / view.pressureInterval : 0.;
        }
    }
    // Observation of native MAC faces: interpolate inside coarse cells and use
    // the resolved side at a level join. This raster is not a restart/checkpoint.
    auto face = [&](int axis, int x, int y)
    {
        if ((axis == 0 && (x == 0 || x == width)) || (axis == 1 && (y == 0 || y == height))) return 0.;
        auto c = view.at(x, y);
        const auto other = view.at(x - (axis == 0), y - (axis == 1));
        if (other.patch->scale < c.patch->scale) c = other;
        const double t = double(axis ? y : x) / c.patch->scale - (axis ? c.y : c.x);
        return std::lerp(c.face(axis, 0), c.face(axis, 1), t);
    };
    for (int y = 0; y < height; ++y) for (int x = 0; x <= width; ++x) state.u[y * (width + 1) + x] = face(0, x, y);
    for (int y = 0; y <= height; ++y) for (int x = 0; x < width; ++x) state.v[y * width + x] = face(1, x, y);
    return state;
}

}
