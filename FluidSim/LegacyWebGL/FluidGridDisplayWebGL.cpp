#include "FluidGridDisplay.h"
#include "FluidGridWebGL.h"

namespace fluid
{
std::uint32_t FluidGridDisplay::texture(const FluidGridWebGL& grid) const
{
    return grid.colorTexture();
}
}
