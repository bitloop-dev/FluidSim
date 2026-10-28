#pragma once
#include <bitloop/util/json.h>
#include <cstddef>
#include <vector>

namespace fluid
{

// Backends own their versioned metadata and native payload; the scene owns file I/O.
struct FluidCheckpoint
{
    bl::JSON::json metadata;
    std::vector<std::byte> data;
};

}
