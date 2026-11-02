#include "FluidCoarseningPreview.h"
#include "AmrHierarchy2D.h"

namespace fluid
{
AmrLayout FluidCoarseningPreview::clusterRequests(
    int width, int height, const std::vector<Requests>& requests, Trace* trace)
{
    if (trace)
        trace->fallback_scale = 0;

    auto mark = [&](std::vector<std::uint8_t>& tags, int tile, int x0, int y0, int x1, int y1)
    {
        const int nx = (width + tile - 1) / tile;
        for (int y = std::max(0, y0) / tile; y <= std::min(height - 1, y1) / tile; ++y)
        {
            for (int x = std::max(0, x0) / tile; x <= std::min(width - 1, x1) / tile; ++x)
                tags[y * nx + x] = 1;
        }
    };

    auto cluster = [&](const std::vector<std::uint8_t>& tags, int tile)
    {
        const int nx = (width + tile - 1) / tile;
        const int ny = (height + tile - 1) / tile;
        std::vector<std::array<int, 2>> points;

        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                if (tags[y * nx + x])
                    points.push_back({x, y});
            }
        }

        std::vector<AmrPatch> patches;
        if (points.empty())
            return patches;

        for (const auto& box : clusterAmrTags(std::move(points), 0.7))
        {
            const int x = tile * box.smallEnd(0);
            const int y = tile * box.smallEnd(1);
            patches.push_back({x, y, std::min(width, tile * (box.bigEnd(0) + 1)) - x,
                std::min(height, tile * (box.bigEnd(1) + 1)) - y});
        }

        return patches;
    };

    auto nearlyFull = [&](const std::vector<AmrPatch>& patches)
    {
        int area = 0;
        for (const auto& p : patches)
            area += p.width * p.height;

        return area >= 3 * width * height / 4;
    };

    AmrLayout layout{requests.back().scale, {}};

    // Parent padding is topological; retained history never becomes a new detail source.
    for (size_t lev = 0; lev < requests.size(); ++lev)
    {
        const auto& request = requests[lev];
        const int tile = std::max(16, request.scale);
        auto tags = request.tiles;
        if (!layout.refinedPatches.empty())
        {
            for (const auto& p : layout.refinedPatches.back())
            {
                mark(tags, tile, p.x - request.scale, p.y - request.scale, p.x + p.width + request.scale - 1,
                    p.y + p.height + request.scale - 1);
            }
        }
        if (trace)
            trace->levels[lev].nested = tags;

        auto patches = cluster(tags, tile);
        if (nearlyFull(patches))
        {
            layout.baseRatio = request.scale / 2;
            if (trace)
                trace->fallback_scale = layout.baseRatio;
            break;
        }

        layout.refinedPatches.push_back(std::move(patches));
    }

    std::reverse(layout.refinedPatches.begin(), layout.refinedPatches.end());

    while (!layout.refinedPatches.empty() && layout.refinedPatches.back().empty())
        layout.refinedPatches.pop_back();

    return layout;
}
}
