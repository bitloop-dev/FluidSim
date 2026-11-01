#pragma once
#include "Shared/WebGPU/FluidWebGPUCompute.h"
#include "FluidGridEdit.h"

namespace fluid::webgpu
{

// Sparse MAC faces adjacent to a replacement reservoir; allocated only for velocity edits.
struct EditVelocityGPU
{
    Kernel apply;
    Field records;
    std::vector<std::array<float, 4>> horizontal;
    std::vector<std::array<float, 4>> vertical;
    std::vector<float> upload;

    explicit EditVelocityGPU(Compute& compute)
    {
        compute.build(apply, "fluid edit momentum", 2, {"density", "edits"},
            R"wgsl(
                fn execute()
                {
                    let record = edits(cell());
                    let p = vec2i(record.xy);
                    let horizontal = cell().x < params.grid.w;
                    let axis = select(vec2i(0, 1), vec2i(1, 0), horizontal);
                    let coordinate = select(p.y, p.x, horizontal);
                    let extent = select(params.grid.y, params.grid.x, horizontal);

                    // The reservoir does not open the domain's impermeable normal boundaries.
                    if (coordinate == 0 || coordinate == extent)
                    {
                        if (horizontal)
                        {
                            write0(p, vec4f(0));
                        }
                        else
                        {
                            write1(p, vec4f(0));
                        }

                        return;
                    }

                    var velocity = 0.0;
                    if (horizontal)
                    {
                        velocity = output0Data[p.y * (params.grid.x + 1) + p.x];
                    }
                    else
                    {
                        velocity = output1Data[p.y * params.grid.x + p.x];
                    }

                    let a = density(p - axis).xy;
                    let b = density(p).xy;
                    let edited = record.zw > vec2f(0);
                    let rho = select(vec2f(a.x + a.y, b.x + b.y), record.zw, edited);
                    let incoming = select(params.value.y, params.value.x, horizontal);
                    let speeds = select(vec2f(velocity), vec2f(incoming), edited);
                    let result = dot(rho, speeds) / (rho.x + rho.y);
                    if (horizontal)
                    {
                        write0(p, vec4f(result, 0, 0, 0));
                    }
                    else
                    {
                        write1(p, vec4f(result, 0, 0, 0));
                    }
                }
            )wgsl",
            {}, false, {64, 1});
    }

    void run(Compute& compute, const detail::CellEditBatch& batch, double cell_area, int width, int height, Field& u,
        Field& v, const Field& old_density)
    {
        horizontal.clear();
        vertical.clear();
        for (const auto& cell : batch.cells)
        {
            const float x = float(cell.x);
            const float y = float(cell.y);
            const float rho = float(cell.mass / cell_area);

            horizontal.push_back({x, y, 0, rho});
            horizontal.push_back({x + 1, y, rho, 0});
            vertical.push_back({x, y, 0, rho});
            vertical.push_back({x, y + 1, rho, 0});
        }

        auto merge = [](auto& faces)
        {
            std::sort(faces.begin(), faces.end(), [](const auto& a, const auto& b)
            {
                return a[1] < b[1] || (a[1] == b[1] && a[0] < b[0]);
            });

            size_t kept = 0;
            for (const auto face : faces)
            {
                if (kept && faces[kept - 1][0] == face[0] && faces[kept - 1][1] == face[1])
                {
                    faces[kept - 1][2] += face[2];
                    faces[kept - 1][3] += face[3];
                }
                else
                    faces[kept++] = face;
            }

            faces.resize(kept);
        };

        merge(horizontal);
        merge(vertical);
        const int count = int(horizontal.size() + vertical.size());
        if (records.width < count)
        {
            apply.bindings.clear();
            records = compute.field(count, 1, 4);
        }

        upload.clear();
        upload.reserve(size_t(count) * 4);

        for (const auto* faces : {&horizontal, &vertical})
        {
            for (const auto& face : *faces)
                upload.insert(upload.end(), face.begin(), face.end());
        }

        compute.write(records, upload);
        Parameters params;
        params.grid = {width, height, 0, int(horizontal.size())};
        params.value = {float(batch.velocity->x), float(batch.velocity->y), 0, 0};

        // One invocation owns each face, so in-place read/write has no neighbouring races.
        compute.run(apply, {&u, &v}, {{"density", &old_density}, {"edits", &records}}, params, {unsigned(count), 1, 1});
    }
};

}
