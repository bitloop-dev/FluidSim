#pragma once

namespace fluid {

inline constexpr const char* fluid_shader_src = R"(#version 300 es
    precision highp float;
    precision highp int;

    // RGBA layers contain four ingredient volume fractions each, liquid first
    // and then gas. Layers are stacked vertically, each u_grid_size.y rows high.
    uniform sampler2D u_fractions;
    uniform sampler2D u_liquid;
    // Each palette texel holds RGB colour and its ElementId in alpha.
    uniform sampler2D u_palette;
    uniform vec2 u_ingredient_counts; // total ingredients, liquid ingredients
    uniform vec2 u_grid_size;
    uniform sampler2D u_gas_seeds;
    uniform float u_depth_shading;
    uniform float u_depth_scale; // e-folding distance in grid cells
    uniform float u_deep_brightness;
    uniform sampler2D u_coarsening;
    uniform float u_coarsening_opacity;
    uniform sampler2D u_resolution;
    uniform float u_mask_opacity;
    in vec2 v_uv;
    out vec4 outColor;

    ivec2 gridCell(ivec2 cell) {
        return clamp(cell, ivec2(0), ivec2(u_grid_size) - 1);
    }
    float ingredientFraction(ivec2 cell, int ingredient) {
        ivec2 p = gridCell(cell) + ivec2(0, (ingredient / 4) * int(u_grid_size.y));
        return texelFetch(u_fractions, p, 0)[ingredient % 4];
    }
    int ingredientElement(int ingredient) {
        return int(round(texelFetch(u_palette, ivec2(ingredient, 0), 0).a));
    }
    vec3 ingredientColor(int ingredient) {
        return texelFetch(u_palette, ivec2(ingredient, 0), 0).rgb;
    }
    bool liquidIngredient(int ingredient) {
        return ingredient < int(u_ingredient_counts.y);
    }
    // Geometry is separate from composition. Uniform engines supply their
    // tracked liquid_fraction; AMR supplies its conserved liquid volume.
    float liquidFraction(ivec2 cell) {
        return texelFetch(u_liquid, gridCell(cell), 0).r;
    }
    vec3 materialColor(ivec2 cell) {
        vec3 color = vec3(0);
        for (int k = 0; k < int(u_ingredient_counts.x); ++k)
            color += ingredientFraction(cell, k) * ingredientColor(k);
        return color;
    }

    float distanceToGas(vec2 point) {
        vec4 encoded=texelFetch(u_gas_seeds,gridCell(ivec2(point)),0);
        // A completely liquid domain has no gas surface to illuminate it.
        if(all(equal(encoded,vec4(1.0)))) return 8.0*u_depth_scale;
        vec4 bytes=round(encoded*255.0);
        vec2 gas=vec2(bytes.x+256.0*bytes.y,bytes.z+256.0*bytes.w)/65534.0*u_grid_size;
        return max(0.0,length(point-gas)-0.5);
    }
    vec3 shadeLiquid(vec2 point,vec3 color) {
        ivec2 cell=ivec2(point);
        vec3 liquidColor=vec3(0);
        float liquidVolume=0.0;
        for(int k=0;k<int(u_ingredient_counts.y);++k) {
            float fraction=ingredientFraction(cell,k);
            liquidColor+=fraction*ingredientColor(k);
            liquidVolume+=fraction;
        }
        if(liquidVolume<=0.0) return color;
        vec3 base=liquidColor/liquidVolume;
        float depth=distanceToGas(point)/u_depth_scale;
        // A stylised absorption tint: a cool, light surface fades into the
        // ingredient's darker bulk colour. No scene background is sampled.
        vec3 shallow=mix(base,vec3(0.48,0.85,0.95),0.30);
        vec3 deep=base*u_deep_brightness;
        vec3 transmission=exp(-depth*vec3(1.25,0.90,0.75));
        vec3 shaded=mix(deep,shallow,transmission);
        return color+liquidVolume*(shaded-base);
    }

    void main()
    {
        vec2 cell = v_uv * u_grid_size;
        vec3 color=materialColor(ivec2(cell));
        if(u_depth_shading>0.5) color=shadeLiquid(cell,color);
        // The transparency illusion comes from colour; the image stays opaque.
        outColor = vec4(color, 1.0);
        if (u_coarsening_opacity > 0.0)
        {
            int label = int(round(255.0 * texelFetch(u_coarsening, ivec2(cell) / 2, 0).r));
            vec3 fill = vec3(0.55); // unassessed block/physical edge
            if (label == 2) fill = vec3(0.65, 0.3, 0.85); // protected liquid/buffer
            if (label == 3) fill = vec3(1.0, 0.7, 0.05); // retain fine detail / wait for quiet
            if (label == 4) fill = vec3(0.15, 0.85, 0.3); // candidate merge
            if (label == 5) fill = vec3(1.0, 0.15, 0.15); // resolved coarse detail
            if (label == 6) fill = vec3(1.0, 0.45, 0.05); // approaching feature
            // No fill means no detected refinement trigger, not known lost detail.
            if (label != 0) outColor.rgb = mix(outColor.rgb, fill, u_coarsening_opacity);
        }
        float scale = round(255.0 * texelFetch(u_resolution, ivec2(cell) / 2, 0).r);
        vec3 tint = scale > 16.5 ? vec3(0.95, 0.25, 0.2) : scale > 8.5 ? vec3(0.1, 0.85, 0.9) : scale > 4.5 ? vec3(0.85, 0.3, 0.9) : scale > 2.5 ? vec3(0.95, 0.55, 0.15) : scale > 1.5 ? vec3(0.15, 0.8, 0.25) : vec3(0.15, 0.35, 0.95);
        vec2 q = cell / scale;
        vec2 edge = min(fract(q), 1.0 - fract(q));
        vec2 line = 1.0 - smoothstep(vec2(0.0), fwidth(q) * 1.25, edge);
        if (scale >= 2.0) outColor.rgb = mix(outColor.rgb, tint, u_mask_opacity * max(line.x, line.y));
    }
)";

} // namespace fluid
