#pragma once
#include "FluidWebGPUCompute.h"

namespace fluid::webgpu
{
struct PressureRecoveryKernels
{
    Kernel transport, geometry, color, rhs, smooth, copy, retain_pressure, health;

    void build(Compute &compute)
    {
        compute.build(rhs, "fluid temporal volume feedback", 1, {"u", "v", "occupied"}, R"wgsl(
fn execute() {
    let p = cell();
    let div = (u(p+vec2i(1,0)).x-u(p).x + v(p+vec2i(0,1)).x-v(p).x) / cellSize();
    let q = occupied(p).xy;
    let requested_divergence = params.value.w * (q.x+q.y-1.0);
    write0(p, vec4f((requested_divergence-div)*cellSize()*cellSize()/timeStep()));
}
)wgsl");
        compute.build(smooth, "fluid local recovery pressure", 1, {"source", "rhs", "weights"}, R"wgsl(
fn execute() {
    let p = cell(); let a = weights(p); let center = source(p).x;
    let diagonal = (a.x+a.y)+(a.z+a.w);
    let product = (a.x*(center-source(p-vec2i(1,0)).x)+a.y*(center-source(p+vec2i(1,0)).x))
        + (a.z*(center-source(p-vec2i(0,1)).x)+a.w*(center-source(p+vec2i(0,1)).x));
    var value = center;
    if (diagonal > 0.0) { value += (2.0/3.0)*(rhs(p).x-product)/diagonal; }
    write0(p, vec4f(value));
}
)wgsl");
        compute.build(copy, "fluid remember local recovery pressure", 1, {"source"}, R"wgsl(
fn execute() { write0(cell(),source(cell())); }
)wgsl");
        compute.build(health, "fluid recovery volume statistics", 1, {"occupied", "solver_state", "denominator"},
                      R"wgsl(
fn execute() {
    let base = cell()*4; var result = vec4f(0);
    let s=solver_state(vec2i(0)); let d=denominator(vec2i(0));
    if (mode()!=0 && (s.x<0.0 || d.w!=0.0 || (s.z>tolerance() && d.x<=0.0))) { result.w=1; }
    for (var y=0; y<4; y++) { for (var x=0; x<4; x++) {
        let p = base+vec2i(x,y); if (any(p>=gridSize())) { continue; }
        let q = occupied(p).xy; let e = q.x+q.y-1.0;
        result.x += e*e; result.y = max(result.y,abs(e));
        if (!finite(q.x) || !finite(q.y) || q.x+q.y<=0.0 || min(q.x,q.y)<-1e-6) { result.w=1; }
    } }
    write0(cell(),result);
}
)wgsl");
        compute.build(retain_pressure, "fluid recovery pressure memory", 1, {"base", "local", "correction"}, R"wgsl(
fn execute() {
    let p = cell(); let local_update = local(p).x;
    write0(p, vec4f(base(p).x + local_update + params.value.w*(correction(p).x-local_update)));
}
)wgsl");
        compute.build(transport, "fluid conservative recovery transport", 1, {"liquid", "masses", "transfers", "flow"},
                      R"wgsl(
fn transfer(p: vec2i, axis: vec2i) -> vec4f {
    let coordinate = select(p.y, p.x, mode() == 0);
    let extent = select(gridSize().y, gridSize().x, mode() == 0);
    if (coordinate == 0 || coordinate == extent) { return vec4f(0); }
    let volume = flow(p).x * timeStep() / cellSize();
    let donor = select(p, p-axis, volume > 0.0);
    let f = liquid(donor).x;
    let liquid_volume = transfers(p).x;
    var ratio = vec4f(0);
    if (f > 0.0) { ratio += params.vectors[0] * (liquid_volume / f); }
    if (f < 1.0) { ratio += params.vectors[1] * ((volume-liquid_volume) / (1.0-f)); }
    return masses(donor) * ratio;
}
fn execute() {
    let p = cell();
    let axis = select(vec2i(0,1), vec2i(1,0), mode() == 0);
    write0(p, masses(p) + (transfer(p,axis) - transfer(p+axis,axis)));
}
)wgsl");
        compute.build(geometry, "fluid compressed phase geometry", 1, {"occupied"}, R"wgsl(
fn execute() {
    let q = occupied(cell()).xy;
    write0(cell(), vec4f(q.x / (q.x+q.y)));
}
)wgsl");
        compute.build(color, "fluid compressed material colors", 1, {"source", "occupied"}, R"wgsl(
fn execute() {
    let value = source(cell());
    let q = occupied(cell()).xy;
    write0(cell(), vec4f(value.xyz / (q.x+q.y), value.w));
}
)wgsl");
    }
};
} // namespace fluid::webgpu
