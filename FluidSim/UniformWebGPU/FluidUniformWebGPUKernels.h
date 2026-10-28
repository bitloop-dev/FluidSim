#pragma once
#include "FluidWebGPUCompute.h"
#include "FluidWebGPUShaders.h"

namespace fluid::webgpu {

struct UniformKernels
{
    Kernel fill, density_pass, color_pass, pressure_coefficient, divergence, pressure_operator;
    Kernel predict_pressure, add_pressure, subtract_mean, residual, reduce_pass, precondition, coefficients, direction, update_pressure;
    Kernel correct_velocity, gravity, damping, velocity_stats, reconstruct, freeze_weights, transfers;
    Kernel transport_components_x, transport_components_y, transport_momentum, transport_liquid, momentum, recover;

    void build(Compute& compute)
    {
        compute.build(fill, "fluid fill", 1, {}, R"wgsl(
fn execute() { write0(cell(), select(vec4f(timeStep()), vec4f(0, 0, 0, 1), mode() == 1)); }
)wgsl");
        compute.build(density_pass, "fluid phase densities", 1, {"masses0", "masses1", "masses2", "masses3", "previous"}, R"wgsl(
fn phases(mass: vec4f, group: u32) -> vec2f {
    return vec2f(dot(mass, params.vectors[group]), dot(mass, params.vectors[group + 4u])) / (cellSize() * cellSize());
}
fn execute() {
    let p = cell();
    var sum = phases(masses0(p), 0u);
    if (params.grid.w > 1) { sum += phases(masses1(p), 1u); }
    if (params.grid.w > 2) { sum += phases(masses2(p), 2u); }
    if (params.grid.w > 3) { sum += phases(masses3(p), 3u); }
    if (mode() != 0) { sum += previous(p).xy; }
    write0(p, vec4f(sum, 0, 0));
}
)wgsl");
        compute.build(color_pass, "fluid catalogue colors", 1, {"masses0", "masses1", "masses2", "masses3", "previous"}, R"wgsl(
fn color(mass: vec4f, group: u32) -> vec3f {
    return vec3f(dot(mass, params.vectors[group]), dot(mass, params.vectors[group + 4u]), dot(mass, params.vectors[group + 8u]));
}
fn execute() {
    let p = cell();
    var sum = color(masses0(p), 0u);
    if (params.grid.w > 1) { sum += color(masses1(p), 1u); }
    if (params.grid.w > 2) { sum += color(masses2(p), 2u); }
    if (params.grid.w > 3) { sum += color(masses3(p), 3u); }
    if (mode() != 0) { sum += previous(p).xyz; }
    write0(p, vec4f(sum, 1));
}
)wgsl");
        compute.build(pressure_coefficient, "fluid pressure weights", 1, {"density"}, R"wgsl(
fn rho(p: vec2i) -> f32 { let phases = density(p).xy; return phases.x + phases.y; }
fn execute() {
    let p = cell();
    let center = rho(p);
    var weights = vec4f(2) / (vec4f(center) + vec4f(rho(p-vec2i(1,0)), rho(p+vec2i(1,0)), rho(p-vec2i(0,1)), rho(p+vec2i(0,1))));
    if (p.x == 0) { weights.x = 0; }
    if (p.x + 1 == gridSize().x) { weights.y = 0; }
    if (p.y == 0) { weights.z = 0; }
    if (p.y + 1 == gridSize().y) { weights.w = 0; }
    write0(p, weights);
}
)wgsl");
        compute.build(divergence, "fluid divergence", 1, {"u", "v"}, R"wgsl(
fn execute() {
    let p = cell();
    var value = (u(p+vec2i(1,0)).x-u(p).x + v(p+vec2i(0,1)).x-v(p).x) / cellSize();
    if (mode() != 0) { value = -value * cellSize() * cellSize() / timeStep(); }
    write0(p, vec4f(value));
}
)wgsl");
        compute.build(velocity_stats, "fluid divergence and CFL", 1, {"u", "v"}, R"wgsl(
fn execute() {
    let base = cell()*4;
    var result = vec4f(0);
    for (var y=0; y<4; y++) { for (var x=0; x<4; x++) {
        let p = base+vec2i(x,y);
        if (any(p >= gridSize())) { continue; }
        let left = u(p).x; let right = u(p+vec2i(1,0)).x;
        let up = v(p).x; let down = v(p+vec2i(0,1)).x;
        let div = (right-left+down-up)/cellSize();
        let outflow = max(right,0.0)+max(-left,0.0)+max(down,0.0)+max(-up,0.0);
        if (!finite(div) || !finite(outflow)) { result.w=1; continue; }
        result.x += div; result.y=max(result.y,abs(div)); result.z=max(result.z,outflow);
    } }
    write0(cell(), result);
}
)wgsl");
        compute.build(pressure_operator, "fluid pressure operator", 1, {"source", "weights"}, R"wgsl(
fn execute() {
    let p=cell(); let center=source(p).x; let a=weights(p);
    let value=(a.x*(center-source(p-vec2i(1,0)).x)+a.y*(center-source(p+vec2i(1,0)).x)) +
              (a.z*(center-source(p-vec2i(0,1)).x)+a.w*(center-source(p+vec2i(0,1)).x));
    write0(p,vec4f(value));
}
)wgsl");
        compute.build(predict_pressure, "fluid pressure predictor", 1, {"latest", "previous"}, R"wgsl(
fn execute() { write0(cell(),vec4f(2.0*latest(cell()).x-previous(cell()).x)); }
)wgsl");
        compute.build(add_pressure, "fluid accumulate pressure", 1, {"base", "correction"}, R"wgsl(
fn execute() { write0(cell(),vec4f(base(cell()).x+correction(cell()).x)); }
)wgsl");
        compute.build(subtract_mean, "fluid pressure mean", 1, {"source", "stats"}, R"wgsl(
fn execute() { write0(cell(),vec4f(source(cell()).x-stats(vec2i(0)).x/f32(gridSize().x*gridSize().y))); }
)wgsl");
        compute.build(residual, "fluid true residual", 1, {"rhs", "product"}, R"wgsl(
fn execute() { write0(cell(),vec4f(rhs(cell()).x-product(cell()).x)); }
)wgsl");
        compute.build(reduce_pass, "fluid reduction", 1, {"source", "other"}, R"wgsl(
fn execute() {
    let base=cell()*4; var result=vec4f(0);
    for (var y=0; y<4; y++) { for (var x=0; x<4; x++) {
        let p=base+vec2i(x,y); if (any(p>=sourceSize())) { continue; }
        var value=source(p);
        if (mode()==0) { value=vec4f(value.x,abs(value.x),0,select(1.0,0.0,finite(value.x))); }
        if (mode()==1) { value=vec4f(value.x*other(p).x,0,0,0); }
        if (mode()==2) { value=vec4f(value.x,0,value.x*other(p).x,0); }
        if (!finite(value.x)||!finite(value.y)||!finite(value.z)) { value=vec4f(0,0,0,1); }
        result.x+=value.x; result.y=max(result.y,value.y); result.w=max(result.w,value.w);
        if (mode()==4) { result.z=max(result.z,value.z); } else { result.z+=value.z; }
    } }
    write0(cell(),result);
}
)wgsl");
        compute.build(precondition, "fluid diagonal preconditioner", 1, {"residual", "stats", "weights"}, R"wgsl(
fn execute() {
    let mean=stats(vec2i(0)).x/f32(gridSize().x*gridSize().y); let a=weights(cell());
    let diagonal=(a.x+a.y)+(a.z+a.w); var value=0.0;
    if (diagonal>0.0) { value=(residual(cell()).x-mean)/diagonal; }
    write0(cell(),vec4f(value));
}
)wgsl");
        compute.build(coefficients, "fluid PCG scalars", 1, {"residual_stats", "correction_stats", "previous", "denominator"}, R"wgsl(
fn execute() {
    let r=residual_stats(vec2i(0)); let z=correction_stats(vec2i(0)); let old=previous(vec2i(0));
    let mean=z.x/f32(gridSize().x*gridSize().y); let rho=z.z-mean*r.x; var failed=max(r.w,z.w);
    if (!finite(rho)||(rho<=0.0 && r.y>tolerance())) { failed=1; }
    if (mode()==0) {
        let d=denominator(vec2i(0)); failed=max(failed,select(0.0,1.0,old.x<0.0));
        if (old.z>tolerance()&&(d.x<=0.0||d.w!=0.0)) { failed=1; }
    }
    var iterations=0.0;
    if (mode()!=1) { iterations=old.w+select(0.0,1.0,old.z>tolerance()&&old.x>0.0); }
    write0(cell(),vec4f(select(rho,-1.0,failed!=0.0),mean,r.y,iterations));
}
)wgsl");
        compute.build(direction, "fluid PCG direction", 1, {"correction", "old_direction", "state", "previous"}, R"wgsl(
fn execute() {
    let s=state(vec2i(0)); let old=previous(vec2i(0));
    if (s.x<0.0) { write0(cell(),old_direction(cell())); return; }
    var beta=0.0; if (mode()!=1 && old.x>0.0) { beta=s.x/old.x; }
    write0(cell(),vec4f(correction(cell()).x-s.y+beta*old_direction(cell()).x));
}
)wgsl");
        compute.build(update_pressure, "fluid PCG update", 2, {"pressure", "residual", "direction", "product", "state", "denominator"}, R"wgsl(
fn execute() {
    let p=cell(); let s=state(vec2i(0)); let d=denominator(vec2i(0));
    if (s.z<=tolerance()||s.x<0.0||d.x<=0.0||d.w!=0.0) { write0(p,pressure(p)); write1(p,residual(p)); return; }
    let alpha=s.x/d.x;
    write0(p,vec4f(pressure(p).x+alpha*direction(p).x));
    write1(p,vec4f(residual(p).x-alpha*product(p).x));
}
)wgsl");
        compute.build(correct_velocity, "fluid pressure gradient", 1, {"velocity", "pressure", "weights"}, R"wgsl(
fn execute() {
    let p=cell(); let axis=select(vec2i(0,1),vec2i(1,0),mode()==0);
    let coordinate=select(p.y,p.x,mode()==0); let extent=select(gridSize().y,gridSize().x,mode()==0);
    if (coordinate==0||coordinate==extent) { write0(p,vec4f(0)); return; }
    let gradient=(pressure(p).x-pressure(p-axis).x)/cellSize(); let a=weights(p);
    write0(p,vec4f(velocity(p).x-gradient*timeStep()*select(a.z,a.x,mode()==0)));
}
)wgsl");
        compute.build(damping, "fluid gas damping", 1, {"velocity", "liquid"}, R"wgsl(
fn execute() {
    let p=cell(); let offset=select(vec2i(0,1),vec2i(1,0),mode()==0);
    let inside=select((p.y > 0 && p.y < gridSize().y), (p.x > 0 && p.x < gridSize().x), mode() == 0);
    let gasOnly=inside && liquid(p).x<=0.0 && liquid(p-offset).x<=0.0;
    write0(p,velocity(p)*select(1.0,timeStep(),gasOnly));
}
)wgsl");
        compute.build(gravity, "fluid gravity", 1, {"velocity"}, R"wgsl(
fn execute() { write0(cell(),vec4f(select(velocity(cell()).x+timeStep(),0.0,cell().y==0||cell().y==gridSize().y))); }
)wgsl");
        compute.build(reconstruct, "fluid MYC reconstruction", 1, {"liquid"}, R"wgsl(
fn execute() {
    let p=cell(); let f=liquid(p).x;
    if (f<=0.0||f>=1.0) { write0(p,vec4f(0)); return; }
    var samples: array<f32,9>;
    for (var y=0; y<3; y++) { for (var x=0; x<3; x++) { samples[y*3+x]=liquid(p+vec2i(x-1,y-1)).x; } }
    let normal=plicNormal(samples); write0(p,vec4f(normal,plicOffset(normal,f),0));
}
)wgsl", geometry);
        compute.build(transfers, "fluid geometric phase transfer", 1, {"liquid", "interfaces", "flow", "density"}, R"wgsl(
fn execute() {
    let p=cell(); let axis=select(vec2i(0,1),vec2i(1,0),mode()==0);
    let coordinate=select(p.y,p.x,mode()==0); let extent=select(gridSize().y,gridSize().x,mode()==0);
    if (coordinate==0||coordinate==extent) { write0(p,vec4f(0)); return; }
    let fluid=flow(p).x*timeStep()/cellSize(); let donor=select(p,p-axis,fluid>0.0);
    let fraction=liquid(donor).x; var amount=0.0;
    if (fraction>=1.0) { amount=abs(fluid); }
    else if (fraction>0.0 && fluid!=0.0) {
        var lo=vec2f(0); var hi=vec2f(1);
        if (fluid>0.0) { lo=vec2f(axis)*(1.0-abs(fluid)); } else { hi-=vec2f(axis)*(1.0-abs(fluid)); }
        let g=interfaces(donor); amount=plicArea(g.xy,g.z,lo,hi);
    }
    let donor_liquid=clamp(fraction,0.0,1.0); let strip=abs(fluid);
    amount=clamp(amount,max(0.0,strip-(1.0-donor_liquid)),min(strip,donor_liquid))*sign(fluid);
    var ratios=vec2f(0);
    if (fraction>0.0) { ratios.x=amount/fraction; }
    if (fraction<1.0) { ratios.y=(fluid-amount)/(1.0-fraction); }
    write0(p,vec4f(amount,dot(density(donor).xy,ratios),0,0));
}
)wgsl", geometry);
        compute.build(freeze_weights, "fluid split weights", 1, {"liquid", "density"}, R"wgsl(
fn execute() {
    let f=liquid(cell()).x; let phases=density(cell()).xy;
    if (f>0.5) { write0(cell(),vec4f(1,phases.x/f,0,0)); }
    else { write0(cell(),vec4f(0,phases.y/(1.0-f),0,0)); }
}
)wgsl");
        const std::string component_helpers = R"wgsl(
fn transfer(p: vec2i, axis: vec2i) -> vec4f {
    let coordinate=select(p.y,p.x,mode()==0); let extent=select(gridSize().y,gridSize().x,mode()==0);
    if (coordinate==0||coordinate==extent) { return vec4f(0); }
    let volume=flow(p).x*timeStep()/cellSize(); let donor=select(p,p-axis,volume>0.0);
    let f=liquid(donor).x; let liquid_volume=transfers(p).x; var ratio=vec4f(0);
    if (f>0.0) { ratio+=params.vectors[0]*(liquid_volume/f); }
    if (f<1.0) { ratio+=params.vectors[1]*((volume-liquid_volume)/(1.0-f)); }
    return masses(donor)*ratio;
}
fn transport(p: vec2i, weight: vec4f) -> vec4f {
    let axis=select(vec2i(0,1),vec2i(1,0),mode()==0);
    let before=transfer(p,axis)-weight*(flow(p).x*timeStep()/cellSize());
    let after=transfer(p+axis,axis)-weight*(flow(p+axis).x*timeStep()/cellSize());
    return masses(p)+(before-after);
}
)wgsl";
        compute.build(transport_components_x, "fluid ingredients X", 2, {"liquid", "masses", "transfers", "flow"}, R"wgsl(
fn execute() {
    let p=cell(); let f=liquid(p).x;
    let weight=masses(p)*select(params.vectors[1],params.vectors[0],f>0.5)/select(1.0-f,f,f>0.5);
    write1(p,weight); write0(p,transport(p,weight));
}
)wgsl", component_helpers);
        compute.build(transport_components_y, "fluid ingredients Y", 1, {"liquid", "masses", "transfers", "flow", "corrections"}, R"wgsl(
fn execute() { write0(cell(),transport(cell(),corrections(cell()))); }
)wgsl", component_helpers);
        const std::string momentum_helpers = R"wgsl(
fn momentumValue(p: vec2i) -> f32 {
    let horizontal=mode()==0||mode()==2; let axis=select(vec2i(0,1),vec2i(1,0),horizontal);
    let coordinate=select(p.y,p.x,horizontal); let extent=select(gridSize().y,gridSize().x,horizontal);
    if (coordinate==0||coordinate==extent) { return 0.0; }
    let a=density(p-axis).xy; let b=density(p).xy; let rho=0.5*((a.x+a.y)+(b.x+b.y));
    if (mode()<2) { return rho*source(p).x; } return source(p).x/rho;
}
)wgsl";
        compute.build(momentum, "fluid initialize momentum", 2, {"density", "source"}, R"wgsl(
fn execute() { write0(cell(),vec4f(momentumValue(cell()))); write1(cell(),source(cell())); }
)wgsl", momentum_helpers);
        compute.build(recover, "fluid recover velocity", 1, {"density", "source"}, R"wgsl(
fn execute() { write0(cell(),vec4f(momentumValue(cell()))); }
)wgsl", momentum_helpers);
        compute.build(transport_momentum, "fluid staggered momentum transfer", 1, {"momentum", "donor_velocity", "original_velocity", "flow", "weights", "transfers"}, R"wgsl(
fn massCorrection(p: vec2i, axis: vec2i) -> f32 { return weights(p).y*(timeStep()/cellSize())*(flow(p+axis).x-flow(p).x); }
fn momentumTransfer(amount: f32, before: f32, after: f32) -> f32 { return amount*select(after,before,amount>=0.0); }
fn execute() {
    let hc=mode()==0||mode()==2; let ht=mode()<2; let p=cell();
    let ca=select(vec2i(0,1),vec2i(1,0),hc); let ta=select(vec2i(0,1),vec2i(1,0),ht);
    let normal=select(p.y,p.x,hc); let normal_extent=select(gridSize().y,gridSize().x,hc);
    if (normal==0||normal==normal_extent) { write0(p,vec4f(0)); return; }
    let before_mass=0.5*(transfers(p-ca).y+transfers(p).y);
    let after_mass=0.5*(transfers(p-ca+ta).y+transfers(p+ta).y);
    let coordinate=select(p.y,p.x,ht); let extent=select(donor_velocitySize().y,donor_velocitySize().x,ht);
    var before=0.0; var after=0.0;
    if (coordinate>0) { before=momentumTransfer(before_mass,donor_velocity(p-ta).x,donor_velocity(p).x); }
    if (coordinate+1<extent) { after=momentumTransfer(after_mass,donor_velocity(p).x,donor_velocity(p+ta).x); }
    let correction=0.5*(massCorrection(p-ca,ta)+massCorrection(p,ta));
    write0(p,vec4f(momentum(p).x+before-after+original_velocity(p).x*correction));
}
)wgsl");
        compute.build(transport_liquid, "fluid split geometric update", 1, {"liquid", "transfers", "flow", "weights"}, R"wgsl(
fn execute() {
    let p=cell(); let axis=select(vec2i(0,1),vec2i(1,0),mode()==0); let weight=weights(p).x;
    let before=transfers(p).x-weight*(flow(p).x*timeStep()/cellSize());
    let after=transfers(p+axis).x-weight*(flow(p+axis).x*timeStep()/cellSize());
    write0(p,vec4f(liquid(p).x+(before-after)));
}
)wgsl");
    }
};

} // namespace fluid::webgpu
