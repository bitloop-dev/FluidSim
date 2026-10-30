#include "AdaptiveWebGPU/FluidGridAdaptiveGPU.h"
#include "Adaptive/FluidCellView.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

using namespace fluid;
namespace {
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
struct Fixture {
    SubstanceCatalogue catalogue;
    FluidMixture liquid, gas;
    Fixture(bool wet, bool contrast, int count = 6) {
        for (int n = 0; n < count; ++n) {
            auto component =
                MixtureComponent{catalogue.add({"Ingredient " + std::to_string(n), {float((n % 3) == 0), float((n % 3) == 1), float((n % 3) == 2)}}),
                                 wet && n < 2 ? 100. + 100. * n
                                 : contrast   ? .5 + .2 * n
                                              : 1.};
            (wet && n < 2 ? liquid : gas).components.push_back(component);
        }
    }
    FluidGridAdaptive::State state(int n, double h, double speed) const {
        FluidGridAdaptive::State s;
        s.liquid_mass.assign(liquid.components.size(), std::vector<double>(n * n));
        s.gas_mass.assign(gas.components.size(), std::vector<double>(n * n));
        s.liquid_fraction.resize(n * n);
        s.u.resize((n + 1) * n);
        s.v.resize(n * (n + 1));
        const double pi = std::numbers::pi;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const int i = y * n + x;
                const double f = liquid.components.empty() ? 0. : std::clamp(y + .5 - n * (.48 + .08 * std::sin(2 * pi * (x + .5) / n)), 0., 1.);
                s.liquid_fraction[i] = float(f);
                for (int k = 0; k < int(liquid.components.size()); ++k)
                    s.liquid_mass[k][i] = float(f * h * h * liquid.components[k].density / liquid.components.size());
                std::vector<double> weights(gas.components.size());
                double sum = 0.;
                for (int k = 0; k < int(weights.size()); ++k) {
                    weights[k] = .8 + .2 * std::sin((x + .5) * (k + 1) * pi / n) * std::cos((y + .5) * pi / n);
                    sum += weights[k];
                }
                for (int k = 0; k < int(weights.size()); ++k)
                    s.gas_mass[k][i] = float((1 - f) * h * h * gas.components[k].density * weights[k] / sum);
            }
        auto psi = [&](int x, int y) {
            return x == 0 || x == n || y == 0 || y == n ? 0. : speed * (n * h) / pi * std::pow(std::sin(pi * x / n) * std::sin(pi * y / n), 2);
        };
        for (int y = 0; y < n; ++y)
            for (int x = 0; x <= n; ++x)
                s.u[y * (n + 1) + x] = float((psi(x, y + 1) - psi(x, y)) / h);
        for (int y = 0; y <= n; ++y)
            for (int x = 0; x < n; ++x)
                s.v[y * n + x] = float(-(psi(x + 1, y) - psi(x, y)) / h);
        return s;
    }
};
double compare(FluidGridAdaptiveGPU& gpu, const FluidGridAdaptive& cpu) {
    double error = 0.;
    for (int lev = 0; lev < int(gpu.levels().size()); ++lev) {
        const auto& g = *gpu.levels()[lev];
        const auto& c = *cpu.levels()[lev];
        const auto values = gpu.commands().read(g.conserved);
        for (int p = 0; p < int(g.qLayout.patches.size()); ++p) {
            const auto& patch = g.qLayout.patches[p];
            const auto in = c.conserved.const_array(p);
            forCells(patch.box, [&](int x, int y) {
                for (int n = 0; n < gpu.ingredientCount() + 2; ++n) {
                    const double scale = n < gpu.ingredientCount() ? gpu.materialMixture().components[n].density : 100.;
                    error = std::max(error, std::abs(values[patch.offset + patch.index(x, y) + n * patch.plane] - in(x, y, n)) / scale);
                }
            });
        }
    }
    return error;
}
void prescribeFaces(FluidGridAdaptiveGPU& gpu, FluidGridAdaptive& cpu, double speed) {
    const double world = gpu.finestWidth() * gpu.finestCellSize();
    for (int lev = 0; lev < int(gpu.levels().size()); ++lev) {
        const auto& g = *gpu.levels()[lev];
        auto& c = *cpu.levels()[lev];
        const double h = g.geometry.cell_size;
        const double pi = std::numbers::pi;
        auto psi = [&](int x, int y) {
            return x == 0 || y == 0 || x == g.geometry.width || y == g.geometry.height
                       ? 0.
                       : speed * world / pi * std::pow(std::sin(pi * x / g.geometry.width) * std::sin(pi * y / g.geometry.height), 2);
        };
        for (int d = 0; d < 2; ++d) {
            std::vector<float> values(g.faceLayout[d].elements);
            for (int p = 0; p < int(g.faceLayout[d].patches.size()); ++p) {
                const auto& patch = g.faceLayout[d].patches[p];
                auto out = c.velocity[d].array(p);
                forCells(patch.box, [&](int x, int y) {
                    const float v = float(d == 0 ? (psi(x, y + 1) - psi(x, y)) / h : -(psi(x + 1, y) - psi(x, y)) / h);
                    values[patch.offset + patch.index(x, y)] = v;
                    out(x, y) = v;
                });
            }
            gpu.commands().write(g.velocity[d], values);
        }
    }
}
void conservation(const GasTotals& before, const GasTotals& after, double limit) {
    require(after.finite, "Nonfinite adaptive conserved state");
    require(after.minMassDensity >= -2e-5, "Negative adaptive ingredient density");
    for (int n = 0; n < int(before.ingredientMass.size()); ++n)
        require(std::abs(after.ingredientMass[n] - before.ingredientMass[n]) <= limit * std::max(before.ingredientMass[n], 1.),
                "Adaptive ingredient mass drift");
}
void transportCases() {
    for (bool wet : {false, true})
        for (bool contrast : {false, true})
            for (bool mixed : {false, true}) {
                Fixture f(wet, contrast);
                const int n = 32;
                const GasLayout layout = mixed ? GasLayout{4, {{{8, 0, 16, 32}}, {{12, 4, 8, 24}}}} : GasLayout{};
                FluidGridAdaptive cpu(n, n, 1., f.catalogue, f.liquid, f.gas, layout, 8);
                FluidGridAdaptiveGPU gpu(n, n, 1., f.catalogue, f.liquid, f.gas, layout, 8);
                auto state = f.state(n, 1., 0.);
                cpu.upload(state, 1e-4);
                gpu.upload(state, 1e-4);
                prescribeFaces(gpu, cpu, 1.);
                const double a = cpu.stableTimeStep(), b = gpu.stableTimeStep();
                require(std::abs(a - b) < 1e-6, "Adaptive donor/mixed CFL differs from CPU");
                const auto before = gpu.totals();
                for (int tick = 0; tick < 24; ++tick) {
                    cpu.transport(.15);
                    gpu.transport(.15);
                    if (tick == 0)
                        require(compare(gpu, cpu) < 1e-6, "First adaptive transport update differs from CPU");
                }
                const auto after = gpu.totals();
                const double error = compare(gpu, cpu);
                std::cout << "transport wet=" << wet << " contrast=" << contrast << " mixed=" << mixed << " error=" << error
                          << " volume=" << after.maxVolumeError << '\n';
                // Common mixture limiters can choose different near-flat branches after float roundoff accumulates.
                require(error < 5e-4, "Repeated adaptive transport differs from CPU");
                conservation(before, after, 8e-6);
            }
}
void projectionCases() {
    for (bool wet : {false, true})
        for (bool contrast : {false, true}) {
            Fixture f(wet, contrast);
            const int n = 32;
            GasLayout layout{4, {{{8, 0, 16, 32}}, {{12, 4, 8, 24}}}};
            FluidGridAdaptive cpu(n, n, .25, f.catalogue, f.liquid, f.gas, layout, 8);
            FluidGridAdaptiveGPU gpu(n, n, .25, f.catalogue, f.liquid, f.gas, layout, 8);
            auto state = f.state(n, .25, .4);
            cpu.upload(state, 1e-4);
            gpu.upload(state, 1e-4);
            auto initial = gpu.totals();
            require(compare(gpu, cpu) < 1e-6, "Initial adaptive projection differs from CPU");
            for (int tick = 0; tick < 16; ++tick) {
                auto a = cpu.step(.016, 1e-4, .1, {0, wet ? .5 : 0.});
                auto b = gpu.step(.016, 1e-4, .1, {0, wet ? .5 : 0.});
                require(a.projection.converged && b.projection.converged, "Adaptive full step failed projection");
            }
            double materialError = 0., velocityError = 0.;
            for (int lev = 0; lev < int(gpu.levels().size()); ++lev) {
                const auto& l = *gpu.levels()[lev];
                const auto values = gpu.commands().read(l.conserved);
                for (int p = 0; p < int(l.qLayout.patches.size()); ++p) {
                    const auto& patch = l.qLayout.patches[p];
                    const auto q = cpu.levels()[lev]->conserved.const_array(p);
                    forCells(patch.box, [&](int x, int y) {
                        const int i = patch.offset + patch.index(x, y);
                        double rho = 0., reference = 0.;
                        for (int k = 0; k < gpu.ingredientCount(); ++k) {
                            rho += values[i + k * patch.plane];
                            reference += q(x, y, k);
                            materialError =
                                std::max(materialError, std::abs(values[i + k * patch.plane] - q(x, y, k)) / gpu.materialMixture().components[k].density);
                        }
                        for (int d = 0; d < 2; ++d)
                            velocityError = std::max(velocityError, std::abs(values[i + (gpu.ingredientCount() + d) * patch.plane] / rho -
                                                                             q(x, y, gpu.ingredientCount() + d) / reference));
                    });
                }
            }
            auto totals = gpu.totals();
            std::cout << "physical wet=" << wet << " contrast=" << contrast << " material_error=" << materialError << " velocity_error=" << velocityError
                      << " volume=" << totals.maxVolumeError << " div=" << gpu.maxDivergence() << '\n';
            require(materialError < 1e-3 && velocityError < 5e-3, "Adaptive repeated physical integration differs from CPU");
            conservation(initial, totals, 2e-5);
            require(gpu.pressureInterval() == .016, "Physical pressure interval lost");
            std::vector<std::vector<float>> before;
            for (const auto& l : gpu.levels())
                before.push_back(gpu.commands().read(l->conserved));
            gpu.step(0., 1e-4);
            for (int lev = 0; lev < int(before.size()); ++lev)
                require(before[lev] == gpu.commands().read(gpu.levels()[lev]->conserved), "Zero dt changed conserved state");
        }
}
void regridCases() {
    Fixture f(false, true);
    constexpr int n = 64;
    FluidGridAdaptive cpu(n, n, 1., f.catalogue, f.gas, {4, {{{16, 0, 32, 64}}, {{24, 8, 16, 48}}}}, 16);
    FluidGridAdaptiveGPU gpu(n, n, 1., f.catalogue, f.gas, cpu.layout(), 16);
    auto state = f.state(n, 1., 0.);
    cpu.upload(state, 1e-4);
    gpu.upload(state, 1e-4);
    const auto before = gpu.totals();
    const std::vector<GasLayout> layouts = {{4, {{{8, 0, 48, 64}}, {{16, 8, 32, 48}}}}, {8, {}}, {32, {}}, {1, {}}, {16, {}},
                                            {4, {{{16, 0, 32, 64}}, {{24, 8, 16, 48}}}}};
    for (int repeat = 0; repeat < 2; ++repeat)
        for (const auto& layout : layouts) {
            require(cpu.regrid(layout), "Changed CPU layout not recognized");
            const auto start = std::chrono::steady_clock::now();
            require(gpu.regrid(layout), "Changed GPU layout not recognized");
            gpu.synchronize();
            const double regridMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            const double error = compare(gpu, cpu);
            const auto after = gpu.totals();
            std::cout << "regrid ratio=" << layout.baseRatio << " levels=" << layout.refinedPatches.size() + 1 << " error=" << error
                      << " remap_setup_ms=" << regridMs << '\n';
            require(error < 3e-6, "Conservative GPU regrid differs from CPU");
            conservation(before, after, 3e-6);
            require(gpu.pressureInterval() == 0., "Regrid retained physical pressure interval");
            require(cpu.project(1e-4).converged && gpu.project(1e-4).converged, "Regrid restoration failed");
            require(!gpu.regrid(layout), "Identical normalized layout is not a no-op");
        }
}
void subcyclingAndTraces() {
    for (bool wet : {false, true}) {
        Fixture f(wet, true);
        constexpr int n = 16;
        FluidGridAdaptiveGPU gpu(n, n, .25, f.catalogue, f.liquid, f.gas, {2, {{{4, 0, 8, 16}}}}, 8);
        gpu.upload(f.state(n, .25, 2.), 1e-4);
        const auto before = gpu.totals();
        const auto result = gpu.step(.2, 1e-4, .1);
        const auto after = gpu.totals();
        require(result.projection.converged && result.substeps > 1, "Adaptive physical step did not CFL subcycle");
        require(gpu.pressureInterval() == .2, "Subcycled step projected with a transport interval");
        conservation(before, after, 2e-5);
        require(after.maxVolumeError < 2e-4, "Subcycled phase volume closure failed");
        std::cout << "subcycle wet=" << wet << " subdivisions=" << result.substeps << " volume=" << after.maxVolumeError << '\n';
    }
    Fixture f(true, true);
    constexpr int n = 16;
    auto state = f.state(n, 1., 0.);
    for (int k = 0; k < 2; ++k)
        std::fill(state.liquid_mass[k].begin(), state.liquid_mass[k].end(), .5 * f.liquid.components[k].density);
    for (int k = 0; k < int(f.gas.components.size()); ++k)
        std::fill(state.gas_mass[k].begin(), state.gas_mass[k].end(), float(1e-13 * f.gas.components[k].density / f.gas.components.size()));
    FluidGridAdaptive cpu(n, n, 1., f.catalogue, f.liquid, f.gas, {}, 8);
    FluidGridAdaptiveGPU gpu(n, n, 1., f.catalogue, f.liquid, f.gas, {}, 8);
    cpu.upload(state, 1e-4);
    gpu.upload(state, 1e-4);
    prescribeFaces(gpu, cpu, 1.);
    const auto before = gpu.totals();
    for (int tick = 0; tick < 24; ++tick)
        gpu.transport(.15);
    const auto after = gpu.totals();
    std::cout << "minority min_density=" << after.minMassDensity << " volume=" << after.maxVolumeError << '\n';
    const double traceRoundoff = 64. * std::numeric_limits<float>::epsilon() * 1.5e-13;
    require(after.finite && after.minMassDensity >= -traceRoundoff, "Tiny minority phase exceeded float roundoff");
    for (int k = 2; k < gpu.ingredientCount(); ++k)
        require(std::abs(after.ingredientMass[k] / before.ingredientMass[k] - 1.) < 2e-5, "Tiny gas ingredient was discarded");
    std::cout << "minority gas volume=1e-13 preserved through 24 transport updates\n";
}
void colorCases() {
    Fixture f(false, true);
    constexpr int n = 32;
    GasLayout layout{4, {{{8, 0, 16, 32}}, {{12, 4, 8, 24}}}};
    FluidGridAdaptive cpu(n, n, 1., f.catalogue, f.gas, layout, 8);
    FluidGridAdaptiveGPU gpu(n, n, 1., f.catalogue, f.gas, layout, 8);
    auto state = f.state(n, 1., 0.);
    cpu.upload(state, 1e-4);
    gpu.upload(state, 1e-4);
    const auto pixels = gpu.colorPixels();
    FluidCellView view(cpu);
    double error = 0.;
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            std::array<double, 3> expected{};
            for (int k = 0; k < gpu.ingredientCount(); ++k) {
                const auto& c = f.catalogue[f.gas.components[k].substance].color;
                const double fraction = view.materialSample(x, y, k);
                for (int d = 0; d < 3; ++d)
                    expected[d] += fraction * c[d];
            }
            for (int d = 0; d < 3; ++d)
                error = std::max(error, std::abs(expected[d] - pixels[(y * n + x) * 4 + d]));
            require(pixels[(y * n + x) * 4 + 3] == 1.f, "Adaptive color uncovered pixel");
        }
    std::cout << "color reconstructed mixed-join error=" << error << '\n';
    require(error < 2e-6, "Adaptive material display differs from bounded CPU observer");
}
void reconstructionCases() {
    constexpr int n=96;
    for(bool contrast:{false,true}) {
        Fixture f(true,contrast,4);
        FluidGridAdaptive cpu(n,n,1.,f.catalogue,f.liquid,f.gas,{16,{}},32);
        FluidGridAdaptiveGPU gpu(n,n,1.,f.catalogue,f.liquid,f.gas,{16,{}},32);
        auto state=f.state(n,1.,0.);
        std::fill(state.liquid_fraction.begin(),state.liquid_fraction.end(),0.);
        for(auto& lane:state.liquid_mass)std::fill(lane.begin(),lane.end(),0.);
        for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
            const double share=.5+.15*((x+.5)/n-.5)+.1*((y+.5)/n-.5);const int i=y*n+x;
            state.gas_mass[0][i]=(1-share)*f.gas.components[0].density;state.gas_mass[1][i]=share*f.gas.components[1].density;
        }
        cpu.upload(state,1e-4);gpu.upload(state,1e-4);
        const auto pixels=gpu.colorPixels();
        auto setMomentum=[&](int tick,bool constant) {
            for(int lev=0;lev<int(gpu.levels().size());++lev) {
                const auto& l=*gpu.levels()[lev];auto values=gpu.commands().read(l.conserved);
                for(int p=0;p<int(l.qLayout.patches.size());++p) {
                    const auto& patch=l.qLayout.patches[p];const auto out=cpu.levels()[lev]->conserved.array(p);
                    forCells(patch.box,[&](int x,int y) {
                        const int i=patch.offset+patch.index(x,y);double rho=0.,reference=0.;
                        for(int k=0;k<4;++k){rho+=values[i+k*patch.plane];reference+=out(x,y,k);}
                        for(int d=0;d<2;++d) {
                            const double v=(d?-.3:.7)+(constant?0.:.4*std::sin(.4*tick+(d?y:x)*.8));
                            values[i+(4+d)*patch.plane]=float(rho*v);out(x,y,4+d)=reference*v;
                        }
                    });
                }
                gpu.commands().write(l.conserved,values);
            }
        };
        for(int tick=0;tick<16;++tick) {
            setMomentum(tick,false);
            require(gpu.colorPixels()==pixels,"Moving momentum extrema changed an unchanged GPU material image");
            FluidCellView view(cpu);double error=0.;
            for(int y=0;y<n;y+=3)for(int x=0;x<n;x+=3)for(int d=0;d<3;++d) {
                double expected=0.;for(int k=0;k<4;++k)expected+=view.materialSample(x,y,k)*cpu.substances()[cpu.materialMixture().components[k].substance].color[d];
                error=std::max(error,std::abs(expected-pixels[(y*n+x)*4+d]));
            }
            require(error<2e-6,"CPU/GPU smooth-label display differs");
        }
        setMomentum(0,true);const auto before=gpu.totals();
        for(const GasLayout layout:{GasLayout{8,{}},GasLayout{4,{{{16,16,64,64}},{{24,24,48,48}}}},GasLayout{1,{}},GasLayout{16,{}}}) {
            cpu.regrid(layout);gpu.regrid(layout);require(compare(gpu,cpu)<3e-6,"Density-consistent regrid differs on CPU/GPU");
            const auto after=gpu.totals();conservation(before,after,2e-6);
            for(int d=0;d<2;++d)require(std::abs(after.momentum[d]-before.momentum[d])<2e-6*std::max(1.,std::abs(before.momentum[d])),
                                      "GPU regrid changed total momentum");
            for(const auto& ptr:gpu.levels()) {
                const auto& l=*ptr;const auto values=gpu.commands().read(l.conserved);
                for(const auto& patch:l.qLayout.patches)forCells(patch.box,[&](int x,int y) {
                    const int i=patch.offset+patch.index(x,y);double rho=0.;for(int k=0;k<4;++k)rho+=values[i+k*patch.plane];
                    for(int d=0;d<2;++d)require(std::abs(values[i+(4+d)*patch.plane]/rho-(d?-.3:.7))<2e-6,"GPU regrid changed constant velocity");
                });
            }
        }
    }
    std::cout<<"reconstruction moving-extrema images bitwise stable; conservative density/velocity transfers pass\n";
}
void behavior(int ticks) {
    for (bool wet : {false, true}) {
        Fixture f(wet, true);
        constexpr int n = 16;
        GasLayout layout{2, {{{4, 0, 8, 16}}}};
        FluidGridAdaptiveGPU gpu(n, n, .25, f.catalogue, f.liquid, f.gas, layout, 8);
        gpu.upload(f.state(n, .25, .5), 1e-4);
        const auto before = gpu.totals();
        double maximumVolume = 0., maximumDiv = 0., maximumMassDrift = 0., minimumDensity = std::numeric_limits<double>::infinity();
        int totalSubsteps = 0;
        // dt=1/60; 24 s spans three nominal L/U=8 s crossing scales and several gravity-response times.
        for (int tick = 0; tick < ticks; ++tick) {
            auto step = gpu.step(1. / 60., 1e-4, wet ? .1 : 0., {0, wet ? 1. : 0.});
            require(step.projection.converged, "Behavioral projection failed");
            maximumDiv = std::max(maximumDiv, step.projection.max_divergence);
            totalSubsteps += step.substeps;
            if (tick % 16 == 0 || tick + 1 == ticks) {
                auto totals = gpu.totals();
                conservation(before, totals, 2e-3);
                maximumVolume = std::max(maximumVolume, totals.maxVolumeError);
                minimumDensity = std::min(minimumDensity, totals.minMassDensity);
                for (int k = 0; k < gpu.ingredientCount(); ++k)
                    maximumMassDrift = std::max(maximumMassDrift, std::abs(totals.ingredientMass[k] / before.ingredientMass[k] - 1.));
                require(totals.maxVolumeError < .01, "Behavioral volume closure failed");
            }
            if ((tick + 1) % 240 == 0)
                std::cout << "behavior wet=" << wet << " ticks=" << tick + 1 << " volume=" << maximumVolume << " div=" << maximumDiv << '\n';
        }
        std::cout << "behavior PASS wet=" << wet << " seconds=" << ticks / 60. << " substeps=" << totalSubsteps << " volume=" << maximumVolume
                  << " div=" << maximumDiv << " relative_mass_drift=" << maximumMassDrift << " min_density=" << minimumDensity << '\n';
    }
}
void wavePressureCase() {
    constexpr int n = 512;
    SubstanceCatalogue catalogue;
    FluidMixture liquid, gas;
    liquid.components = {{catalogue.add({"Water", {0, 0, 1}}), 100.}, {catalogue.add({"Red", {1, 0, 0}}), 300.}};
    gas.components = {{catalogue.add({"Air", {0, 0, 0}}), .1}, {catalogue.add({"Cloud", {1, 1, 1}}), .1}};
    FluidGridAdaptive::State state;
    state.liquid_fraction.resize(n * n);
    state.liquid_mass.assign(2, std::vector<double>(n * n)); state.gas_mass.assign(2, std::vector<double>(n * n));
    state.u.resize((n + 1) * n); state.v.resize(n * (n + 1));
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            const double alpha = std::clamp(y + 1. - (.7 * n + double(x) * .5), 0., 1.);
            const int i = y * n + x; state.liquid_fraction[i] = alpha;
            state.liquid_mass[0][i] = 100. * alpha; state.gas_mass[0][i] = .1 * (1. - alpha);
        }
    FluidGridAdaptiveGPU gpu(n, n, 1., catalogue, liquid, gas, {}, 64);
    for (int reset = 0; reset < 2; ++reset) {
        gpu.upload(state, 1.e-3);
        for (const auto& level : gpu.levels()) {
            require(level->pressure.channels == 2, "Physical pressure has no compensated low plane");
            const auto values = gpu.commands().read(level->pressure);
            require(std::all_of(values.begin(), values.end(), [](float value) { return value == 0.; }), "Upload retained stale pressure low bits");
        }
        const auto before = gpu.totals();
        const auto step = gpu.step(.016, 1.e-3, 2., {0, 100.});
        require(step.projection.converged && step.projection.max_divergence <= 1.e-3, "Physical 512-square wave projection failed");
        conservation(before, gpu.totals(), 1.e-6);
        std::cout << "wave512 reset=" << reset << " iterations=" << step.projection.iterations
                  << " div=" << step.projection.max_divergence << '\n';
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;
        const std::string mode = argc > 1 ? argv[1] : "all";
        if (mode == "all" || mode == "transport")
            transportCases();
        if (mode == "all" || mode == "projection")
            projectionCases();
        if (mode == "all" || mode == "regrid")
            regridCases();
        if (mode == "all" || mode == "reconstruction")
            reconstructionCases();
        if (mode == "all" || mode == "color")
            colorCases();
        if (mode == "all" || mode == "subcycle")
            subcyclingAndTraces();
        if (mode == "wave")
            wavePressureCase();
        if (mode == "behavior")
            behavior(argc > 2 ? std::stoi(argv[2]) : 1440);
        std::cout << "PASS adaptive WebGPU " << mode << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
