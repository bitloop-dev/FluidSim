#pragma once
#include "FluidGridCPU.h"
#include <cmath>
#include <numeric>
#include <type_traits>
#include <utility>

namespace fluid::test
{

inline void requireEdit(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class State>
bool sameEditState(const State& a, const State& b)
{
    return a.liquid_fraction == b.liquid_fraction && a.liquid_mass == b.liquid_mass && a.gas_mass == b.gas_mass &&
           a.u == b.u && a.v == b.v && a.pressure == b.pressure;
}

template<class Grid>
void materialEditingIsSparseAndReusable()
{
    using Scalar = typename Grid::Scalar;
    for (const auto [liquids, gases] : {std::pair{1, 0}, std::pair{0, 1}, std::pair{2, 2}, std::pair{9, 10}})
    {
        Grid grid(13, 9, 2.5);
        const auto shared = grid.elements.add({"Shared", {1, 0, 1}});
        for (int k = 0; k < liquids; ++k)
        {
            const auto element = k == liquids - 1 ? shared : grid.elements.add({"Liquid", {0, 0, 1}});
            grid.liquid_mixture.components.push_back({element, 100.0 + 20.0 * k});
        }
        for (int k = 0; k < gases; ++k)
        {
            const auto element = k == gases - 1 ? shared : grid.elements.add({"Gas", {0, 1, 0}});
            grid.gas_mixture.components.push_back({element, 0.125 + 0.125 * k});
        }

        auto edit = grid.beginEdit();
        edit.commit();
        edit.setCell(0, 0, liquids ? Phase::Liquid : Phase::Gas, shared);
        bool needs_upload = false;
        try { edit.commit(); }
        catch (const std::logic_error&) { needs_upload = true; }
        requireEdit(needs_upload && edit.size() == 1, "nonempty edit accepted an uninitialized grid");
        edit.clear();
        typename Grid::State initial;
        initial.liquid_fraction.resize(117);
        initial.liquid_mass.assign(liquids, std::vector<Scalar>(117));
        initial.gas_mass.assign(gases, std::vector<Scalar>(117));
        initial.u.resize(14 * 9);
        initial.v.resize(13 * 10);
        for (size_t i = 0; i < initial.u.size(); ++i) initial.u[i] = Scalar(0.03 * std::sin(double(i)));
        for (size_t i = 0; i < initial.v.size(); ++i) initial.v[i] = Scalar(0.02 * std::cos(double(i)));
        for (size_t i = 0; i < 117; ++i)
        {
            const double fraction = liquids == 0 ? 0.0 : gases == 0 ? 1.0 : 0.25 * (i % 5);
            initial.liquid_fraction[i] = Scalar(fraction);
            for (int k = 0; k < liquids; ++k)
                initial.liquid_mass[k][i] = Scalar(grid.liquid_mixture.components[k].density * 6.25 * fraction / liquids);
            for (int k = 0; k < gases; ++k)
                initial.gas_mass[k][i] = Scalar(grid.gas_mixture.components[k].density * 6.25 * (1.0 - fraction) / gases);
        }
        grid.upload(initial);
        const double tolerance = std::is_same_v<Scalar, double> ? 1e-8 : 1e-3;
        requireEdit(grid.project(0.004, 2000, tolerance).converged, "edit fixture projection failed");
        const auto before = grid.download();
        auto expected = before;
        const Phase a = liquids ? Phase::Liquid : Phase::Gas;
        const Phase b = gases ? Phase::Gas : Phase::Liquid;
        const ElementId first = liquids ? grid.liquid_mixture.components.front().element : grid.gas_mixture.components.front().element;
        edit.reserve(64);
        auto paint = [&](int x, int y, Phase phase, ElementId element)
        {
            edit.setCell(x, y, phase, element);
            const size_t i = size_t(y) * 13 + x;
            expected.liquid_fraction[i] = phase == Phase::Liquid ? Scalar(1) : Scalar(0);
            for (auto& mass : expected.liquid_mass) mass[i] = 0;
            for (auto& mass : expected.gas_mass) mass[i] = 0;
            const auto& mixture = phase == Phase::Liquid ? grid.liquid_mixture : grid.gas_mixture;
            const auto found = std::find_if(mixture.components.begin(), mixture.components.end(),
                [&](const MixtureComponent& component) { return component.element == element; });
            auto& masses = phase == Phase::Liquid ? expected.liquid_mass : expected.gas_mass;
            masses[size_t(found - mixture.components.begin())][i] = Scalar(found->density * 6.25);
        };
        paint(12, 8, a, shared);
        paint(0, 0, b, shared);
        paint(4, 2, a, first);
        paint(5, 2, b, shared);
        paint(4, 3, a, shared);
        paint(5, 3, b, shared);
        paint(7, 3, a, shared);
        paint(11, 5, b, shared);
        paint(12, 5, b, shared);
        paint(0, 6, a, shared);
        paint(1, 6, a, shared);
        paint(4, 2, b, shared);
        requireEdit(edit.size() == 12 && sameEditState(grid.download(), before), "staging edits mutated the live grid");
        auto moved = std::move(edit);
        moved.commit();
        requireEdit(moved.empty() && sameEditState(grid.download(), expected), "sparse material replacement or last-write ordering failed");
        moved.commit();
        requireEdit(sameEditState(grid.download(), expected), "empty commit changed the state");

        moved.setCell(1, 1, a, shared);
        moved.clear();
        moved.commit();
        {
            auto abandoned = grid.beginEdit();
            abandoned.setCell(2, 2, b, shared);
        }
        requireEdit(sameEditState(grid.download(), expected), "discarded edits were committed");

        for (int invalid = 0; invalid < 4; ++invalid)
        {
            moved.setCell(1, 1, a, shared);
            if (invalid == 0) moved.setCell(-1, 0, a, shared);
            else if (invalid == 1) moved.setCell(13, 0, a, shared);
            else if (invalid == 2) moved.setCell(0, 9, b, shared);
            else moved.setCell(0, 0, a, ElementId(1000000));
            bool rejected = false;
            try { moved.commit(); }
            catch (const std::out_of_range&) { rejected = true; }
            catch (const std::invalid_argument&) { rejected = true; }
            requireEdit(rejected && moved.size() == 2 && sameEditState(grid.download(), expected), "invalid edit partially committed or lost its commands");
            moved.clear();
        }

        // Reuse the same session with already-sorted writes and a different phase/material.
        moved.setCell(3, 4, a, first);
        moved.setCell(4, 4, a, shared);
        moved.setCell(3, 5, b, shared);
        moved.setCell(4, 5, b, shared);
        moved.commit();

        // A fresh upload is an independent oracle for density and pressure cache invalidation.
        Grid fresh(13, 9, 2.5);
        fresh.elements = grid.elements;
        fresh.liquid_mixture = grid.liquid_mixture;
        fresh.gas_mixture = grid.gas_mixture;
        fresh.upload(grid.download());
        grid.applyGravity(0.004);
        fresh.applyGravity(0.004);
        requireEdit(grid.project(0.004, 2000, tolerance).converged && fresh.project(0.004, 2000, tolerance).converged,
            "post-edit projection failed");
        const auto cached = grid.download(), rebuilt = fresh.download();
        const double velocity_tolerance = std::is_same_v<Scalar, double> ? 1e-6 : 0.003;
        for (size_t i = 0; i < cached.u.size(); ++i)
            requireEdit(std::abs(double(cached.u[i]) - rebuilt.u[i]) < velocity_tolerance, "edit left stale horizontal density coefficients");
        for (size_t i = 0; i < cached.v.size(); ++i)
            requireEdit(std::abs(double(cached.v[i]) - rebuilt.v[i]) < velocity_tolerance, "edit left stale vertical density coefficients");
    }
}

template<class Grid>
void bulkMaterialEditsPreserveEveryCell()
{
    using Scalar = typename Grid::Scalar;
    Grid grid(257, 129, 1);
    const auto water = grid.elements.add({"Water", {0, 0, 1}});
    const auto air = grid.elements.add({"Air", {0, 0, 0}});
    grid.liquid_mixture.components = {{water, 100}};
    grid.gas_mixture.components = {{air, 0.125}};
    const size_t count = size_t(grid.width) * grid.height;
    typename Grid::State state;
    state.liquid_fraction.assign(count, 0);
    state.liquid_mass.assign(1, std::vector<Scalar>(count, 0));
    state.gas_mass.assign(1, std::vector<Scalar>(count, Scalar(0.125)));
    grid.upload(state);
    auto edit = grid.beginEdit();
    edit.reserve(count);
    for (int y = 0; y < grid.height; ++y)
    for (int x = 0; x < grid.width; ++x)
    {
        const bool liquid = (x + 3 * y) % 7 != 0;
        edit.setCell(x, y, liquid ? Phase::Liquid : Phase::Gas, liquid ? water : air);
    }
    edit.commit();
    state = grid.download();
    for (size_t i = 0; i < count; ++i)
    {
        const bool liquid = (i % 257 + 3 * (i / 257)) % 7 != 0;
        requireEdit(state.liquid_fraction[i] == Scalar(liquid) && state.liquid_mass[0][i] == Scalar(liquid ? 100 : 0) &&
                    state.gas_mass[0][i] == Scalar(liquid ? 0 : 0.125), "bulk edit lost a cell or selected the wrong material");
    }
}

template<class Grid>
void repeatedMaterialEditsPreservePhaseVolumes()
{
    using Scalar = typename Grid::Scalar;
    Grid grid(17, 13, 2.5);
    const auto water = grid.elements.add({"Water", {0, 0, 1}});
    const auto air = grid.elements.add({"Air", {0, 0, 0}});
    grid.liquid_mixture.components = {{water, 100}};
    grid.gas_mixture.components = {{air, 0.1}};
    typename Grid::State initial;
    initial.liquid_fraction.resize(17 * 13);
    initial.liquid_mass.assign(1, std::vector<Scalar>(17 * 13));
    initial.gas_mass.assign(1, std::vector<Scalar>(17 * 13));
    for (int y = 0; y < 13; ++y)
    for (int x = 0; x < 17; ++x)
    {
        const int i = y * 17 + x;
        initial.liquid_fraction[i] = y >= 9 ? Scalar(1) : Scalar(0);
        initial.liquid_mass[0][i] = Scalar(625 * initial.liquid_fraction[i]);
        initial.gas_mass[0][i] = Scalar(0.625 * (1 - initial.liquid_fraction[i]));
    }
    grid.upload(initial);
    auto totals = [](const auto& state) {
        return std::pair{std::accumulate(state.liquid_mass[0].begin(), state.liquid_mass[0].end(), 0.0),
                         std::accumulate(state.gas_mass[0].begin(), state.gas_mass[0].end(), 0.0)};
    };
    auto expected = totals(initial);
    auto edit = grid.beginEdit();
    edit.reserve(2);
    const double tolerance = std::is_same_v<Scalar, double> ? 1e-8 : 1e-3;
    const double budget = std::is_same_v<Scalar, double> ? 1e-9 : 2e-3;
    // Two seconds exercises repeated spawning and gas replacement during pool motion.
    for (int tick = 0; tick < 240; ++tick)
    {
        if (tick % 20 == 0)
        {
            const auto before = grid.download();
            const int x = 2 + (tick / 20) % 12, y = 1;
            const size_t i = size_t(y) * 17 + x;
            expected.first += 625.0 - before.liquid_mass[0][i];
            expected.second -= before.gas_mass[0][i];
            edit.setCell(x, y, Phase::Liquid, water);
            const size_t j = 11 * 17 + x;
            expected.first -= before.liquid_mass[0][j];
            expected.second += double(Scalar(0.625)) - before.gas_mass[0][j];
            edit.setCell(x, 11, Phase::Gas, air);
            edit.commit();
        }
        requireEdit(grid.step(1.0 / 120, 2000, tolerance).converged, "repeated-edit physics projection failed");
        const auto state = grid.download();
        const auto mass = totals(state);
        requireEdit(std::abs(mass.first - expected.first) < std::abs(expected.first) * budget &&
                    std::abs(mass.second - expected.second) < std::abs(expected.second) * budget, "mass drift exceeded the edit source/sink budget");
        for (size_t i = 0; i < state.liquid_fraction.size(); ++i)
        {
            const double fraction = state.liquid_fraction[i];
            requireEdit(std::isfinite(fraction) && fraction >= -budget && fraction <= 1 + budget, "post-edit liquid fraction is invalid");
            requireEdit(std::abs(state.liquid_mass[0][i] / 625.0 - fraction) < budget &&
                        std::abs(state.gas_mass[0][i] / 0.625 - (1 - fraction)) < budget, "post-edit masses disagree with phase volumes");
        }
    }
}

}
