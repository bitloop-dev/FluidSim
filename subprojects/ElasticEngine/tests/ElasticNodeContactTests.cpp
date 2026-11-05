#include "ElasticEngine.h"
#include <iostream>
#include <stdexcept>

namespace
{
using Engine = ElasticEngine::ElasticEngine;
using bl::DVec2;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double sweptDistance(DVec2 initial, DVec2 final)
{
    DVec2 travel = final - initial;
    double time = travel.isZero() ? 0.0 : std::clamp(-initial.dot(travel) / travel.mag2(), 0.0, 1.0);
    return (initial + time * travel).mag();
}

void nodePairs()
{
    for (double dt : {0.25, 1.0, 2.0})
    {
        for (bool bonded : {false, true})
        {
            Engine engine;
            engine.setMinimumNodeDistance(8.0);
            const DVec2 drift{0.75, -0.25};
            auto a = engine.addNode({-10.0, 0.0}, drift + DVec2{40.0 / dt, 0.0});
            auto b = engine.addNode({10.0, 0.0}, drift - DVec2{40.0 / dt, 0.0});
            if (bonded)
                engine.addBond(a, b, 8.0, 1.0, 0.0);

            engine.step(dt);
            const auto& first = engine.getNode(a);
            const auto& second = engine.getNode(b);
            require(sweptDistance({20.0, 0.0}, second.position - first.position) >= 8.0 - 1e-9,
                "Fast node pair crossed its clearance during the step");
            require(std::abs((second.position - first.position).mag() - 8.0) < 1e-9,
                "Head-on nodes stopped at the wrong distance");
            require(
                (first.velocity + second.velocity - 2.0 * drift).mag() < 1e-10, "Node contact changed total momentum");
            require(first.velocity.mag2() + second.velocity.mag2() <= 2.0 * drift.mag2() + 3200.0 / (dt * dt),
                "Node contact added kinetic energy");
        }
    }
}

void fixedAndGlancingContacts()
{
    for (double y : {0.0, 6.0})
    {
        Engine engine;
        engine.setMinimumNodeDistance(8.0);
        auto fixed = engine.addNode({0.0, 0.0}, {}, true);
        DVec2 position{-12.0, y};
        auto moving = engine.addNode(position, {30.0, 0.0});
        engine.step();
        require(engine.getNode(fixed).position.isZero() && engine.getNode(fixed).velocity.isZero(),
            "Node contact moved an anchor");
        require(sweptDistance(position, engine.getNode(moving).position) >= 8.0 - 1e-9,
            "Glancing node contact passed through its clearance");
        require(engine.getNode(moving).velocity.mag2() <= 900.0 + 1e-9, "Fixed contact added kinetic energy");
    }
}

void separatedNodesStillRespectBonds()
{
    Engine engine;
    engine.setMinimumNodeDistance(8.0);
    auto a = engine.addNode({-40.0, 0.0}, {}, true);
    auto b = engine.addNode({40.0, 0.0}, {}, true);
    auto p = engine.addNode({0.0, 10.0}, {0.0, -30.0});
    engine.addBoundary(a, b);
    engine.step();
    require(engine.getNode(p).position.y > 0.0, "Node clearance disabled long-bond containment");

    // Every detached/reused node remains in the node-pair search.
    engine.removeNode(p);
    auto reused = engine.addNode({-50.0, 0.0}, {20.0, 0.0});
    require(reused == p, "Node reuse fixture did not reuse the slot");
    engine.step();
    require(engine.getNode(reused).position.x <= -48.0 + 1e-9, "Reused node missed a fixed endpoint");
}

void restingAndSeparatingNodes()
{
    Engine engine;
    engine.setMinimumNodeDistance(8.0);
    auto a = engine.addNode({-4.0, 0.0}, {-1.0, 0.0});
    auto b = engine.addNode({4.0, 0.0}, {1.0, 0.0});
    engine.addBond(a, b, 10.0, 1.0, 0.0);
    engine.step();
    require(engine.getNode(a).velocity == DVec2{-1.0, 0.0} && engine.getNode(b).velocity == DVec2{1.0, 0.0},
        "Clearance attracted separating nodes");

    engine.removeBond(0);
    engine.getNode(a).velocity = {10.0, 0.0};
    engine.getNode(b).velocity = {-10.0, 0.0};
    engine.step();
    require((engine.getNode(b).position - engine.getNode(a).position).mag() >= 8.0 - 1e-9,
        "Removing a bond removed node clearance");
}

void overlapRecovery()
{
    for (double dt : {0.25, 1.0, 2.0})
    {
        Engine engine;
        engine.setMinimumNodeDistance(8.0);
        DVec2 drift{0.75, -0.25};
        auto a = engine.addNode({0.0, 0.0}, drift);
        auto b = engine.addNode({7.98, 0.0}, drift);
        engine.step(dt);
        // Critical damping, omega * dt = pi / 2: implicit spring displacement for an isolated pair.
        double frequency_step = 0.5 * std::acos(-1.0);
        double recovered = 0.02 * frequency_step * frequency_step / ((1.0 + frequency_step) * (1.0 + frequency_step));
        require(std::abs((engine.getNode(b).position - engine.getNode(a).position).mag() - 7.98 - recovered) < 1e-9,
            "Soft contact disagrees with the isolated damped spring response");
        require(
            (engine.getNode(a).velocity - drift).mag() < 1e-10 && (engine.getNode(b).velocity - drift).mag() < 1e-10,
            "Relaxation retained overlap recovery energy");
        require((engine.getNode(a).velocity + engine.getNode(b).velocity - 2.0 * drift).mag() < 1e-10,
            "Overlap recovery changed total momentum");

        // Damping away recovery motion exposes whether small spacing errors accumulate or settle.
        double previous = 7.98 + recovered;
        for (int step = 0; step < 64; ++step)
        {
            engine.getNode(a).velocity = drift;
            engine.getNode(b).velocity = drift;
            engine.step(dt);
            double spacing = (engine.getNode(b).position - engine.getNode(a).position).mag();
            require(spacing >= previous - 1e-10 && spacing <= 8.0, "Recovery attracted or overshot resting nodes");
            previous = spacing;
        }
        require(previous >= 7.99, "Small overlap did not settle within the spacing allowance");

        engine.getNode(a).velocity = drift - DVec2{1.0, 0.0};
        engine.getNode(b).velocity = drift + DVec2{1.0, 0.0};
        engine.step(dt);
        require((engine.getNode(a).velocity - drift + DVec2{1.0, 0.0}).mag() < 1e-10 &&
                    (engine.getNode(b).velocity - drift - DVec2{1.0, 0.0}).mag() < 1e-10,
            "Overlap recovery attracted separating nodes");
    }
}

void recoveryRespectsMembranes()
{
    for (bool fixed : {false, true})
    {
        Engine engine;
        engine.setMinimumNodeDistance(8.0);
        auto a = engine.addNode({-40.0, 0.0}, {}, fixed);
        auto b = engine.addNode({40.0, 0.0}, {}, fixed);
        auto inner = engine.addNode({0.0, 0.001});
        auto outer = engine.addNode({0.0, 7.981});
        engine.addBoundary(a, b);
        for (int step = 0; step < 256; ++step)
        {
            engine.step(0.25, {0.0, -0.02});
            require(engine.getNode(inner).position.y > engine.getNode(a).position.y,
                "Spacing recovery pushed a node through a wall or moving membrane");
            require((engine.getNode(outer).position - engine.getNode(inner).position).mag() >= 7.98 - 1e-9,
                "Pressure against a membrane deepened the initial overlap");
            if (fixed)
            {
                require(engine.getNode(a).position.y == 0.0 && engine.getNode(b).position.y == 0.0,
                    "Recovery moved fixed membrane endpoints");
            }
        }
        require((engine.getNode(outer).position - engine.getNode(inner).position).mag() >= 7.99,
            "Membrane contact prevented spacing recovery");
    }
}

void restingContactHistory()
{
    Engine engine;
    engine.setMinimumNodeDistance(8.0);
    auto a = engine.addNode({-40.0, 0.0}, {}, true);
    auto b = engine.addNode({40.0, 0.0}, {}, true);
    engine.addBoundary(a, b);
    for (int i = 0; i < 32; ++i)
        engine.addNode({0.0, 0.001 + 8.0 * i});

    // Resting support must survive changing substep durations without stale impulse strength.
    for (int step = 0; step < 4096; ++step)
    {
        engine.step(step % 2 == 0 ? 0.0625 : 0.125, {0.0, -0.02});
        require(engine.getNode(2).position.y > 0.0, "Cached support crossed the floor");
        for (ElasticEngine::NodeId id = 3; id < engine.getNodes().size(); ++id)
        {
            const auto& node = engine.getNode(id);
            require(node.position.isFinite() && node.velocity.isFinite(), "Contact history became nonfinite");
            require(node.position.y - engine.getNode(id - 1).position.y > 7.84,
                "Persistent support accumulated excessive overlap");
        }
    }

    // Deleting a loaded wall must remove its cached support as well as its geometry.
    engine.removeBond(0);
    for (int i = 0; i < 64; ++i)
        engine.step(0.125, {0.0, -0.02});
    require(engine.getNode(2).position.y < 0.0, "A removed wall retained cached contact forces");
}

void approachingMembraneGap()
{
    // Ending just short of a crossing must preserve the gap, before roundoff can erase the side.
    for (double side : {-1.0, 1.0})
    {
        Engine engine;
        auto a = engine.addNode({-40.0, 200.0}, {}, true);
        auto b = engine.addNode({40.0, 200.0}, {}, true);
        auto node = engine.addNode({0.0, 200.0 + side * 0.0101}, {0.0, -side * 0.16});
        engine.addBoundary(a, b);
        engine.step(0.0625);
        require(side * (engine.getNode(node).position.y - 200.0) >= 0.0009,
            "An approaching membrane contact lost its numerical safety gap");
    }
}
}

void runNodeContactTests()
{
    nodePairs();
    fixedAndGlancingContacts();
    separatedNodesStillRespectBonds();
    restingAndSeparatingNodes();
    overlapRecovery();
    recoveryRespectsMembranes();
    restingContactHistory();
    approachingMembraneGap();
    std::cout << "PASS node clearance, bonded neighbours, continuous contact and fixed supports\n";
}
