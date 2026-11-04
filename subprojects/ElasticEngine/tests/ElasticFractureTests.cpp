#include "ElasticEngine.h"
#include <iostream>
#include <stdexcept>

namespace
{
using Engine = ElasticEngine::ElasticEngine;
using ElasticEngine::NodeId;
using bl::DVec2;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void tensileThreshold()
{
    // A binary-exact limit tests strict exceedance without decimal rounding ambiguity.
    for (double length : {5.0, 12.0, 12.5, 12.75})
    {
        Engine engine;
        NodeId a = engine.addNode({0.0, 0.0});
        NodeId b = engine.addNode({length, 0.0});
        engine.addBond(a, b, 10.0, 1.0, 0.0, 0.25);
        engine.step();
        require(engine.getBonds().empty() == (length > 12.5), "Tensile threshold or compression behaviour is wrong");
        require(engine.getNodes().size() == 2 && engine.getNode(a).active && engine.getNode(b).active,
            "Fracture deleted a node");
    }

    Engine unbreakable;
    NodeId a = unbreakable.addNode({0.0, 0.0});
    NodeId b = unbreakable.addNode({100.0, 0.0});
    unbreakable.addBond(a, b);
    unbreakable.step();
    require(unbreakable.getBonds().size() == 1, "Existing bonds are no longer unbreakable by default");
}

void brokenBondsReleaseWithoutImpulse()
{
    Engine engine;
    NodeId a = engine.addNode({0.0, 0.0}, {-4.0, 0.0});
    NodeId b = engine.addNode({10.0, 0.0});
    NodeId c = engine.addNode({20.0, 0.0}, {4.0, 0.0});
    engine.addBond(a, b, 10.0, 10.0, 5.0, 0.30);
    engine.addBond(b, c, 10.0, 10.0, 5.0, 0.30);
    const auto initial = engine.getNodes();
    for (int step = 1; step <= 1024; ++step)
    {
        engine.step();
        require(engine.getBonds().empty(), "Simultaneous failures left a bond behind");
        require(engine.getNodes().size() == initial.size(), "Detached nodes were removed");
        for (NodeId id = 0; id < initial.size(); ++id)
        {
            const auto& node = engine.getNode(id);
            require(node.active && node.velocity == initial[id].velocity,
                "Broken spring or damping changed fragment velocity");
            require(node.position == initial[id].position + double(step) * initial[id].velocity,
                "Detached fragment failed to drift freely");
        }
    }
}

void survivorCompaction()
{
    Engine engine;
    DVec2 drift{0.25, -0.125};
    NodeId a = engine.addNode({0.0, 0.0}, drift);
    NodeId b = engine.addNode({10.0, 0.0}, drift);
    NodeId c = engine.addNode({24.0, 0.0}, drift);
    NodeId d = engine.addNode({24.0, 10.0}, drift);
    engine.addBond(a, b, 8.0, 0.25, 0.2);
    engine.addBond(b, c, 10.0, 10.0, 5.0, 0.30);
    engine.addBond(c, d, 12.0, 0.5, 0.7);

    Engine reference = engine;
    reference.removeBond(1);
    for (int step = 0; step < 1024; ++step)
    {
        engine.step();
        reference.step();
        require(engine.getBonds().size() == 2, "Fracture removed a surviving segment");
        require(engine.getNodes().size() == 4, "Two-node fragments were deleted");
        for (std::size_t index = 0; index < 2; ++index)
        {
            const auto& bond = engine.getBonds()[index];
            const auto& expected = reference.getBonds()[index];
            require(bond.a == expected.a && bond.b == expected.b && bond.rest_length == expected.rest_length &&
                        bond.stiffness == expected.stiffness && bond.damping == expected.damping &&
                        bond.breaking_strain == expected.breaking_strain,
                "Compaction changed surviving bond properties");
        }
        for (NodeId id = 0; id < 4; ++id)
        {
            require((engine.getNode(id).position - reference.getNode(id).position).mag() < 1e-10 &&
                        (engine.getNode(id).velocity - reference.getNode(id).velocity).mag() < 1e-10,
                "Fracture left a force, damping weight or direction from the removed bond");
        }
    }
}

void zeroStiffnessReleasesStructure()
{
    for (bool initially_zero : {false, true})
    {
        Engine engine;
        engine.setMinimumNodeDistance(8.0);
        auto a = engine.addNode({-20.0, 0.0}, {-0.25, 0.0});
        auto b = engine.addNode({20.0, 0.0}, {0.5, 0.0});
        engine.addNode({0.0, 10.0}, {0.0, -30.0});
        Engine reference = engine;
        engine.addBond(a, b, 40.0, initially_zero ? 0.0 : 1.0, 100.0);
        engine.step(0.0);
        require(engine.getBonds().size() == 1, "Zero-duration step released topology");
        engine.getBond(0).stiffness = 0.0;

        // The third node must cross the former segment with exactly the bondless trajectory.
        for (int step = 0; step < 1024; ++step)
        {
            engine.step();
            reference.step();
            require(engine.getBonds().empty(), "Zero-stiffness material retained structure");
            for (NodeId id = 0; id < engine.getNodes().size(); ++id)
            {
                require(engine.getNode(id).position == reference.getNode(id).position &&
                            engine.getNode(id).velocity == reference.getNode(id).velocity && engine.getNode(id).active,
                    "Released material still applied damping, contact or changed node motion");
            }
        }
    }
}

void cupFracturesEveryDirection()
{
    Engine engine;
    ElasticEngine::setupCupDrop(engine, 1.0, 0.02);
    for (const auto& bond : engine.getBonds())
    {
        if (!bond.boundary)
            require(bond.breaking_strain == 0.02, "A lattice direction missed the configured breaking strain");
    }

    // Uniform dilation extends horizontal and diagonal bonds equally.
    for (NodeId id = 4; id < engine.getNodes().size(); ++id)
        engine.getNode(id).position *= 1.03;
    engine.step();
    require(engine.getBonds().size() == 3, "Horizontal or diagonal material bonds survived above the strain limit");
    for (const auto& bond : engine.getBonds())
        require(bond.boundary, "Fracture removed a cup boundary");
    for (const auto& node : engine.getNodes())
        require(node.active && node.velocity.isZero(), "Fracture applied a final force or removed a node");
}
}

void runFractureTests()
{
    tensileThreshold();
    brokenBondsReleaseWithoutImpulse();
    survivorCompaction();
    zeroStiffnessReleasesStructure();
    cupFracturesEveryDirection();
    std::cout << "PASS tensile fracture, survivor responses and persistent small fragments\n";
}
