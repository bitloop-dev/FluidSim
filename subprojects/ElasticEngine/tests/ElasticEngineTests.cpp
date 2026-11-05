#include "ElasticEngine.h"
#include <cmath>
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

bool vectorsClose(DVec2 a, DVec2 b, double tolerance = 1e-10)
{
    return (a - b).mag() <= tolerance;
}

void nodeSlotsSurviveGrowthAndReuse()
{
    Engine engine;
    NodeId a = engine.addNode({1.0, 2.0});
    NodeId b = engine.addNode({15.0, 2.0});
    engine.addBond(a, b);

    const std::size_t capacity = engine.getNodes().capacity();
    for (int i = 0; i < 4096; ++i)
        engine.addNode({100.0 + i, 50.0});

    require(engine.getNodes().capacity() > capacity, "Fixture did not grow node storage");
    engine.step();
    require(
        vectorsClose(engine.getNode(a).velocity, {0.2, 0.0}) && vectorsClose(engine.getNode(b).velocity, {-0.2, 0.0}),
        "Growth changed spring endpoint identity");

    NodeId c = engine.addNode({8.0, 14.0});
    engine.addBond(b, c);
    engine.addBond(a, c);
    const DVec2 old_position = engine.getNode(b).position;
    const DVec2 old_velocity = engine.getNode(b).velocity;
    engine.removeNode(b);

    require(!engine.getNode(b).active && engine.getBonds().size() == 1 && engine.getBonds()[0].a == a &&
                engine.getBonds()[0].b == c,
        "Node deletion removed the wrong bonds");
    engine.step();
    require(engine.getNode(b).position == old_position && engine.getNode(b).velocity == old_velocity,
        "Inactive slot was advanced");

    const std::size_t slots = engine.getNodes().size();
    for (int i = 0; i < 512; ++i)
    {
        NodeId reused = engine.addNode({30.0, 40.0}, {0.5, -0.25});
        require(reused == b && engine.getNodes().size() == slots && engine.getNode(reused).active &&
                    engine.getNode(reused).position == DVec2(30.0, 40.0) &&
                    engine.getNode(reused).velocity == DVec2(0.5, -0.25),
            "Free slot was not reused and reinitialised");
        require(engine.getBonds().size() == 1, "Reused node inherited old bonds");
        engine.addBond(a, reused);
        engine.removeNode(reused);
        require(engine.getBonds().size() == 1, "Repeated deletion left an incident bond");
    }

    engine.removeNode(a);
    engine.removeNode(c);
    require(engine.getBonds().empty(), "Endpoint deletion left dangling bonds");
}

void bondsStayPacked()
{
    Engine engine;
    NodeId a = engine.addNode({0.0, 0.0});
    NodeId b = engine.addNode({10.0, 0.0});
    NodeId c = engine.addNode({5.0, 9.0});
    engine.addBond(a, b);
    engine.addBond(b, c);
    engine.addBond(c, a, 12.0, 0.3, 0.4);
    engine.removeBond(1);

    const auto& moved = engine.getBonds()[1];
    require(engine.getBonds().size() == 2 && moved.a == c && moved.b == a && moved.rest_length == 12.0 &&
                moved.stiffness == 0.3 && moved.damping == 0.4,
        "Removing a middle bond lost the last bond or its parameters");
    engine.removeBond(1);
    require(engine.getBonds().size() == 1 && engine.getBonds()[0].a == a && engine.getBonds()[0].b == b,
        "Removing the last bond changed a survivor");
    engine.removeBond(0);
    require(engine.getBonds().empty() && engine.getNodes().size() == 3, "Breaking bonds changed node storage");
}

void springResponseAndRelativeDamping()
{
    for (double distance : {8.0, 12.0})
    {
        Engine engine;
        NodeId a = engine.addNode({0.0, 0.0});
        NodeId b = engine.addNode({distance, 0.0});
        engine.addBond(a, b);
        engine.step();

        require(engine.getNode(a).velocity.x * (distance - 10.0) > 0.0 &&
                    vectorsClose(engine.getNode(a).velocity + engine.getNode(b).velocity, {}),
            "Spring has the wrong restoring direction or creates momentum");
    }

    Engine damping;
    NodeId a = damping.addNode({0.0, 0.0}, {0.25, 0.0});
    NodeId b = damping.addNode({9.0, 0.0}, {1.25, 0.0});
    damping.addBond(a, b, 10.0, 1.0, 0.5);
    damping.step();
    require(
        vectorsClose(damping.getNode(a).velocity, {0.5, 0.0}) && vectorsClose(damping.getNode(b).velocity, {1.0, 0.0}),
        "Bond damping did not reduce relative speed while retaining drift");

    Engine transverse;
    a = transverse.addNode({0.0, -1.0}, {0.0, 1.0});
    b = transverse.addNode({10.0, 1.0}, {0.0, -1.0});
    transverse.addBond(a, b);
    transverse.step();
    require(transverse.getNode(a).velocity == DVec2(0.0, 1.0) && transverse.getNode(b).velocity == DVec2(0.0, -1.0),
        "Axial damping resisted transverse motion");

    Engine coincident;
    a = coincident.addNode({0.0, 0.0});
    b = coincident.addNode({0.0, 0.0});
    coincident.addBond(a, b);
    coincident.step();
    require(coincident.getNode(a).velocity.isFinite() && coincident.getNode(b).velocity.isFinite(),
        "Coincident endpoints produced non-finite velocities");
}

void driftingTriangleSettlesAndBreaks()
{
    Engine resting;
    Engine drifting;
    const DVec2 drift{0.125, -0.25};
    for (DVec2 position : {DVec2{10.0, 10.0}, DVec2{20.0, 10.0}, DVec2{15.0, 28.660254}})
    {
        resting.addNode(position);
        drifting.addNode(position, drift);
    }
    for (Engine* engine : {&resting, &drifting})
    {
        engine->addBond(0, 1);
        engine->addBond(1, 2);
        engine->addBond(2, 0);
    }

    const DVec2 initial_center{15.0, (20.0 + 28.660254) / 3.0};
    for (int step = 1; step <= 4096; ++step)
    {
        resting.step();
        drifting.step();
        DVec2 center{};
        DVec2 momentum{};
        for (NodeId id = 0; id < 3; ++id)
        {
            const auto& a = resting.getNode(id);
            const auto& b = drifting.getNode(id);
            require(a.position.isFinite() && a.velocity.isFinite() && b.position.isFinite() && b.velocity.isFinite(),
                "Triangle developed non-finite state");
            require(vectorsClose(b.position - a.position, drift * double(step), 1e-7) &&
                        vectorsClose(b.velocity - a.velocity, drift, 1e-9),
                "Shared drift changed the triangle's relative dynamics");
            center += b.position / 3.0;
            momentum += b.velocity;
        }
        require(vectorsClose(center, initial_center + drift * double(step), 1e-7) &&
                    vectorsClose(momentum, drift * 3.0, 1e-9),
            "Internal forces changed centre-of-mass motion");

        if (step == 256 || step == 4096)
        {
            double length_error = 0.0;
            for (const auto& bond : resting.getBonds())
            {
                double length = (resting.getNode(bond.b).position - resting.getNode(bond.a).position).mag();
                length_error = std::max(length_error, std::abs(length - bond.rest_length));
            }
            require(length_error < 1e-6, "Triangle failed to settle to its rest lengths");
            std::cout << "triangle step=" << step << " maximum_length_error=" << length_error << '\n';
        }
    }

    // Remove both bonds attached to node 0; the other pair remains connected.
    drifting.removeBond(0);
    drifting.removeBond(0);
    const DVec2 start = drifting.getNode(0).position;
    const DVec2 velocity{-0.25, 0.5};
    drifting.getNode(0).velocity = velocity;
    for (int step = 1; step <= 1024; ++step)
    {
        drifting.step();
        require(drifting.getNode(0).velocity == velocity &&
                    vectorsClose(drifting.getNode(0).position, start + velocity * double(step), 1e-8),
            "Broken bonds still influenced the detached node");
    }
}
}

void runCollisionTests();
void runSymmetryTests();
void runFractureTests();
void runNodeContactTests();
void runCupTests(bool extreme = false);
void runMaterialCupTests();
void runCurrentSceneTests(int ticks = 400);

int main(int argc, char** argv)
{
    try
    {
        runNodeContactTests();
        if (argc == 2 && std::string_view(argv[1]) == "--symmetry")
        {
            runSymmetryTests();
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--current-scene")
        {
            runCurrentSceneTests(argc >= 3 ? std::stoi(argv[2]) : 400);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--material-cup")
        {
            runMaterialCupTests();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--cup-extreme")
        {
            runCupTests(true);
            return 0;
        }
        nodeSlotsSurviveGrowthAndReuse();
        bondsStayPacked();
        springResponseAndRelativeDamping();
        driftingTriangleSettlesAndBreaks();
        runFractureTests();
        runCollisionTests();
        runSymmetryTests();
        runCurrentSceneTests();
        runCupTests();
        std::cout << "PASS stable node slots, bond removal, relative damping and sustained drift\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
