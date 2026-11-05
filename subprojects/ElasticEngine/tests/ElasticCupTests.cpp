#include "ElasticEngine.h"
#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace
{
using Engine = ElasticEngine::ElasticEngine;
using ElasticEngine::ElasticEngine_Scene;
using ElasticEngine::ElasticScenario;
using ElasticEngine::NodeId;
using bl::DVec2;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void fixedContactsAndSprings()
{
    for (double dt : {0.25, 1.0, 2.0})
    {
        Engine engine;
        auto a = engine.addNode({-100.0, 0.0}, {}, true);
        auto b = engine.addNode({100.0, 0.0}, {}, true);
        auto c = engine.addNode({0.0, 5.0}, {2.0, -10.0 / dt});
        engine.addBoundary(a, b);
        engine.step(dt);
        require(engine.getNode(a).position == DVec2{-100.0, 0.0} && engine.getNode(b).position == DVec2{100.0, 0.0} &&
                    engine.getNode(a).velocity.isZero() && engine.getNode(b).velocity.isZero(),
            "Collision moved a fixed wall");
        require(std::abs(engine.getNode(c).position.y - 0.001) < 1e-9 &&
                    std::abs(engine.getNode(c).velocity.x - 2.0) < 1e-10,
            "Fixed wall lost containment or changed tangential motion");
    }

    Engine pinned;
    auto a = pinned.addNode({0.0, 0.0}, {50.0, 0.0}, true);
    auto b = pinned.addNode({12.0, 0.0});
    pinned.addBond(a, b);
    pinned.step(0.0);
    require(pinned.getNode(b).position == DVec2{12.0, 0.0} && pinned.getNode(b).velocity.isZero(),
        "Zero timestep changed the state");
    for (int tick = 0; tick < 1024; ++tick)
        pinned.step();
    require(pinned.getNode(a).position.isZero() && pinned.getNode(a).velocity.isZero() &&
                std::abs(pinned.getNode(b).position.x - 10.0) < 1e-8,
        "Pinned spring did not relax against its anchor");

    pinned.removeNode(a);
    require(pinned.addNode({20.0, 0.0}) == a && !pinned.getNode(a).fixed, "Reused slot inherited its anchor flag");
}

void sceneControls()
{
    ElasticEngine_Scene::Config config;
    ElasticEngine_Scene scene(config);
    scene.sceneStart();
    require(scene.elastic_engine.getNodes().size() == 1028 && scene.elastic_engine.getBonds().size() == 2948,
        "Cup scene does not contain four anchors and a 32 by 32 lattice");
    const auto initial = scene.elastic_engine.getNodes();
    scene.sceneProcess(1.0);
    scene.setStiffness(100.0);
    require(scene.elastic_engine.getNode(4).position != initial[4].position, "Changing stiffness restarted the scene");
    require(
        scene.elastic_engine.getBonds()[0].stiffness == 0.0 && scene.elastic_engine.getBonds()[3].stiffness == 100.0,
        "Live stiffness changed the cup or missed its lattice");
    scene.restartScenario();
    require(scene.elastic_engine.getNode(4).position == initial[4].position && scene.stiffness == 100.0,
        "Restart did not preserve settings and restore geometry");

    scene.setStiffness(0.0);
    scene.sceneProcess(1.0);
    require(scene.elastic_engine.getBonds().size() == 3, "Zero stiffness did not release the cup material");
    for (const auto& bond : scene.elastic_engine.getBonds())
        require(bond.boundary && bond.stiffness == 0.0, "Zero stiffness removed or changed the cup boundary");
    scene.setStiffness(30.0);
    scene.sceneProcess(1.0);
    require(scene.elastic_engine.getBonds().size() == 3, "Raising stiffness restored obsolete material bonds");
    scene.restartScenario();
    require(scene.elastic_engine.getBonds().size() == 2948, "Restart did not rebuild released material");

    scene.selectScenario(ElasticScenario::TwoBalls);
    require(scene.elastic_engine.getNodes().size() == 182 && scene.elastic_engine.getBonds().size() == 480 &&
                scene.gravity == 0.0 && scene.stiffness == 1.0 &&
                scene.elastic_engine.getBonds()[0].breaking_strain == config.breaking_strain,
        "Scenario switch lost the existing two-ball setup");
    scene.selectScenario(ElasticScenario::CupDrop);
    require(scene.elastic_engine.getNodes().size() == 1028 && scene.gravity == 0.02,
        "Switching back did not restore the cup defaults");
}

void timestepRefinement()
{
    double errors[2];
    for (int level = 0; level < 2; ++level)
    {
        Engine engine;
        auto anchor = engine.addNode({0.0, 0.0}, {}, true);
        auto bob = engine.addNode({12.0, 0.0});
        engine.addBond(anchor, bob, 10.0, 100.0, 0.0);
        int steps = 20 << level;
        for (int step = 0; step < steps; ++step)
            engine.advance(1.0 / steps);

        // One pinned axial spring: x'' = -(0.5 * stiffness / rest_length) * (x - rest_length).
        double exact = 10.0 + 2.0 * std::cos(std::sqrt(5.0));
        errors[level] = std::abs(engine.getNode(bob).position.x - exact);
    }
    require(errors[1] < 0.55 * errors[0], "Smaller timesteps did not improve the stiff spring trajectory");
}

void cupDrop(double stiffness, int ticks, double minimum_node_distance = 8.0,
    double breaking_strain = std::numeric_limits<double>::infinity())
{
    ElasticEngine_Scene::Config config;
    config.minimum_node_distance = minimum_node_distance;
    config.breaking_strain = breaking_strain;
    ElasticEngine_Scene scene(config);
    scene.sceneStart();
    scene.setStiffness(stiffness);
    scene.restartScenario();
    Engine& engine = scene.elastic_engine;
    const auto initial = engine.getNodes();
    std::vector<std::array<NodeId, 3>> faces;
    for (int row = 0; row < 31; ++row)
    {
        for (int col = 0; col < 31; ++col)
        {
            NodeId a = 4 + row * 32 + col;
            NodeId b = a + 32;
            if (row % 2 == 0)
            {
                faces.push_back({a, a + 1, b});
                faces.push_back({b, a + 1, b + 1});
            }
            else
            {
                faces.push_back({a, b + 1, b});
                faces.push_back({a, a + 1, b + 1});
            }
        }
    }
    double total_ms = 0.0;
    double maximum_ms = 0.0;
    std::size_t maximum_contacts = 0;
    double minimum_spacing = 10.0;
    double final_window_spacing = 10.0;
    double final_window_ms = 0.0;
    std::size_t previous_bonds = engine.getBonds().size();
    for (int tick = 1; tick <= ticks; ++tick)
    {
        auto start = std::chrono::steady_clock::now();
        try
        {
            scene.sceneProcess(1.0);
            require(scene.simulation_error.empty(), scene.simulation_error.c_str());
        }
        catch (...)
        {
            std::cerr << "cup failure stiffness=" << stiffness << " tick=" << tick << '\n';
            throw;
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        total_ms += ms;
        maximum_ms = std::max(maximum_ms, ms);
        if (tick > ticks - std::min(ticks, 100))
            final_window_ms += ms;
        maximum_contacts = std::max(maximum_contacts, engine.getCollisionStats().contacts);
        for (NodeId id = 0; id < initial.size(); ++id)
        {
            const auto& node = engine.getNode(id);
            require(node.position.isFinite() && node.velocity.isFinite(), "Cup developed nonfinite state");
            if (node.fixed)
            {
                require(node.position == initial[id].position && node.velocity.isZero(), "Cup anchor moved");
                continue;
            }
            require(node.position.y <= 200.0 + 1e-8, "Node passed through the cup floor");
            if (node.position.y > -180.0)
                require(std::abs(node.position.x) <= 220.0 + 1e-8, "Node passed through a cup wall");
        }
        require(engine.getNodes().size() == initial.size(), "Cup lost node slots");
        require(engine.getBonds().size() <= previous_bonds, "Cup created new bonds");
        if (stiffness == 0.0)
            require(engine.getBonds().size() == 3, "Zero-stiffness cup retained material bonds");
        else if (std::isinf(breaking_strain))
            require(engine.getBonds().size() == 2948, "Unbreakable cup fixture lost bonds");

        if (engine.getBonds().size() != previous_bonds)
        {
            // Released edges no longer enclose triangles; only surviving faces must resist inversion.
            std::set<std::pair<NodeId, NodeId>> edges;
            for (const auto& bond : engine.getBonds())
                edges.insert(std::minmax(bond.a, bond.b));
            std::erase_if(faces, [&](const auto& face)
            {
                return !edges.contains(std::minmax(face[0], face[1])) ||
                       !edges.contains(std::minmax(face[1], face[2])) || !edges.contains(std::minmax(face[2], face[0]));
            });
            previous_bonds = engine.getBonds().size();
        }
        for (const auto& bond : engine.getBonds())
        {
            require(bond.a < initial.size() && bond.b < initial.size(), "Cup has an invalid bond endpoint");
            if (bond.boundary)
                continue;
            double strain =
                (engine.getNode(bond.b).position - engine.getNode(bond.a).position).mag() / bond.rest_length - 1.0;
            require(strain <= breaking_strain + 1e-12, "Cup retained an overstretched material bond");
        }

        // Only nearby buckets need spacing checks; this remains cheap for the full 1,024-node fixture.
        if (minimum_node_distance > 0.0)
        {
            std::map<std::pair<int, int>, std::vector<NodeId>> buckets;
            for (NodeId id = 0; id < initial.size(); ++id)
            {
                const auto& node = engine.getNode(id);
                int x = static_cast<int>(std::floor(node.position.x / minimum_node_distance));
                int y = static_cast<int>(std::floor(node.position.y / minimum_node_distance));
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        auto bucket = buckets.find({x + dx, y + dy});
                        if (bucket == buckets.end())
                            continue;
                        for (NodeId other : bucket->second)
                        {
                            double spacing = (node.position - engine.getNode(other).position).mag();
                            minimum_spacing = std::min(minimum_spacing, spacing);
                            if (tick > ticks - std::min(ticks, 100))
                                final_window_spacing = std::min(final_window_spacing, spacing);

                            // Soft contacts have a load-dependent spacing error; keep cup compression below 2%.
                            if (spacing < minimum_node_distance * 0.98 - 1e-7)
                            {
                                std::cerr << "spacing failure tick=" << tick << " spacing=" << spacing << '\n';
                                require(false, "Soft node overlap exceeded 2% of clearance");
                            }
                        }
                    }
                }
                buckets[{x, y}].push_back(id);
            }
        }
        for (auto face : faces)
        {
            DVec2 ab = engine.getNode(face[1]).position - engine.getNode(face[0]).position;
            DVec2 ac = engine.getNode(face[2]).position - engine.getNode(face[0]).position;
            if (ab.x * ac.y - ab.y * ac.x >= 0.0)
            {
                std::cerr << "inverted tick=" << tick << " face=" << face[0] << ',' << face[1] << ',' << face[2]
                          << " area=" << ab.x * ac.y - ab.y * ac.x << '\n';
                for (auto id : face)
                {
                    const auto& node = engine.getNode(id);
                    std::cerr << id << " position=" << node.position.x << ',' << node.position.y
                              << " velocity=" << node.velocity.x << ',' << node.velocity.y << '\n';
                }
            }
            require(ab.x * ac.y - ab.y * ac.x < 0.0, "Cup lattice triangle inverted");
        }
        if ((stiffness < 1.0 && tick % 32 == 0) || tick == 256 || tick == 512 || tick == 1024 || tick == ticks)
        {
            double bottom = 1e100;
            double top = -1e100;
            double speed = 0.0;
            for (const auto& node : engine.getNodes())
            {
                if (node.fixed)
                    continue;
                bottom = std::min(bottom, node.position.y);
                top = std::max(top, node.position.y);
                speed = std::max(speed, node.velocity.mag());
            }
            std::cout << "cup stiffness=" << stiffness << " tick=" << tick << " bottom=" << bottom << " top=" << top
                      << " max_speed=" << speed << " max_contacts=" << maximum_contacts
                      << " bonds=" << engine.getBonds().size() << " minimum_spacing=" << minimum_spacing
                      << " mean_ms=" << total_ms / tick << " peak_ms=" << maximum_ms << std::endl;
        }
    }
    std::cout << "cup final_window=" << std::min(ticks, 100) << " minimum_spacing=" << final_window_spacing
              << " mean_ms=" << final_window_ms / std::min(ticks, 100) << std::endl;
}

}

void runMaterialCupTests()
{
    bool passed = true;
    for (double stiffness : {0.0, 30.0})
    {
        try
        {
            cupDrop(stiffness, 2048, 8.0, 0.02);
        }
        catch (const std::exception& error)
        {
            std::cerr << error.what() << std::endl;
            passed = false;
        }
    }
    require(passed, "Breakable material cup sweep failed");
}

void runCurrentSceneTests(int ticks)
{
    ElasticEngine_Scene::Config config;
    cupDrop(30.0, ticks, config.minimum_node_distance, config.breaking_strain);
}

void runCupTests(bool extreme)
{
    fixedContactsAndSprings();
    sceneControls();
    timestepRefinement();
    bool passed = true;
    for (double stiffness : extreme ? std::array{0.0, 0.1} : std::array{30.0, 100.0})
    {
        try
        {
            cupDrop(stiffness, 2048);
        }
        catch (const std::exception& error)
        {
            std::cerr << error.what() << std::endl;
            passed = false;
        }
    }
    require(passed, "Cup stiffness sweep failed");
}
