#include "ElasticEngine.h"
#include <iostream>
#include <stdexcept>

namespace
{
using Engine = ElasticEngine::ElasticEngine;
using ElasticEngine::ElasticEngine_Scene;
using ElasticEngine::NodeId;
using bl::DVec2;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

Engine copyScene(const Engine& source, bool reorder, bool reflect, bool flip_bonds = false)
{
    Engine result;
    result.setMinimumNodeDistance(source.getMinimumNodeDistance());
    const auto& nodes = source.getNodes();
    const auto& bonds = source.getBonds();
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        const auto& node = nodes[reorder ? nodes.size() - 1 - i : i];
        DVec2 position = node.position;
        DVec2 velocity = node.velocity;
        if (reflect)
        {
            position.y = -position.y;
            velocity.y = -velocity.y;
        }
        result.addNode(position, velocity, node.fixed);
    }
    for (std::size_t i = 0; i < bonds.size(); ++i)
    {
        const auto& bond = bonds[reorder ? bonds.size() - 1 - i : i];
        NodeId a = reorder ? NodeId(nodes.size() - 1 - bond.a) : bond.a;
        NodeId b = reorder ? NodeId(nodes.size() - 1 - bond.b) : bond.b;
        if (flip_bonds)
            std::swap(a, b);
        std::size_t index = result.addBond(a, b, bond.rest_length, bond.stiffness, bond.damping, bond.breaking_strain);
        result.getBond(index).boundary = bond.boundary;
    }
    return result;
}

void impactSymmetry(double breaking_strain, double minimum_node_distance = 0.0)
{
    ElasticEngine_Scene::Config config;
    config.scenario = ElasticEngine::ElasticScenario::TwoBalls;
    config.breaking_strain = breaking_strain;
    config.minimum_node_distance = minimum_node_distance;
    ElasticEngine_Scene scene(config);
    scene.sceneStart();
    auto& engine = scene.elastic_engine;
    auto reordered = copyScene(engine, true, false);
    auto reflected = copyScene(engine, false, true);
    auto flipped = copyScene(engine, false, false, true);
    const auto initial = engine.getNodes();
    std::vector<NodeId> partners(initial.size());
    for (NodeId id = 0; id < initial.size(); ++id)
    {
        double best = 1e100;
        DVec2 target{initial[id].position.x, -initial[id].position.y};
        for (NodeId other = 0; other < initial.size(); ++other)
        {
            double distance = (initial[other].position - target).mag2();
            if (distance < best)
            {
                best = distance;
                partners[id] = other;
            }
        }
        require(best < 1e-20, "Impact fixture is not reflection symmetric");
    }

    double max_symmetry_error = 0.0;
    double max_order_error = 0.0;
    double max_reflection_error = 0.0;
    double max_endpoint_error = 0.0;
    double impact_symmetry_error = 0.0;
    for (int step = 1; step <= 2048; ++step)
    {
        // Exercise the same substep schedule as the displayed scene.
        engine.advance(1.0);
        reordered.advance(1.0);
        reflected.advance(1.0);
        flipped.advance(1.0);
        require(engine.getBonds().size() == reflected.getBonds().size() &&
                    engine.getBonds().size() == flipped.getBonds().size(),
            "Fracture count depends on reflection or endpoint orientation");
        double squared_error = 0.0;
        double reordered_squared_error = 0.0;
        double projectile_y = 0.0;
        int projectile_count = 0;
        for (NodeId id = 0; id < initial.size(); ++id)
        {
            const auto& node = engine.getNode(id);
            const auto& opposite = engine.getNode(partners[id]);
            const auto& other = reordered.getNode(NodeId(initial.size() - 1 - id));
            const auto& reordered_opposite = reordered.getNode(NodeId(initial.size() - 1 - partners[id]));
            const auto& reflection = reflected.getNode(id);
            squared_error += (node.position - DVec2{opposite.position.x, -opposite.position.y}).mag2();
            reordered_squared_error +=
                (other.position - DVec2{reordered_opposite.position.x, -reordered_opposite.position.y}).mag2();
            max_order_error = std::max(max_order_error, (node.position - other.position).mag());
            max_endpoint_error = std::max(max_endpoint_error, (node.position - flipped.getNode(id).position).mag());
            max_reflection_error = std::max(
                max_reflection_error, (node.position - DVec2{reflection.position.x, -reflection.position.y}).mag());
            if (initial[id].velocity.x > 0.0)
            {
                projectile_y += node.position.y;
                ++projectile_count;
            }
        }
        double symmetry_error = std::sqrt(std::max(squared_error, reordered_squared_error) / initial.size());
        max_symmetry_error = std::max(max_symmetry_error, symmetry_error);
        if (step <= 128 || std::isinf(breaking_strain))
            impact_symmetry_error = std::max(impact_symmetry_error, symmetry_error);
        if (step == 80 || step == 512 || step == 2048)
        {
            std::cout << "symmetry step=" << step << " breaking_strain=" << breaking_strain
                      << " bonds=" << engine.getBonds().size() << " max_rms=" << max_symmetry_error
                      << " max_order_error=" << max_order_error << " max_reflection_error=" << max_reflection_error
                      << " max_endpoint_error=" << max_endpoint_error
                      << " projectile_y=" << projectile_y / projectile_count << '\n';
        }
    }
    // Sequential impulses permit small ordering errors; 1 unit is one tenth of a lattice edge.
    // Fracture can amplify tiny ordering differences over long free flight. Compare early bias there.
    require(impact_symmetry_error < 1.0, "Impact developed visible directional bias");
    require(max_reflection_error < 1e-5, "Reflected impact did not reflect the result");
    require(max_endpoint_error < 1e-5, "Bond endpoint orientation changed the impact");
}

void dampingStability()
{
    Engine engine;
    NodeId center = engine.addNode({0.0, 0.0}, {0.25, -0.5});
    const DVec2 drift{0.25, -0.5};
    for (int index = 0; index < 32; ++index)
    {
        double angle = index * 2.0 * std::acos(-1.0) / 32.0;
        DVec2 direction{std::cos(angle), std::sin(angle)};
        NodeId node = engine.addNode(20.0 * direction, drift + direction);
        engine.addBond(center, node, 20.0, 1.0, 0.99);
    }
    auto reordered = copyScene(engine, true, false);
    double previous_energy = 1e100;
    for (int step = 0; step < 256; ++step)
    {
        // Isolate damping by putting each spring at rest at its next force evaluation.
        for (Engine* network : {&engine, &reordered})
        {
            for (std::size_t index = 0; index < network->getBonds().size(); ++index)
            {
                auto& bond = network->getBond(index);
                const auto& a = network->getNode(bond.a);
                const auto& b = network->getNode(bond.b);
                bond.rest_length = ((b.position + b.velocity) - (a.position + a.velocity)).mag();
            }
        }
        engine.step();
        reordered.step();
        double energy = 0.0;
        DVec2 momentum{};
        for (NodeId id = 0; id < engine.getNodes().size(); ++id)
        {
            const auto& node = engine.getNode(id);
            const auto& other = reordered.getNode(NodeId(engine.getNodes().size() - 1 - id));
            energy += 0.5 * (node.velocity - drift).mag2();
            momentum += node.velocity;
            require(node.position.isFinite() && node.velocity.isFinite(), "High-degree damping became non-finite");
            require((node.position - other.position).mag() < 1e-9, "Damping depends on bond traversal");
        }
        require(energy <= previous_energy + 1e-10, "Simultaneous damping added energy");
        require((momentum - double(engine.getNodes().size()) * drift).mag() < 1e-10, "Damping changed shared drift");
        previous_energy = energy;
    }
}
}

void runSymmetryTests()
{
    dampingStability();
    impactSymmetry(std::numeric_limits<double>::infinity());
    impactSymmetry(0.10);
    impactSymmetry(0.05, 8.0);
    std::cout << "PASS stable simultaneous damping and impact symmetry\n";
}
