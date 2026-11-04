#include "ElasticEngine.h"
#include <chrono>
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

double cross(DVec2 a, DVec2 b)
{
    return a.x * b.y - a.y * b.x;
}

DVec2 momentum(const Engine& engine)
{
    DVec2 total{};
    for (const auto& node : engine.getNodes())
    {
        if (node.active)
            total += node.velocity;
    }
    return total;
}

double kineticEnergy(const Engine& engine)
{
    double total = 0.0;
    for (const auto& node : engine.getNodes())
        total += 0.5 * node.velocity.mag2();
    return total;
}

double angularMomentum(const Engine& engine)
{
    double total = 0.0;
    for (const auto& node : engine.getNodes())
        total += cross(node.position, node.velocity);
    return total;
}

void assertDissipativeStep(Engine& engine)
{
    DVec2 before = momentum(engine);
    double energy = kineticEnergy(engine);
    engine.step();
    require((momentum(engine) - before).mag() < 1e-9, "Contact created linear momentum");
    require(kineticEnergy(engine) <= energy + 1e-9, "Frictionless inelastic contact added energy");
    for (const auto& node : engine.getNodes())
        require(node.position.isFinite() && node.velocity.isFinite(), "Contact produced non-finite state");
}

void sweptContacts()
{
    // A point crosses a long segment in one step; include a moving segment and a shared boost.
    for (bool moving_bond : {false, true})
    {
        for (DVec2 boost : {DVec2{}, DVec2{100.0, -50.0}})
        {
            Engine engine;
            NodeId a = engine.addNode({-30.0, 0.0}, boost + (moving_bond ? DVec2{0.0, -20.0} : DVec2{}));
            NodeId b = engine.addNode({30.0, 0.0}, engine.getNode(a).velocity);
            NodeId p = engine.addNode({0.0, -5.0}, boost + (moving_bond ? DVec2{} : DVec2{0.0, 20.0}));
            engine.addBoundary(a, b);
            assertDissipativeStep(engine);
            require(engine.getCollisionStats().contacts > 0, "Swept point/bond collision was missed");
            require(engine.getNode(p).position.y < engine.getNode(a).position.y,
                "Fast point tunneled through a moving or stretched bond");
            require((engine.getNode(a).velocity - engine.getNode(b).velocity).mag() < 1e-9,
                "Midpoint impact did not share its impulse equally");
        }
    }

    // The signed area has roots at t=.25 and t=.75; both ends of the step have the same sign.
    Engine rotating;
    NodeId a = rotating.addNode({0.0, 0.0});
    NodeId b = rotating.addNode({10.0, 0.0}, {0.0, 10.0});
    NodeId p = rotating.addNode({10.0, 1.875}, {-10.0, 0.0});
    rotating.addBoundary(a, b);
    auto initial = rotating.getNodes();
    assertDissipativeStep(rotating);
    require(rotating.getCollisionStats().contacts > 0, "Rotating bond's intermediate crossing was missed");
    for (int sample = 0; sample <= 1000; ++sample)
    {
        double t = double(sample) / 1000.0;
        DVec2 pa = initial[a].position + t * rotating.getNode(a).velocity;
        DVec2 pb = initial[b].position + t * rotating.getNode(b).velocity;
        DVec2 pp = initial[p].position + t * rotating.getNode(p).velocity;
        require(cross(pb - pa, pp - pa) > 0.0, "Corrected rotating-bond trajectory still crosses");
    }

    // Collinear motion must hit an endpoint instead of entering through the end of a bond.
    Engine parallel;
    a = parallel.addNode({0.0, 0.0});
    b = parallel.addNode({10.0, 0.0});
    p = parallel.addNode({15.0, 0.0}, {-20.0, 0.0});
    parallel.addBoundary(a, b);
    assertDissipativeStep(parallel);
    require(parallel.getNode(p).position.x > parallel.getNode(b).position.x, "Collinear endpoint contact tunneled");

    Engine parallel_bonds;
    a = parallel_bonds.addNode({-20.0, 0.0});
    b = parallel_bonds.addNode({20.0, 0.0});
    NodeId c = parallel_bonds.addNode({-5.0, -5.0}, {0.0, 20.0});
    NodeId d = parallel_bonds.addNode({5.0, -5.0}, {0.0, 20.0});
    parallel_bonds.addBoundary(a, b);
    parallel_bonds.addBoundary(c, d);
    assertDissipativeStep(parallel_bonds);
    for (NodeId id : {c, d})
    {
        require(cross(parallel_bonds.getNode(b).position - parallel_bonds.getNode(a).position,
                    parallel_bonds.getNode(id).position - parallel_bonds.getNode(a).position) < 0.0,
            "Short parallel bond passed through a longer bond");
    }

    Engine particles;
    a = particles.addNode({-1.0, 0.0}, {10.0, 0.0});
    b = particles.addNode({1.0, 0.0}, {-10.0, 0.0});
    for (int step = 0; step < 128; ++step)
    {
        assertDissipativeStep(particles);
        require(particles.getNode(a).position.x < particles.getNode(b).position.x,
            "Detached particles crossed or swapped identities");
    }

    // The opposite edge must stay eligible even though all three nodes are connected.
    Engine triangle;
    a = triangle.addNode({-5.0, 0.0});
    b = triangle.addNode({5.0, 0.0});
    p = triangle.addNode({0.0, 5.0}, {0.0, -20.0});
    triangle.addBoundary(a, b);
    triangle.addBoundary(b, p);
    triangle.addBoundary(p, a);
    assertDissipativeStep(triangle);
    require(cross(triangle.getNode(b).position - triangle.getNode(a).position,
                triangle.getNode(p).position - triangle.getNode(a).position) > 0.0,
        "Triangle inverted through its opposite bond");
}

struct Triangle
{
    NodeId a;
    NodeId b;
    NodeId c;
    double orientation;
};

std::vector<Triangle> triangles(const Engine& engine)
{
    const auto& nodes = engine.getNodes();
    std::vector<std::vector<bool>> adjacent(nodes.size(), std::vector<bool>(nodes.size(), false));
    for (const auto& bond : engine.getBonds())
        adjacent[bond.a][bond.b] = adjacent[bond.b][bond.a] = true;

    std::vector<Triangle> result;
    for (NodeId a = 0; a < nodes.size(); ++a)
    {
        for (NodeId b = a + 1; b < nodes.size(); ++b)
        {
            if (!adjacent[a][b])
                continue;

            for (NodeId c = b + 1; c < nodes.size(); ++c)
            {
                if (adjacent[a][c] && adjacent[b][c])
                    result.push_back(
                        {a, b, c, cross(nodes[b].position - nodes[a].position, nodes[c].position - nodes[a].position)});
            }
        }
    }
    return result;
}

bool intersects(DVec2 a, DVec2 b, DVec2 c, DVec2 d)
{
    if (std::max(a.x, b.x) < std::min(c.x, d.x) || std::max(c.x, d.x) < std::min(a.x, b.x) ||
        std::max(a.y, b.y) < std::min(c.y, d.y) || std::max(c.y, d.y) < std::min(a.y, b.y))
    {
        return false;
    }

    double ac = cross(b - a, c - a);
    double ad = cross(b - a, d - a);
    if (std::abs(ac) < 1e-10 && std::abs(ad) < 1e-10)
    {
        bool horizontal = std::abs(b.x - a.x) >= std::abs(b.y - a.y);
        double a0 = horizontal ? a.x : a.y;
        double a1 = horizontal ? b.x : b.y;
        double b0 = horizontal ? c.x : c.y;
        double b1 = horizontal ? d.x : d.y;
        return std::min(std::max(a0, a1), std::max(b0, b1)) - std::max(std::min(a0, a1), std::min(b0, b1)) > 1e-10;
    }

    return ac * ad < 0.0 && cross(d - c, a - c) * cross(d - c, b - c) < 0.0;
}

void ballImpact(int steps, bool transformed, double projectile_speed,
    double breaking_strain = std::numeric_limits<double>::infinity(), double minimum_node_distance = 0.0)
{
    ElasticEngine_Scene::Config config;
    config.scenario = ElasticEngine::ElasticScenario::TwoBalls;
    config.breaking_strain = breaking_strain;
    config.minimum_node_distance = minimum_node_distance;
    ElasticEngine_Scene scene(config);
    scene.sceneStart();
    auto& engine = scene.elastic_engine;
    for (NodeId id = 0; id < engine.getNodes().size(); ++id)
    {
        auto& node = engine.getNode(id);
        node.velocity *= projectile_speed / 2.0;
        if (transformed)
        {
            node.position = {
                0.6 * node.position.x - 0.8 * node.position.y, 0.8 * node.position.x + 0.6 * node.position.y};
            node.velocity =
                DVec2{0.6 * node.velocity.x - 0.8 * node.velocity.y, 0.8 * node.velocity.x + 0.6 * node.velocity.y} +
                DVec2{0.125, -0.25};
        }
    }

    auto faces = triangles(engine);
    const std::size_t initial_nodes = engine.getNodes().size();
    const std::size_t initial_bonds = engine.getBonds().size();
    std::size_t previous_bonds = initial_bonds;
    int first_break = 0;
    int last_break = 0;
    const DVec2 initial_momentum = momentum(engine);
    const double initial_angular_momentum = angularMomentum(engine);
    const auto& bonds = engine.getBonds();
    std::size_t contacts = 0;
    std::size_t candidates = 0;
    int max_passes = 0;
    double elapsed = 0.0;
    double peak_step = 0.0;
    double maximum_strain = 0.0;
    for (int step = 0; step < steps; ++step)
    {
        const auto before = engine.getNodes();
        auto start = std::chrono::steady_clock::now();
        try
        {
            engine.step();
        }
        catch (const std::exception&)
        {
            std::cerr << "impact failed at step=" << step + 1 << " speed=" << projectile_speed
                      << " contacts=" << engine.getCollisionStats().contacts << '\n';
            throw;
        }
        double duration = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        elapsed += duration;
        peak_step = std::max(peak_step, duration);
        contacts += engine.getCollisionStats().contacts;
        candidates += engine.getCollisionStats().candidate_tests;
        max_passes = std::max(max_passes, engine.getCollisionStats().passes);
        require((momentum(engine) - initial_momentum).mag() < 1e-8, "Ball impact lost linear momentum");
        require(engine.getNodes().size() == initial_nodes, "Fracture removed node slots");
        require(bonds.size() <= previous_bonds, "Fracture unexpectedly added bonds");
        for (const auto& node : engine.getNodes())
        {
            require(node.active, "Fracture removed an active node");
            require(node.position.isFinite() && node.velocity.isFinite(), "Ball impact developed non-finite state");
        }
        for (const auto& bond : bonds)
        {
            require(bond.a < initial_nodes && bond.b < initial_nodes, "Fracture corrupted a bond endpoint");
            double length = (engine.getNode(bond.b).position - engine.getNode(bond.a).position).mag();
            double strain = (length - bond.rest_length) / bond.rest_length;
            maximum_strain = std::max(maximum_strain, strain);
            require(strain <= bond.breaking_strain + 1e-12, "Overstretched bond survived fracture");
        }
        for (const auto& face : faces)
        {
            DVec2 a = engine.getNode(face.a).position;
            DVec2 b = engine.getNode(face.b).position;
            DVec2 c = engine.getNode(face.c).position;
            require(cross(b - a, c - a) * face.orientation > 0.0, "Ball impact inverted a triangle");

            // The signed area is quadratic during the committed linear motion.
            DVec2 edge = before[face.b].position - before[face.a].position;
            DVec2 offset = before[face.c].position - before[face.a].position;
            DVec2 edge_change = b - a - edge;
            DVec2 offset_change = c - a - offset;
            double quadratic = cross(edge_change, offset_change) * face.orientation;
            double linear = (cross(edge_change, offset) + cross(edge, offset_change)) * face.orientation;
            if (quadratic > 0.0)
            {
                double time = std::clamp(-linear / (2.0 * quadratic), 0.0, 1.0);
                require(cross(edge + time * edge_change, offset + time * offset_change) * face.orientation > 0.0,
                    "Triangle inverted between committed steps");
            }
        }
        for (std::size_t i = 0; i < bonds.size(); ++i)
        {
            const auto& a = bonds[i];
            for (std::size_t j = i + 1; j < bonds.size(); ++j)
            {
                const auto& b = bonds[j];
                if (a.a == b.a || a.a == b.b || a.b == b.a || a.b == b.b)
                    continue;

                require(!intersects(engine.getNode(a.a).position, engine.getNode(a.b).position,
                            engine.getNode(b.a).position, engine.getNode(b.b).position),
                    "Ball impact crossed nonadjacent bonds");
            }
        }

        if (bonds.size() < previous_bonds)
        {
            if (first_break == 0)
                first_break = step + 1;

            last_break = step + 1;
            previous_bonds = bonds.size();

            // The deleted edges stop bounding triangles on subsequent moves.
            faces = triangles(engine);
        }
    }

    require(contacts > 0, "Ball fixture never made contact");
    std::cout << "impact steps=" << steps << " transformed=" << transformed << " speed=" << projectile_speed
              << " breaking_strain=" << breaking_strain << " mean_us=" << elapsed / steps << " peak_us=" << peak_step
              << " max_passes=" << max_passes << " mean_candidates=" << double(candidates) / steps
              << " contacts=" << contacts << '\n';
    std::cout << "impact angular_momentum_change=" << angularMomentum(engine) - initial_angular_momentum << '\n';
    std::cout << "fracture broken=" << initial_bonds - bonds.size() << " first_step=" << first_break
              << " last_step=" << last_break << " remaining_nodes=" << engine.getNodes().size()
              << " maximum_surviving_strain=" << maximum_strain << '\n';
    if (std::isfinite(breaking_strain))
        require(first_break > 0, "Breakable impact never exercised fracture");
}

void localSearchAndRemoval()
{
    Engine engine;
    for (int i = 0; i < 400; ++i)
    {
        DVec2 offset{double(i % 20) * 100.0, double(i / 20) * 100.0};
        NodeId a = engine.addNode(offset);
        NodeId b = engine.addNode(offset + DVec2{10.0, 0.0});
        NodeId c = engine.addNode(offset + DVec2{5.0, std::sqrt(75.0)});
        engine.addBond(a, b);
        engine.addBond(b, c);
        engine.addBond(c, a);
    }
    engine.step();
    require(engine.getCollisionStats().candidate_tests < 12000, "Broad phase fell back to all-pairs search");
    require(engine.getCollisionStats().contacts == 0, "Separated resting triangles generated contact");
    engine.removeNode(0);
    NodeId reused = engine.addNode({-100.0, -100.0}, {-1.0, 0.0});
    require(reused == 0, "Collision fixture failed to reuse its slot");
    engine.step();
    require(engine.getNode(reused).velocity == DVec2(-1.0, 0.0), "Collision search retained a removed bond");
}
}

void runCollisionTests()
{
    require(intersects({-1.0, 0.0}, {1.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}), "Crossing oracle missed an intersection");
    require(!intersects({-1.0, 0.0}, {1.0, 0.0}, {-1.0, 2.0}, {1.0, 2.0}), "Crossing oracle rejected separated edges");
    require(intersects({0.0, 0.0}, {10.0, 0.0}, {5.0, 0.0}, {15.0, 0.0}), "Crossing oracle missed collinear overlap");
    sweptContacts();
    localSearchAndRemoval();
    ballImpact(2048, false, 2.0);
    ballImpact(1024, true, 2.0);
    ballImpact(1024, false, 8.0);
    ballImpact(2048, false, 2.0, 0.10);
    ballImpact(2048, false, 2.0, 0.05, 8.0);
    std::cout << "PASS continuous contacts, self-collision, impact conservation and spatial search\n";
}
