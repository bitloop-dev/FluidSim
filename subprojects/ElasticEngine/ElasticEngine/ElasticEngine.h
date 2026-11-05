#pragma once
#include "ElasticCollisions.h"
#include "ElasticScenarios.h"
#include <algorithm>
#include <vector>

SIM_BEG;

using namespace bl;

class ElasticEngine
{
    std::vector<ElasticNode> nodes;
    std::vector<NodeId> free_nodes;
    std::vector<ElasticBond> bonds;
    ElasticCollisions collisions;

    std::vector<DVec2> bond_directions;
    std::vector<ElasticVectorSum> spring_impulses;
    std::vector<ElasticVectorSum> velocity_changes;
    std::vector<double> damping_weights;

    std::vector<DVec2> step_velocities;

    void applySprings(double dt);

    static constexpr double node_radius = 2.0;
    static constexpr double bond_thickness = 1.0;

public:
    NodeId addNode(DVec2 position, DVec2 velocity = {}, bool fixed = false)
    {
        ElasticNode node{position, fixed ? DVec2{} : velocity, true, fixed};

        if (!free_nodes.empty())
        {
            NodeId id = free_nodes.back();
            free_nodes.pop_back();

            nodes[id] = node;
            return id;
        }

        NodeId id = static_cast<NodeId>(nodes.size());
        nodes.push_back(node);

        return id;
    }

    // The caller supplies an existing, active node. Edit topology between steps.
    void removeNode(NodeId id)
    {
        collisions.clearHistory();

        // Remove references before allowing this slot to be reused.
        std::erase_if(bonds, [id](const ElasticBond& bond)
        {
            return bond.a == id || bond.b == id;
        });

        nodes[id].active = false;
        free_nodes.push_back(id);
    }

    std::size_t addBond(NodeId a, NodeId b, double rest_length = 10.0, double stiffness = 1.0, double damping = 0.99,
        double breaking_strain = std::numeric_limits<double>::infinity())
    {
        std::size_t index = bonds.size();
        bonds.push_back({a, b, rest_length, stiffness, damping, breaking_strain});

        return index;
    }

    std::size_t addBoundary(NodeId a, NodeId b)
    {
        double length = (nodes[b].position - nodes[a].position).mag();
        std::size_t index = addBond(a, b, length, 0.0, 0.0);
        bonds[index].boundary = true;
        return index;
    }

    // Bond indices are temporary: removal, zero stiffness or tensile fracture can change them.
    void removeBond(std::size_t index)
    {
        collisions.clearHistory();

        if (index != bonds.size() - 1)
            bonds[index] = bonds.back();

        bonds.pop_back();
    }

    // Retain IDs across node insertion; vector growth can invalidate this reference.
    ElasticNode& getNode(NodeId id)
    {
        return nodes[id];
    }

    const ElasticNode& getNode(NodeId id) const
    {
        return nodes[id];
    }

    const std::vector<ElasticNode>& getNodes() const
    {
        return nodes;
    }

    const std::vector<ElasticBond>& getBonds() const
    {
        return bonds;
    }

    ElasticBond& getBond(std::size_t index)
    {
        return bonds[index];
    }

    const ElasticCollisionStats& getCollisionStats() const
    {
        return collisions.getStats();
    }

    void setMinimumNodeDistance(double distance) { collisions.setMinimumNodeDistance(distance); }
    double getMinimumNodeDistance() const { return collisions.getMinimumNodeDistance(); }

    // The default retains the original playground's one-unit update.
    void step(double dt = 1.0, DVec2 acceleration = {});

    // Cover an interval with stiffness-limited explicit steps.
    void advance(double dt, DVec2 acceleration = {});

    void draw(Viewport* ctx) const
    {
        for (const ElasticNode& node : nodes)
        {
            if (!node.active)
                continue;

            ctx->fillEllipse(node.position, node.fixed ? 3.5 : node_radius);
        }

        ctx->setLineWidth(bond_thickness);

        for (const ElasticBond& bond : bonds)
        {
            const ElasticNode& a = nodes[bond.a];
            const ElasticNode& b = nodes[bond.b];

            ctx->strokeLine(a.position, b.position);
        }
    }
};

struct ElasticEngine_Scene : public Scene<ElasticEngine_Scene>
{
    CameraInfo       camera;
    CameraNavigator  navigator;

    /// ─────── Variables ───────
    ElasticEngine elastic_engine;
    ElasticScenario scenario;
    double stiffness = 1.0;
    double gravity = 0.0;
    double breaking_strain;
    double minimum_node_distance;
    bool frame_scenario = false;
    std::string simulation_error;

    void selectScenario(ElasticScenario selected);
    void restartScenario();
    void setStiffness(double value);
    void frameScenario(Viewport* ctx);

    /// ─────── Methods ───────


    /// ─────── Launch config ───────
    struct Config
    {
        ElasticScenario scenario = ElasticScenario::CupDrop;
        double breaking_strain = 0.3; // Material lattice; the projectile and cup walls stay unbreakable.
        double minimum_node_distance = 8.0;
    };

    ElasticEngine_Scene(Config& info) : scenario(info.scenario), breaking_strain(info.breaking_strain),
        minimum_node_distance(info.minimum_node_distance)
    {}

    struct UI : BufferedInterfaceModel
    {
        using BufferedInterfaceModel::BufferedInterfaceModel;
        void sidebar() override;
        // void overlay() override; // ImGui viewport overlay

        /// ─────── Your UI-only variables ───────

    };

    /// ─────── Scene methods ───────
    void sceneStart() override;
    void sceneMounted(Viewport* viewport) override;
    void sceneUnmounted(Viewport* viewport) override;
    void sceneDestroy() override;
    void sceneProcess(double dt) override;

    /// ─────── Viewport methods ───────
    void viewportProcess(Viewport* ctx, double dt) override;
    void viewportDraw(Viewport* ctx) const override;

    /// ─────── Input handling ───────
    void onEvent(Event e) override; // receives all event types


    // ─────── AI hooks ───────
    #ifdef BITLOOP_DEV_MODE
    void traceFrame(SimTraceFrame& frame) const override;
    bool captureTraceSnapshot(std::string_view route, SimSnapshotImage& image) const override;
    bool saveCheckpoint(SimCheckpointWriter& writer) const override;
    bool loadCheckpoint(const SimCheckpointReader& reader) override;
    #endif
};

struct ElasticEngine_Project : public Project<ElasticEngine_Project>
{
    static ProjectInfo info()
    {
        return ProjectInfo({ "New Projects", "ElasticEngine" });
    }

    struct UI : BufferedInterfaceModel
    {
        using BufferedInterfaceModel::BufferedInterfaceModel;
        void sidebar();
        // void overlay();
    };

    int viewport_count = 1;

    void projectPrepare(Layout& layout) override;
};

SIM_END;
