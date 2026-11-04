#pragma once
#include "ElasticSum.h"
#include <vector>

SIM_BEG;

struct ElasticCollisionStats
{
    std::size_t candidate_tests = 0;
    std::size_t contacts = 0;
    int passes = 0;
};

class ElasticCollisions
{
    struct Bounds
    {
        bl::DVec2 lower;
        bl::DVec2 upper;

        void include(bl::DVec2 point);
        bool overlaps(const Bounds& other) const;
    };

    struct Segment
    {
        NodeId a;
        NodeId b;
        Bounds bounds;
    };

    struct Entry
    {
        std::uint64_t cell;
        std::size_t segment;
        std::size_t next;
    };

    struct Contact
    {
        NodeId node;
        NodeId a;
        NodeId b;
        double coordinate; // -0.5 at a, +0.5 at b; centered to preserve endpoint symmetry.
        bl::DVec2 normal;
        double speed_limit;
        double speed_tolerance;
        double response_scale = 0.0;
        double closing = 0.0;
        double speed_change = 0.0;
    };

    std::vector<Segment> segments;
    std::vector<Entry> entries;
    std::vector<std::size_t> buckets;
    std::vector<NodeId> visited;
    std::vector<bool> connected;
    std::vector<Contact> contacts;
    std::vector<bl::DVec2> original_velocities;
    std::vector<ElasticVectorSum> velocity_changes;
    std::vector<ElasticSum> node_weights;
    std::vector<NodeId> contact_nodes;
    bl::DVec2 frame_velocity{};
    double step_dt = 1.0;
    double minimum_node_distance = 0.0;
    ElasticCollisionStats stats;

    void buildGrid(const std::vector<ElasticNode>& nodes, const std::vector<ElasticBond>& bonds);
    void findContacts(const std::vector<ElasticNode>& nodes);
    void resolveContacts(std::vector<ElasticNode>& nodes);

public:
    // Small node-spacing overlaps recover gradually; intersecting membranes are not supported.
    void setMinimumNodeDistance(double distance)
    {
        minimum_node_distance = distance;
    }

    double getMinimumNodeDistance() const
    {
        return minimum_node_distance;
    }

    // False restores input velocities; the caller must not commit an unchecked move.
    bool resolve(std::vector<ElasticNode>& nodes, const std::vector<ElasticBond>& bonds, double dt = 1.0);

    const ElasticCollisionStats& getStats() const
    {
        return stats;
    }
};

SIM_END;
