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
        double clearance;
        double mass = 0.0;
        double impulse = 0.0;
    };

    std::vector<Segment> segments;
    std::vector<Entry> entries;
    std::vector<std::size_t> buckets;
    std::vector<NodeId> visited;
    std::vector<bool> connected;
    std::vector<Contact> contacts;
    std::vector<Contact> previous_contacts;
    std::vector<Contact> barriers;
    std::vector<ElasticVectorSum> velocity_changes;
    std::vector<ElasticSum> node_weights;
    std::vector<NodeId> contact_nodes;
    std::vector<bl::DVec2> original_velocities;
    bl::DVec2 frame_velocity{};
    double step_dt = 1.0;
    double previous_dt = 0.0;
    double minimum_node_distance = 0.0;
    bool reverse_order = false;
    ElasticCollisionStats stats;

    void buildGrid(const std::vector<ElasticNode>& nodes, const std::vector<ElasticBond>& bonds);
    void findContacts(const std::vector<ElasticNode>& nodes, std::vector<Contact>& result, bool safety);
    void prepareContacts(std::vector<ElasticNode>& nodes);
    void solveContacts(std::vector<ElasticNode>& nodes, bool use_bias);
    bool protectCrossings(std::vector<ElasticNode>& nodes, const std::vector<ElasticBond>& bonds);
    static double contactMass(const Contact& contact, const std::vector<ElasticNode>& nodes);
    static void applyImpulse(const Contact& contact, std::vector<ElasticNode>& nodes, double impulse);

public:
    // Contact softness allows small spacing errors; intersecting initial membranes are unsupported.
    void setMinimumNodeDistance(double distance)
    {
        minimum_node_distance = distance;
        clearHistory();
    }

    double getMinimumNodeDistance() const
    {
        return minimum_node_distance;
    }

    // False restores input velocities; the caller must not commit an unchecked move.
    bool resolve(std::vector<ElasticNode>& nodes, const std::vector<ElasticBond>& bonds, double dt = 1.0);

    // Run after checked movement, before spring forces for the next step.
    void relax(std::vector<ElasticNode>& nodes);

    void clearHistory()
    {
        contacts.clear();
        previous_contacts.clear();
        previous_dt = 0.0;
    }

    const ElasticCollisionStats& getStats() const
    {
        return stats;
    }
};

SIM_END;
