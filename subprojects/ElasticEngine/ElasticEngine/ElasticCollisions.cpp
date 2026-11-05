#include "ElasticCollisions.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <tuple>

SIM_BEG;

using bl::DVec2;

namespace
{
// Search scale and numerical contact gap; neither depends on the drawing radius.
constexpr double cell_size = 10.0;
constexpr double separation = 0.001;

// The tolerance stays inside the bond safety gap; a swept crossing must still be resolved.
constexpr double position_tolerance = 0.1 * separation;
constexpr double contact_hertz = 4.0;
constexpr double damping_ratio = 1.0;
constexpr double contact_margin = 0.2;

double cross(DVec2 a, DVec2 b)
{
    return a.x * b.y - a.y * b.x;
}

std::uint64_t cellKey(int x, int y)
{
    return (std::uint64_t(std::uint32_t(x)) << 32) | std::uint32_t(y);
}

std::size_t bucketIndex(std::uint64_t cell, std::size_t count)
{
    // Mix both signed cell coordinates before using a power-of-two table.
    cell ^= cell >> 30;
    cell *= 0xbf58476d1ce4e5b9ULL;
    cell ^= cell >> 27;
    cell *= 0x94d049bb133111ebULL;
    cell ^= cell >> 31;
    return static_cast<std::size_t>(cell) & (count - 1);
}

int cellIndex(double position)
{
    return static_cast<int>(std::floor(position / cell_size));
}

// Stable quadratic roots, including linear and tangential motion.
int roots(double a, double b, double c, double (&values)[2])
{
    if (a == 0.0)
    {
        if (b == 0.0)
            return 0;

        values[0] = -c / b;
        return 1;
    }

    double discriminant = b * b - 4.0 * a * c;
    double roundoff = 16.0 * std::numeric_limits<double>::epsilon() * (b * b + std::abs(4.0 * a * c));
    if (discriminant < -roundoff)
        return 0;

    double q = -0.5 * (b + std::copysign(std::sqrt(std::max(0.0, discriminant)), b));
    if (q == 0.0)
    {
        values[0] = -b / (2.0 * a);
        return 1;
    }

    values[0] = q / a;
    values[1] = c / q;
    if (values[1] < values[0])
        std::swap(values[0], values[1]);

    return 2;
}

struct Hit
{
    double time = std::numeric_limits<double>::infinity();
    double coordinate = 0.0;
    DVec2 normal{};
    double clearance = separation;
};

void considerHit(const ElasticNode& node, const ElasticNode& a, const ElasticNode& b, double time, double coordinate,
    DVec2 normal, double distance, double dt, Hit& hit, bool safety)
{
    DVec2 offset = node.position - (0.5 * (a.position + b.position) + coordinate * (b.position - a.position));
    DVec2 velocity = node.velocity - (0.5 * (a.velocity + b.velocity) + coordinate * (b.velocity - a.velocity));
    double gap = offset.dot(normal);

    // Safety tests use a gap larger than their tolerance: accepting overlap cannot accept crossing.
    if (safety && velocity.dot(normal) >= (distance - gap - position_tolerance) / dt)
        return;

    if (time < hit.time)
        hit = {time, coordinate, normal, distance};
}

void endpointHit(const ElasticNode& node, const ElasticNode& a, const ElasticNode& b, double coordinate,
    double clearance, double dt, Hit& hit, bool safety)
{
    const auto& endpoint = coordinate < 0.0 ? a : b;
    DVec2 offset = node.position - endpoint.position;
    DVec2 velocity = node.velocity - endpoint.velocity;
    double distance = offset.mag();

    if (distance <= clearance * (safety ? 1.0 : 1.0 + contact_margin))
    {
        if (distance > 0.0)
            considerHit(node, a, b, 0.0, coordinate, offset / distance, clearance, dt, hit, safety);
        else if (!velocity.isZero())
            considerHit(node, a, b, 0.0, coordinate, -velocity.normalized(), clearance, dt, hit, safety);

        return;
    }

    double times[2];
    int count = roots(velocity.mag2(), 2.0 * offset.dot(velocity), offset.mag2() - clearance * clearance, times);
    for (int i = 0; i < count; ++i)
    {
        double time = times[i];
        if (time < 0.0 || time > dt || time >= hit.time)
            continue;

        DVec2 difference = offset + time * velocity;

        // Very fast endpoint motion can round a tiny separation down to zero.
        DVec2 normal = difference.isZero() ? -velocity.normalized() : difference.normalized();
        considerHit(node, a, b, time, coordinate, normal, clearance, dt, hit, safety);
    }
}

Hit sweepNode(const ElasticNode& node, const ElasticNode& a, const ElasticNode& b, bool point,
    double minimum_node_distance, double dt, bool safety)
{
    Hit hit;
    if (point)
    {
        endpointHit(node, a, b, -0.5, std::max(separation, minimum_node_distance), dt, hit, safety);
        return hit;
    }

    // Centered interpolation treats both endpoint orders identically.
    DVec2 edge = b.position - a.position;
    DVec2 edge_velocity = b.velocity - a.velocity;
    DVec2 offset = node.position - 0.5 * (a.position + b.position);
    DVec2 velocity = node.velocity - 0.5 * (a.velocity + b.velocity);

    // Interior normals come from the edge; include a margin around the numerical safety gap.
    double length_squared = edge.mag2();
    if (length_squared > 0.0)
    {
        double coordinate = offset.dot(edge) / length_squared;
        if (coordinate > -0.5 && coordinate < 0.5)
        {
            double length = std::sqrt(length_squared);
            DVec2 normal{-edge.y / length, edge.x / length};
            double gap = offset.dot(normal);
            if (gap < 0.0)
                normal = -normal;

            if (std::abs(gap) <= 2.0 * separation)
                considerHit(node, a, b, 0.0, coordinate, normal, separation, dt, hit, safety);
        }
    }

    // Prefer the closest interior feature for resting contact, before checking round endcaps.
    // With node clearance, the explicit node pairs already cover both endpoints.
    if (minimum_node_distance == 0.0)
    {
        endpointHit(node, a, b, -0.5, separation, dt, hit, safety);
        endpointHit(node, a, b, 0.5, separation, dt, hit, safety);
    }

    // Collinearity is quadratic when all three nodes move linearly.
    // Persistent collinear motion is covered by endpoint contacts.
    double times[2];
    int count = roots(cross(edge_velocity, velocity), cross(edge, velocity) + cross(edge_velocity, offset),
        cross(edge, offset), times);
    // Look ahead by at most one safety gap of relative travel. A root just beyond dt can
    // otherwise round onto the edge during movement and erase which side the node occupied.
    double relative_speed = std::max((node.velocity - a.velocity).mag(), (node.velocity - b.velocity).mag());
    double horizon = relative_speed > 0.0 ? dt + separation / relative_speed : dt;
    for (int i = 0; i < count; ++i)
    {
        double time = times[i];
        if (time < 0.0 || time > horizon || std::min(time, dt) >= hit.time)
            continue;

        DVec2 moving_edge = edge + time * edge_velocity;
        double length_squared = moving_edge.mag2();
        if (length_squared == 0.0)
            continue;

        double coordinate = (offset + time * velocity).dot(moving_edge) / length_squared;
        if (coordinate < -0.5 || coordinate > 0.5)
            continue;

        DVec2 normal{-moving_edge.y, moving_edge.x};
        normal /= std::sqrt(length_squared);
        DVec2 initial_offset = offset - coordinate * edge;
        if (initial_offset.dot(normal) < 0.0)
            normal = -normal;

        considerHit(node, a, b, std::min(time, dt), coordinate, normal, separation, dt, hit, safety);
    }

    return hit;
}
}

void ElasticCollisions::Bounds::include(DVec2 point)
{
    lower.x = std::min(lower.x, point.x);
    lower.y = std::min(lower.y, point.y);
    upper.x = std::max(upper.x, point.x);
    upper.y = std::max(upper.y, point.y);
}

bool ElasticCollisions::Bounds::overlaps(const Bounds& other) const
{
    return lower.x <= other.upper.x && upper.x >= other.lower.x && lower.y <= other.upper.y && upper.y >= other.lower.y;
}

void ElasticCollisions::buildGrid(const std::vector<ElasticNode>& nodes, const std::vector<ElasticBond>& bonds)
{
    // Reuse a sparse chained table instead of sorting every swept entry each pass.
    std::size_t bucket_count = std::bit_ceil(std::max({std::size_t{64}, nodes.size() * 8, entries.size()}));
    buckets.assign(bucket_count, std::numeric_limits<std::size_t>::max());
    segments.clear();
    entries.clear();
    connected.assign(nodes.size(), false);

    // Shared drift should not inflate the swept bounds: use a translating frame.
    ElasticVectorSum frame_sum;
    std::size_t active_count = 0;
    bool has_fixed = false;
    for (const auto& node : nodes)
    {
        if (!node.active)
            continue;

        has_fixed |= node.fixed;
        frame_sum.add(node.velocity);
        ++active_count;
    }

    // Fixed supports keep the search in the world frame.
    frame_velocity = has_fixed ? DVec2{} : frame_sum.value();
    if (active_count != 0)
        frame_velocity /= double(active_count);

    auto add_segment = [&](NodeId a, NodeId b)
    {
        Bounds bounds{nodes[a].position, nodes[a].position};
        bounds.include(nodes[b].position);
        bounds.include(nodes[a].position + step_dt * (nodes[a].velocity - frame_velocity));
        bounds.include(nodes[b].position + step_dt * (nodes[b].velocity - frame_velocity));
        double padding =
            a == b ? (1.0 + contact_margin) * std::max(separation, minimum_node_distance) : 2.0 * separation;
        bounds.lower -= DVec2{padding, padding};
        bounds.upper += DVec2{padding, padding};
        std::size_t index = segments.size();
        segments.push_back({a, b, bounds});

        for (int x = cellIndex(bounds.lower.x); x <= cellIndex(bounds.upper.x); ++x)
        {
            for (int y = cellIndex(bounds.lower.y); y <= cellIndex(bounds.upper.y); ++y)
            {
                std::uint64_t key = cellKey(x, y);
                std::size_t bucket = bucketIndex(key, buckets.size());
                entries.push_back({key, index, buckets[bucket]});
                buckets[bucket] = entries.size() - 1;
            }
        }
    };

    for (const auto& bond : bonds)
    {
        connected[bond.a] = true;
        connected[bond.b] = true;
        add_segment(bond.a, bond.b);
    }

    // Node pairs use the same swept search. With no clearance, only detached points need entries.
    for (NodeId id = 0; id < nodes.size(); ++id)
    {
        if (nodes[id].active && (minimum_node_distance > 0.0 || !connected[id]))
            add_segment(id, id);
    }

    visited.assign(segments.size(), std::numeric_limits<NodeId>::max());
}

void ElasticCollisions::findContacts(const std::vector<ElasticNode>& nodes, std::vector<Contact>& result, bool safety)
{
    for (NodeId id = 0; id < nodes.size(); ++id)
    {
        const auto& node = nodes[id];
        if (!node.active)
            continue;

        Bounds bounds{node.position, node.position};
        bounds.include(node.position + step_dt * (node.velocity - frame_velocity));
        for (int x = cellIndex(bounds.lower.x); x <= cellIndex(bounds.upper.x); ++x)
        {
            for (int y = cellIndex(bounds.lower.y); y <= cellIndex(bounds.upper.y); ++y)
            {
                std::uint64_t key = cellKey(x, y);
                std::size_t first = buckets[bucketIndex(key, buckets.size())];
                for (std::size_t index = first; index != std::numeric_limits<std::size_t>::max();
                    index = entries[index].next)
                {
                    const Entry& entry = entries[index];
                    if (entry.cell != key || visited[entry.segment] == id)
                        continue;

                    visited[entry.segment] = id;
                    const auto& segment = segments[entry.segment];
                    if (id == segment.a || id == segment.b || !bounds.overlaps(segment.bounds))
                        continue;

                    if (node.fixed && nodes[segment.a].fixed && nodes[segment.b].fixed)
                        continue;

                    bool point = segment.a == segment.b;
                    if (point && (id > segment.a || (minimum_node_distance == 0.0 && connected[id])))
                        continue;

                    ++stats.candidate_tests;
                    Hit hit = sweepNode(node, nodes[segment.a], nodes[segment.b], point,
                        safety ? 0.0 : minimum_node_distance, step_dt, safety);
                    if (hit.time <= step_dt)
                    {
                        Contact contact{id, segment.a, segment.b, hit.coordinate, hit.normal, hit.clearance};
                        if (std::abs(contact.coordinate) == 0.5)
                        {
                            // The same endpoint pair can be found through several incident bonds.
                            NodeId endpoint = contact.coordinate < 0.0 ? contact.a : contact.b;
                            contact.a = endpoint;
                            contact.b = endpoint;
                            contact.coordinate = 0.0;
                            if (contact.node > endpoint)
                            {
                                contact.a = contact.node;
                                contact.b = contact.node;
                                contact.node = endpoint;
                                contact.normal = -contact.normal;
                            }
                        }
                        else if (contact.a > contact.b)
                        {
                            std::swap(contact.a, contact.b);
                            contact.coordinate = -contact.coordinate;
                        }

                        // Canonicalizing an endpoint can leave only fixed participants.
                        if (nodes[contact.node].fixed && nodes[contact.a].fixed && nodes[contact.b].fixed)
                            continue;

                        // The contact coordinate cannot affect an immobile segment's response.
                        if (nodes[contact.a].fixed && nodes[contact.b].fixed)
                            contact.coordinate = 0.0;

                        result.push_back(contact);
                    }
                }
            }
        }
    }
}

double ElasticCollisions::contactMass(const Contact& contact, const std::vector<ElasticNode>& nodes)
{
    double inverse = nodes[contact.node].fixed ? 0.0 : 1.0;
    if (contact.a == contact.b)
        inverse += nodes[contact.a].fixed ? 0.0 : 1.0;
    else
    {
        double wa = 0.5 - contact.coordinate;
        double wb = 0.5 + contact.coordinate;
        inverse += nodes[contact.a].fixed ? 0.0 : wa * wa;
        inverse += nodes[contact.b].fixed ? 0.0 : wb * wb;
    }
    return 1.0 / inverse;
}

void ElasticCollisions::applyImpulse(const Contact& contact, std::vector<ElasticNode>& nodes, double impulse)
{
    DVec2 change = impulse * contact.normal;
    if (!nodes[contact.node].fixed)
        nodes[contact.node].velocity += change;
    if (!nodes[contact.a].fixed)
        nodes[contact.a].velocity -= (0.5 - contact.coordinate) * change;
    if (!nodes[contact.b].fixed)
        nodes[contact.b].velocity -= (0.5 + contact.coordinate) * change;
}

void ElasticCollisions::prepareContacts(std::vector<ElasticNode>& nodes)
{
    auto key = [](const Contact& contact)
    {
        return std::tie(contact.node, contact.a, contact.b);
    };
    std::sort(contacts.begin(), contacts.end(), [&](const Contact& a, const Contact& b)
    {
        if (key(a) != key(b))
            return key(a) < key(b);
        return a.clearance > b.clearance;
    });
    auto end = std::unique(contacts.begin(), contacts.end(), [&](const Contact& a, const Contact& b)
    {
        return key(a) == key(b);
    });
    contacts.erase(end, contacts.end());

    // Only freshly discovered features can inherit impulses. Removed bonds cannot be resurrected.
    std::size_t previous = 0;
    for (auto& contact : contacts)
    {
        contact.mass = contactMass(contact, nodes);
        while (previous < previous_contacts.size() && key(previous_contacts[previous]) < key(contact))
            ++previous;
        if (previous == previous_contacts.size() || key(previous_contacts[previous]) != key(contact))
            continue;

        const auto& old = previous_contacts[previous];
        if (old.normal.dot(contact.normal) > 0.95 && std::abs(old.coordinate - contact.coordinate) < 0.1)
        {
            contact.impulse = old.impulse * step_dt / previous_dt;
            applyImpulse(contact, nodes, contact.impulse);
        }
    }
}

void ElasticCollisions::solveContacts(std::vector<ElasticNode>& nodes, bool use_bias)
{
    // Soft Step equations: Erin Catto, Solver2D (MIT). See LICENSES/Solver2D.txt.
    // https://github.com/erincatto/solver2d/blob/main/src/solve_tgs_soft.c
    double omega = 2.0 * std::numbers::pi * std::min(contact_hertz, 0.25 / step_dt);
    double a1 = 2.0 * damping_ratio + omega * step_dt;
    double a2 = step_dt * omega * a1;
    double bias_rate = omega / a1;
    double impulse_scale = 1.0 / (1.0 + a2);
    double mass_scale = a2 * impulse_scale;

    for (int pass = 0; pass < 2; ++pass)
    {
        for (std::size_t index = 0; index < contacts.size(); ++index)
        {
            auto& contact = contacts[(reverse_order != (pass % 2 == 1)) ? contacts.size() - 1 - index : index];
            double wa = 0.5 - contact.coordinate;
            double wb = 0.5 + contact.coordinate;
            DVec2 offset =
                nodes[contact.node].position - (wa * nodes[contact.a].position + wb * nodes[contact.b].position);
            DVec2 velocity =
                nodes[contact.node].velocity - (wa * nodes[contact.a].velocity + wb * nodes[contact.b].velocity);
            double gap = offset.dot(contact.normal) - contact.clearance;
            double speed = velocity.dot(contact.normal);
            double impulse;
            if (gap > 0.0)
                impulse = -contact.mass * (speed + gap / step_dt);
            else if (use_bias)
                impulse = -mass_scale * contact.mass * (speed + bias_rate * gap) - impulse_scale * contact.impulse;
            else
                impulse = -contact.mass * speed;

            // Clamp the total, allowing a later constraint to retract an earlier excessive push.
            double accumulated = std::max(0.0, contact.impulse + impulse);
            applyImpulse(contact, nodes, accumulated - contact.impulse);
            contact.impulse = accumulated;
        }
    }
}

bool ElasticCollisions::protectCrossings(std::vector<ElasticNode>& nodes, const std::vector<ElasticBond>& bonds)
{
    // Soft spacing is allowed to compress. Only swept crossings of the numerical barrier are repaired here.
    barriers.clear();
    for (int pass = 0; pass < 128; ++pass)
    {
        ++stats.passes;
        buildGrid(nodes, bonds);
        std::size_t previous = barriers.size();
        findContacts(nodes, barriers, true);
        if (barriers.size() == previous)
            return true;

        stats.contacts += barriers.size() - previous;

        // Keep earlier repairs active when a later response exposes another crossing.
        auto key = [](const Contact& contact)
        {
            return std::tie(contact.node, contact.a, contact.b, contact.coordinate, contact.normal.x, contact.normal.y);
        };
        std::sort(barriers.begin(), barriers.end(), [&](const Contact& a, const Contact& b)
        {
            return key(a) < key(b);
        });
        auto end = std::unique(barriers.begin(), barriers.end(), [&](const Contact& a, const Contact& b)
        {
            return key(a) == key(b);
        });
        barriers.erase(end, barriers.end());

        node_weights.assign(nodes.size(), {});
        velocity_changes.resize(nodes.size());
        contact_nodes.clear();
        auto add_weight = [&](NodeId id, double weight)
        {
            if (nodes[id].fixed || weight == 0.0)
                return;

            if (node_weights[id].value() == 0.0)
                contact_nodes.push_back(id);
            node_weights[id].add(weight);
        };
        for (const auto& contact : barriers)
        {
            add_weight(contact.node, 1.0);
            add_weight(contact.a, 0.5 - contact.coordinate);
            add_weight(contact.b, 0.5 + contact.coordinate);
        }
        for (auto& contact : barriers)
        {
            double weight = node_weights[contact.node].value() +
                            (0.5 - contact.coordinate) * node_weights[contact.a].value() +
                            (0.5 + contact.coordinate) * node_weights[contact.b].value();
            contact.mass = 1.0 / weight;
        }

        // Simultaneous corrections keep the strict barrier repair from favouring one side.
        // Shared-node weights bound the combined response; soft spacing never enters this loop.
        for (int iteration = 0; iteration < 32; ++iteration)
        {
            for (NodeId id : contact_nodes)
                velocity_changes[id] = {};

            bool resolved = true;
            for (const auto& contact : barriers)
            {
                double wa = 0.5 - contact.coordinate;
                double wb = 0.5 + contact.coordinate;
                DVec2 offset =
                    nodes[contact.node].position - (wa * nodes[contact.a].position + wb * nodes[contact.b].position);
                DVec2 velocity =
                    nodes[contact.node].velocity - (wa * nodes[contact.a].velocity + wb * nodes[contact.b].velocity);
                double limit = (contact.clearance - offset.dot(contact.normal)) / step_dt;
                double closing = std::max(0.0, limit - velocity.dot(contact.normal));
                resolved &= closing <= 0.5 * position_tolerance / step_dt;
                DVec2 impulse = (contact.mass * closing) * contact.normal;
                if (!nodes[contact.node].fixed)
                    velocity_changes[contact.node].add(impulse);
                if (!nodes[contact.a].fixed)
                    velocity_changes[contact.a].add(-wa * impulse);
                if (!nodes[contact.b].fixed)
                    velocity_changes[contact.b].add(-wb * impulse);
            }
            if (resolved)
                break;

            for (NodeId id : contact_nodes)
                nodes[id].velocity += velocity_changes[id].value();
        }
    }
    return false;
}

bool ElasticCollisions::resolve(std::vector<ElasticNode>& nodes, const std::vector<ElasticBond>& bonds, double dt)
{
    step_dt = dt;
    stats = {};
    original_velocities.resize(nodes.size());
    for (NodeId id = 0; id < nodes.size(); ++id)
        original_velocities[id] = nodes[id].velocity;

    previous_contacts.swap(contacts);
    contacts.clear();
    buildGrid(nodes, bonds);
    findContacts(nodes, contacts, false);
    prepareContacts(nodes);
    stats.contacts = contacts.size();
    solveContacts(nodes, true);
    if (protectCrossings(nodes, bonds))
    {
        previous_dt = dt;
        return true;
    }

    // Never commit an unchecked move, even when the soft response itself has a fixed work budget.
    for (NodeId id = 0; id < nodes.size(); ++id)
        nodes[id].velocity = original_velocities[id];
    clearHistory();
    return false;
}

void ElasticCollisions::relax(std::vector<ElasticNode>& nodes)
{
    // Remove recovery energy from velocities without undoing the accepted positional correction.
    solveContacts(nodes, false);
    reverse_order = !reverse_order;
}

SIM_END;
