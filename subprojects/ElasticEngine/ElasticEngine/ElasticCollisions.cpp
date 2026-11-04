#include "ElasticCollisions.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
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
constexpr double node_tolerance_fraction = 0.00025;
constexpr double overlap_recovery = 0.2;

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
    double speed_limit = 0.0;
    double speed_tolerance = 0.0;
};

void considerHit(const ElasticNode& node, const ElasticNode& a, const ElasticNode& b, double time, double coordinate,
    DVec2 normal, double distance, double dt, Hit& hit, bool node_clearance = false)
{
    DVec2 offset = node.position - (0.5 * (a.position + b.position) + coordinate * (b.position - a.position));
    DVec2 velocity = node.velocity - (0.5 * (a.velocity + b.velocity) + coordinate * (b.velocity - a.velocity));
    double gap = offset.dot(normal);

    // Nodes recover overlap gradually; membrane recovery targets its safety gap in one step.
    double recovery = node_clearance ? overlap_recovery : 1.0;
    double speed_limit = (recovery * std::max(0.0, distance - gap) - std::max(0.0, gap - distance)) / dt;
    double tolerance = node_clearance ? node_tolerance_fraction * distance : position_tolerance;

    // Separating and resting contacts must not pull nodes together.
    if (velocity.dot(normal) >= speed_limit - tolerance / dt)
        return;

    if (time < hit.time)
        hit = {time, coordinate, normal, speed_limit, tolerance / dt};
}

void endpointHit(const ElasticNode& node, const ElasticNode& a, const ElasticNode& b, double coordinate,
    double clearance, double dt, Hit& hit, bool node_clearance = false)
{
    const auto& endpoint = coordinate < 0.0 ? a : b;
    DVec2 offset = node.position - endpoint.position;
    DVec2 velocity = node.velocity - endpoint.velocity;
    double distance = offset.mag();

    if (distance <= clearance)
    {
        if (distance > 0.0)
            considerHit(node, a, b, 0.0, coordinate, offset / distance, clearance, dt, hit, node_clearance);
        else if (!velocity.isZero())
            considerHit(node, a, b, 0.0, coordinate, -velocity.normalized(), clearance, dt, hit, node_clearance);

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
        considerHit(node, a, b, time, coordinate, normal, clearance, dt, hit, node_clearance);
    }
}

Hit sweepNode(const ElasticNode& node, const ElasticNode& a, const ElasticNode& b, bool point,
    double minimum_node_distance, double dt)
{
    Hit hit;
    if (point)
    {
        endpointHit(
            node, a, b, -0.5, std::max(separation, minimum_node_distance), dt, hit, minimum_node_distance > 0.0);
        return hit;
    }

    // Centered interpolation treats both endpoint orders identically.
    DVec2 edge = b.position - a.position;
    DVec2 edge_velocity = b.velocity - a.velocity;
    DVec2 offset = node.position - 0.5 * (a.position + b.position);
    DVec2 velocity = node.velocity - 0.5 * (a.velocity + b.velocity);

    // Interior contact normals come from the edge, avoiding cancellation near its endpoints.
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

            if (std::abs(gap) <= separation)
                considerHit(node, a, b, 0.0, coordinate, normal, separation, dt, hit);
        }
    }

    // Prefer the closest interior feature for resting contact, before checking round endcaps.
    // With node clearance, the explicit node pairs already cover both endpoints.
    if (minimum_node_distance == 0.0)
    {
        endpointHit(node, a, b, -0.5, separation, dt, hit);
        endpointHit(node, a, b, 0.5, separation, dt, hit);
    }

    // Collinearity is quadratic when all three nodes move linearly.
    // Persistent collinear motion is covered by endpoint contacts.
    double times[2];
    int count = roots(cross(edge_velocity, velocity), cross(edge, velocity) + cross(edge_velocity, offset),
        cross(edge, offset), times);
    for (int i = 0; i < count; ++i)
    {
        double time = times[i];
        if (time < 0.0 || time > dt || time >= hit.time)
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

        considerHit(node, a, b, time, coordinate, normal, separation, dt, hit);
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

    // Stationary supports exchange momentum with the world: bound contact energy in that frame.
    frame_velocity = has_fixed ? DVec2{} : frame_sum.value();
    if (active_count != 0)
        frame_velocity /= double(active_count);

    auto add_segment = [&](NodeId a, NodeId b)
    {
        Bounds bounds{nodes[a].position, nodes[a].position};
        bounds.include(nodes[b].position);
        bounds.include(nodes[a].position + step_dt * (nodes[a].velocity - frame_velocity));
        bounds.include(nodes[b].position + step_dt * (nodes[b].velocity - frame_velocity));
        double padding = a == b ? std::max(separation, minimum_node_distance) : separation;
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

void ElasticCollisions::findContacts(const std::vector<ElasticNode>& nodes)
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
                    Hit hit =
                        sweepNode(node, nodes[segment.a], nodes[segment.b], point, minimum_node_distance, step_dt);
                    if (hit.time <= step_dt)
                    {
                        Contact contact{
                            id, segment.a, segment.b, hit.coordinate, hit.normal, hit.speed_limit, hit.speed_tolerance};
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

                        // The contact coordinate cannot affect an immobile segment's response.
                        if (nodes[contact.a].fixed && nodes[contact.b].fixed)
                            contact.coordinate = 0.0;

                        contacts.push_back(contact);
                    }
                }
            }
        }
    }
}

void ElasticCollisions::resolveContacts(std::vector<ElasticNode>& nodes)
{
    // Equivalent constraints have one response, regardless of the endpoint's bond count.
    auto key = [](const Contact& contact)
    {
        return std::tie(contact.node, contact.a, contact.b, contact.coordinate, contact.normal.x, contact.normal.y);
    };
    std::sort(contacts.begin(), contacts.end(), [&](const Contact& a, const Contact& b)
    {
        return key(a) < key(b);
    });
    std::size_t count = 0;
    for (const auto& contact : contacts)
    {
        if (count > 0 && key(contacts[count - 1]) == key(contact))
        {
            contacts[count - 1].speed_limit = std::max(contacts[count - 1].speed_limit, contact.speed_limit);
            contacts[count - 1].speed_tolerance =
                std::min(contacts[count - 1].speed_tolerance, contact.speed_tolerance);
        }
        else
            contacts[count++] = contact;
    }
    contacts.resize(count);

    node_weights.assign(nodes.size(), {});
    velocity_changes.assign(nodes.size(), {});
    contact_nodes.clear();
    auto add_weight = [&](NodeId id, double weight)
    {
        if (weight == 0.0 || nodes[id].fixed)
            return;

        if (node_weights[id].value() == 0.0)
            contact_nodes.push_back(id);

        node_weights[id].add(weight);
    };
    for (const auto& contact : contacts)
    {
        add_weight(contact.node, 1.0);
        add_weight(contact.a, 0.5 - contact.coordinate);
        add_weight(contact.b, 0.5 + contact.coordinate);
    }

    // Bound each row of the coupled contact matrix using its shared-node weights.
    for (auto& contact : contacts)
    {
        double wa = 0.5 - contact.coordinate;
        double wb = 0.5 + contact.coordinate;
        double weight = node_weights[contact.node].value() +
                        (wa * node_weights[contact.a].value() + wb * node_weights[contact.b].value());
        contact.response_scale = 1.0 / weight;
    }

    for (int iteration = 0; iteration < 256; ++iteration)
    {
        for (NodeId id : contact_nodes)
            velocity_changes[id] = {};

        bool resolved = true;
        ElasticSum initial_error;
        for (auto& contact : contacts)
        {
            double wa = 0.5 - contact.coordinate;
            double wb = 0.5 + contact.coordinate;
            DVec2 velocity =
                nodes[contact.node].velocity - (wa * nodes[contact.a].velocity + wb * nodes[contact.b].velocity);
            contact.closing = contact.speed_limit - velocity.dot(contact.normal);
            double closing = std::max(0.0, contact.closing);
            resolved &= closing <= 0.5 * contact.speed_tolerance;
            initial_error.add(closing * closing * contact.response_scale);
            DVec2 impulse = (closing * contact.response_scale) * contact.normal;
            if (!nodes[contact.node].fixed)
                velocity_changes[contact.node].add(impulse);
            if (!nodes[contact.a].fixed)
                velocity_changes[contact.a].add(-wa * impulse);
            if (!nodes[contact.b].fixed)
                velocity_changes[contact.b].add(-wb * impulse);
        }
        if (resolved)
            break;

        ElasticSum slope;
        ElasticSum energy_slope;
        for (NodeId id : contact_nodes)
        {
            DVec2 change = velocity_changes[id].value();
            slope.add(change.mag2());
            energy_slope.add(-(nodes[id].velocity - frame_velocity).dot(change));
        }

        // Opposing corrections can cancel at numerical precision; the outer sweep still checks safety.
        if (slope.value() == 0.0)
            break;

        ElasticSum curvature;
        ElasticSum recovery_work;
        for (auto& contact : contacts)
        {
            double wa = 0.5 - contact.coordinate;
            double wb = 0.5 + contact.coordinate;
            DVec2 change = velocity_changes[contact.node].value() -
                           (wa * velocity_changes[contact.a].value() + wb * velocity_changes[contact.b].value());
            contact.speed_change = change.dot(contact.normal);
            if (contact.closing > 0.0)
            {
                curvature.add(contact.speed_change * contact.speed_change * contact.response_scale);
                recovery_work.add(std::max(0.0, contact.speed_limit) * contact.closing * contact.response_scale);
            }
        }

        // Take a longer simultaneous step when the conservative row bound is too small.
        // Recovery may do separation work; ordinary collision response remains dissipative.
        double scale = std::max(1.0, std::min(slope.value() / curvature.value(),
                                         (energy_slope.value() + recovery_work.value()) / slope.value()));
        while (scale > 1.0)
        {
            ElasticSum error;
            for (const auto& contact : contacts)
            {
                double closing = std::max(0.0, contact.closing - scale * contact.speed_change);
                error.add(closing * closing * contact.response_scale);
            }
            if (error.value() <= initial_error.value() - scale * slope.value())
                break;

            scale = std::max(1.0, scale * 0.5);
        }
        for (NodeId id : contact_nodes)
            nodes[id].velocity += scale * velocity_changes[id].value();
    }
}

bool ElasticCollisions::resolve(std::vector<ElasticNode>& nodes, const std::vector<ElasticBond>& bonds, double dt)
{
    step_dt = dt;
    stats = {};
    contacts.clear();
    for (int pass = 0; pass < 64; ++pass)
    {
        ++stats.passes;
        buildGrid(nodes, bonds);
        std::size_t previous_contacts = contacts.size();
        findContacts(nodes);
        if (contacts.size() == previous_contacts)
            return true;

        if (pass == 0)
        {
            original_velocities.resize(nodes.size());
            for (NodeId id = 0; id < nodes.size(); ++id)
                original_velocities[id] = nodes[id].velocity;
        }

        stats.contacts += contacts.size() - previous_contacts;

        resolveContacts(nodes);
    }

    // Never commit an unchecked trajectory after the contact budget is exhausted.
    for (NodeId id = 0; id < nodes.size(); ++id)
        nodes[id].velocity = original_velocities[id];

    return false;
}

SIM_END;
