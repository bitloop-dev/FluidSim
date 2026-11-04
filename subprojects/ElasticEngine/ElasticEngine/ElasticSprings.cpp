#include "ElasticEngine.h"
#include <cmath>
#include <stdexcept>

SIM_BEG;

void ElasticEngine::advance(double dt, DVec2 acceleration)
{
    if (dt == 0.0)
        return;

    // Bound the explicit spring frequency by the incident axial stiffness per unit test inertia.
    damping_weights.assign(nodes.size(), 0.0);
    for (const auto& bond : bonds)
    {
        if (bond.boundary)
            continue;

        double rate = bond.stiffness / bond.rest_length;
        if (!nodes[bond.a].fixed)
            damping_weights[bond.a] += rate;
        if (!nodes[bond.b].fixed)
            damping_weights[bond.b] += rate;
    }

    double maximum_rate = 0.0;
    for (double rate : damping_weights)
        maximum_rate = std::max(maximum_rate, rate);

    int substeps = std::max(1, static_cast<int>(std::ceil(dt * std::sqrt(maximum_rate))));
    double step_dt = dt / substeps;
    for (int step = 0; step < substeps; ++step)
        this->step(step_dt, acceleration);
}

void ElasticEngine::step(double dt, DVec2 acceleration)
{
    if (dt == 0.0)
        return;

    // Zero stiffness releases material structure before it can apply contact or damping.
    std::erase_if(bonds, [](const ElasticBond& bond)
    {
        return !bond.boundary && bond.stiffness == 0.0;
    });

    step_velocities.resize(nodes.size());
    for (NodeId id = 0; id < nodes.size(); ++id)
        step_velocities[id] = nodes[id].velocity;

    for (auto& node : nodes)
    {
        if (!node.active)
            continue;

        if (node.fixed)
            node.velocity = {};
        else
            node.velocity += dt * acceleration;
    }

    if (!collisions.resolve(nodes, bonds, dt))
    {
        // Contact failure must leave the last accepted substep intact, including gravity.
        for (NodeId id = 0; id < nodes.size(); ++id)
            nodes[id].velocity = step_velocities[id];

        throw std::runtime_error("Contact solver did not converge.");
    }

    for (auto& node : nodes)
    {
        if (node.active && !node.fixed)
            node.position += dt * node.velocity;
    }

    applySprings(dt);
}

void ElasticEngine::applySprings(double dt)
{
    bond_directions.assign(bonds.size(), {});
    spring_impulses.assign(nodes.size(), {});
    damping_weights.assign(nodes.size(), 0.0);
    velocity_changes.resize(nodes.size());

    // Break and compact bonds using the same geometry, before committing any velocity response.
    std::size_t retained = 0;
    for (std::size_t index = 0; index < bonds.size(); ++index)
    {
        const auto& bond = bonds[index];
        if (bond.boundary)
        {
            bonds[retained++] = bond;
            continue;
        }

        DVec2 difference = nodes[bond.b].position - nodes[bond.a].position;
        double distance = difference.mag();
        double strain = (distance - bond.rest_length) / bond.rest_length;
        if (strain > bond.breaking_strain)
            continue;

        std::size_t slot = retained++;
        if (slot != index)
            bonds[slot] = bond;

        // A collapsed bond has no defined axial direction.
        if (distance == 0.0)
            continue;

        DVec2 direction = difference / distance;
        bond_directions[slot] = direction;
        double force = bond.stiffness * strain;
        DVec2 impulse = (0.5 * dt * force) * direction;
        spring_impulses[bond.a].add(impulse);
        spring_impulses[bond.b].add(-impulse);
        if (!nodes[bond.a].fixed)
            damping_weights[bond.a] += bond.damping;
        if (!nodes[bond.b].fixed)
            damping_weights[bond.b] += bond.damping;
    }
    bonds.resize(retained);
    bond_directions.resize(retained);

    // Nonnegative axial damping has maximum rate <= max(sum of incident damping).
    // Velocity-only substeps bound every damping mode to [0, 1], even at high degree.
    double maximum_rate = 0.0;
    for (double weight : damping_weights)
        maximum_rate = std::max(maximum_rate, weight);

    int substeps = static_cast<int>(std::ceil(dt * maximum_rate));
    if (substeps > 0)
    {
        double scale = 0.5 * dt / substeps;
        for (int step = 0; step < substeps; ++step)
        {
            std::fill(velocity_changes.begin(), velocity_changes.end(), ElasticVectorSum{});
            for (std::size_t index = 0; index < bonds.size(); ++index)
            {
                const auto& bond = bonds[index];
                if (bond.boundary)
                    continue;

                DVec2 direction = bond_directions[index];
                double speed = (nodes[bond.b].velocity - nodes[bond.a].velocity).dot(direction);
                DVec2 impulse = (scale * bond.damping * speed) * direction;
                velocity_changes[bond.a].add(impulse);
                velocity_changes[bond.b].add(-impulse);
            }

            for (NodeId id = 0; id < nodes.size(); ++id)
            {
                if (nodes[id].active && !nodes[id].fixed)
                    nodes[id].velocity += velocity_changes[id].value();
            }
        }
    }

    // Keep the elastic kick separate from the damping solve.
    for (NodeId id = 0; id < nodes.size(); ++id)
    {
        if (nodes[id].active && !nodes[id].fixed)
            nodes[id].velocity += spring_impulses[id].value();
    }
}

SIM_END;
