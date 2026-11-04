#pragma once
#include "ElasticTypes.h"
#include <cmath>

SIM_BEG;

// Retain small contributions when many equal/opposite responses meet at a node.
struct ElasticSum
{
    double sum = 0.0;
    double correction = 0.0;

    void add(double value)
    {
        double next = sum + value;
        correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
        sum = next;
    }

    double value() const
    {
        return sum + correction;
    }
};

struct ElasticVectorSum
{
    ElasticSum x;
    ElasticSum y;

    void add(bl::DVec2 value)
    {
        x.add(value.x);
        y.add(value.y);
    }

    bl::DVec2 value() const
    {
        return {x.value(), y.value()};
    }
};

SIM_END;
