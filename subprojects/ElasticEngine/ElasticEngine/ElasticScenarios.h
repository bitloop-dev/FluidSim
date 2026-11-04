#pragma once
#include <bitloop.h>

SIM_BEG;

class ElasticEngine;

enum class ElasticScenario
{
    TwoBalls,
    CupDrop
};

void setupTwoBalls(ElasticEngine& engine, double breaking_strain);
void setupCupDrop(ElasticEngine& engine, double stiffness, double breaking_strain);

SIM_END;
