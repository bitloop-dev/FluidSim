#pragma once
#include <bitloop.h>

SIM_BEG;

struct FluidSim_Project : bl::Project<FluidSim_Project>
{
    static bl::ProjectInfo info()
    {
        return bl::ProjectInfo({ "New Projects", "FluidSim" });
    }

    struct UI : BufferedInterfaceModel
    {
        using BufferedInterfaceModel::BufferedInterfaceModel;
        void sidebar() override;
    };

    int viewport_count = 1;
    int scenario = 0; // SplashCubes; CrashingWave remains selectable.

    void projectPrepare(bl::Layout& layout) override;
};

SIM_END;
