#include <emscripten.h>

int fluidWebGpuTestMain(int argc, char** argv);

int main(int argc, char** argv)
{
    const int result = fluidWebGpuTestMain(argc, argv);
    // Finished tests must exit even while the browser resolves destroyed devices' loss promises.
    emscripten_force_exit(result);
}
