cmake_minimum_required(VERSION 3.25)
project(FluidCpu2D LANGUAGES CXX)

# The complete CPU engine builds without the framework or any AMReX package/source.
add_library(FluidCpu2D STATIC
    ../../FluidSim/AmrGas.cpp ../../FluidSim/AmrField2D.cpp
    ../../FluidSim/AmrTransfer2D.cpp ../../FluidSim/AmrPressure2D.cpp
    ../../FluidSim/AmrGasRuntime.cpp)
target_include_directories(FluidCpu2D PUBLIC ../../FluidSim)
target_compile_features(FluidCpu2D PUBLIC cxx_std_20)
if(MSVC)
    set(OpenMP_RUNTIME_MSVC llvm)
    set(OpenMP_CXX_FLAGS "/openmp:llvm" CACHE STRING "" FORCE)
endif()
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(FluidCpu2D PUBLIC OpenMP::OpenMP_CXX)
endif()
