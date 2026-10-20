cmake_minimum_required(VERSION 3.21)
project(FluidSim NONE)

include("${CMAKE_CURRENT_LIST_DIR}/tooling/findBitloop.cmake")

bitloop_enable_languages(LANGUAGES C CXX)


set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/FluidSim")

file(GLOB_RECURSE SIM_SOURCES CONFIGURE_DEPENDS
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.h"
    "${SRC_DIR}/*.hpp"
)
list(REMOVE_ITEM SIM_SOURCES 
    "${SRC_DIR}/FluidGrid.cpp" 
    "${SRC_DIR}/PressureMultigrid.cpp"
    "${SRC_DIR}/FluidGridGPU.cpp"
    "${SRC_DIR}/FluidGridHybrid.cpp"
    "${SRC_DIR}/PressureMultigridGPU.cpp")

bitloop_new_project(FluidSim ${SIM_SOURCES})

bitloop_finalize()

if(NOT BITLOOP_DISCOVERY)
    add_library(FluidSimEngine STATIC "${SRC_DIR}/FluidGrid.cpp" "${SRC_DIR}/PressureMultigrid.cpp")
    target_include_directories(FluidSimEngine PUBLIC "${SRC_DIR}")
    target_compile_features(FluidSimEngine PUBLIC cxx_std_23)
    target_link_libraries(FluidSimEngine PUBLIC bitloop::bitloop)
    target_link_libraries(FluidSim PRIVATE FluidSimEngine)

    add_library(FluidSimEngineGPU STATIC "${SRC_DIR}/FluidGridGPU.cpp" "${SRC_DIR}/PressureMultigridGPU.cpp")
    target_include_directories(FluidSimEngineGPU PUBLIC "${SRC_DIR}")
    target_compile_features(FluidSimEngineGPU PUBLIC cxx_std_23)
    target_link_libraries(FluidSimEngineGPU PUBLIC bitloop::bitloop)
    target_link_libraries(FluidSim PRIVATE FluidSimEngineGPU)

    target_link_libraries(FluidSimEngineGPU PRIVATE FluidSimEngine)
    add_library(FluidSimEngineHybrid STATIC "${SRC_DIR}/FluidGridHybrid.cpp")
    target_link_libraries(FluidSimEngineHybrid PUBLIC FluidSimEngineGPU FluidSimEngine)
    target_link_libraries(FluidSim PRIVATE FluidSimEngineHybrid)

    option(FLUIDSIM_BUILD_TESTS "Build the fluid engine regression tests" OFF)
    if(FLUIDSIM_BUILD_TESTS)
        enable_testing()
        add_executable(FluidSimTests tests/FluidGridTests.cpp)
        target_link_libraries(FluidSimTests PRIVATE FluidSimEngine)
        add_test(NAME FluidGridTests COMMAND FluidSimTests)
        if(NOT EMSCRIPTEN)
            add_executable(FluidSimHybridTests tests/FluidGridHybridTests.cpp "${SRC_DIR}/FluidGridDisplay.cpp")
            target_link_libraries(FluidSimHybridTests PRIVATE FluidSimEngineHybrid)
            add_test(NAME FluidGridHybridTests COMMAND FluidSimHybridTests)
            add_executable(FluidSimGPUTests tests/FluidGridGPUTests.cpp "${SRC_DIR}/FluidGridDisplay.cpp")
            target_link_libraries(FluidSimGPUTests PRIVATE FluidSimEngineHybrid)
            add_test(NAME FluidGridGPUTests COMMAND FluidSimGPUTests)
        endif()
    endif()
endif()

