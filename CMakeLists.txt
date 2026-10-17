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
list(REMOVE_ITEM SIM_SOURCES "${SRC_DIR}/FluidGrid.cpp")

bitloop_new_project(FluidSim ${SIM_SOURCES})

bitloop_finalize()

if(NOT BITLOOP_DISCOVERY)
    add_library(FluidSimEngine STATIC "${SRC_DIR}/FluidGrid.cpp")
    target_include_directories(FluidSimEngine PUBLIC "${SRC_DIR}")
    target_compile_features(FluidSimEngine PUBLIC cxx_std_23)
    target_link_libraries(FluidSimEngine PUBLIC bitloop::bitloop)
    target_link_libraries(FluidSim PRIVATE FluidSimEngine)

    option(FLUIDSIM_BUILD_TESTS "Build the fluid engine regression tests" OFF)
    if(FLUIDSIM_BUILD_TESTS)
        enable_testing()
        add_executable(FluidSimTests tests/FluidGridTests.cpp)
        target_link_libraries(FluidSimTests PRIVATE FluidSimEngine)
        add_test(NAME FluidGridTests COMMAND FluidSimTests)
    endif()
endif()

