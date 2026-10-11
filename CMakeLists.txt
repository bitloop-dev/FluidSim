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

bitloop_new_project(FluidSim ${SIM_SOURCES})

bitloop_finalize()

