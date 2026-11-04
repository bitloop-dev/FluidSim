cmake_minimum_required(VERSION 3.21)
project(ElasticEngine NONE)

include("${CMAKE_CURRENT_LIST_DIR}/tooling/findBitloop.cmake")

bitloop_enable_languages(LANGUAGES C CXX)


set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/ElasticEngine")
file(GLOB_RECURSE SIM_SOURCES CONFIGURE_DEPENDS
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.h"
    "${SRC_DIR}/*.hpp"
)

bitloop_new_project(ElasticEngine ${SIM_SOURCES})

bitloop_finalize()

if(NOT BITLOOP_DISCOVERY)
    option(ELASTICENGINE_BUILD_TESTS "Build ElasticEngine regression tests" OFF)
    if(ELASTICENGINE_BUILD_TESTS AND NOT EMSCRIPTEN)
        enable_testing()
        add_executable(ElasticEngineTests
            tests/ElasticEngineTests.cpp
            tests/ElasticCollisionTests.cpp
            tests/ElasticSymmetryTests.cpp
            tests/ElasticFractureTests.cpp
            tests/ElasticCupTests.cpp
            tests/ElasticNodeContactTests.cpp
            ElasticEngine/ElasticCollisions.cpp
            ElasticEngine/ElasticSprings.cpp
            ElasticEngine/ElasticScenarios.cpp
            ElasticEngine/ElasticEngine.cpp)
        target_compile_definitions(ElasticEngineTests PRIVATE SIM_NAME=ElasticEngine)
        target_include_directories(ElasticEngineTests PRIVATE "${SRC_DIR}")
        target_compile_features(ElasticEngineTests PRIVATE cxx_std_23)
        target_link_libraries(ElasticEngineTests PRIVATE bitloop::bitloop)
        add_test(NAME ElasticEngineTests COMMAND ElasticEngineTests)
    endif()
endif()

