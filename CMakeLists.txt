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
    "${SRC_DIR}/AmrPressure2D.cpp"
    "${SRC_DIR}/AmrGas.cpp"
    "${SRC_DIR}/AmrField2D.cpp"
    "${SRC_DIR}/AmrTransfer2D.cpp"
    "${SRC_DIR}/AmrGasRuntime.cpp"
    "${SRC_DIR}/FluidCoarseningPreview.cpp"
    "${SRC_DIR}/FluidAmbientMask.cpp"
    "${SRC_DIR}/FluidGrid.cpp" 
    "${SRC_DIR}/PressureMultigrid.cpp"
    "${SRC_DIR}/FluidGridGPU.cpp"
    "${SRC_DIR}/PressureMultigridGPU.cpp")

bitloop_new_project(FluidSim ${SIM_SOURCES})

bitloop_finalize()

if(NOT BITLOOP_DISCOVERY)
    add_library(FluidPressure2D STATIC "${SRC_DIR}/AmrPressure2D.cpp")
    target_include_directories(FluidPressure2D PUBLIC "${SRC_DIR}")
    target_compile_features(FluidPressure2D PUBLIC cxx_std_20)
    add_library(FluidSimEngine STATIC "${SRC_DIR}/FluidGrid.cpp" "${SRC_DIR}/PressureMultigrid.cpp" "${SRC_DIR}/FluidAmbientMask.cpp" "${SRC_DIR}/FluidCoarseningPreview.cpp")
    target_include_directories(FluidSimEngine PUBLIC "${SRC_DIR}")
    target_compile_features(FluidSimEngine PUBLIC cxx_std_23)
    target_link_libraries(FluidSimEngine PUBLIC bitloop::bitloop)
    if(NOT EMSCRIPTEN)
        # Keep the existing native worker implementation.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(OpenMP_RUNTIME_MSVC llvm)
            set(OpenMP_CXX_FLAGS "/openmp:llvm" CACHE STRING "" FORCE)
        endif()
        find_package(OpenMP REQUIRED)
        target_link_libraries(FluidPressure2D PUBLIC OpenMP::OpenMP_CXX)
        add_library(FluidAmrGas STATIC "${SRC_DIR}/AmrGas.cpp" "${SRC_DIR}/AmrGasRuntime.cpp" "${SRC_DIR}/AmrField2D.cpp" "${SRC_DIR}/AmrTransfer2D.cpp")
        target_include_directories(FluidAmrGas PUBLIC "${SRC_DIR}")
        target_compile_features(FluidAmrGas PUBLIC cxx_std_23)
        target_link_libraries(FluidAmrGas PUBLIC FluidPressure2D)
        target_compile_definitions(FluidAmrGas PUBLIC FLUIDSIM_HAS_AMR=1)
        target_link_libraries(FluidSimEngine PUBLIC FluidAmrGas)
    endif()
    target_link_libraries(FluidSim PRIVATE FluidSimEngine)

    add_library(FluidSimEngineGPU STATIC "${SRC_DIR}/FluidGridGPU.cpp" "${SRC_DIR}/PressureMultigridGPU.cpp")
    target_include_directories(FluidSimEngineGPU PUBLIC "${SRC_DIR}")
    target_compile_features(FluidSimEngineGPU PUBLIC cxx_std_23)
    target_link_libraries(FluidSimEngineGPU PUBLIC bitloop::bitloop)
    target_link_libraries(FluidSim PRIVATE FluidSimEngineGPU)

    target_link_libraries(FluidSimEngineGPU PRIVATE FluidSimEngine)

    option(FLUIDSIM_BUILD_TESTS "Build the fluid engine regression tests" OFF)
    if(FLUIDSIM_BUILD_TESTS)
        enable_testing()
        add_executable(FluidSimTests tests/FluidGridTests.cpp)
        target_link_libraries(FluidSimTests PRIVATE FluidSimEngine)
        add_test(NAME FluidGridTests COMMAND FluidSimTests)
        add_executable(FluidCoarseningPreviewTests tests/FluidCoarseningPreviewTests.cpp)
        target_link_libraries(FluidCoarseningPreviewTests PRIVATE FluidSimEngine)
        add_test(NAME FluidCoarseningPreviewTests COMMAND FluidCoarseningPreviewTests)
        if(NOT EMSCRIPTEN)
            add_executable(FluidRefinementTests tests/FluidRefinementTests.cpp)
            target_link_libraries(FluidRefinementTests PRIVATE FluidSimEngine)
            add_test(NAME FluidRefinementTests COMMAND FluidRefinementTests)
            set_tests_properties(FluidRefinementTests PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=8;OMP_DYNAMIC=FALSE")
        endif()
        add_executable(FluidAmbientTests tests/FluidAmbientTests.cpp)
        target_link_libraries(FluidAmbientTests PRIVATE FluidSimEngine)
        add_test(NAME FluidAmbientTests COMMAND FluidAmbientTests)
        add_executable(FluidAmbientMaskTests tests/FluidAmbientMaskTests.cpp)
        target_link_libraries(FluidAmbientMaskTests PRIVATE FluidSimEngine)
        add_test(NAME FluidAmbientMaskTests COMMAND FluidAmbientMaskTests)
        if(NOT EMSCRIPTEN)
            add_executable(FluidSimGPUTests tests/FluidGridGPUTests.cpp tests/FluidGridGPUCoarseTests.cpp
                tests/FluidAmbientMaskDisplayTests.cpp "${SRC_DIR}/FluidGridDisplay.cpp" "${SRC_DIR}/FluidAmbientMaskDisplay.cpp")
            target_link_libraries(FluidSimGPUTests PRIVATE FluidSimEngineGPU FluidSimEngine)
            add_test(NAME FluidGridGPUTests COMMAND FluidSimGPUTests)
            add_test(NAME FluidAmbientMaskDisplayTests COMMAND FluidSimGPUTests --mask-display-only)
        endif()
    endif()
endif()

