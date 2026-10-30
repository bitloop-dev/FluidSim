cmake_minimum_required(VERSION 3.21)
project(FluidSim NONE)

# FluidSim uses compute only; Bitloop's OpenGL renderer remains independent.
set(BITLOOP_WEBGPU_ASYNC_WAIT ON CACHE BOOL "Allow cooperative WebGPU waits in browser builds")
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
    "${SRC_DIR}/AmrPressureTopology2D.cpp"
    "${SRC_DIR}/AmrPressureGPU.cpp"
    "${SRC_DIR}/FluidGridAdaptiveGPU.cpp"
    "${SRC_DIR}/FluidCoarseningGPU.cpp"
    "${SRC_DIR}/FluidGridAdaptive.cpp"
    "${SRC_DIR}/AmrField2D.cpp"
    "${SRC_DIR}/AmrTransfer2D.cpp"
    "${SRC_DIR}/AmrGasRuntime.cpp"
    "${SRC_DIR}/FluidCoarseningPreview.cpp"
    "${SRC_DIR}/FluidAmbientMask.cpp"
    "${SRC_DIR}/FluidGrid.cpp" 
    "${SRC_DIR}/PressureMultigrid.cpp"
    "${SRC_DIR}/FluidGridWebGL.cpp"
    "${SRC_DIR}/PressureMultigridWebGL.cpp"
    "${SRC_DIR}/FluidGridGPU.cpp"
    "${SRC_DIR}/PressureMultigridGPU.cpp")

bitloop_new_project(FluidSim ${SIM_SOURCES})

bitloop_finalize()

if(NOT BITLOOP_DISCOVERY)
    add_library(FluidPressure2D STATIC "${SRC_DIR}/AmrPressure2D.cpp" "${SRC_DIR}/AmrPressureTopology2D.cpp")
    target_include_directories(FluidPressure2D PUBLIC "${SRC_DIR}")
    target_compile_features(FluidPressure2D PUBLIC cxx_std_20)
    add_library(FluidSimEngine STATIC "${SRC_DIR}/FluidGrid.cpp" "${SRC_DIR}/PressureMultigrid.cpp" "${SRC_DIR}/FluidAmbientMask.cpp" "${SRC_DIR}/FluidCoarseningPreview.cpp")
    target_include_directories(FluidSimEngine PUBLIC "${SRC_DIR}")
    target_compile_features(FluidSimEngine PUBLIC cxx_std_23)
    target_link_libraries(FluidSimEngine PUBLIC bitloop::bitloop)
    if(EMSCRIPTEN)
        # Device/solver failures must reach the scene's existing error handler in browsers.
        target_compile_options(FluidPressure2D PUBLIC -pthread -fexceptions)
        target_compile_options(FluidSimEngine PUBLIC -pthread -fexceptions)
        target_link_options(FluidSimEngine PUBLIC -fexceptions)
    endif()
    if(NOT EMSCRIPTEN)
        # Keep the existing native worker implementation.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(OpenMP_RUNTIME_MSVC llvm)
            set(OpenMP_CXX_FLAGS "/openmp:llvm" CACHE STRING "" FORCE)
        endif()
        find_package(OpenMP REQUIRED)
        target_link_libraries(FluidPressure2D PUBLIC OpenMP::OpenMP_CXX)
    endif()
    add_library(FluidAmrGas STATIC "${SRC_DIR}/FluidGridAdaptive.cpp" "${SRC_DIR}/AmrGasRuntime.cpp" "${SRC_DIR}/AmrField2D.cpp" "${SRC_DIR}/AmrTransfer2D.cpp")
    target_include_directories(FluidAmrGas PUBLIC "${SRC_DIR}")
    target_compile_features(FluidAmrGas PUBLIC cxx_std_23)
    target_link_libraries(FluidAmrGas PUBLIC FluidPressure2D)
    target_compile_definitions(FluidAmrGas PUBLIC FLUIDSIM_HAS_AMR=1)
    target_link_libraries(FluidSimEngine PUBLIC FluidAmrGas)
    target_link_libraries(FluidSim PRIVATE FluidSimEngine)

    add_library(FluidSimEngineGPU STATIC "${SRC_DIR}/FluidGridWebGL.cpp" "${SRC_DIR}/PressureMultigridWebGL.cpp" "${SRC_DIR}/FluidGridGPU.cpp" "${SRC_DIR}/PressureMultigridGPU.cpp" "${SRC_DIR}/AmrPressureGPU.cpp" "${SRC_DIR}/FluidGridAdaptiveGPU.cpp" "${SRC_DIR}/FluidCoarseningGPU.cpp")
    target_include_directories(FluidSimEngineGPU PUBLIC "${SRC_DIR}")
    target_compile_features(FluidSimEngineGPU PUBLIC cxx_std_23)
    target_link_libraries(FluidSimEngineGPU PUBLIC bitloop::bitloop bitloop::webgpu FluidSimEngine)
    target_link_libraries(FluidSim PRIVATE FluidSimEngineGPU)
    if(PROJECT_IS_TOP_LEVEL)
        bitloop_deploy_webgpu_runtime(FluidSim)
    endif()

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
        add_executable(FluidSimGPUTests tests/FluidGridWebGPUTests.cpp)
        target_link_libraries(FluidSimGPUTests PRIVATE FluidSimEngineGPU FluidSimEngine)
        bitloop_deploy_webgpu_runtime(FluidSimGPUTests)
        add_executable(FluidPressureWebGPUTests tests/AmrPressureWebGPUTests.cpp)
        target_link_libraries(FluidPressureWebGPUTests PRIVATE FluidSimEngineGPU FluidPressure2D)
        bitloop_deploy_webgpu_runtime(FluidPressureWebGPUTests)
        add_executable(FluidAmrWebGPUTests tests/AmrGasWebGPUTests.cpp)
        target_link_libraries(FluidAmrWebGPUTests PRIVATE FluidSimEngineGPU)
        bitloop_deploy_webgpu_runtime(FluidAmrWebGPUTests)
        add_executable(FluidCoarseningWebGPUTests tests/FluidCoarseningWebGPUTests.cpp)
        target_link_libraries(FluidCoarseningWebGPUTests PRIVATE FluidSimEngineGPU)
        bitloop_deploy_webgpu_runtime(FluidCoarseningWebGPUTests)
        if(EMSCRIPTEN)
            set_source_files_properties(tests/FluidGridWebGPUTests.cpp tests/AmrPressureWebGPUTests.cpp tests/AmrGasWebGPUTests.cpp tests/FluidCoarseningWebGPUTests.cpp
                PROPERTIES COMPILE_DEFINITIONS main=fluidWebGpuTestMain)
            foreach(test FluidSimGPUTests FluidPressureWebGPUTests FluidAmrWebGPUTests FluidCoarseningWebGPUTests)
                target_sources(${test} PRIVATE tests/webgpu/main.cpp)
                target_link_options(${test} PRIVATE -sPROXY_TO_PTHREAD=1 -sPTHREAD_POOL_SIZE=1 -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=1)
                set_target_properties(${test} PROPERTIES SUFFIX ".html")
            endforeach()
        else()
            add_executable(FluidPressureWebGPUReference reference/pressure_gpu/webgpu.cpp)
            target_link_libraries(FluidPressureWebGPUReference PRIVATE FluidSimEngineGPU FluidPressure2D)
            bitloop_deploy_webgpu_runtime(FluidPressureWebGPUReference)
            add_executable(FluidUniformBenchmark tests/FluidUniformBenchmark.cpp)
            target_link_libraries(FluidUniformBenchmark PRIVATE FluidSimEngineGPU)
            add_test(NAME FluidGridGPUTests COMMAND FluidSimGPUTests)
            add_test(NAME FluidPressureWebGPUTests COMMAND FluidPressureWebGPUTests)
            add_test(NAME FluidAmrWebGPUTests COMMAND FluidAmrWebGPUTests)
            add_test(NAME FluidCoarseningWebGPUTests COMMAND FluidCoarseningWebGPUTests)
            add_executable(FluidDisplayTests tests/FluidDisplayTests.cpp tests/FluidAmbientMaskDisplayTests.cpp
                "${SRC_DIR}/FluidGridDisplay.cpp" "${SRC_DIR}/FluidAmbientMaskDisplay.cpp")
            target_link_libraries(FluidDisplayTests PRIVATE FluidSimEngineGPU)
            bitloop_deploy_webgpu_runtime(FluidDisplayTests)
            add_test(NAME FluidDisplayTests COMMAND FluidDisplayTests)
        endif()
    endif()
endif()

