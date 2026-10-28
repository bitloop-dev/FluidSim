cmake_minimum_required(VERSION 3.21)
# Keep the existing Dawn shader/compiler packaging inside this project.
list(APPEND VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/ports")

# The four-backend project requires these, including in existing build caches.
# Apply them before project/toolchain/provider discovery creates Bitloop targets.
set(BITLOOP_WITH_WEBGPU ON CACHE BOOL "Build Bitloop WebGPU support" FORCE)
set(BITLOOP_WEBGPU_ASYNC_WAIT ON CACHE BOOL "Allow cooperative WebGPU waits in browser builds" FORCE)
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
    "${SRC_DIR}/FluidGridCPU.cpp"
    "${SRC_DIR}/FluidGridAdaptiveCPU.cpp"
    "${SRC_DIR}/FluidGridWebGPU.cpp"
    "${SRC_DIR}/FluidGridAdaptiveWebGPU.cpp"
    "${SRC_DIR}/PressureMultigrid.cpp"
    "${SRC_DIR}/FluidGridWebGL.cpp"
    "${SRC_DIR}/PressureMultigridGPU.cpp")

list(FILTER SIM_SOURCES EXCLUDE REGEX "/Adaptive/.*\\.cpp$")
list(FILTER SIM_SOURCES EXCLUDE REGEX "/UniformWebGPU/.*\\.cpp$")
list(FILTER SIM_SOURCES EXCLUDE REGEX "/AdaptiveWebGPU/.*\\.cpp$")

bitloop_new_project(FluidSim ${SIM_SOURCES})

bitloop_finalize()

if(NOT BITLOOP_DISCOVERY)
    add_library(FluidAdaptiveCore STATIC
        "${SRC_DIR}/Adaptive/AmrField2D.cpp" "${SRC_DIR}/Adaptive/AmrTransfer2D.cpp"
        "${SRC_DIR}/Adaptive/AmrPressure2D.cpp" "${SRC_DIR}/Adaptive/AmrPressureTopology2D.cpp"
        "${SRC_DIR}/Adaptive/FluidGridAdaptive.cpp" "${SRC_DIR}/Adaptive/AmrGasRuntime.cpp")
    target_include_directories(FluidAdaptiveCore PUBLIC "${SRC_DIR}/Adaptive" "${SRC_DIR}")
    target_compile_features(FluidAdaptiveCore PUBLIC cxx_std_23)
    target_compile_definitions(FluidAdaptiveCore PUBLIC FLUIDSIM_HAS_AMR=1)
    if(EMSCRIPTEN)
        target_compile_options(FluidAdaptiveCore PUBLIC -pthread -fexceptions)
        target_link_options(FluidAdaptiveCore PUBLIC -fexceptions)
    else()
        if(MSVC)
            set(OpenMP_RUNTIME_MSVC llvm)
            set(OpenMP_CXX_FLAGS "/openmp:llvm" CACHE STRING "" FORCE)
        endif()
        find_package(OpenMP REQUIRED)
        target_link_libraries(FluidAdaptiveCore PUBLIC OpenMP::OpenMP_CXX)
    endif()
    add_library(FluidUniformWebGPUCore STATIC
        "${SRC_DIR}/UniformWebGPU/FluidGridGPU.cpp" "${SRC_DIR}/UniformWebGPU/PressureMultigridGPU.cpp")
    target_include_directories(FluidUniformWebGPUCore PUBLIC "${SRC_DIR}")
    target_compile_features(FluidUniformWebGPUCore PUBLIC cxx_std_23)
    target_link_libraries(FluidUniformWebGPUCore PUBLIC bitloop::bitloop bitloop::webgpu)
    if(EMSCRIPTEN)
        target_compile_options(FluidUniformWebGPUCore PUBLIC -pthread -fexceptions)
        target_link_options(FluidUniformWebGPUCore PUBLIC -fexceptions)
    endif()
    add_library(FluidAdaptiveWebGPUCore STATIC
        "${SRC_DIR}/AdaptiveWebGPU/FluidGridAdaptiveGPU.cpp" "${SRC_DIR}/AdaptiveWebGPU/AmrPressureGPU.cpp")
    target_include_directories(FluidAdaptiveWebGPUCore PUBLIC "${SRC_DIR}")
    target_compile_features(FluidAdaptiveWebGPUCore PUBLIC cxx_std_23)
    target_link_libraries(FluidAdaptiveWebGPUCore PUBLIC FluidAdaptiveCore FluidUniformWebGPUCore)
    add_library(FluidSimEngine STATIC "${SRC_DIR}/FluidGrid.cpp" "${SRC_DIR}/FluidGridCPU.cpp" "${SRC_DIR}/PressureMultigrid.cpp" "${SRC_DIR}/FluidGridAdaptiveCPU.cpp" "${SRC_DIR}/Adaptive/FluidCoarseningPreview.cpp" "${SRC_DIR}/FluidGridWebGPU.cpp" "${SRC_DIR}/FluidGridAdaptiveWebGPU.cpp" "${SRC_DIR}/AdaptiveWebGPU/FluidCoarseningGPU.cpp")
    target_include_directories(FluidSimEngine PUBLIC "${SRC_DIR}")
    target_compile_features(FluidSimEngine PUBLIC cxx_std_23)
    target_link_libraries(FluidSimEngine PUBLIC bitloop::bitloop FluidAdaptiveCore FluidUniformWebGPUCore FluidAdaptiveWebGPUCore)
    target_link_libraries(FluidSim PRIVATE FluidSimEngine)
    bitloop_deploy_webgpu_runtime(FluidSim)

    add_library(FluidSimEngineWebGL STATIC "${SRC_DIR}/FluidGridWebGL.cpp" "${SRC_DIR}/PressureMultigridGPU.cpp")
    target_include_directories(FluidSimEngineWebGL PUBLIC "${SRC_DIR}")
    target_compile_features(FluidSimEngineWebGL PUBLIC cxx_std_23)
    target_link_libraries(FluidSimEngineWebGL PUBLIC bitloop::bitloop)
    target_link_libraries(FluidSim PRIVATE FluidSimEngineWebGL)

    target_link_libraries(FluidSimEngineWebGL PRIVATE FluidSimEngine)

    option(FLUIDSIM_BUILD_TESTS "Build the fluid engine regression tests" OFF)
    if(FLUIDSIM_BUILD_TESTS)
        enable_testing()
        add_executable(FluidSimTests tests/FluidGridTests.cpp)
        target_link_libraries(FluidSimTests PRIVATE FluidSimEngine)
        add_test(NAME FluidGridTests COMMAND FluidSimTests)
        add_executable(FluidSimAPITests tests/FluidGridAPITests.cpp)
        target_link_libraries(FluidSimAPITests PRIVATE FluidSimEngine)
        add_test(NAME FluidGridAPITests COMMAND FluidSimAPITests)
        add_executable(FluidCoarseningPreviewTests tests/FluidCoarseningPreviewTests.cpp)
        target_link_libraries(FluidCoarseningPreviewTests PRIVATE FluidSimEngine)
        add_test(NAME FluidCoarseningPreviewTests COMMAND FluidCoarseningPreviewTests)
        add_executable(FluidAdaptiveAPITests tests/FluidAdaptiveAPITests.cpp)
        target_link_libraries(FluidAdaptiveAPITests PRIVATE FluidSimEngine)
        add_test(NAME FluidAdaptiveAPITests COMMAND FluidAdaptiveAPITests)
        add_executable(FluidUniformWebGPUTests tests/FluidGridWebGPUTests.cpp)
        target_link_libraries(FluidUniformWebGPUTests PRIVATE FluidSimEngine)
        bitloop_deploy_webgpu_runtime(FluidUniformWebGPUTests)
        add_executable(FluidWebGPUAPITests tests/FluidWebGPUAPITests.cpp "${SRC_DIR}/FluidGridDisplay.cpp")
        target_link_libraries(FluidWebGPUAPITests PRIVATE FluidSimEngine)
        bitloop_deploy_webgpu_runtime(FluidWebGPUAPITests)
        add_executable(FluidAdaptiveWebGPUAPITests tests/FluidAdaptiveWebGPUAPITests.cpp)
        target_link_libraries(FluidAdaptiveWebGPUAPITests PRIVATE FluidSimEngine)
        foreach(test AmrGasWebGPUTests FluidCoarseningWebGPUTests AmrPressureWebGPUTests)
            add_executable(${test} tests/${test}.cpp)
            target_link_libraries(${test} PRIVATE FluidSimEngine)
        endforeach()
        foreach(test FluidAdaptiveWebGPUAPITests AmrGasWebGPUTests FluidCoarseningWebGPUTests AmrPressureWebGPUTests)
            bitloop_deploy_webgpu_runtime(${test})
            if(EMSCRIPTEN)
                set_source_files_properties(tests/${test}.cpp PROPERTIES COMPILE_DEFINITIONS main=fluidWebGpuTestMain)
                target_sources(${test} PRIVATE tests/webgpu/main.cpp)
                target_link_options(${test} PRIVATE -sPROXY_TO_PTHREAD=1 -sPTHREAD_POOL_SIZE=1 -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=1)
                set_target_properties(${test} PROPERTIES SUFFIX ".html")
            else()
                add_test(NAME ${test} COMMAND ${test})
            endif()
        endforeach()
        if(EMSCRIPTEN)
            # Native hidden-window display checks are conditional in the API fixture.
            set_source_files_properties(tests/FluidGridWebGPUTests.cpp tests/FluidWebGPUAPITests.cpp PROPERTIES COMPILE_DEFINITIONS main=fluidWebGpuTestMain)
            foreach(test FluidUniformWebGPUTests FluidWebGPUAPITests)
                target_sources(${test} PRIVATE tests/webgpu/main.cpp)
                target_link_options(${test} PRIVATE -sPROXY_TO_PTHREAD=1 -sPTHREAD_POOL_SIZE=1 -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=1)
                set_target_properties(${test} PROPERTIES SUFFIX ".html")
            endforeach()
        else()
            add_test(NAME FluidUniformWebGPUTests COMMAND FluidUniformWebGPUTests)
            add_test(NAME FluidWebGPUAPITests COMMAND FluidWebGPUAPITests)
            add_test(NAME FluidWebGPUEvolvingParity COMMAND FluidWebGPUAPITests --evolving)
        endif()
        if(NOT EMSCRIPTEN)
            add_executable(FluidRefinementTests tests/FluidRefinementTests.cpp)
            target_link_libraries(FluidRefinementTests PRIVATE FluidSimEngine)
            foreach(case reconstruction cfl liquid-short selection)
                add_test(NAME FluidRefinement_${case} COMMAND FluidRefinementTests ${case})
                set_tests_properties(FluidRefinement_${case} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=8;OMP_DYNAMIC=FALSE")
            endforeach()
            add_executable(FluidSimWebGLTests tests/FluidGridWebGLTests.cpp tests/FluidGridWebGLCoarseTests.cpp "${SRC_DIR}/FluidGridDisplay.cpp")
            target_link_libraries(FluidSimWebGLTests PRIVATE FluidSimEngineWebGL FluidSimEngine)
            add_test(NAME FluidGridWebGLTests COMMAND FluidSimWebGLTests)
        endif()
    endif()
    foreach(test FluidSimTests FluidSimAPITests FluidCoarseningPreviewTests FluidAdaptiveAPITests FluidRefinementTests FluidSimWebGLTests)
        if(TARGET ${test})
            bitloop_deploy_webgpu_runtime(${test})
        endif()
    endforeach()
endif()

