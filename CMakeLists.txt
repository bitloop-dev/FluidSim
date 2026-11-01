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
    "${SRC_DIR}/*.inl"
)
# Engine sources belong exclusively to their libraries; the project owns scenes/rendering.
list(REMOVE_ITEM SIM_SOURCES
    "${SRC_DIR}/FluidField.cpp"
    "${SRC_DIR}/FluidGrid.cpp")
list(FILTER SIM_SOURCES EXCLUDE REGEX "/(UniformCPU|UniformWebGPU|AdaptiveCPU|AdaptiveWebGPU|Shared|LegacyWebGL)/.*\\.cpp$")

bitloop_new_project(FluidSim ${SIM_SOURCES})

bitloop_finalize()

if(NOT BITLOOP_DISCOVERY)
    find_package(unofficial-brotli CONFIG REQUIRED)
    # Shared host geometry and GPU execution infrastructure do not own an engine.
    add_library(FluidAdaptiveShared STATIC
        "${SRC_DIR}/Shared/Adaptive/AmrField2D.cpp"
        "${SRC_DIR}/Shared/Adaptive/AmrTransfer2D.cpp"
        "${SRC_DIR}/Shared/Adaptive/AmrPressureTopology2D.cpp"
        "${SRC_DIR}/Shared/Adaptive/FluidCoarseningLayout.cpp"
        "${SRC_DIR}/Shared/Adaptive/FluidCoarseningPreview.cpp")
    target_link_libraries(FluidAdaptiveShared PUBLIC bitloop::bitloop)
    if(NOT EMSCRIPTEN)
        if(MSVC)
            set(OpenMP_RUNTIME_MSVC llvm)
            set(OpenMP_CXX_FLAGS "/openmp:llvm" CACHE STRING "" FORCE)
        endif()
        find_package(OpenMP REQUIRED)
        target_link_libraries(FluidAdaptiveShared PUBLIC OpenMP::OpenMP_CXX)
    endif()

    add_library(FluidFieldCore STATIC "${SRC_DIR}/FluidField.cpp")
    target_link_libraries(FluidFieldCore PUBLIC bitloop::bitloop bitloop::webgpu)

    # Four independent implementations of the common grid contract.
    add_library(FluidUniformCPUCore STATIC
        "${SRC_DIR}/UniformCPU/FluidGridCPU.cpp"
        "${SRC_DIR}/UniformCPU/PressureMultigrid.cpp")
    target_link_libraries(FluidUniformCPUCore PUBLIC FluidFieldCore)

    add_library(FluidUniformWebGPUCore STATIC
        "${SRC_DIR}/UniformWebGPU/FluidGridWebGPU.cpp"
        "${SRC_DIR}/UniformWebGPU/FluidGridGPU.cpp"
        "${SRC_DIR}/UniformWebGPU/PressureMultigridGPU.cpp"
        "${SRC_DIR}/UniformWebGPU/ViscositySolverGPU.cpp")
    target_link_libraries(FluidUniformWebGPUCore PUBLIC FluidFieldCore)

    add_library(FluidAdaptiveCPUCore STATIC
        "${SRC_DIR}/AdaptiveCPU/FluidGridAdaptiveCPU.cpp"
        "${SRC_DIR}/AdaptiveCPU/FluidGridAdaptive.cpp"
        "${SRC_DIR}/AdaptiveCPU/AmrPressure2D.cpp")
    target_link_libraries(FluidAdaptiveCPUCore PUBLIC FluidAdaptiveShared FluidFieldCore)

    add_library(FluidAdaptiveWebGPUCore STATIC
        "${SRC_DIR}/AdaptiveWebGPU/FluidGridAdaptiveWebGPU.cpp"
        "${SRC_DIR}/AdaptiveWebGPU/FluidGridAdaptiveGPU.cpp"
        "${SRC_DIR}/AdaptiveWebGPU/AmrPressureGPU.cpp"
        "${SRC_DIR}/AdaptiveWebGPU/FluidCoarseningGPU.cpp")
    target_link_libraries(FluidAdaptiveWebGPUCore PUBLIC FluidAdaptiveShared FluidFieldCore)

    # The facade is the only place that selects a concrete engine.
    add_library(FluidSimEngine STATIC "${SRC_DIR}/FluidGrid.cpp")
    target_link_libraries(FluidSimEngine PUBLIC
        FluidUniformCPUCore FluidUniformWebGPUCore
        FluidAdaptiveCPUCore FluidAdaptiveWebGPUCore)
    target_link_libraries(FluidSim PRIVATE FluidSimEngine)
    target_link_libraries(FluidSim PRIVATE unofficial::brotli::brotlienc unofficial::brotli::brotlidec)
    bitloop_deploy_webgpu_runtime(FluidSim)

    # Retained reference implementation, linked only by its regression tests.
    add_library(FluidSimEngineWebGL STATIC
        "${SRC_DIR}/LegacyWebGL/FluidGridWebGL.cpp"
        "${SRC_DIR}/LegacyWebGL/FluidGridDisplayWebGL.cpp"
        "${SRC_DIR}/LegacyWebGL/PressureMultigridGPU.cpp")
    target_link_libraries(FluidSimEngineWebGL PUBLIC FluidUniformCPUCore)

    foreach(engine FluidAdaptiveShared FluidFieldCore FluidUniformCPUCore FluidUniformWebGPUCore
                   FluidAdaptiveCPUCore FluidAdaptiveWebGPUCore FluidSimEngine FluidSimEngineWebGL)
        target_include_directories(${engine} PUBLIC "${SRC_DIR}")
        target_compile_features(${engine} PUBLIC cxx_std_23)
        if(EMSCRIPTEN)
            target_compile_options(${engine} PUBLIC -pthread -fexceptions)
            target_link_options(${engine} PUBLIC -fexceptions)
        endif()
    endforeach()

    option(FLUIDSIM_BUILD_TESTS "Build the fluid engine regression tests" OFF)
    if(FLUIDSIM_BUILD_TESTS)
        enable_testing()
        if(BITLOOP_DEV_MODE)
            foreach(test FluidSurfaceTensionWave FluidSnapshotSceneTests FluidSceneTests FluidSplashCubesTests FluidHoneyTapTests)
                add_executable(${test} tests/${test}.cpp
                    "${SRC_DIR}/scenarios/CrashingWave.cpp" "${SRC_DIR}/scenarios/SplashCubes.cpp"
                    "${SRC_DIR}/FluidGridDisplay.cpp"
                    "${SRC_DIR}/FluidCheckpointIO.cpp" "${SRC_DIR}/FluidSnapshot.cpp")
                target_compile_definitions(${test} PRIVATE SIM_NAME=FluidSim)
                target_include_directories(${test} PRIVATE "${SRC_DIR}")
                target_link_libraries(${test} PRIVATE FluidSimEngine
                    unofficial::brotli::brotlienc unofficial::brotli::brotlidec)
                bitloop_deploy_webgpu_runtime(${test})
            endforeach()
            target_sources(FluidHoneyTapTests PRIVATE "${SRC_DIR}/scenarios/HoneyTap.cpp")
            add_test(NAME FluidSceneTests COMMAND FluidSceneTests)
            add_test(NAME FluidSplashCubesTests COMMAND FluidSplashCubesTests)
            add_test(NAME FluidHoneyTapTests COMMAND FluidHoneyTapTests)
            add_test(NAME FluidSnapshotSceneTests COMMAND FluidSnapshotSceneTests)
            add_executable(FluidCheckpointTests tests/FluidCheckpointTests.cpp
                "${SRC_DIR}/FluidCheckpointIO.cpp" "${SRC_DIR}/FluidSnapshot.cpp")
            target_include_directories(FluidCheckpointTests PRIVATE "${SRC_DIR}")
            target_compile_features(FluidCheckpointTests PRIVATE cxx_std_23)
            target_link_libraries(FluidCheckpointTests PRIVATE bitloop::bitloop unofficial::brotli::brotlienc unofficial::brotli::brotlidec)
            add_test(NAME FluidCheckpointTests COMMAND FluidCheckpointTests)
        endif()
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
        add_executable(FluidPressureRecoveryTests tests/FluidPressureRecoveryTests.cpp)
        target_link_libraries(FluidPressureRecoveryTests PRIVATE FluidSimEngine)
        bitloop_deploy_webgpu_runtime(FluidPressureRecoveryTests)
        add_executable(FluidViscosityTests tests/FluidViscosityTests.cpp)
        target_link_libraries(FluidViscosityTests PRIVATE FluidUniformWebGPUCore)
        bitloop_deploy_webgpu_runtime(FluidViscosityTests)
        add_executable(FluidViscosityIntegrationTests tests/FluidViscosityIntegrationTests.cpp)
        add_executable(FluidBoundaryTests tests/FluidBoundaryTests.cpp)
        target_link_libraries(FluidBoundaryTests PRIVATE FluidSimEngine)
        bitloop_deploy_webgpu_runtime(FluidBoundaryTests)
        target_link_libraries(FluidViscosityIntegrationTests PRIVATE FluidSimEngine)
        bitloop_deploy_webgpu_runtime(FluidViscosityIntegrationTests)
        add_executable(FluidSurfaceTensionTests tests/FluidSurfaceTensionTests.cpp)
        target_link_libraries(FluidSurfaceTensionTests PRIVATE FluidSimEngine)
        bitloop_deploy_webgpu_runtime(FluidSurfaceTensionTests)
        add_executable(FluidAccelerationTests tests/FluidAccelerationTests.cpp)
        target_link_libraries(FluidAccelerationTests PRIVATE FluidSimEngine)
        bitloop_deploy_webgpu_runtime(FluidAccelerationTests)
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
            set_source_files_properties(tests/FluidGridWebGPUTests.cpp tests/FluidWebGPUAPITests.cpp tests/FluidSurfaceTensionTests.cpp tests/FluidAccelerationTests.cpp PROPERTIES COMPILE_DEFINITIONS main=fluidWebGpuTestMain)
            set_source_files_properties(tests/FluidPressureRecoveryTests.cpp tests/FluidViscosityTests.cpp tests/FluidViscosityIntegrationTests.cpp PROPERTIES COMPILE_DEFINITIONS main=fluidWebGpuTestMain)
            set_source_files_properties(tests/FluidBoundaryTests.cpp PROPERTIES COMPILE_DEFINITIONS main=fluidWebGpuTestMain)
            foreach(test FluidUniformWebGPUTests FluidWebGPUAPITests FluidSurfaceTensionTests FluidAccelerationTests FluidPressureRecoveryTests FluidViscosityTests FluidViscosityIntegrationTests FluidBoundaryTests)
                target_sources(${test} PRIVATE tests/webgpu/main.cpp)
                target_link_options(${test} PRIVATE -sPROXY_TO_PTHREAD=1 -sPTHREAD_POOL_SIZE=1 -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=1)
                set_target_properties(${test} PROPERTIES SUFFIX ".html")
            endforeach()
        else()
            add_test(NAME FluidUniformWebGPUTests COMMAND FluidUniformWebGPUTests)
            add_test(NAME FluidPressureRecoveryTests COMMAND FluidPressureRecoveryTests)
            add_test(NAME FluidViscosityTests COMMAND FluidViscosityTests)
            add_test(NAME FluidViscosityIntegrationTests COMMAND FluidViscosityIntegrationTests)
            add_test(NAME FluidBoundaryTests COMMAND FluidBoundaryTests)
            add_test(NAME FluidSurfaceTensionTests COMMAND FluidSurfaceTensionTests)
            add_test(NAME FluidAccelerationTests COMMAND FluidAccelerationTests)
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

