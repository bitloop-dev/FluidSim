# Keep both packages on the same validated dimensionality, precision and runtime.
set(VCPKG_LIBRARY_LINKAGE static)
set(platform_patches)
if(VCPKG_TARGET_IS_WINDOWS)
    list(APPEND platform_patches msvc-openmp.patch)
endif()
vcpkg_from_github(
    OUT_SOURCE_PATH amrex_source
    REPO AMReX-Codes/amrex
    REF 015c14b65715b277fbc54706ebe7455b196d2254
    SHA512 ccdaffd5709b7351e8fa5c2c5b617ecbb285a7f4b7fb455218c742cc76d01e1d54c25c2c21881792a5d5b3b591c941ab573857b34e7d6609e38ef12f0f49f941
    PATCHES ${platform_patches}
)
vcpkg_from_github(
    OUT_SOURCE_PATH hydro_source
    REPO AMReX-Fluids/AMReX-Hydro
    REF 5829a90a6cd521b378ebb7bce1390937beb49506
    SHA512 4cefece7210dfc2cc994104621cf92336695860a5767eedf6b2ee248ec5811da1945b77238b327b817e41673b3dcc967e244e929f89c6e3f5350bf0cc16b9efe
)
vcpkg_cmake_configure(
    SOURCE_PATH "${CURRENT_PORT_DIR}"
    OPTIONS "-DFLUID_AMREX_SOURCE=${amrex_source}" "-DFLUID_HYDRO_SOURCE=${hydro_source}"
)
vcpkg_cmake_install()
if(VCPKG_TARGET_IS_WINDOWS)
    vcpkg_cmake_config_fixup(PACKAGE_NAME AMReX CONFIG_PATH cmake)
else()
    vcpkg_cmake_config_fixup(PACKAGE_NAME AMReX CONFIG_PATH lib/cmake/AMReX)
endif()
vcpkg_cmake_config_fixup(PACKAGE_NAME AMReX-Hydro CONFIG_PATH lib/cmake/AMReX-Hydro)
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/AMReX/AMReXConfig.cmake"
    "\${PACKAGE_PREFIX_DIR}/cmake/AMReXCMakeModules" "\${CMAKE_CURRENT_LIST_DIR}/AMReXCMakeModules")
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/AMReX/AMReXConfig.cmake"
    "\${PACKAGE_PREFIX_DIR}/lib/cmake/AMReX/AMReXCMakeModules" "\${CMAKE_CURRENT_LIST_DIR}/AMReXCMakeModules")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${amrex_source}/LICENSE" "${hydro_source}/LICENSE")
