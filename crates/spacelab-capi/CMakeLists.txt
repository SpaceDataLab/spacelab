# The CMake side of the C ABI: the imported targets a consumer links, an install rule that lays
# down a `find_package(spacelab)` config, and the smoke tests that prove the headers work.
#
# This does not build the Rust half. Cargo owns that, and a CMake rule that shelled out to it would
# be a second build system with an opinion about profiles -- so the library path comes in as a cache
# variable and the caller is responsible for having built it. `just capi` drives the whole thing and
# puts the build tree under `target/capi-build`, which is already ignored.
#
# The smoke tests are built once per language standard the compiler supports, from C99 to the
# newest C and from C++20 to the newest C++, and once more with `-fshort-enums`. A header that is a
# public contract is compiled by toolchains this repository does not own; the standard the consumer
# happens to pass is theirs to choose, so every rung of that ladder is a test rather than a hope.

cmake_minimum_required(VERSION 3.20)
project(
    spacelab
    VERSION 0.1.0
    DESCRIPTION "C ABI and C++20 header for the spacelab space-data computation core"
    LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(SPACELAB_CAPI_LIB "" CACHE FILEPATH "Path to libspacelab_capi.a")
if(NOT SPACELAB_CAPI_LIB)
    message(FATAL_ERROR "Set -DSPACELAB_CAPI_LIB=<path to libspacelab_capi.a>; build it with "
                        "`cargo build -p spacelab-capi`")
endif()

# Two targets, because the artifact is two things. `spacelab::spacelab_c` is the C ABI: a library,
# an include directory and the system libraries a Rust `staticlib` needs. `spacelab::spacelab` is
# that plus the C++20 the header-only wrapper requires -- and it has to be a separate target,
# because a `cxx_std_20` compile feature attached to the C library would follow into a pure C
# consumer's build, where it means nothing and is an error to require.
#
# UNKNOWN rather than STATIC because the same rule then works for the `cdylib` the crate also
# builds; CMake reads the file and decides.
add_library(spacelab_capi UNKNOWN IMPORTED GLOBAL)
set_target_properties(spacelab_capi PROPERTIES IMPORTED_LOCATION "${SPACELAB_CAPI_LIB}")
target_include_directories(spacelab_capi INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# A Rust `staticlib` brings its own `std`, which wants the dynamic loader, threads and libm.
if(UNIX AND NOT APPLE)
    target_link_libraries(spacelab_capi INTERFACE dl pthread m)
elseif(APPLE)
    target_link_libraries(spacelab_capi INTERFACE m)
endif()

add_library(spacelab_cxx INTERFACE)
target_link_libraries(spacelab_cxx INTERFACE spacelab_capi)
target_compile_features(spacelab_cxx INTERFACE cxx_std_20)

add_library(spacelab::spacelab_c ALIAS spacelab_capi)
add_library(spacelab::spacelab ALIAS spacelab_cxx)

# ---- The smoke tests, one per standard ----
#
# -Werror on these translation units, because the first warning is the cheapest one to fix and
# these headers are what consumers compile with their own flags. Not on the INTERFACE above: a
# consumer's warning policy is the consumer's to choose.
set(SPACELAB_WARNINGS -Wall -Wextra -Wpedantic -Werror)

# `-fshort-enums` is the default on most ARM bare-metal toolchains and it shrinks a C enum to the
# smallest type that fits. The `_FORCE_INT32` enumerators in `spacelab.h` exist to stop that, and
# these variants are what proves they still do -- the C test checks the widths at run time and the
# C++ test static-asserts them.
check_c_compiler_flag(-fshort-enums SPACELAB_HAVE_SHORT_ENUMS_C)
check_cxx_compiler_flag(-fshort-enums SPACELAB_HAVE_SHORT_ENUMS_CXX)

enable_testing()

# One executable and one test per (language, standard, enum-width) rung. `label` names the rung in
# ctest output, so a failure says which standard broke without a rebuild to find out.
function(spacelab_smoke_test language standard short_enums)
    set(label "${language}${standard}")
    if(short_enums)
        set(label "${label}_short_enums")
    endif()
    string(TOLOWER "${label}" label)
    string(REPLACE "+" "x" label "${label}")
    set(target "spacelab_smoke_${label}")

    if(language STREQUAL "C")
        add_executable(${target} tests/c/smoke.c)
        target_link_libraries(${target} PRIVATE spacelab::spacelab_c)
        set_target_properties(${target} PROPERTIES C_STANDARD ${standard} C_STANDARD_REQUIRED ON
                                                   C_EXTENSIONS OFF)
    elseif(language STREQUAL "CXXNOTHROW")
        add_executable(${target} tests/cpp/nothrow.cpp)
        target_link_libraries(${target} PRIVATE spacelab::spacelab)
        target_compile_options(${target} PRIVATE -fno-exceptions)
        set_target_properties(${target} PROPERTIES CXX_STANDARD ${standard} CXX_STANDARD_REQUIRED ON
                                                   CXX_EXTENSIONS OFF)
    else()
        add_executable(${target} tests/cpp/smoke.cpp)
        target_link_libraries(${target} PRIVATE spacelab::spacelab)
        set_target_properties(${target} PROPERTIES CXX_STANDARD ${standard} CXX_STANDARD_REQUIRED ON
                                                   CXX_EXTENSIONS OFF)
    endif()

    target_compile_options(${target} PRIVATE ${SPACELAB_WARNINGS})
    if(short_enums)
        target_compile_options(${target} PRIVATE -fshort-enums)
    endif()

    # An IMPORTED library's file is not, by itself, something CMake will relink for: rebuilding the
    # Rust half and running `cmake --build` again left every executable here linked against the
    # previous archive, so the tests passed against code that no longer existed. `LINK_DEPENDS`
    # names the file as a dependency of the link step, which is what makes `just capi` honest.
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${SPACELAB_CAPI_LIB}")

    add_test(NAME ${label} COMMAND ${target})
endfunction()

# C99 is the floor the header is written to; the rest are every newer standard this compiler knows.
# `c_std_NN` in CMAKE_C_COMPILE_FEATURES is CMake saying it can ask for that standard here, so a
# consumer on an older toolchain configures fine and simply tests fewer rungs.
set(SPACELAB_C_STANDARDS_TESTED "")
foreach(standard IN ITEMS 99 11 17 23)
    if("c_std_${standard}" IN_LIST CMAKE_C_COMPILE_FEATURES)
        spacelab_smoke_test(C ${standard} OFF)
        list(APPEND SPACELAB_C_STANDARDS_TESTED ${standard})
    endif()
endforeach()

# Each C++ standard is run twice: once with exceptions, once without. The `-fno-exceptions` rung
# compiles a different translation unit, because a build with exceptions off cannot see the
# throwing half of the header at all -- which is the property being tested. `result<T>` is
# `std::expected` from C++23 on and a stand-in on C++20, so both of those paths are compiled too.
check_cxx_compiler_flag(-fno-exceptions SPACELAB_HAVE_NO_EXCEPTIONS)

set(SPACELAB_CXX_STANDARDS_TESTED "")
foreach(standard IN ITEMS 20 23 26)
    if("cxx_std_${standard}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        spacelab_smoke_test(CXX ${standard} OFF)
        list(APPEND SPACELAB_CXX_STANDARDS_TESTED ${standard})
        if(SPACELAB_HAVE_NO_EXCEPTIONS)
            spacelab_smoke_test(CXXNOTHROW ${standard} OFF)
        endif()
    endif()
endforeach()

if(NOT SPACELAB_C_STANDARDS_TESTED OR NOT SPACELAB_CXX_STANDARDS_TESTED)
    message(FATAL_ERROR "No usable C or C++ standard found; the headers cannot be tested here")
endif()

# The enum-width rung runs on the newest standard of each language, which is where a regression
# would land first.
list(GET SPACELAB_C_STANDARDS_TESTED -1 SPACELAB_NEWEST_C)
list(GET SPACELAB_CXX_STANDARDS_TESTED -1 SPACELAB_NEWEST_CXX)
if(SPACELAB_HAVE_SHORT_ENUMS_C)
    spacelab_smoke_test(C ${SPACELAB_NEWEST_C} ON)
endif()
if(SPACELAB_HAVE_SHORT_ENUMS_CXX)
    spacelab_smoke_test(CXX ${SPACELAB_NEWEST_CXX} ON)
endif()

message(STATUS "spacelab: C standards tested ${SPACELAB_C_STANDARDS_TESTED}, "
               "C++ standards tested ${SPACELAB_CXX_STANDARDS_TESTED}")

# ---- Installation ----
#
# `cmake --install` lays down the two headers, the static library and a package config, so that a
# consumer writes:
#
#     find_package(spacelab REQUIRED)
#     target_link_libraries(my_app PRIVATE spacelab::spacelab)     # C++
#     target_link_libraries(my_c_app PRIVATE spacelab::spacelab_c) # C
#
# The config is generated rather than hand-written so that the version check is real: a consumer
# asking for `find_package(spacelab 0.1.0)` gets a hard error against a different one, which is the
# C++ half of what `SL_ABI_VERSION` does at run time.

install(FILES include/spacelab.h include/spacelab.hpp
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES "${SPACELAB_CAPI_LIB}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# The installed config cannot reuse the IMPORTED targets above -- those point at a build-tree path.
# It names the installed library instead, which is the only path that survives the install.
file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/spacelabConfig.cmake.in"
    "@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
get_filename_component(SPACELAB_PREFIX \"\${CMAKE_CURRENT_LIST_DIR}/../../..\" ABSOLUTE)

add_library(spacelab::spacelab_c UNKNOWN IMPORTED)
set_target_properties(spacelab::spacelab_c PROPERTIES
    IMPORTED_LOCATION \"\${SPACELAB_PREFIX}/@CMAKE_INSTALL_LIBDIR@/@SPACELAB_LIB_NAME@\"
    INTERFACE_INCLUDE_DIRECTORIES \"\${SPACELAB_PREFIX}/@CMAKE_INSTALL_INCLUDEDIR@\")
if(UNIX AND NOT APPLE)
    set_property(TARGET spacelab::spacelab_c APPEND PROPERTY
                 INTERFACE_LINK_LIBRARIES dl pthread m)
elseif(APPLE)
    set_property(TARGET spacelab::spacelab_c APPEND PROPERTY INTERFACE_LINK_LIBRARIES m)
endif()

add_library(spacelab::spacelab INTERFACE IMPORTED)
set_target_properties(spacelab::spacelab PROPERTIES
    INTERFACE_LINK_LIBRARIES spacelab::spacelab_c
    INTERFACE_COMPILE_FEATURES cxx_std_20)

check_required_components(spacelab)
")

get_filename_component(SPACELAB_LIB_NAME "${SPACELAB_CAPI_LIB}" NAME)
configure_package_config_file(
    "${CMAKE_CURRENT_BINARY_DIR}/spacelabConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/spacelabConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/spacelab")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/spacelabConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/spacelabConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/spacelabConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/spacelab")
