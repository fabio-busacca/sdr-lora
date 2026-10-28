# Copyright 2024 SDR-LoRa contributors. SPDX-License-Identifier: GPL-3.0-only
cmake_minimum_required(VERSION 3.16)
project(gr-sdrlora VERSION 2.0.0 LANGUAGES CXX C)

# GNU Radio 3.10's headers need C++17, so the installable module is built at
# 17. The PHY CORE (lora_phy, lora_rx, lora_sync) is deliberately written to
# C++11 with no GNU Radio dependency, so it also compiles on its own with
# -std=c++11 under a bare compiler. The difference is not an accident.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# COMPONENTS blocks is not needed by the library itself. Nothing under lib/
# includes a gnuradio/blocks header or links gnuradio-blocks -- the decoder
# and encoder are plain gr::sync_block and gr::block. What needs it is any
# flowgraph built around them: the usual sources, sinks and message_debug
# that drive the blocks all live in gnuradio-blocks.
#
# Kept REQUIRED, because without it a GNU Radio install lacking
# gnuradio-blocks would configure and build cleanly and then fail with an
# ImportError the first time a flowgraph is run, which is a much worse place
# to learn about a missing dependency than configure time.
# OFF builds only the C++ PHY library, sdrlora-phy, with no GNU Radio.
option(SDRLORA_ENABLE_GNURADIO
       "Build the GNU Radio blocks and their Python bindings" ON)
if(SDRLORA_ENABLE_GNURADIO)
find_package(Gnuradio "3.10" REQUIRED COMPONENTS blocks)
find_package(pybind11 QUIET)
else()
    message(STATUS "SDRLORA_ENABLE_GNURADIO is OFF: building the C++ PHY library only")
endif()

include(GNUInstallDirs)

if(SDRLORA_ENABLE_GNURADIO)

########################################################################
# Can a GNU Radio header actually be included?
#
# GNU Radio's headers pull in spdlog, which pulls in fmt, and spdlog only
# compiles against the fmt MAJOR version it was built for. find_package() in
# config mode also searches prefixes derived from the PATH environment
# variable, so anything on PATH that ships its own fmt with cmake config files
# -- an active conda being the common case -- can win that search. Its include
# directory then arrives as -isystem on every GR-facing translation unit and
# shadows the system fmt.
#
# A typical case: with a conda environment's bin/ on PATH, cmake picks
# conda's fmt over the system's, and every file including
# <gnuradio/block.h> fails with
#
#     error: 'basic_format_string' is not a member of 'fmt'
#
# amid template noise pointing at spdlog's headers, with no hint that the
# cause is an unrelated prefix on PATH. Diagnose it here instead, while the
# include directories about to be used are still known.
########################################################################
# Compile only, never link. check_cxx_source_compiles() links as well, and a
# translation unit that merely includes the GR headers pulls in inline fmt code
# whose symbols live in libfmt -- so a link-based probe reports failure on a
# perfectly healthy system. STATIC_LIBRARY makes try_compile stop after the
# compile, which is the only step whose success is in question.
#
# LINK_LIBRARIES on the real imported target rather than a hand-rebuilt flag
# list: SPDLOG_FMT_EXTERNAL, SPDLOG_COMPILED_LIB and FMT_SHARED come from the
# fmt and spdlog targets GNU Radio links, not from GNU Radio's own interface,
# so reconstructing the flags by hand gets the wrong answer.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sdrlora_gr_probe.cc"
     "#include <gnuradio/block.h>\nint probe() { return 0; }\n")
set(_saved_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
try_compile(SDRLORA_GR_HEADERS_USABLE
            "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/sdrlora_gr_probe"
            "${CMAKE_CURRENT_BINARY_DIR}/sdrlora_gr_probe.cc"
            LINK_LIBRARIES gnuradio::gnuradio-runtime
            OUTPUT_VARIABLE _gr_probe_out)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_saved_target_type}")
if(SDRLORA_GR_HEADERS_USABLE)
    message(STATUS "GNU Radio headers compile cleanly")
endif()

if(NOT SDRLORA_GR_HEADERS_USABLE)
    get_target_property(_gr_incs gnuradio::gnuradio-runtime
                        INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _gr_incs)
        set(_gr_incs "")
    endif()
    # The offending directory usually arrives via the fmt or spdlog imported
    # target rather than GNU Radio's own interface, so it is not in _gr_incs.
    # Recover it from the command line the probe actually ran.
    set(_hint "")
    string(REGEX MATCHALL "-isystem +[^ \n]+" _isystems "${_gr_probe_out}")
    foreach(_flag IN LISTS _isystems)
        string(REGEX REPLACE "-isystem +" "" _dir "${_flag}")
        if(NOT _dir STREQUAL "/usr/include" AND EXISTS "${_dir}/fmt")
            set(_hint "${_hint}\n  a competing fmt lives in ${_dir}/fmt -- this is almost certainly the culprit")
        endif()
    endforeach()
    foreach(_i IN LISTS _gr_incs)
        if(EXISTS "${_i}/fmt" AND NOT _i STREQUAL "/usr/include")
            set(_hint "${_hint}\n  a competing fmt lives in ${_i}/fmt")
        endif()
    endforeach()
    string(REGEX MATCH "error:[^\n]*" _first_error "${_gr_probe_out}")
    if(_first_error)
        set(_hint "${_hint}\n  first compiler error: ${_first_error}")
    endif()
    message(FATAL_ERROR
        "<gnuradio/block.h> does not compile with the include paths cmake "
        "selected, so the GNU Radio blocks cannot be built.\n"
        "This is almost always a second copy of fmt or spdlog shadowing the "
        "one the system spdlog was built against, pulled in because "
        "find_package searches prefixes taken from PATH.${_hint}\n"
        "Include paths in use: ${_gr_incs}\n"
        "Fixes, cheapest first:\n"
        "  1. Build from a shell without that prefix on PATH "
        "(`conda deactivate` if it is conda).\n"
        "  2. Re-run cmake with -DCMAKE_IGNORE_PREFIX_PATH=<that prefix> "
        "(needs cmake 3.23+).\n"
        "In a development checkout, `bash test/diagnose_env.sh` run from the "
        "failing shell prints a full report.")
endif()
endif()  # SDRLORA_ENABLE_GNURADIO

########################################################################
# The PHY core and the GNU Radio blocks
########################################################################
# The PHY core is compiled once: the C++ library sdrlora-phy and the GNU
# Radio module are both built from these objects.
add_library(sdrlora_phy_objects OBJECT
    lib/lora_phy.cc
    lib/lora_rx.cc
    lib/lora_sync.cc)
set_target_properties(sdrlora_phy_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(sdrlora_phy_objects
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# The C++ PHY library: no GNU Radio, usable from C++11 on. Static unless
# BUILD_SHARED_LIBS is set. find_package(sdrlora-phy) finds it once
# installed, as the target sdrlora::sdrlora-phy.
add_library(sdrlora-phy $<TARGET_OBJECTS:sdrlora_phy_objects>)
add_library(sdrlora::sdrlora-phy ALIAS sdrlora-phy)
target_include_directories(sdrlora-phy
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(sdrlora-phy PUBLIC cxx_std_11)
set_target_properties(sdrlora-phy PROPERTIES
    LINKER_LANGUAGE CXX
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
install(TARGETS sdrlora-phy EXPORT sdrlora-phy-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/sdrlora/lora_phy.h
              include/sdrlora/lora_rx.h
              include/sdrlora/lora_tables.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sdrlora)
install(EXPORT sdrlora-phy-targets
        NAMESPACE sdrlora::
        FILE sdrlora-phyConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdrlora-phy)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sdrlora-phyConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sdrlora-phyConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdrlora-phy)

# The C++ example, built here so that it is known to compile against the
# library alone, and installed with its own CMakeLists.txt.
add_executable(sdrlora_hello examples/cpp/hello_lora.cc)
target_link_libraries(sdrlora_hello PRIVATE sdrlora-phy)
install(FILES examples/cpp/hello_lora.cc examples/cpp/CMakeLists.txt
        DESTINATION share/sdrlora/examples/cpp)

if(SDRLORA_ENABLE_GNURADIO)
add_library(gnuradio-sdrlora SHARED
    $<TARGET_OBJECTS:sdrlora_phy_objects>
    lib/encoder_impl.cc
    lib/decoder_impl.cc)

target_include_directories(gnuradio-sdrlora
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)
target_link_libraries(gnuradio-sdrlora PUBLIC gnuradio::gnuradio-runtime)
set_target_properties(gnuradio-sdrlora PROPERTIES
    DEFINE_SYMBOL "gnuradio_sdrlora_EXPORTS"
    SOVERSION 1)

install(TARGETS gnuradio-sdrlora
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/sdrlora/api.h
              include/sdrlora/decoder.h
              include/sdrlora/encoder.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sdrlora)
# Relative DESTINATIONs, so they resolve against CMAKE_INSTALL_PREFIX and
# the GRC files install under whatever prefix is asked for.
install(FILES grc/sdrlora_encoder.block.yml
              grc/sdrlora_decoder.block.yml
              grc/sdrlora_packet_print.block.yml
        DESTINATION share/gnuradio/grc/blocks)
# The example flowgraphs, to open in GRC.
install(DIRECTORY examples/
        DESTINATION share/sdrlora/examples
        FILES_MATCHING PATTERN "*.grc")
endif()  # SDRLORA_ENABLE_GNURADIO

########################################################################
# Self-checks, built only in a development checkout that has test/.
#
# This distribution ships without test/, so the block below is skipped and
# the module, the GRC files and the bindings build on their own.
########################################################################
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/conformance_main.cc")
add_executable(sdrlora_conformance test/conformance_main.cc lib/lora_phy.cc)
target_include_directories(sdrlora_conformance
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_testing()
add_test(NAME conformance
         COMMAND sdrlora_conformance ${CMAKE_CURRENT_SOURCE_DIR}/../conformance)

add_executable(sdrlora_rx_conformance
    test/rx_conformance_main.cc lib/lora_phy.cc lib/lora_rx.cc lib/lora_sync.cc)
target_include_directories(sdrlora_rx_conformance
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME rx_conformance
         COMMAND sdrlora_rx_conformance ${CMAKE_CURRENT_SOURCE_DIR}/../conformance)

# End to end: no known offset or CFO, so this is what exercises the sync search.
add_executable(sdrlora_e2e
    test/e2e_main.cc lib/lora_phy.cc lib/lora_rx.cc lib/lora_sync.cc)
target_include_directories(sdrlora_e2e
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME end_to_end COMMAND sdrlora_e2e)

# The remaining PHY-only programs: SF5/SF6, third-party captures, accepted
# rates and real-time cost.
foreach(_t lowsf interop rate_check rt_check)
    add_executable(sdrlora_${_t}
        test/${_t}_main.cc lib/lora_phy.cc lib/lora_rx.cc lib/lora_sync.cc)
    target_include_directories(sdrlora_${_t}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach()
# rt_check AT -O2, the level its budgets are calibrated at; Release's -O3
# shifts the timings and eats into the margin. A target's own options follow
# the configuration's flags, so this -O2 is the one in force.
target_compile_options(sdrlora_rt_check PRIVATE -O2)

# SF5 and SF6, which exist only in implicit-header mode. A driver script
# requires BOTH passes in the output AND a zero exit code: the exit code
# alone would pass a program that ran only SF5, and a
# PASS_REGULAR_EXPRESSION alone would make ctest ignore the exit code.
add_test(NAME low_sf
         COMMAND ${CMAKE_COMMAND} -DPROG=$<TARGET_FILE:sdrlora_lowsf>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/test/run_lowsf.cmake)
# Decodes the third-party gr-lora_sdr captures. Skips itself, passing, when
# the capture directory is absent.
add_test(NAME interop
         COMMAND sdrlora_interop ${CMAKE_CURRENT_SOURCE_DIR}/../conformance)
# Sample rates and bandwidths the demodulator accepts.
add_test(NAME rate_check COMMAND sdrlora_rate_check)
# Steady-state decode cost against the machine's own encode speed -- a ratio,
# so it does not flake with runner speed the way a wall-clock budget does.
add_test(NAME realtime COMMAND sdrlora_rt_check)
# ...and that the budget can actually fail, which is the point of --selftest.
add_test(NAME realtime_selftest COMMAND sdrlora_rt_check --selftest)
# ALONE: both time CPU work, which another test running on the other cores
# would slow under ctest -j.
set_tests_properties(realtime realtime_selftest PROPERTIES RUN_SERIAL TRUE)
# The decoder BLOCK, driven through work() with the call sizes a scheduler
# may choose: its documented delivery-latency interval has to hold for every
# chunking, not only for calls of exactly one window. Links the block
# library, unlike the PHY-only programs above.
if(SDRLORA_ENABLE_GNURADIO)
add_executable(sdrlora_cadence test/cadence_main.cc)
target_link_libraries(sdrlora_cadence PRIVATE gnuradio-sdrlora)
add_test(NAME block_cadence COMMAND sdrlora_cadence)
endif()
else()
    message(STATUS "test/ not present: building the module without its checks")
endif()

########################################################################
# Python bindings (optional)
########################################################################
if(SDRLORA_ENABLE_GNURADIO AND pybind11_FOUND)
    pybind11_add_module(sdrlora_python
        python/bindings/python_bindings.cc
        python/bindings/encoder_python.cc
        python/bindings/decoder_python.cc)
    target_link_libraries(sdrlora_python PRIVATE gnuradio-sdrlora
                                                 gnuradio::gnuradio-runtime)
    target_include_directories(sdrlora_python
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    message(STATUS "pybind11 found: Python bindings will be built")

    # The extension is installed alongside __init__.py, so `import sdrlora`
    # works from an installed tree.
    #
    # Destination follows GNU Radio's own convention -- its Python package
    # directory, relative to the prefix -- because grc/*.block.yml emits
    # `from gnuradio import sdrlora`. That import only resolves when this
    # lands inside the SAME gnuradio/ package directory as GNU Radio itself,
    # since a regular (non-namespace) package does not merge across
    # sys.path entries. In practice: configure with the prefix GNU Radio is
    # installed under (usually /usr) if you want the blocks in GRC.
    #
    # A USER-PREFIX INSTALL NEEDS TWO THINGS:
    #
    #   1. the extension lands in <prefix>/lib/python3/dist-packages/
    #      gnuradio/sdrlora, which is not on sys.path -- so that directory
    #      has to be on PYTHONPATH;
    #   2. it links libgnuradio-sdrlora, installed to CMAKE_INSTALL_LIBDIR,
    #      and CMake strips the build-tree RPATH on install, so without an
    #      install RPATH dlopen cannot find it outside a system prefix.
    #
    # (2) is handled below. Under --prefix=/usr neither applies: LIBDIR is
    # lib/<triplet>, which is in ld.so.cache, and GRC finds the package by
    # the path above.
    # Derived from where GNU Radio's OWN package sits, relative to GNU Radio's
    # prefix -- not from sysconfig. Debian patches sysconfig to a "posix_local"
    # scheme whose purelib is /usr/local/lib/python3.12/dist-packages while the
    # prefix is /usr, so a relpath against it yields a spurious leading
    # "local/" AND the wrong interpreter directory: the package would land in
    # <prefix>/local/lib/python3.12/dist-packages, which with --prefix=/usr
    # means /usr/local/... where GRC does not look. GNU Radio itself is in
    # /usr/lib/python3/dist-packages, so asking it is the only portable answer.
    if(NOT DEFINED GR_PYTHON_RELDIR)
        if(NOT DEFINED PYTHON_EXECUTABLE)
            find_program(PYTHON_EXECUTABLE NAMES python3 python)
        endif()
        execute_process(COMMAND gnuradio-config-info --prefix
                        OUTPUT_VARIABLE _gr_prefix
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                "import gnuradio,os;print(os.path.dirname(os.path.dirname(gnuradio.__file__)))"
                        OUTPUT_VARIABLE _gr_pypkg
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
        string(REGEX REPLACE "/+$" "" _gr_prefix "${_gr_prefix}")
        if(_gr_prefix AND _gr_pypkg)
            file(RELATIVE_PATH GR_PYTHON_RELDIR "${_gr_prefix}" "${_gr_pypkg}")
        endif()
    endif()
    if(NOT GR_PYTHON_RELDIR OR GR_PYTHON_RELDIR MATCHES "^\.\.")
        # Could not ask GNU Radio, or it lives outside its own prefix.
        set(GR_PYTHON_RELDIR "lib/python3/dist-packages")
        message(WARNING "could not locate GNU Radio's Python directory; "
                        "falling back to ${GR_PYTHON_RELDIR}")
    endif()
    message(STATUS "Python package directory: ${GR_PYTHON_RELDIR}/gnuradio/sdrlora")

    # RPATH, so the installed extension can find the library it links.
    # Computed rather than hardcoded as $ORIGIN/../../../..: LIBDIR is
    # "lib" at a user prefix and "lib/<triplet>" under /usr on Debian,
    # which are different depths from the package directory.
    file(RELATIVE_PATH _sdrlora_libdir_rel
         "${CMAKE_INSTALL_PREFIX}/${GR_PYTHON_RELDIR}/gnuradio/sdrlora"
         "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    set_target_properties(sdrlora_python PROPERTIES
        INSTALL_RPATH "$ORIGIN/${_sdrlora_libdir_rel}")
    message(STATUS "Python extension RPATH: "
                   "$ORIGIN/${_sdrlora_libdir_rel}")

    install(TARGETS sdrlora_python
            LIBRARY DESTINATION ${GR_PYTHON_RELDIR}/gnuradio/sdrlora)
    install(FILES python/sdrlora/__init__.py
                  python/sdrlora/packet_print.py
            DESTINATION ${GR_PYTHON_RELDIR}/gnuradio/sdrlora)
elseif(SDRLORA_ENABLE_GNURADIO)
    message(STATUS "pybind11 not found: skipping Python bindings")
endif()

message(STATUS "")
message(STATUS "gr-sdrlora configured. After building, run:  ctest --output-on-failure")
message(STATUS "")
