##
##  Copyright (c) 2018-2025, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information.
##

include ( ../ExamplesCommon.cmake )

##  Set the minimum CMake version
cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  ===== For most examples you should not need to modify anything ABOVE this line =====

##  Set the project name - usually the same as the folder name
project ( fortran ${_lang_add} Fortran )

##  Put Fortran .mod files in a named directory, to help cmake better serialize
set ( CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
file ( MAKE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY} )

##  Set suffix to F90 for Fortran files
set ( _desired_suffix F90 )

set ( _stem fftx )
set ( _prefixes  )
set ( BUILD_PROGS fortran_main )

##  List of Fortran source files required for the example
set ( FORTRAN_SOURCES
    math_constants_mod.F90
    mpi_utils_mod.F90
    problem_dimensions_mod.F90
    fft_mod.F90
    data_functions_mod.F90
    transform_tests_mod.F90
    convolution_mod.F90
)

##  The C++ file that needs to be compiled as part of the example
set ( CPP_SOURCES test.cpp )

##  Get around problem with MPI calls on different types.
include ( CheckFortranCompilerFlag )
check_fortran_compiler_flag ( "-fallow-argument-mismatch" _fallow-argument-mismatch )
if ( _fallow-argument-mismatch )
    set ( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch" )
endif ()

check_fortran_compiler_flag ( "-fPIC" _fPIC )
if ( _fPIC )
    set ( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC" )
endif ()

##  Get around SYCL problem linking C++ and Fortran code.
if ( ${_codegen} STREQUAL "SYCL" )
   string ( REPLACE "CMAKE_CXX_COMPILER" "CMAKE_Fortran_COMPILER" _new_link_executable ${CMAKE_CXX_LINK_EXECUTABLE} )
   set ( CMAKE_CXX_LINK_EXECUTABLE "${_new_link_executable}" )
endif ()

list ( APPEND _all_build_srcs ${FORTRAN_SOURCES} ${CPP_SOURCES} )
message ( STATUS "_all_build_srcs: ${_all_build_srcs}" )

##  ===== For most examples you should not need to modify anything BELOW this line =====

##  Loop through the programs defined in BUILD_PROGS
foreach ( _prog ${BUILD_PROGS} )
    ##  Specify Fortran as the language for Fortran source files
    set_source_files_properties ( ${FORTRAN_SOURCES} PROPERTIES LANGUAGE Fortran )

    if ( ${_codegen} STREQUAL "HIP" )
        set_source_files_properties ( ${CPP_SOURCES} PROPERTIES LANGUAGE CXX )
    elseif ( ${_codegen} STREQUAL "CUDA" )
        set_source_files_properties ( ${CPP_SOURCES} PROPERTIES LANGUAGE CUDA )
    endif ()

    ##  Manage dependencies and add include directories/libraries for the program
    manage_deps_codegen ( ${_codegen} ${_stem} "${_prefixes}" )
    add_includes_libs_to_target ( ${_prog} ${_stem} "${_prefixes}" )
    fort_add_mpi_decorations_to_target ( ${_prog} )

    ##  Use JOB_POOL "fortran_pool" to ensure Fortran code building is limited to single job
    set_property ( TARGET ${_prog} PROPERTY JOB_POOL_COMPILE fortran_pool )

    ##  Tell cmake where .mod files are written to better manage Fortran dependencies
    set_property ( TARGET ${_prog} PROPERTY Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY} )

    if ( WIN32 AND CMAKE_Fortran_COMPILER MATCHES "ifx" )
        target_link_libraries ( ${_prog} PRIVATE ucrt.lib msvcrt.lib vcruntime.lib oldnames.lib )
        message ( STATUS "Linked MSVC CRTs explicitly for Fortran target to satisfy Intel LLVM ifx" )
    endif ()

    get_target_property ( LINKED_LIBS ${_prog} LINK_LIBRARIES )
    message ( STATUS "Libraries linked to target ${_prog}: ${LINKED_LIBS}" )

endforeach ()
