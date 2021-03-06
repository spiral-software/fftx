##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

if ( "x${CMAKE_MINIMUM_REQUIRED_VERSION}" STREQUAL "x" )
    ##  CMake minimum version is not defined -- which means this cmake was invoked
    ##  standalone (i.e., via the buildexamples script).
    ##  FFTX_PROJECT_SOURCE_DIR must be set, get it from FFTX_PROJ_DIR which must be
    ##  (either environment or command line)
    if ( DEFINED ENV{FFTX_PROJ_DIR} )
	message ( STATUS "FFTX_PROJ_DIR = $ENV{FFTX_PROJ_DIR}" )
	set ( FFTX_PROJECT_SOURCE_DIR $ENV{FFTX_PROJ_DIR} )
    else ()
	if ( "x${FFTX_PROJ_DIR}" STREQUAL "x" )
	    message ( FATAL_ERROR "FFTX_PROJ_DIR variable must be defined" )
	endif ()
	set ( FFTX_PROJECT_SOURCE_DIR ${FFTX_PROJ_DIR} )
    endif ()

    ##  Languages aren't defined yet so add c & c++
    set ( _lang_base C CXX )

    ##  _codegen will be passed on the cmake command line: will be one of CPU | GPU | HIP
    ##  set ( _codegen GPU )

    ##  Setup / determine the necessary paths and variable to continue)
    set ( FFTX_CMAKE_INCLUDE_DIR ${FFTX_PROJECT_SOURCE_DIR}/CMakeIncludes )
    include ( "${FFTX_CMAKE_INCLUDE_DIR}/FFTXSetup.cmake" )
endif ()

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  _codegen specifies CPU or GPU (serial or CUDA) code generation.  Will be set
##  in parent so don't change here.

if ( ${_codegen} STREQUAL "GPU" )
    set ( _lang_add LANGUAGES CUDA )
endif ()

##  ===== For most examples you should not need to modify anything ABOVE this line =====

##  Set the project name.  Preferred name is just the *name* of the example folder 
project ( compare_cufft ${_lang_add} ${_lang_base} )

##  File nameing convention is: <prefix>.<stem>.xxxxx; set _stem and _prefix(es)
##  Test program name is:  test<project name>.{cpp|cu}

set ( _stem fftx )
set ( _prefixes spiralmddft )
set ( BUILD_PROGRAM test${PROJECT_NAME} )

##  ===== For most examples you should not need to modify anything BELOW this line =====

manage_deps_codegen ( ${_codegen} ${_stem} "${_prefixes}" )
list ( APPEND _all_build_srcs ${BUILD_PROGRAM}.${_suffix} )

add_executable   ( ${BUILD_PROGRAM} ${_all_build_srcs} )
add_dependencies ( ${BUILD_PROGRAM} ${_all_build_deps} )

target_compile_options ( ${BUILD_PROGRAM} PRIVATE ${ADDL_COMPILE_FLAGS} )

target_include_directories ( ${BUILD_PROGRAM} PRIVATE ${${PROJECT_NAME}_BINARY_DIR} ${CMAKE_BINARY_DIR} )

target_link_libraries      ( ${BUILD_PROGRAM} ${LIBS_FOR_CUDA} )

set ( INSTALL_DIR_TARGET ${CMAKE_BINARY_DIR}/bin )

install ( TARGETS ${BUILD_PROGRAM} DESTINATION ${INSTALL_DIR_TARGET} )

