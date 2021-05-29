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
    project ( FFTX_DUMMY LANGUAGES ${_lang_base} )
    
    ##  Setup / determine the necessary paths and variable to continue)
    set ( FFTX_CMAKE_INCLUDE_DIR ${FFTX_PROJECT_SOURCE_DIR}/CMakeIncludes )
    include ( "${FFTX_CMAKE_INCLUDE_DIR}/FFTXSetup.cmake" )
endif ()

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  _codegen specifies CPU | GPU | HIP code generation.  Will be set in parent.

if ( ${_codegen} STREQUAL "GPU" )
    set ( _lang_add LANGUAGES CUDA )
endif ()

##  ===== For most examples you should not need to modify anything ABOVE this line =====

##  Set the project name.  Preferred name is just the *name* of the example folder 
project ( mddft ${_lang_add} ${_lang_base} )

##  File nameing convention is: <prefix>.<stem>.xxxxx; set _stem and _prefix(es)
##  Test program name is:  test<project name>.{cpp|cu}

set ( _stem fftx )
set ( _prefixes mddft imddft )
set ( BUILD_PROGRAM test${PROJECT_NAME} )

##  ===== For most examples you should not need to modify anything BELOW this line =====

manage_deps_codegen ( ${_codegen} ${_stem} "${_prefixes}" )
add_includes_libs_to_target ( ${BUILD_PROGRAM} ${_stem} "${_prefixes}" )
