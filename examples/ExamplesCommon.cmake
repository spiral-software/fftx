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

    message ( STATUS "About to include ${FFTX_PROJECT_SOURCE_DIR}/CMakeIncludes/FFTXSetup.cmake" )
    
    ##  Setup / determine the necessary paths and variable to continue)
    set ( FFTX_CMAKE_INCLUDE_DIR ${FFTX_PROJECT_SOURCE_DIR}/CMakeIncludes )
    include ( "${FFTX_CMAKE_INCLUDE_DIR}/FFTXSetup.cmake" )

    message ( STATUS "FFTXSetup.cmake included; CMAKE_MINIMUM_REQUIRED_VERSION = ${CMAKE_MINIMUM_REQUIRED_VERSION}" ) 
endif ()

##  _codegen specifies CPU | CUDA | HIP code generation.  Will be set in parent.

if ( ${_codegen} STREQUAL "CUDA" )
    set ( _lang_add LANGUAGES CUDA )
endif ()
