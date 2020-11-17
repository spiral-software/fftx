##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required (VERSION 3.14)

set (CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

project (FFTX_PROJECT
         VERSION 1.0.1
                DESCRIPTION "FFTX Project"
                LANGUAGES C CXX )

##  Prevent building directly into the source tree
string (COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" DISALLOW_COMPILE_INPLACE)
if ( DISALLOW_COMPILE_INPLACE )
    message (FATAL_ERROR, "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
                 rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
                 mkdir <build>
                 cd <build>
                 cmake ..
                 make | make install")
endif ()

##  Get SPIRAL home...

if ( DEFINED ENV{SPIRAL_HOME} )
    message ( STATUS "SPIRAL_HOME = $ENV{SPIRAL_HOME}" )
    set ( SPIRAL_SOURCE_DIR $ENV{SPIRAL_HOME} )
else ()
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
        message ( FATAL "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
endif ()

include ("${SPIRAL_SOURCE_DIR}/config/CMakeIncludes/RunSpiral.cmake")

set ( FFTX_CMAKE_INCLUDE_DIR ${FFTX_PROJECT_SOURCE_DIR}/CMakeIncludes )
set ( BACKEND_SOURCE_DIR ${FFTX_PROJECT_SOURCE_DIR}/examples/backend )

include ("${FFTX_CMAKE_INCLUDE_DIR}/run_driver_program.cmake")
include ("${FFTX_CMAKE_INCLUDE_DIR}/create_generator_file.cmake")

set ( FFTX_INCLUDE ${FFTX_PROJECT_SOURCE_DIR}/include )
set ( CMAKE_C_STANDARD 14)
set ( CMAKE_CXX_STANDARD 14)

include_directories ( ${FFTX_INCLUDE} ${SPIRAL_SOURCE_DIR}/profiler/targets )

if ( (NOT DEFINED CMAKE_BUILD_TYPE) OR (NOT CMAKE_BUILD_TYPE) )
    set ( CMAKE_BUILD_TYPE Release )
endif ()

add_subdirectory ( src )
add_subdirectory ( examples )

##  add_subdirectory ( test )

enable_testing()

