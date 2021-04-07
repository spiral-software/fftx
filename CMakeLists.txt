##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required (VERSION 3.14)

set (CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

project (FFTX_PROJECT
         VERSION 0.9.1
                DESCRIPTION "FFTX Project"
                LANGUAGES C CXX )

##  Prevent building directly into the source tree
string (COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" DISALLOW_COMPILE_INPLACE)
if ( DISALLOW_COMPILE_INPLACE )
    message (FATAL_ERROR "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
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
        message ( FATAL_ERROR "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
endif ()

##  Find python3 -- used to marshall/run examples

find_package (Python3 COMPONENTS Interpreter)
if (${Python3_FOUND})
    ##  It exists, executable is ${Python3_EXECUTABLE}
    message ( STATUS "Found Python3: Version = ${Python3_VERSION}, Executable = ${Python3_EXECUTABLE}")
else ()
    message ( SEND_ERROR "Python3 NOT FOUND: Python is required to build/run examples")
endif ()

set ( SPIRAL_INCLUDE_PATH ${SPIRAL_SOURCE_DIR}/config/CMakeIncludes )
include ("${SPIRAL_INCLUDE_PATH}/RunSpiral.cmake")

set ( FFTX_CMAKE_INCLUDE_DIR ${FFTX_PROJECT_SOURCE_DIR}/CMakeIncludes )
set ( BACKEND_SOURCE_DIR ${FFTX_PROJECT_SOURCE_DIR}/examples/backend )

include ( "${FFTX_CMAKE_INCLUDE_DIR}/FFTXCmakeFunctions.cmake" )

##  Check the code generation option ...

if ( "X${_codegen}" STREQUAL "X" )
    message ( STATUS "_codegen is undefined, CPU assumed" )
    set ( _codegen CPU )
else ()
    message ( STATUS "_codegen = ${_codegen}" )
endif ()

set ( ADDL_COMPILE_FLAGS -D_USE_MATH_DEFINES )
if ( ${_codegen} STREQUAL "GPU" )
    if (WIN32)
	set ( CUDA_COMPILE_FLAGS -rdc=false )
	set ( GPU_COMPILE_DEFNS )			## -Xptxas -v
	set ( LIBS_FOR_CUDA cufft )
	set ( ADDL_COMPILE_FLAGS -DWIN64 -D_USE_MATH_DEFINES )
    else ()
	set ( CUDA_COMPILE_FLAGS -m64 -rdc=true )
	set ( GPU_COMPILE_DEFNS -dc )		## -Xptxas -v
	set ( LIBS_FOR_CUDA cufft_static culibos )
	set ( ADDL_COMPILE_FLAGS )
    endif ()

    set ( CMAKE_CUDA_ARCHITECTURES 52 )
    set ( GPU_EXTRAS _CUDAGEN )
endif ()

set ( FFTX_INCLUDE ${FFTX_PROJECT_SOURCE_DIR}/include )
set ( CMAKE_C_STANDARD 14)
set ( CMAKE_CXX_STANDARD 14)

include_directories ( ${FFTX_INCLUDE} ${SPIRAL_SOURCE_DIR}/profiler/targets )

if ( (NOT DEFINED CMAKE_BUILD_TYPE) OR (NOT CMAKE_BUILD_TYPE) )
    set ( CMAKE_BUILD_TYPE Release )
endif ()

add_subdirectory ( src )
add_subdirectory ( examples )

##  We will only build the documentation if the following conditions are satisfied:
##    FFTX_MAKE_DOCS is defined when cmake is invoked (e.g., cmake -DFFTX_MAKE_DOCS=True ..)
##    Sphinx is installed (also breathe & sp[hinx_rtd_theme)
##    Doxygen is installed

if ( "X${FFTX_MAKE_DOCS}X" STREQUAL "XX" OR NOT FFTX_MAKE_DOCS )
    message ( STATUS "FFTX_MAKE_DOCS is not defined, do NOT build documentation" )
else ()
    ##  Add the CMakeIncludes folder so the FindSphinx module is found
    set ( CMAKE_MODULE_PATH "${SPIRAL_SOURCE_DIR}/config/CMakeIncludes" ${CMAKE_MODULE_PATH} )

    find_package ( Sphinx )		##  don't make it REQUIRED; skip docs if not found
    if ( ${Sphinx_FOUND} )
	##  Check for Doxygen -- also required to build docs 
	find_package ( Doxygen )	##  don't make it REQUIRED; skip docs if not found
	if ( ${Doxygen_FOUND} )
	    message ( STATUS "Found Sphinx & Doxygen, add subdirectory 'docs' to build documentation" )
	    add_subdirectory (docs)
	else ()
	    message ( STATUS "Sphinx found, Doxygen not found -- skipping 'docs'" )
	endif ()
    else ()
	message ( STATUS "Sphinx not found -- skipping 'docs', do NOT build documentation" )
    endif ()
endif ()

enable_testing()

