##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required (VERSION 3.14)

##  Check the code generation option ...

if ( "X${_codegen}" STREQUAL "X" )
    message ( STATUS "_codegen is undefined, CPU assumed" )
    set ( _codegen CPU )
else ()
    message ( STATUS "_codegen = ${_codegen}" )
endif ()

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

##  Define CMake include directory and include the setup information
set ( FFTX_CMAKE_INCLUDE_DIR ${FFTX_PROJECT_SOURCE_DIR}/CMakeIncludes )
include ( "${FFTX_CMAKE_INCLUDE_DIR}/FFTXSetup.cmake" )

##  Setup CMAKE_INSTALL_RPATH to point to CMAKE_INSTALL_PREFIX/lib
set ( CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )
message ( STATUS "Defined: INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}, INSTALL_RPATH = ${CMAKE_INSTALL_RPATH}" )

##  Walk the following subdirectories...

add_subdirectory ( src )
add_subdirectory ( examples )

##  We will only build the documentation if the following conditions are satisfied:
##    FFTX_MAKE_DOCS is defined when cmake is invoked (e.g., cmake -DFFTX_MAKE_DOCS=True ..)
##    Sphinx is installed (also breathe & sphinx_rtd_theme)
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

