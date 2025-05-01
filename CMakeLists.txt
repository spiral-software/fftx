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
         VERSION 1.0.3
                DESCRIPTION "FFTX Project"
                LANGUAGES C CXX )

##  --------------------------------------------------------------------------------
##  Try to explicitly find Fortran compiler by name.
##  Do this *early* to avoid issues on Windows with Intel ifx...
##  --------------------------------------------------------------------------------

set ( Fortran_ENABLED FALSE )
find_program ( FORTRAN_COMPILER NAMES ifx mpifort )   ##  gfortran )
if ( FORTRAN_COMPILER )
    message ( STATUS "Found Fortran compiler candidate: ${FORTRAN_COMPILER}" )
    ##  Tell CMake to use our candidate
    set ( CMAKE_Fortran_COMPILER "${FORTRAN_COMPILER}" CACHE FILEPATH "Fortran compiler" FORCE )
    enable_language ( Fortran OPTIONAL )
    
    if ( CMAKE_Fortran_COMPILER_WORKS )
        message ( STATUS "Working Fortran compiler detected: ${CMAKE_Fortran_COMPILER}" )
        set ( Fortran_ENABLED TRUE )
    else ()
        message ( STATUS "Fortran compiler found (${CMAKE_Fortran_COMPILER}), but it does not work with current generator." )
        set ( Fortran_ENABLED FALSE )
    endif ()
else ()
    message ( STATUS "No Fortran compiler found. Fortran support disabled." )
endif ()

if ( WIN32 )
    set ( FINDMPIFLAG NO_MODULE )
else ()
    set ( FINDMPIFLAG )
endif ()

##  --------------------------------------------------------------------------------
##  Intel's ifx compiler introduces some invalid flags for an MSVC linker
##  Clean them up and force it to use MSVC linker [ifx seems to assume it will
##  compile and link]
##  --------------------------------------------------------------------------------

if ( WIN32 AND CMAKE_Fortran_COMPILER MATCHES "ifx" )
    message ( STATUS "Windows + Intel ifx detected - clearing bad /Qoption,link flags" )

    ##  Override linker executable to prevent ifx injecting /Qoption,link
    set ( CMAKE_LINKER "link" CACHE FILEPATH "Use MSVC linker instead of ifx" FORCE )

    ##  Clean the linker flags
    foreach ( var IN ITEMS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS )
        if ( DEFINED ${var} )
            string ( REGEX REPLACE "/Qoption,link,[^ ]+" "" cleaned_flag "${${var}}" )
            set ( ${var} "${cleaned_flag}" CACHE STRING "" FORCE )
            message ( STATUS "${var} cleaned: ${${var}}" )
        endif ()
    endforeach ()

    ##  Make sure Fortran linking is done via MSVC link.exe directly
    set ( CMAKE_Fortran_LINK_EXECUTABLE "<CMAKE_Fortran_COMPILER> <FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>" )

    ##  set ( CMAKE_LINKER "link.exe" CACHE STRING "Use MSVC linker with Intel ifx/icx" FORCE )
    ##  set ( CMAKE_CXX_LINK_EXECUTABLE    "<CMAKE_LINKER> <FLAGS> <OBJECTS> -out:<TARGET> <LINK_LIBRARIES>" )
    message ( STATUS "CMAKE_Fortran_LINK_EXECUTABLE overridden to avoid ifx driving linking" )

    ##  Silence MSVC warnings (C/C++)
    add_compile_options ( /W0 /w /nologo )

    ##  Silence Intel ifx Fortran warnings
    set ( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} /warn:none /Qdiag-disable:remark /nologo" )

    ##  Allow find ( MPI ) to access non standard module
    set ( FINDMPIFLAG )
endif ()

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

# if ( MSVC )
#     message ( STATUS "MSVC detected -- suppressing all warnings" )

#     ##  Silence MSVC warnings (C/C++)
#     add_compile_options ( /W0 /w /nologo )

#     ##  Silence Intel ifx Fortran warnings
#     set ( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} /warn:none /Qdiag-disable:remark /nologo" )
# endif ()

##  NOTE: We use ADDL_COMPILE_FLAGS to hold additional ad-hoc compile time
##  flags.  Any child cmake lists should APPEND to the list (i.e., don't assume
##  it is empty).

##  Check if we're on Apple/MAC
if ( APPLE )
    ##  Check if arch should be x86_64 or arm64
    message ( STATUS "Running on Apple / MAC" )
    message ( STATUS "CMAKE_OSX_ARCHITECTURES = ${CMAKE_OSX_ARCHITECTURES}" )
    message ( STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}" )
    if ( ${CMAKE_OSX_ARCHITECTURES} MATCHES "arm64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64.*" )
	set ( ADDL_COMPILE_FLAGS -arch arm64 )
        message ( STATUS "Set addl compile flags: ${ADDL_COMPILE_FLAGS}" )
    elseif ( ${CMAKE_OSX_ARCHITECTURES} MATCHES "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64.*")
	set ( ADDL_COMPILE_FLAGS -arch x86_64 )
        message ( STATUS " set addl compile flags: ${ADDL_COMPILE_FLAGS}" )
    endif ()
endif ()	

##  To enable debug printing for the APIs uncomment the next line:
##  list ( APPEND ADDL_COMPILE_FLAGS -DFFTX_PRINTDEBUG )

##  options.cmake (automatically generated by config-fftx-libs.sh) defines the libraries to be built
##  Will exist when FFTX is being built normally -- but may not if just trying to install documentation
if ( EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/options.cmake" )
    message ( STATUS "Found file options.cmake")
    include ( "${CMAKE_CURRENT_SOURCE_DIR}/options.cmake" )
else ()
    message ( STATUS "No such file: options.cmake")
endif ()

##  Define CMake include directory and include the setup information
set ( FFTX_CMAKE_INCLUDE_DIR ${FFTX_PROJECT_SOURCE_DIR}/CMakeIncludes )
include ( "${FFTX_CMAKE_INCLUDE_DIR}/FFTXSetup.cmake" )

##  Setup CMAKE_INSTALL_RPATH to point to CMAKE_INSTALL_PREFIX/lib
set ( CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )
message ( STATUS "Defined: INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}, INSTALL_RPATH = ${CMAKE_INSTALL_RPATH}" )

##  MPI library and example depend on MPI being installed & accessable

if ( DEFINED ENV{MPI_HOME} )
    message ( STATUS "MPI_HOME = $ENV{MPI_HOME}" )
    ##  Append MPI_HOME to CMAKE_PREFIX_PATH
    list ( APPEND CMAKE_PREFIX_PATH $ENV{MPI_HOME} )
    message ( STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}" )
endif ()

message ( STATUS "Try to find an MPI installation" )
find_package ( MPI QUIET ${FINDMPIFLAG} )
if ( ${MPI_FOUND} )
    message ( STATUS "MPI installation found" )
    ##  MPI installation found
    setup_mpi_variables ()
else ()
    message ( STATUS "MPI NOT found: No MPI examples will be built" )
endif ()

##  Walk the following subdirectories...

add_subdirectory ( src )

if ( BUILD_EXAMPLES )
    ##  Build the FFTX example programs -- traverse the examples sub-directlry
    add_subdirectory ( examples )
else ()
    message ( "Build examples option = ${BUILD_EXAMPLES}; no example programs will be built" )
endif ()

add_subdirectory ( CMakeIncludes )

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

