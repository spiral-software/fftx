##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

project ( examples )

##  All examples are expected to support CPU code generation, and ultimately
##  most (if not all) will support GPU code generation also.  Call
##  manage_add_subdir with the directory name and True/False for building for
##  CPU and GPU

##                  subdir name   CPU       GPU

manage_add_subdir ( mddft         TRUE      TRUE )
manage_add_subdir ( hockney       TRUE      FALSE )
manage_add_subdir ( rconv         TRUE      TRUE )
manage_add_subdir ( verify        TRUE      TRUE )
manage_add_subdir ( warpx         TRUE      FALSE )
manage_add_subdir ( warpx_basic   FALSE     TRUE )
manage_add_subdir ( compare_cufft FALSE     TRUE )

##  MPI examples depend on MPI being installed & accessable

find_package ( MPI COMPONENTS MPICXX )
if ( ${MPI_FOUND} )
    ##  MPI installation found
    message ( STATUS "Found MPI: mpicxx = ${MPI_MPICXX_FOUND}, Version = ${MPI_CXX_VERSION}" )
    ##  message ( STATUS "MPI_CXX_FOUND = ${MPI_CXX_FOUND}" )
    message ( STATUS "MPI_CXX_COMPILER = ${MPI_CXX_COMPILER}" )
    ##  message ( STATUS "MPI_CXX_COMPILE_OPTIONS = ${MPI_CXX_COMPILE_OPTIONS}" )
    ##  message ( STATUS "MPI_CXX_COMPILE_DEFINITIONS = ${MPI_CXX_COMPILE_DEFINITIONS}" )
    ##  message ( STATUS "MPI_CXX_INCLUDE_DIRS = ${MPI_CXX_INCLUDE_DIRS}" )
    message ( STATUS "MPI_CXX_LINK_FLAGS = ${MPI_CXX_LINK_FLAGS}" )
    message ( STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}" )
    ##  message ( STATUS "MPI_CXX_VERSION = ${MPI_CXX_VERSION}" )

    manage_add_subdir ( MPI/3DFFT     FALSE     TRUE )

else ()
    message ( STATUS "MPI NOT found: No MPI examples will be built" )
endif ()
