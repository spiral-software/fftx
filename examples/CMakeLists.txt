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
##  CPU and GPU.  Building for GPU will build for the GPU type (e.g., CUDA or
##  HIP) depending on the toolchain/compilers present

##                  subdir name   CPU       GPU

set ( _library_includes library)
manage_add_subdir ( library       TRUE      TRUE )

message ( STATUS "Library include dirs = ${_library_includes}" )
message ( STATUS "Library names = ${_library_names}" )

include_directories ( ${_library_includes} )

manage_add_subdir ( compare       FALSE     TRUE )

manage_add_subdir ( mddft         TRUE      TRUE )
manage_add_subdir ( hockney       TRUE      FALSE )
manage_add_subdir ( rconv         TRUE      TRUE )
manage_add_subdir ( verify        TRUE      TRUE )
manage_add_subdir ( warpx         TRUE      FALSE )
##  manage_add_subdir ( warpx_basic   FALSE     TRUE )
manage_add_subdir ( compare_cufft FALSE     TRUE )

##  MPI examples depend on MPI being installed & accessable

find_package ( MPI COMPONENTS MPICXX )
if ( ${MPI_FOUND} )
    ##  MPI installation found
    setup_mpi_variables ()

    if ( ${_codegen} STREQUAL "HIP" )
	message ( STATUS "MPI example is not built for HIP" )
    else ()
	manage_add_subdir ( MPI/3DFFT     FALSE     FALSE )
    endif ()
else ()
    message ( STATUS "MPI NOT found: No MPI examples will be built" )
endif ()
