##
## Copyright (c) 2018-2022, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

project ( examples )

##  Examples are generally expected to support CPU code generation, and ultimately
##  most (if not all) will support GPU code generation also.  Call
##  manage_add_subdir with the directory name and True/False for building for
##  CPU and GPU.  Building for GPU will build for the GPU type (e.g., CUDA or
##  HIP) depending on the toolchain/compilers present

##                  subdir name   CPU       GPU

manage_add_subdir ( batch1ddft    TRUE      TRUE )
manage_add_subdir ( batch1dprdft  TRUE      TRUE )
manage_add_subdir ( mddft         TRUE      TRUE )
manage_add_subdir ( mdprdft       TRUE      TRUE )

manage_add_subdir ( rconv         TRUE      TRUE )
manage_add_subdir ( verify        TRUE      TRUE )

##  MPI examples depend on MPI being installed & accessable
##  Looked for MPI at top level CMake
if ( ${MPI_FOUND} AND NOT ${_codegen} STREQUAL "SYCL")
    ##  MPI installation found
    manage_add_subdir ( 3DDFT_mpi     FALSE     TRUE )
else ()
    message ( STATUS "MPI NOT found: No MPI examples will be built" )
endif ()
