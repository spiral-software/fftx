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

##  Setup dependent options for those examples that depend on FFTX libraries.  If a
##  pre-requisite library is not configured and built then don't attempt to build that
##  example (NOTE: not all examples depend on libraries)

include ( CMakeDependentOption )
cmake_dependent_option ( BLD_COMPARE_EX "Build compare example" ON "MDDFT_LIB;MDPRDFT_LIB" OFF )
cmake_dependent_option ( BLD_RCONV_EX "Build rconv example" ON "RCONV_LIB" OFF )
cmake_dependent_option ( BLD_VERIFY_EX "Build verify example" ON "MDDFT_LIB;MDPRDFT_LIB" OFF )

##                  subdir name   CPU       GPU

if ( BLD_COMPARE_EX )
    manage_add_subdir ( compare       FALSE     TRUE )
endif ()

manage_add_subdir ( mddft         TRUE      TRUE )
manage_add_subdir ( mdprdft         TRUE      TRUE )
manage_add_subdir ( hockney       TRUE      FALSE )

manage_add_subdir ( rconv         TRUE      TRUE )

if ( BLD_VERIFY_EX )
    manage_add_subdir ( verify        TRUE      TRUE )
endif ()

manage_add_subdir ( warpx         TRUE      FALSE )
##  manage_add_subdir ( warpx_basic   FALSE     TRUE )
manage_add_subdir ( compare_cufft FALSE     TRUE )

##  MPI examples depend on MPI being installed & accessable
##  Looked for MPI at top level CMake
if ( ${MPI_FOUND} )
    ##  MPI installation found
        manage_add_subdir ( 3DDFT_mpi     FALSE     TRUE )
else ()
    message ( STATUS "MPI NOT found: No MPI examples will be built" )
endif ()
