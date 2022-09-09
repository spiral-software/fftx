##
## Copyright (c) 2018-2022, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

project ( src )

##  The main folder with source code is the 'library' folder, from which the various transform libraries are generated.

##  All examples are expected to support CPU code generation, and ultimately
##  most (if not all) will support GPU code generation also.  Call
##  manage_add_subdir with the directory name and True/False for building for
##  CPU and GPU.  Building for GPU will build for the GPU type (e.g., CUDA or
##  HIP) depending on the toolchain/compilers present

##                  subdir name   CPU       GPU

set ( _library_includes ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/library )
include_directories ( ${_library_includes} )

manage_add_subdir ( library       TRUE      TRUE )

##  message ( STATUS "Library include dirs = ${_library_includes}" )
##  message ( STATUS "Library names = ${_library_names}" )

set ( _library_names ${_library_names} PARENT_SCOPE )
set ( _library_includes ${_library_includes} PARENT_SCOPE )

manage_add_subdir ( include       TRUE      TRUE )
