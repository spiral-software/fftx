##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

##
## First set cube-sizes.txt with list of domain sizes.
## 
## Thse environment variables need to be set:
## SPIRAL_HOME for root Spiral directory
## FFTX_HOME for root FFTX directory
##
## Then do (with HIP instead of CUDA for HIP):
## python gen_files.py fftx_mddft CUDA
## python gen_files.py fftx_imddft CUDA
## python gen_files.py fftx_mdprdft CUDA
## python gen_files.py fftx_imdprdft CUDA
## python gen_files.py fftx_rconv CUDA
##
## cmake -S . -B builddir
## cmake --build builddir --target install
## For HIP, to that last line, add: -DCMAKE_CXX_COMPILER=hipcc
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.18 )

## The list of libraries to build.
set ( FFTX_LIB_LIST fftx_mddft fftx_imddft fftx_mdprdft fftx_imdprdft fftx_rconv )

set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )

set ( PROJECT fftxlib )

if ( "${CMAKE_CXX_COMPILER}" STREQUAL "hipcc" )
    ##  HIP flavour
    set ( _be_hip ON )
    set ( _add_lang )
    set ( _suffix .cpp )
else ()
    ##  CUDA flavour
    set ( _be_hip OFF )
    set ( _add_lang CUDA )
    set ( _suffix .cu )
endif ()

project ( ${PROJECT}
         VERSION 1.0.1
         DESCRIPTION "Library of pre-compiled FFTX transforms"
         LANGUAGES C CXX ${_add_lang} )

##  Set C standard 11
set ( CMAKE_C_STANDARD 11 )
set ( CMAKE_CXX_STANDARD 11 )

if ( ${_be_hip} )
    ##  Add additional paths for CMake to find HIP
    list ( APPEND CMAKE_PREFIX_PATH
           /opt/rocm/hip /opt/rocm ) ##  /opt/rocm-4.2.0/hip

    find_package ( hip REQUIRED )
    if (${hip_FOUND})
	##  HIP found
	message ( STATUS "Found HIP: Version = ${hip_VERSION}")
    else ()
	message ( SEND_ERROR "HIP NOT FOUND: HIP is required to build/run examples")
    endif ()

    ##  Need to add $ROCM_PATH for includes and libraries
    if ( DEFINED ENV{ROCM_PATH} )
	message ( STATUS "ROCM_PATH is defined: $ENV{ROCM_PATH}" )
	include_directories ( $ENV{ROCM_PATH}/hipfft/include
                              $ENV{ROCM_PATH}/include )
    endif ()
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

##  Get FFTX home...

if ( DEFINED ENV{FFTX_HOME} )
    message ( STATUS "FFTX_HOME = $ENV{FFTX_HOME}" )
    set ( FFTX_SOURCE_DIR $ENV{FFTX_HOME} )
else ()
    if ( "x${FFTX_HOME}" STREQUAL "x" )
        message ( FATAL_ERROR "FFTX_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( FFTX_SOURCE_DIR ${FFTX_HOME} )
endif ()

## Will include library source directories later.
include_directories ( ${CMAKE_SOURCE_DIR}
                      ${CMAKE_BINARY_DIR}
                      ${SPIRAL_SOURCE_DIR}/profiler/targets/include
                      ${FFTX_SOURCE_DIR}/include )

##  Set up compilation flags

if (WIN32)
    if ( ${_be_hip} )
	set ( HIP_COMPILE_FLAGS -DFFTX_HIP )
	set ( LIBS_FOR_HIP hipfft rocfft )
    else ()
	set ( GPU_COMPILE_DEFNS -Xptxas -v -maxrregcount=64 )  ## -Xptxas -v
	set ( LIBS_FOR_CUDA cufft )
    endif ()
    list ( APPEND ADDL_COMPILE_FLAGS -DWIN64 )
else ()
    if ( ${_be_hip} )
	set ( HIP_COMPILE_FLAGS -DFFTX_HIP )
	set ( LIBS_FOR_HIP hipfft rocfft )
    else ()
	set ( GPU_COMPILE_DEFNS -Xptxas -v -maxrregcount=64 )
	set ( CUDA_COMPILE_FLAGS -m64 -rdc=false )
	set ( LIBS_FOR_CUDA cufft culibos )
    endif ()
    list ( APPEND ADDL_COMPILE_FLAGS )
endif ()

if ( NOT ${_be_hip} )
    ##  set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 )
    set ( CMAKE_CUDA_ARCHITECTURES 70 )
endif ()

foreach ( FFTX_LIB_NAME ${FFTX_LIB_LIST} )
   set ( FFTX_LIB_SRC_DIR lib_${FFTX_LIB_NAME}_srcs )
   set ( FFTX_LIB_TARGET ${FFTX_LIB_NAME}_precomp )

   include_directories ( ${CMAKE_SOURCE_DIR}/${FFTX_LIB_SRC_DIR} )

   ##  Walk this subdirectory for the source files to add to the library
   add_subdirectory ( ${FFTX_LIB_SRC_DIR} )

   ## add_library ( ${FFTX_LIB_TARGET} SHARED )

   if (WIN32)
       target_link_directories ( ${FFTX_LIB_TARGET} PUBLIC
          ${CMAKE_BINARY_DIR}/${FFTX_LIB_SRC_DIR}/${CMAKE_BUILD_TYPE} )
   else ()
       target_link_directories ( ${FFTX_LIB_TARGET} PUBLIC
          ${CMAKE_BINARY_DIR}/${FFTX_LIB_SRC_DIR} )
   endif ()

   if ( ${_be_hip} )
       target_compile_options     ( ${FFTX_LIB_TARGET} PRIVATE
                                    ${HIP_COMPILE_FLAGS}
                                    ${ADDL_COMPILE_FLAGS} )
       target_link_directories    ( ${FFTX_LIB_TARGET} PRIVATE
                                    $ENV{ROCM_PATH}/lib )
       target_link_libraries      ( ${FFTX_LIB_TARGET}
                                    hip::host hip::device ${LIBS_FOR_HIP} )
   else ()
       target_compile_options     ( ${FFTX_LIB_TARGET} PRIVATE
                                    ${CUDA_COMPILE_FLAGS}
                                    ${GPU_COMPILE_DEFNS} )
       set_property        ( TARGET ${FFTX_LIB_TARGET}
                             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
       target_link_libraries      ( ${FFTX_LIB_TARGET}
                                    ${LIBS_FOR_CUDA} )
   endif ()

endforeach()
