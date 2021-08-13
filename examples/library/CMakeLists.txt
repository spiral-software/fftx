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
## python gen_files_point.py fftx_mddft CUDA
## python gen_files_point.py fftx_imddft CUDA
## python gen_files_point.py fftx_mdprdft CUDA
## python gen_files_point.py fftx_imdprdft CUDA
## python gen_files_point.py fftx_rconv CUDA
## 
## mkdir build ; pushd build
## Then, for just the libraries, adding for HIP, -DCMAKE_CXX_COMPILER=hipcc
## cmake .. -DPROJECT=dummy
## Or these applications:
## cmake .. -DPROJECT=testrconv
## cmake .. -DPROJECT=testcompare_device
## cmake .. -DPROJECT=testcompare_device -DCMAKE_CXX_COMPILER=hipcc
## cmake .. -DPROJECT=testverify
## 
## Then:
## make install
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.18 )

set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )
# set ( PROJECT fftxlib )  ##  driver to test library of pre-compiled mddft's

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
                DESCRIPTION "Driver to test library of pre-compiled FFTX transforms"
                LANGUAGES C CXX ${_add_lang} )

##  Set C standard 11
set ( CMAKE_C_STANDARD 11 )
set ( CMAKE_CXX_STANDARD 11 )

if ( ${_be_hip} )
    ##  Add additional paths for CMake to find HIP
    list ( APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm )	##  /opt/rocm-4.2.0/hip

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

include_directories ( ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
                      ${CMAKE_SOURCE_DIR}/lib_fftx_mddft_srcs
                      ${CMAKE_SOURCE_DIR}/lib_fftx_imddft_srcs
                      ${CMAKE_SOURCE_DIR}/lib_fftx_mdprdft_srcs
                      ${CMAKE_SOURCE_DIR}/lib_fftx_imdprdft_srcs
                      ${CMAKE_SOURCE_DIR}/lib_fftx_rconv_srcs
                      ${SPIRAL_SOURCE_DIR}/profiler/targets/include
                      ${FFTX_SOURCE_DIR}/include )

##  Setup compilation flags

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

##  Walk the following subdirectory for the source files to add to the library
add_subdirectory ( lib_fftx_mddft_srcs )   ## lib-srcs
add_subdirectory ( lib_fftx_imddft_srcs )
add_subdirectory ( lib_fftx_mdprdft_srcs )
add_subdirectory ( lib_fftx_imdprdft_srcs )
add_subdirectory ( lib_fftx_rconv_srcs )

## set                        ( _lib_precomp mddft3d_precomp )
set                        ( _lib_precomp fftx_mddft_precomp fftx_imddft_precomp fftx_mdprdft_precomp fftx_imdprdft_precomp fftx_rconv_precomp )
set                        (   EXE_TARGET  ${PROJECT} )
add_executable		   ( ${EXE_TARGET} ${PROJECT}${_suffix} )
if (WIN32)
    target_link_directories    ( ${EXE_TARGET} PRIVATE
                                 ${CMAKE_BINARY_DIR}/lib_fftx_mddft_srcs/${CMAKE_BUILD_TYPE}
                                 ${CMAKE_BINARY_DIR}/lib_fftx_imddft_srcs/${CMAKE_BUILD_TYPE}
                                 ${CMAKE_BINARY_DIR}/lib_fftx_mdprdft_srcs/${CMAKE_BUILD_TYPE}
                                 ${CMAKE_BINARY_DIR}/lib_fftx_imdprdft_srcs/${CMAKE_BUILD_TYPE}
                                 ${CMAKE_BINARY_DIR}/lib_fftx_rconv_srcs/${CMAKE_BUILD_TYPE} )
else ()
    target_link_directories    ( ${EXE_TARGET} PRIVATE
                                 ${CMAKE_BINARY_DIR}/lib_fftx_mddft_srcs
                                 ${CMAKE_BINARY_DIR}/lib_fftx_imddft_srcs
                                 ${CMAKE_BINARY_DIR}/lib_fftx_mdprdft_srcs
                                 ${CMAKE_BINARY_DIR}/lib_fftx_imdprdft_srcs
                                 ${CMAKE_BINARY_DIR}/lib_fftx_rconv_srcs )
endif ()
add_dependencies           ( ${EXE_TARGET} ${_lib_precomp} )
## add_dependencies           ( ${EXE_TARGET} fftx_mddft_precomp fftx_imddft_precomp fftx_mdprdft_precomp fftx_imdprdft_precomp fftx_rconv_precomp )

if ( ${_be_hip} )
    target_compile_options     ( ${EXE_TARGET} PRIVATE
                                 ${HIP_COMPILE_FLAGS}
                                 ${ADDL_COMPILE_FLAGS} )
    target_link_directories    ( ${EXE_TARGET} PRIVATE
                                 $ENV{ROCM_PATH}/lib )
    target_link_libraries      ( ${EXE_TARGET}
                                 hip::host hip::device ${LIBS_FOR_HIP}
                                 ${_lib_precomp} )
else ()
    target_compile_options     ( ${EXE_TARGET} PRIVATE
                                 ${CUDA_COMPILE_FLAGS}
                                 ${GPU_COMPILE_DEFNS} )
    set_property        ( TARGET ${EXE_TARGET}
                          PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
    target_link_libraries      ( ${EXE_TARGET}
                                 ${LIBS_FOR_CUDA}
                                 ${_lib_precomp} )
endif ()

install ( TARGETS
    ${EXE_TARGET}
    DESTINATION ${CMAKE_BINARY_DIR}/bin )
