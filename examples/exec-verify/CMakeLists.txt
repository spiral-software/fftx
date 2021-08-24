##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## cmake -S . -B builddir
## For HIP, add: -DCMAKE_CXX_COMPILER=hipcc
## 
## cmake --build builddir --target install
##
## When running, LD_LIBRARY_PATH needs to include $FFTX_HOME/examples/library

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.18 )

## Name of the project.
set ( PROJECT testverify )

## The list of libraries (from ../library) needed by the executable.
set ( FFTX_LIBRARY_LIST fftx_mddft fftx_imddft fftx_mdprdft fftx_imdprdft )
                     
set ( CMAKE_BUILD_TYPE Release
      CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )

## Project needs to know if LANGUAGES will include CUDA.
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
          DESCRIPTION "Driver to test real convolution"
          LANGUAGES C CXX ${_add_lang} )

##  Set C standard 11
set ( CMAKE_C_STANDARD 11 )
set ( CMAKE_CXX_STANDARD 11 )

set                        (   EXE_TARGET  ${PROJECT} )

add_executable		   ( ${EXE_TARGET} ${PROJECT}${_suffix} )

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

if ( ${_be_hip} )
    ##  Add additional paths for CMake to find HIP
    list ( APPEND CMAKE_PREFIX_PATH
           /opt/rocm/hip /opt/rocm )	##  /opt/rocm-4.2.0/hip
    find_package ( hip REQUIRED )
    if (${hip_FOUND})
	message ( STATUS "Found HIP: Version = ${hip_VERSION}")
    else ()
	message ( SEND_ERROR "HIP NOT FOUND: HIP is required to build/run examples")
    endif ()

    ##  Need to add $ROCM_PATH for includes and libraries
    if ( DEFINED ENV{ROCM_PATH} )
	message ( STATUS "ROCM_PATH is defined: $ENV{ROCM_PATH}" )
	target_include_directories ( ${PROJECT} PUBLIC
                                     $ENV{ROCM_PATH}/hipfft/include
                                     $ENV{ROCM_PATH}/include )
    endif ()
endif ()

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

set ( FFTX_LIBRARY_SOURCE_DIR ${FFTX_SOURCE_DIR}/examples/library )

target_include_directories ( ${EXE_TARGET} PUBLIC
                             ${FFTX_LIBRARY_SOURCE_DIR}
                             ${CMAKE_BINARY_DIR}
                             ${SPIRAL_SOURCE_DIR}/profiler/targets/include
                             ${FFTX_SOURCE_DIR}/include )

if (WIN32)
    target_link_directories    ( ${EXE_TARGET} PRIVATE
                                 ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
                                 ${FFTX_LIBRARY_SOURCE_DIR}/${CMAKE_BUILD_TYPE} )
else ()
    target_link_directories    ( ${EXE_TARGET} PRIVATE
                                 ${CMAKE_BINARY_DIR}
                                 ${FFTX_LIBRARY_SOURCE_DIR} )
endif ()

if ( ${_be_hip} )
    target_compile_options     ( ${EXE_TARGET} PRIVATE
                                 ${HIP_COMPILE_FLAGS}
                                 ${ADDL_COMPILE_FLAGS} )
    target_link_directories    ( ${EXE_TARGET} PRIVATE
                                 $ENV{ROCM_PATH}/lib )
    target_link_libraries      ( ${EXE_TARGET}
                                 hip::host hip::device ${LIBS_FOR_HIP} )
else ()
    target_compile_options     ( ${EXE_TARGET} PRIVATE
                                 ${CUDA_COMPILE_FLAGS}
                                 ${GPU_COMPILE_DEFNS} )
    set_property        ( TARGET ${EXE_TARGET}
                          PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
    target_link_libraries      ( ${EXE_TARGET}
                                 ${LIBS_FOR_CUDA} )
endif ()

foreach ( FFTX_LIB ${FFTX_LIBRARY_LIST} )
     target_include_directories ( ${EXE_TARGET} PUBLIC
                            ${FFTX_LIBRARY_SOURCE_DIR}/lib_${FFTX_LIB}_srcs )
     target_link_libraries     ( ${EXE_TARGET}
                                 ${FFTX_LIB}_precomp )
endforeach()

install ( TARGETS ${EXE_TARGET}
          DESTINATION ${CMAKE_SOURCE_DIR}
          PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                      GROUP_EXECUTE GROUP_READ )
