##
## Copyright (c) 2018-2022, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

set ( _lib_root fftx_mpi )
set ( _lib_name ${_lib_root} )
set ( _lib_name ${_lib_root} PARENT_SCOPE )

set ( _source_files fftx_mpi.cpp fftx_gpu.cu )
if ( ${_codegen} STREQUAL "HIP" )
    set_source_files_properties ( fftx_mpi.cpp PROPERTIES LANGUAGE CXX )
    set_source_files_properties ( fftx_gpu.cu  PROPERTIES LANGUAGE CXX )
    list ( APPEND ADDL_COMPILE_FLAGS -DFFTX_HIP )
elseif ( ${_codegen} STREQUAL "CUDA" )
    set_source_files_properties ( fftx_mpi.cpp PROPERTIES LANGUAGE CUDA )
    set_source_files_properties ( fftx_gpu.cu  PROPERTIES LANGUAGE CUDA )
    list ( APPEND ADDL_COMPILE_FLAGS -DFFTX_CUDA )
endif ()

if ( WIN32 )
    list ( APPEND ADDL_COMPILE_FLAGS -D_USE_MATH_DEFINES )
    set ( CMAKE_CUDA_ARCHITECTURES 52 )		##  60 61 62 70 72 75 80 )
else ()
    set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 )
endif ()

add_library                ( ${_lib_name} SHARED ${_source_files} )
target_compile_options     ( ${_lib_name} PRIVATE ${ADDL_COMPILE_FLAGS} )
add_mpi_decorations_to_target ( ${_lib_name} )

if ( ${_codegen} STREQUAL "CUDA" )
    set_property    ( TARGET ${_lib_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
    target_link_libraries ( ${_lib_name} PRIVATE ${LIBS_FOR_CUDA} )
endif ()

if ( WIN32 )
    set_property    ( TARGET ${_lib_name} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON )
endif ()

set ( _incl_files fftx_mpi.hpp fftx_gpu.h fftx_util.h )

install ( TARGETS
          ${_lib_name}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

install ( FILES ${_incl_files}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
