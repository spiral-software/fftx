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

##  List the names of the source files to compile for the library

set ( _source_files fftx_1d_gpu.cpp
                    fftx_1d_mpi.cpp
                    fftx_1d_mpi_default.cpp
                    fftx_1d_mpi_spiral.cpp
                    fftx_gpu.cpp
                    fftx_mpi_default.cpp
                    fftx_mpi_spiral.cpp
                    fftx_mpi.cpp )

foreach ( _src ${_source_files} )
    ##  set the desired language property
    if ( ${_codegen} STREQUAL "HIP" )
	set_source_files_properties ( ${_src} PROPERTIES LANGUAGE CXX )
    elseif ( ${_codegen} STREQUAL "CUDA" )
	set_source_files_properties ( ${_src} PROPERTIES LANGUAGE CUDA )
    endif ()
endforeach ()

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

##  List the names of the public header files for the library

set ( _incl_files fftx_1d_gpu.h
                  fftx_1d_mpi.hpp
                  fftx_1d_mpi_default.hpp
                  fftx_1d_mpi_spiral.hpp
                  fftx_gpu.h
                  fftx_mpi.hpp
                  fftx_mpi_default.hpp
                  fftx_mpi_spiral.hpp
                  fftx_util.h )

install ( TARGETS
          ${_lib_name}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

install ( FILES ${_incl_files}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
