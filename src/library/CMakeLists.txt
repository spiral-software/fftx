##
##  Copyright (c) 2018-2022, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information
##

if ( ${_codegen} STREQUAL "CUDA" )
    set ( _lang_add LANGUAGES CUDA )
endif ()

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

project ( ${PROJECT} library ${_lang_add} ${_lang_base} )

##
##  Build the pre-compiled libraries...
##  Assume the code generation (by Spiral) is complete (see build-lib-code.sh)
##  There'll be sub-folders named lib_<transform> and any valid such folder will
##  contain a CMakeLists.txt.
##  Get the list of such folders and traverse them to build the libraries.
##

set ( _pkg_folder "${CMAKE_CURRENT_SOURCE_DIR}" )
file ( GLOB _pkg_names RELATIVE ${_pkg_folder} CONFIGURE_DEPENDS ${_pkg_folder}/* )
##  message(STATUS "Found in ${CMAKE_CURRENT_SOURCE_DIR}: = ${_pkg_names}" )
set ( _lib_include_dirs ${CMAKE_CURRENT_SOURCE_DIR} )
set ( _library_names )

foreach ( _dir ${_pkg_names} )
    if ( IS_DIRECTORY ${_pkg_folder}/${_dir} AND EXISTS "${_pkg_folder}/${_dir}/CMakeLists.txt" )
        ##  Subdirectory exists and contains a CMakeLists.txt file -- add subir
	if ( "${_dir}" STREQUAL "lib_fftx_mpi" )
	    continue ()			## already handled
	endif ()
        add_subdirectory ( "${_dir}" )
	include_directories ( ${CMAKE_CURRENT_SOURCE_DIR}/${_dir} )
	list ( APPEND _lib_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/${_dir} )
	list ( APPEND _library_names ${_lib_name} )
        message ( STATUS "Added sub-directory ${_dir} for pre-compiled library " )
    endif ()
endforeach ()

##  Don't attempt MPI library unless: MPI found, codegen != CPU
##  lib_fftx_mpi is not generated code, want to get the header path and library first
set ( _fftxmpi_libname )

if ( ${MPI_FOUND} )
    if ( NOT ${_codegen} STREQUAL "CPU" AND NOT ${_codegen} STREQUAL "SYCL" )
	add_subdirectory ( lib_fftx_mpi )
	set ( _fftxmpi_libname ${_lib_name} )
	include_directories ( ${CMAKE_CURRENT_SOURCE_DIR}/lib_fftx_mpi )
	list ( APPEND _lib_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/lib_fftx_mpi )
    endif ()
endif ()

list ( APPEND _library_names ${_fftxmpi_libname} )
set ( _library_names ${_library_names} PARENT_SCOPE )
set ( _library_includes ${_library_includes} ${_lib_include_dirs} PARENT_SCOPE )
message ( STATUS "Library include dirs = ${_lib_include_dirs}" )
message ( STATUS "Library names = ${_library_names}" )


##  List of header files to be copies to .../include on 'make install'
set ( _incl_files imddft.fftx.precompile.hpp
	imdprdft.fftx.precompile.hpp
	mddft.fftx.precompile.hpp
	mdprdft.fftx.precompile.hpp
	rconv.fftx.precompile.hpp
	transformer.fftx.precompile.hpp
	device_macros.h
    )

install ( FILES ${_incl_files}
	  DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
