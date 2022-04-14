##
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information
##

if ( ${_codegen} STREQUAL "CUDA" )
    set ( _lang_add LANGUAGES CUDA )
endif ()

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  ===== For most examples you should not need to modify anything ABOVE this line =====

##  Set the project name.  Preferred name is just the *name* of the example folder 
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
        message ( STATUS "Added sub-directory ${_dir} for pre-compiled library " )
        add_subdirectory ( "${_dir}" )
	include_directories ( ${CMAKE_CURRENT_SOURCE_DIR}/${_dir} )
	list ( APPEND _lib_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/${_dir} )
	list ( APPEND _library_names ${_lib_name} )
    endif ()
endforeach ()

set ( _library_names ${_library_names} PARENT_SCOPE )
set ( _library_includes ${_library_includes} ${_lib_include_dirs} PARENT_SCOPE )
##  message ( STATUS "Library include dirs = ${_lib_include_dirs}" )
