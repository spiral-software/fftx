##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

include ( ../ExamplesCommon.cmake )

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  ===== For most examples you should not need to modify anything ABOVE this line =====

##  Set the project name.  Preferred name is just the *name* of the example folder 
project ( rconv ${_lang_add} ${_lang_base} )

##  File naming convention is: <prefix>.<stem>.xxxxx; set _stem and _prefix(es)
##  Test program name is:  test<project name>.{cpp|cu}

set ( _stem fftx )
set ( _prefixes rconv3 )
set ( _prefixes_lib )
set ( BUILD_PROGS test${PROJECT_NAME} test${PROJECT_NAME}_lib )

##  One .cpp file is coded with device_macros and should build for CUDA & HIP
set ( _desired_suffix cpp )

##  ===== For most examples you should not need to modify anything BELOW this line =====

foreach ( _prog ${BUILD_PROGS} )
    ##  Build the dependencies and get the include directories / libraries for each program
    if ( ${_codegen} STREQUAL "HIP" )
	set_source_files_properties ( ${_prog}.${_desired_suffix} PROPERTIES LANGUAGE CXX )
    elseif ( ${_codegen} STREQUAL "CUDA" )
	set_source_files_properties ( ${_prog}.${_desired_suffix} PROPERTIES LANGUAGE CUDA )
    endif ()
    if ( ${_prog} STREQUAL "test${PROJECT_NAME}" )
	set ( _pref ${_prefixes} )
    else ()
	set ( _pref ${_prefixes_lib} )
    endif ()
    manage_deps_codegen ( ${_codegen} ${_stem} "${_pref}" )
    add_includes_libs_to_target ( ${_prog} ${_stem} "${_pref}" )
endforeach ()
    
# manage_deps_codegen ( ${_codegen} ${_stem} "${_prefixes}" )
# add_includes_libs_to_target ( ${BUILD_PROGRAM} ${_stem} "${_prefixes}" )

# ## Now for the version with _lib suffix.

# set ( BUILD_PROG_LIB test${PROJECT_NAME}_lib )
# set ( _prefixes_lib )

# ##  message ( STATUS "${BUILD_PROG_LIB}: Library include dirs = ${_library_includes}" )
# ##  message ( STATUS "${BUILD_PROG_LIB}: Library names = ${_library_names}" )

# manage_deps_codegen ( ${_codegen} ${_stem} "${_prefixes_lib}" )
# add_includes_libs_to_target ( ${BUILD_PROG_LIB} ${_stem} "${_prefixes_lib}" )

# ##  ===== For most examples you should not need to modify anything BELOW this line =====

# foreach ( _prog ${BUILD_PROGS} )
#     ##  Build the dependencies and get the include directories / libraries for each program
#     if ( ${_codegen} STREQUAL "HIP" )
# 	set_source_files_properties ( ${_prog}.${_desired_suffix} PROPERTIES LANGUAGE CXX )
#     elseif ( ${_codegen} STREQUAL "CUDA" )
# 	set_source_files_properties ( ${_prog}.${_desired_suffix} PROPERTIES LANGUAGE CUDA )
#     endif ()
#     manage_deps_codegen ( ${_codegen} ${_stem} "${_prefixes}" )
#     add_includes_libs_to_target ( ${_prog} ${_stem} "${_prefixes}" )
#     add_mpi_decorations_to_target ( ${_prog} )
# endforeach ()
