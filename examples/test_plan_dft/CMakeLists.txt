##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  _codegen specifies CPU or GPU (serial or CUDA) code generation.  Will be set
##  in parent so don't change here.

if ( ${_codegen} STREQUAL "GPU" )
    set ( _lang_add LANGUAGES CUDA )
endif ()

##  Set the project name.  Preferred name is just the *name* of the example folder 
project ( test_plan_dft ${_lang_add} )

##  File nameing convention is: <prefix>.<stem>.xxxxx; set _stem and _prefix(es)
##  Test program name is:  test<project name>.{cpp|cu}

set ( _stem fftx )
set ( _prefixes mddft imddft )
set ( BUILD_PROGRAM testmddft )

##  ===== For most examples you should not need to modify anything below this line =====

manage_deps_codegen ( ${_codegen} ${_stem} "${_prefixes}" )
list ( APPEND _all_build_srcs ${BUILD_PROGRAM}.${_suffix} )

add_executable   ( ${BUILD_PROGRAM} ${_all_build_srcs} )
add_dependencies ( ${BUILD_PROGRAM} ${_all_build_deps} )

target_compile_options ( ${BUILD_PROGRAM} PRIVATE ${ADDL_COMPILE_FLAGS} )

if ( ${_codegen} STREQUAL "GPU" AND NOT WIN32 )
    target_compile_options ( ${BUILD_PROGRAM} PRIVATE -ccbin=${CMAKE_CXX_COMPILER} )
endif ()

target_include_directories ( ${BUILD_PROGRAM} PRIVATE ${${PROJECT_NAME}_BINARY_DIR} ${CMAKE_BINARY_DIR} )

set ( INSTALL_DIR_TARGET ${CMAKE_BINARY_DIR}/bin )

install ( TARGETS ${BUILD_PROGRAM} DESTINATION ${INSTALL_DIR_TARGET} )

