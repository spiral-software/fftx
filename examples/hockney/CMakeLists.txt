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
project ( hockney ${_lang_add} ${_lang_base} )

##  File naming convention is: <prefix>.<stem>.xxxxx; set _stem and _prefix(es)
##  Test program name is:  test<project name>.{cpp|cu}

set ( _stem fftx )
set ( _prefixes hockney gethat )
set ( BUILD_PROGRAM test${PROJECT_NAME} )

##  ===== For most examples you should not need to modify anything BELOW this line =====

manage_deps_codegen ( ${_codegen} ${_stem} "${_prefixes}" )
add_includes_libs_to_target ( ${BUILD_PROGRAM} ${_stem} "${_prefixes}" )
