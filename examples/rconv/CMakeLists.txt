##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  Set the project name.  Preferred name is just the *name* of the example folder 
project ( rconv )

##  First create and run the driver program(s):  Creates the ~.plan.g and ~.codegen.hpp files
##  File nameing convention is: <prefix>.<stem>.xxxxx; set _stem and _prefix1
##  Test program name is:  test<prefix>.cpp

set ( _stem fftx )
set ( _prefixes rconv2 rconv3 )

set ( BUILD_PROGRAM test${PROJECT_NAME} )

##  ===== For most examples you should not need to modify anything below this line =====

##  build a running list of sources and dependencies...

set ( _all_build_srcs ${BUILD_PROGRAM}.cpp )
set ( _all_build_deps )

foreach ( _prefix ${_prefixes} ) 
    run_driver_program ( ${_prefix} ${_stem} )
    set ( _driver ${PROJECT_NAME}.${${_prefix}_driver} )
    set ( _plan ${${_prefix}_plan} )
    set ( _hdr  ${_prefix}.${_stem}.codegen.hpp )

    ##  Create the generator scripts: ~.generator.g files

    create_generator_file ( ${_prefix} ${_stem} )
    set ( _gen ${${_prefix}_gen} )

    ##  Create the C source code from the SPIRAL generator script(s)

    set                ( _ccode ${_prefix}.${_stem}.source.cpp )
    file               ( TO_NATIVE_PATH ${${PROJECT_NAME}_BINARY_DIR}/${_gen} _gfile )
    create_source_file ( ${_gfile} ${_ccode} )

    ##  append to our running lists
    list ( APPEND _all_build_srcs ${_hdr} ${_ccode} )
    list ( APPEND _all_build_deps ${_driver}
	   NAME.${PROJECT_NAME}.${_plan}
           NAME.${PROJECT_NAME}.${_gen}
	   NAME.${PROJECT_NAME}.${_ccode} )

endforeach ()
   
add_executable   ( ${BUILD_PROGRAM} ${_all_build_srcs} )
add_dependencies ( ${BUILD_PROGRAM} ${_all_build_deps} )

target_include_directories ( ${BUILD_PROGRAM} PRIVATE ${${PROJECT_NAME}_BINARY_DIR} ${CMAKE_BINARY_DIR} )

set ( INSTALL_DIR_TARGET ${CMAKE_BINARY_DIR}/bin )

install ( TARGETS ${BUILD_PROGRAM} DESTINATION ${INSTALL_DIR_TARGET} )
