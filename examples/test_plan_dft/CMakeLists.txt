##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

project ( TEST_PLAN_DFT )

##  First create and run the driver program(s):  Creates the ~.plan.g and ~.codegen.hpp files

set ( _stem fftx )
set ( _prefix1 mddft )
set ( BUILD_PROGRAM test${_prefix1} )
set ( _prefix2 imddft )

run_driver_program ( ${_prefix1} ${_stem} )
set ( _driver1 ${${_prefix1}_driver} )
set ( _plan1 ${${_prefix1}_plan} )
set ( _hdr1  ${_prefix1}.${_stem}.codegen.hpp )

run_driver_program ( ${_prefix2} ${_stem} )
set ( _driver2 ${${_prefix2}_driver} )
set ( _plan2 ${${_prefix2}_plan} )
set ( _hdr2  ${_prefix2}.${_stem}.codegen.hpp )

##  Create the generator scripts: ~.generator.g files

create_generator_file ( ${_prefix1} ${_stem} )
create_generator_file ( ${_prefix2} ${_stem} )
    
set ( _gen1 ${${_prefix1}_gen} )
set ( _gen2 ${${_prefix2}_gen} )

##  Create the C source code from the SPIRAL generator script(s)

set                ( _ccode1 ${_prefix1}.${_stem}.source.cpp )
file               ( TO_NATIVE_PATH ${TEST_PLAN_DFT_BINARY_DIR}/${_gen1} _gfile )
create_source_file ( ${_gfile} ${_ccode1} )

set                ( _ccode2 ${_prefix2}.${_stem}.source.cpp )
file               ( TO_NATIVE_PATH ${TEST_PLAN_DFT_BINARY_DIR}/${_gen2} _gfile )
create_source_file ( ${_gfile} ${_ccode2} )

add_executable   ( ${BUILD_PROGRAM} ${BUILD_PROGRAM}.cpp ${_ccode1} ${_ccode2} ${_hdr1} ${_hdr2} )
add_dependencies ( ${BUILD_PROGRAM} ${_driver1} ${_driver2}
                   NAME.${_plan1} NAME.${_gen1} NAME.${_ccode1}
                   NAME.${_plan2} NAME.${_gen2} NAME.${_ccode2} )

target_include_directories ( ${BUILD_PROGRAM} PRIVATE ${TEST_PLAN_DFT_BINARY_DIR} ${CMAKE_BINARY_DIR} )

set ( INSTALL_DIR_TARGET ${CMAKE_BINARY_DIR}/bin )

install ( TARGETS ${BUILD_PROGRAM} DESTINATION ${INSTALL_DIR_TARGET} )

