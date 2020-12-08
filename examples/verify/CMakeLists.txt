##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  Set the project name.  Preferred name is just the *name* of the example folder 
project ( verify )

##  First create and run the driver program(s):  Creates the ~.plan.g and ~.codegen.hpp files
##  File nameing convention is: <prefix>.<stem>.xxxxx; set _stem and _prefix1
##  Test program name is:  test<prefix>.cpp

set ( _stem fftx )
set ( BUILD_PROGRAM test${PROJECT_NAME} )
set ( _prefix1 mddft2 )
set ( _prefix2 imddft2 )
set ( _prefix3 mddft3 )
set ( _prefix4 imddft3 )

##  ===== For most examples you should not need to modify anything below this line =====

run_driver_program ( ${_prefix1} ${_stem} )
set ( _driver1 ${${_prefix1}_driver} )
set ( _plan1 ${${_prefix1}_plan} )
set ( _hdr1  ${_prefix1}.${_stem}.codegen.hpp )

run_driver_program ( ${_prefix2} ${_stem} )
set ( _driver2 ${${_prefix2}_driver} )
set ( _plan2 ${${_prefix2}_plan} )
set ( _hdr2  ${_prefix2}.${_stem}.codegen.hpp )

run_driver_program ( ${_prefix3} ${_stem} )
set ( _driver3 ${${_prefix3}_driver} )
set ( _plan3 ${${_prefix3}_plan} )
set ( _hdr3  ${_prefix3}.${_stem}.codegen.hpp )

run_driver_program ( ${_prefix4} ${_stem} )
set ( _driver4 ${${_prefix4}_driver} )
set ( _plan4 ${${_prefix4}_plan} )
set ( _hdr4  ${_prefix4}.${_stem}.codegen.hpp )

##  Create the generator scripts: ~.generator.g files

create_generator_file ( ${_prefix1} ${_stem} )
create_generator_file ( ${_prefix2} ${_stem} )
create_generator_file ( ${_prefix3} ${_stem} )
create_generator_file ( ${_prefix4} ${_stem} )
    
set ( _gen1 ${${_prefix1}_gen} )
set ( _gen2 ${${_prefix2}_gen} )
set ( _gen3 ${${_prefix3}_gen} )
set ( _gen4 ${${_prefix4}_gen} )

##  Create the C source code from the SPIRAL generator script(s)

set                ( _ccode1 ${_prefix1}.${_stem}.source.cpp )
file               ( TO_NATIVE_PATH ${${PROJECT_NAME}_BINARY_DIR}/${_gen1} _gfile )
create_source_file ( ${_gfile} ${_ccode1} )

set                ( _ccode2 ${_prefix2}.${_stem}.source.cpp )
file               ( TO_NATIVE_PATH ${${PROJECT_NAME}_BINARY_DIR}/${_gen2} _gfile )
create_source_file ( ${_gfile} ${_ccode2} )

set                ( _ccode3 ${_prefix3}.${_stem}.source.cpp )
file               ( TO_NATIVE_PATH ${${PROJECT_NAME}_BINARY_DIR}/${_gen3} _gfile )
create_source_file ( ${_gfile} ${_ccode3} )

set                ( _ccode4 ${_prefix4}.${_stem}.source.cpp )
file               ( TO_NATIVE_PATH ${${PROJECT_NAME}_BINARY_DIR}/${_gen4} _gfile )
create_source_file ( ${_gfile} ${_ccode4} )

add_executable   ( ${BUILD_PROGRAM} ${BUILD_PROGRAM}.cpp ${_ccode1} ${_ccode2} ${_ccode3} ${_ccode4} ${_hdr1} ${_hdr2} ${_hdr3} ${_hdr4} )
add_dependencies ( ${BUILD_PROGRAM} ${_driver1} ${_driver2} ${_driver3} ${_driver4}
                   NAME.${_plan1} NAME.${_gen1} NAME.${_ccode1}
                   NAME.${_plan2} NAME.${_gen2} NAME.${_ccode2}
                   NAME.${_plan3} NAME.${_gen3} NAME.${_ccode3}
                   NAME.${_plan4} NAME.${_gen4} NAME.${_ccode4} )

target_include_directories ( ${BUILD_PROGRAM} PRIVATE ${${PROJECT_NAME}_BINARY_DIR} ${CMAKE_BINARY_DIR} )

set ( INSTALL_DIR_TARGET ${CMAKE_BINARY_DIR}/bin )

install ( TARGETS ${BUILD_PROGRAM} DESTINATION ${INSTALL_DIR_TARGET} )

