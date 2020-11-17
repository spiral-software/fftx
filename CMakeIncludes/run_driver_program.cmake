##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

##  create and run a driver program, given a <prefix> and a <stem> (filename =
##  <prefix>.<stem>.cpp; driver = <prefix>.<stem>.driver[.exe]).  If an
##  additional argument (beyond the named arguments) is given it is treated as a
##  list of additional include directories, if required.
##  Outputs from running the driver program are: <prefix>.<stem>.plan.g and
##  <prefix>.<stem>.codegen.hpp
##
##  For example:
##      run_driver_program ( "mddft" "fftx" "/path/to/addl/includes" )
##  will compile source file mddft.fftx.cpp to program mddft.fftx.driver[.exe]
##  then run it to create mddft.fftx.plan.g and mddft.fftx.codegen.hpp, adding
##  "/path/to/addl/includes" to the search path for include files.

function ( run_driver_program prefix stem )
    message ( "build and run driver for ${prefix}.${stem}.cpp" )
    set     ( _driver ${prefix}.${stem}.driver )
    add_executable ( ${_driver} ${prefix}.${stem}.cpp )
    set_property ( TARGET ${_driver} PROPERTY CXX_STANDARD 14 )

    if ( ${ARGC} GREATER 2 )
	##  received optional include directories -- add to target
	foreach ( _fil ${ARGN} )
	    message ( "add ${_fil} to include directories for ${_driver}" )
	    target_include_directories ( ${_driver} PRIVATE ${_fil} )
	endforeach ()
    endif ()
    
    ##  Run the driver program to create ~.codegen.hpp and ~.plan.g

    set ( _plan ${prefix}.${stem}.plan.g )
    set ( _header ${prefix}.${stem}.codegen.hpp )
    ##  message ( "define vars: plan = ${_plan}, header = ${_header}" )
    
    if ( WIN32 )
	add_custom_command ( OUTPUT ${_plan} ${_header} 
	COMMAND IF EXIST ${_plan} ( DEL /F ${_plan} )
	COMMAND IF EXIST ${_header} ( DEL /F ${_header} )
	COMMAND ${_driver} ${prefix} > ${_plan}
	DEPENDS ${_driver}
	VERBATIM
	COMMENT "Generating ${_plan}" )
    else ()
	include ( FindUnixCommands )
	add_custom_command ( OUTPUT ${_plan} ${_header}
	    COMMAND ${BASH} -c "rm -f ${_plan} ${_header} ; ${_driver} ${prefix} > ${_plan}"
	    DEPENDS ${_driver}
	    VERBATIM
	    COMMENT "Generating ${_plan}" )
    endif ()

    add_custom_target ( NAME.${_plan} ALL
	DEPENDS ${_plan}
	VERBATIM )
    
endfunction ()

