##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

##  create a generator script, given a <prefix> and a <stem> (script =
##  <prefix>.<stem>.generator.g.
##  This function assumes the plan.g file has already been created (e.g., by
##  calling run_driver_program() which creates the plan and codegen header
##  files).  For example:
##      create_generator_file ( "mddft" "fftx" )
##  will create script mddft.fftx.generator.g.  This script may be consumed in a
##  subsequent step to create a source code file (e.g., see RunSpiral.cmake).
##  Additionally, the variable ${mddft_gen} = "mddft.fftx.generator.g" is defined.
    
##  define standard files... (may need to customize this later)

set ( BACKEND_SPIRAL_DIR      ${BACKEND_SOURCE_DIR}/spiral_cpu_serial )
set ( SPIRAL_BACKEND_PREAMBLE ${BACKEND_SPIRAL_DIR}/preamble.g )
set ( SPIRAL_BACKEND_CODEGEN  ${BACKEND_SPIRAL_DIR}/codegen.g  )

function ( create_generator_file prefix stem )
    message ( "create generator SPIRAL script ${prefix}.${stem}.generator.g" )
    set     ( ${prefix}_gen ${prefix}.${stem}.generator.g PARENT_SCOPE )
    set     ( _gen ${prefix}.${stem}.generator.g )
    set     ( _plan   ${prefix}.${stem}.plan.g )
    ##  message ( "define varible: {_gen} = ${_gen}" )
    
    if ( WIN32 )
	add_custom_command ( OUTPUT ${_gen}
	COMMAND IF EXIST ${_gen} ( DEL /F ${_gen} )
	COMMAND cat ${SPIRAL_BACKEND_PREAMBLE} ${_plan} ${SPIRAL_BACKEND_CODEGEN} > ${_gen}
	DEPENDS ${_plan}
	VERBATIM
	COMMENT "Generating ${_gen}" )
    else ()
	add_custom_command ( OUTPUT ${_gen}
	COMMAND ${BASH} -c "rm -f ${_gen} ; cat ${SPIRAL_BACKEND_PREAMBLE} ${_plan} ${SPIRAL_BACKEND_CODEGEN} >> ${_gen}"
	DEPENDS ${_plan}
	VERBATIM
	COMMENT "Generating ${_gen}" )
    endif ()

    add_custom_target ( NAME.${_gen} ALL
	DEPENDS ${_gen}
	VERBATIM )

endfunction ()

