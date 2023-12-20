##
## Copyright (c) 2018-2021, Carnegie Mellon University
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
    ##  message ( "build and run driver for ${prefix}.${stem}.cpp" )
    set     ( _driver ${PROJECT_NAME}.${prefix}.${stem}.driver )
    set     ( ${prefix}_driver ${prefix}.${stem}.driver PARENT_SCOPE )
    add_executable ( ${_driver} ${prefix}.${stem}.cpp )
    target_compile_options ( ${_driver} PRIVATE ${ADDL_COMPILE_FLAGS} )
    set_property ( TARGET ${_driver} PROPERTY CXX_STANDARD 11 )
    ##  message ( STATUS "Added ${ADDL_COMPILE_FLAGS} to target: ${_driver}" )

    if ( ${ARGC} GREATER 2 )
	##  received optional include directories -- add to target
	foreach ( _fil ${ARGN} )
	    ##  message ( "add ${_fil} to include directories for ${_driver}" )
	    target_include_directories ( ${_driver} PRIVATE ${_fil} )
	endforeach ()
    endif ()
    
    ##  Run the driver program to create ~.codegen.hpp and ~.plan.g

    set ( _plan ${prefix}.${stem}.plan.g )
    set ( ${prefix}_plan ${prefix}.${stem}.plan.g PARENT_SCOPE )
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
	    COMMAND ${BASH} -c "rm -f ${_plan} ${_header} ; ${CMAKE_CURRENT_BINARY_DIR}/${_driver} ${prefix} > ${_plan}"
	    DEPENDS ${_driver}
	    VERBATIM
	    COMMENT "Generating ${_plan}" )
    endif ()

    add_custom_target ( NAME.${PROJECT_NAME}.${_plan} ALL
	DEPENDS ${_driver}
	VERBATIM )
    
endfunction ()


##  create a generator script, given a <prefix> and a <stem> (script =
##  <prefix>.<stem>.generator.g.  This function assumes the plan.g file has
##  already been created (e.g., by calling run_driver_program() which creates
##  the plan and codegen header files).  For example:
##      create_generator_file ( _codefor "mddft" "fftx" )
##  will create script mddft.fftx.generator.g, _codefor indicates if code should
##  be generated for CPU or GPU (CUDA or HIP).  This script may be consumed in a
##  subsequent step to create a source code file (e.g., see RunSpiral.cmake).
##  Additionally, the variable ${mddft_gen} = "mddft.fftx.generator.g" is
##  defined.
    
##  define standard files... (may need to customize this later)

set ( BACKEND_SPIRAL_CPU_DIR      ${BACKEND_SOURCE_DIR}/spiral_cpu_serial )
set ( BACKEND_SPIRAL_CUDA_DIR     ${BACKEND_SOURCE_DIR}/spiral_gpu )
set ( BACKEND_SPIRAL_HIP_DIR      ${BACKEND_SOURCE_DIR}/spiral_hip )

set ( SPIRAL_BACKEND_CPU_PREAMBLE ${BACKEND_SPIRAL_CPU_DIR}/preamble.g )
set ( SPIRAL_BACKEND_CUDA_PREAMBLE ${BACKEND_SPIRAL_CUDA_DIR}/preamble.g )
set ( SPIRAL_BACKEND_HIP_PREAMBLE ${BACKEND_SPIRAL_HIP_DIR}/preamble.g )
set ( SPIRAL_BACKEND_CPU_CODEGEN  ${BACKEND_SPIRAL_CPU_DIR}/codegen.g  )
set ( SPIRAL_BACKEND_CUDA_CODEGEN  ${BACKEND_SPIRAL_CUDA_DIR}/codegen.g  )
set ( SPIRAL_BACKEND_CUDA_CODEGEN_CPP  ${BACKEND_SPIRAL_CUDA_DIR}/codegen_cpp.g  )
set ( SPIRAL_BACKEND_HIP_CODEGEN  ${BACKEND_SPIRAL_HIP_DIR}/codegen.g  )

function ( create_generator_file _codefor prefix stem )
    message ( "create generator SPIRAL script ${prefix}.${stem}.generator.g" )
    set     ( ${prefix}_gen ${prefix}.${stem}.generator.g PARENT_SCOPE )
    set     ( _gen ${prefix}.${stem}.generator.g )
    set     ( _plan   ${prefix}.${stem}.plan.g )

    set ( _preamble ${SPIRAL_BACKEND_${_codefor}_PREAMBLE} )
    set ( _postfix  ${SPIRAL_BACKEND_${_codefor}_CODEGEN} )
    if ( ${_codefor} STREQUAL "CUDA" AND ${_suffix} STREQUAL "cpp" )
	set ( _postfix  ${SPIRAL_BACKEND_${_codefor}_CODEGEN_CPP} )
    endif ()
    if ( "X${_generator_script}" STREQUAL "X" )
	##  The complete script is not defined -- no action required
    elseif ( ${_generator_script} STREQUAL "COMPLETE" )
	##  Don't add preamble and codegen pieces
	set ( _preamble )
	set ( _postfix  )
    endif ()

    if ( WIN32 )
	add_custom_command ( OUTPUT ${_gen}
	    COMMAND ${Python3_EXECUTABLE} ${SPIRAL_SOURCE_DIR}/gap/bin/catfiles.py
	            ${_gen} ${_preamble} ${_plan} ${_postfix}
	    COMMAND IF EXIST ${prefix}.${stem}.source.${_suffix} ( DEL /F ${prefix}.${stem}.source.${_suffix} )
            DEPENDS ${_plan}
	    VERBATIM
	    COMMENT "Generating ${_gen}" )
    else ()
	include ( FindUnixCommands )
	add_custom_command ( OUTPUT ${_gen}
	    COMMAND ${Python3_EXECUTABLE} ${SPIRAL_SOURCE_DIR}/gap/bin/catfiles.py
	            ${_gen} ${_preamble} ${_plan} ${_postfix}
	    COMMAND rm -f ${prefix}.${stem}.source.${_suffix}
	    DEPENDS ${_plan}
	    VERBATIM
	    COMMENT "Generating ${_gen}" )
    endif ()

    add_custom_target ( NAME.${PROJECT_NAME}.${_gen} ALL
	DEPENDS ${_plan}
	VERBATIM )

endfunction ()


##  run_hipify_perl() is a simple function called to run hipify-perl on a CUDA
##  source code file.  The two arguments are the file name root (_program) and
##  suffix (_suffix).  The resulting converted file will be written to
##  ${_program}-hip.${_suffix}.

function ( run_hipify_perl _program _suffix )
    if ( WIN32 )
	## TBD
    else ()
	include ( FindUnixCommands )
	add_custom_command ( OUTPUT ${_program}-hip.${_suffix}
	    COMMAND ${BASH} -c "rm -f ${_program}-hip.${_suffix} && hipify-perl ${CMAKE_CURRENT_SOURCE_DIR}/${_program}.${_suffix} > ${_program}-hip.${_suffix}"
	    DEPENDS ${_program}.${_suffix}
	    VERBATIM
	    COMMENT "Generating ${_program}-hip.${_suffix}" )
    endif ()

endfunction ()


##  add_includes_libs_to_target() is a function called to add include
##  directories, library paths, and libraries to an executable target.  Its
##  purpose is to encapsulate all the information in one place and let each
##  example cmake simply call this to get the appropriate qualifier for each
##  build.

function ( add_includes_libs_to_target _target _stem _prefixes )
    ##  Test _codegen and setup accordingly
    # if ( ${_codegen} STREQUAL "HIP" )
    # 	## run hipify-perl on the test driver
    # 	run_hipify_perl ( ${_target} ${_suffix} )
    # 	list ( APPEND _all_build_srcs ${_target}-hip.${_suffix} )
    # 	set_source_files_properties ( ${_target}-hip.${_suffix} PROPERTIES LANGUAGE CXX )
    # 	foreach ( _pref ${_prefixes} )
    # 	    set_source_files_properties ( ${_pref}.${_stem}.source.${_suffix} PROPERTIES LANGUAGE CXX )
    # 	endforeach ()
    # else ()
	list ( APPEND _all_build_srcs ${_target}.${_suffix} )
    # endif ()

    add_executable   ( ${_target} ${_all_build_srcs} )
    ##  message ( STATUS "executable added: target = ${_target}, depends: = ${_all_build_srcs}" )
    ##  message ( STATUS "dependencies for ${_target} = ${_all_build_deps}" )
    if ( NOT "X${_all_build_deps}" STREQUAL "X" )
	##  we have some dependencies
	add_dependencies ( ${_target} ${_all_build_deps} )
    endif ()
 
    target_compile_options ( ${_target} PRIVATE ${ADDL_COMPILE_FLAGS} )
 
    target_include_directories ( ${_target} PRIVATE
	${${PROJECT_NAME}_BINARY_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${_library_includes} )

    if ( ${_codegen} STREQUAL "HIP" )
	target_link_directories    ( ${_target} PRIVATE $ENV{ROCM_PATH}/lib )
	target_link_libraries      ( ${_target} PRIVATE ${LIBS_FOR_HIP} )
    elseif ( ${_codegen} STREQUAL "CUDA" )
	target_link_libraries      ( ${_target} PRIVATE ${LIBS_FOR_CUDA} )
    elseif ( ${_codegen} STREQUAL "SYCL" )
	target_link_libraries      ( ${_target} PRIVATE ${LIBS_FOR_SYCL} )
        target_link_options        ( ${_target} PRIVATE "-fsycl" )
    elseif ( ${_codegen} STREQUAL "CPU" )
        if ( NOT WIN32 )
	    target_link_libraries      ( ${_target} PRIVATE dl )
        endif ()
    endif ()
    if ( NOT "X{_library_names}" STREQUAL "X" )
	##  Some libraries were built -- add them for linker
	target_link_libraries      ( ${_target} PRIVATE ${_library_names} )
	##  message ( STATUS "${_target}: Libraries added = ${_library_names}" )
    endif ()

    set ( INSTALL_DIR_TARGET ${CMAKE_INSTALL_PREFIX}/bin )
    install ( TARGETS ${_target} DESTINATION ${INSTALL_DIR_TARGET} )

endfunction ()

##  add_mpi_decorations_to_target() -- Add MPI include directories, compile flags,
##  link options, and libraries to target

function ( add_mpi_decorations_to_target _target )
    if (${MPI_FOUND} )
	##  MPI installation found -- add the libraries and include for this target
	target_include_directories ( ${_target} PRIVATE MPI::MPI_CXX )
	##  target_compile_options     ( ${_target} PRIVATE ${MPI_CXX_COMPILE_OPTIONS} )
	##  target_link_options        ( ${_target} PRIVATE ${MPI_CXX_LINK_FLAGS} )
	##  link flags are wrong on thom
	target_link_libraries      ( ${_target} PRIVATE MPI::MPI_CXX ${ADDL_MPI_LIBS} )
        if ( NOT "X{_library_names}" STREQUAL "X" )
            target_link_libraries      ( ${_target} PRIVATE ${_library_names} )
        endif ()
    else ()
	message ( STATUS "MPI was not found -- cannot add decorations for target = ${_target}" )
    endif ()
endfunction ()


##  manage_deps_codegen() is a function called to orchestrate creating the
##  intermediate files (targets) for codegen and build the list of dependencies
##  for a test program.  The following conventions are assumed:
##  File naming convention is: <prefix>.<stem>.xxxxx (e.g., <prefix>.<stem>.cpp)
##  The function is passed a codegen flag (create CPU/CUDA/HIP code), a stem, a list
##  of prefixes (1 or more) and builds lists of all source code files for the
##  test program and a list of dependency names (to ensure cmake builds all
##  targets in the right order).
##

function ( manage_deps_codegen _codefor _stem _prefixes )
    ##  message ( "manage_deps_codegen: # args = ${ARGC}, code for = ${_codefor}, stem = ${_stem}, prefixes = ${_prefixes}" )
    if ( ${ARGC} LESS 3 )
	message ( FATAL_ERROR "manage_deps_codegen() requires at least 1 prefix" )
    endif ()
    
    ##  if ( ( ${_codefor} STREQUAL "CUDA" ) OR ( ${_codefor} STREQUAL "HIP" ) )
    if ( ${_codefor} STREQUAL "CUDA" )
	if ( "X${_desired_suffix}" STREQUAL "X" )
	    ##  desired suffix is not set -- use defaults
	    set ( _suffix cu PARENT_SCOPE )
	    set ( _suffix cu )
	else ()
	    ##  desired suffix is set -- use it
	    set ( _suffix ${_desired_suffix} PARENT_SCOPE )
	    set ( _suffix ${_desired_suffix} )
	endif ()
    else ()
	if ( "X${_desired_suffix}" STREQUAL "X" )
	    ##  desired suffix is not set -- use defaults
	    set ( _suffix cpp PARENT_SCOPE )
	    set ( _suffix cpp )
	else ()
	    ##  desired suffix is set -- use it
	    set ( _suffix ${_desired_suffix} PARENT_SCOPE )
	    set ( _suffix ${_desired_suffix} )
	endif ()
    endif ()

    foreach ( _prefix ${_prefixes} ) 
	run_driver_program ( ${_prefix} ${_stem} ${_library_includes} )
	set ( _driver ${PROJECT_NAME}.${${_prefix}_driver} )
	set ( _plan ${${_prefix}_plan} )
	set ( _hdr  ${_prefix}.${_stem}.codegen.hpp )

	##  Create the generator scripts: ~.generator.g files

	create_generator_file ( ${_codefor} ${_prefix} ${_stem} )
	set ( _gen ${${_prefix}_gen} )

	##  Create the C source code from the SPIRAL generator script(s)

	set                ( _ccode ${_prefix}.${_stem}.source.${_suffix} )
	file               ( TO_NATIVE_PATH ${${PROJECT_NAME}_BINARY_DIR}/${_gen} _gfile )
	create_source_file ( ${_gfile} ${_ccode} )
	##  message ( STATUS "Create source file ${_ccode} from script ${_gfile}" )
	if ( ${_codefor} STREQUAL "CUDA" )
	    set_source_files_properties ( ${_ccode} PROPERTIES LANGUAGE CUDA )
	endif ()

	##  append to our running lists
	list ( APPEND _all_build_srcs ${_hdr} ${_ccode} )
	list ( APPEND _all_build_deps ${_driver}
               NAME.${PROJECT_NAME}.${_plan}
               NAME.${PROJECT_NAME}.${_gen}
               NAME.${PROJECT_NAME}.${_ccode} )

    endforeach ()

    set ( _all_build_srcs ${_all_build_srcs} PARENT_SCOPE )
    set ( _all_build_deps ${_all_build_deps} PARENT_SCOPE )

endfunction ()


##  manage_add_subdir() is a function to add a subdirectory to the list of
##  examples invoked.  It requires three arguments: subdirectory, buildForCPU,
##  buildForGpu; where subdirectory is the name of the subdirectory containing
##  the example, buildForCpu and buildForGpu are logical (True or False) values
##  indicating if the example should be built.
##  NOTE: If buildForGpu is specified a CUDA (Nvidia) or Hip/rocm (AMD) toolkit is required.

function ( manage_add_subdir _subdir _buildForCpu _buildForGpu )

    if ( ${_buildForCpu} AND ${_codegen} STREQUAL "CPU" )
	message ( STATUS "Adding subdirectory ${_subdir} to build for ${_codegen}" )
	add_subdirectory ( ${_subdir} )
	if ( NOT "X${_library_includes}" STREQUAL "X" )
	    set ( _library_includes ${_library_includes} PARENT_SCOPE )
	endif () 
	if ( NOT "X${_library_names}" STREQUAL "X" )
	    set ( _library_names ${_library_names} PARENT_SCOPE )
	endif () 
    elseif ( NOT ${_buildForCpu} AND ${_codegen} STREQUAL "CPU" )
	message ( STATUS "Do NOT build subdirectory ${_subdir} for ${_codegen}" )
    endif ()

    if ( ${_buildForGpu} AND ( ${_codegen} STREQUAL "CUDA" OR ${_codegen} STREQUAL "HIP" OR ${_codegen} STREQUAL "SYCL" ) )
	message ( STATUS "Adding subdirectory ${_subdir} to build for ${_codegen}" )
	add_subdirectory ( ${_subdir} )
	if ( NOT "X${_library_includes}" STREQUAL "X" )
	    set ( _library_includes ${_library_includes} PARENT_SCOPE )
	endif () 
	if ( NOT "X${_library_names}" STREQUAL "X" )
	    set ( _library_names ${_library_names} PARENT_SCOPE )
	endif () 
    elseif ( NOT ${_buildForGpu} AND ( ${_codegen} STREQUAL "CUDA" OR ${_codegen} STREQUAL "HIP" OR ${_codegen} STREQUAL "SYCL" ) )
	message ( STATUS "Do NOT build subdirectory ${_subdir} for ${_codegen}" )
    endif ()

endfunction ()


##  setup_mpi_variables() is a function to perform variable setup so that MPI
##  examples can be built.  It should only be called if MPI is successfully
##  found.  No arguments are required.

function ( setup_mpi_variables )
    ##  We assume MPI installation found
    message ( STATUS "Found MPI: mpicxx = ${MPI_MPICXX_FOUND}, Version = ${MPI_CXX_VERSION}" )
    ##  message ( STATUS "MPI_CXX_COMPILER = ${MPI_CXX_COMPILER}" )
    ##  message ( STATUS "MPI_CXX_COMPILE_OPTIONS = ${MPI_CXX_COMPILE_OPTIONS}" )
    ##  message ( STATUS "MPI_CXX_COMPILE_DEFINITIONS = ${MPI_CXX_COMPILE_DEFINITIONS}" )
    ##  message ( STATUS "MPI_CXX_INCLUDE_DIRS = ${MPI_CXX_INCLUDE_DIRS}" )
    ##  message ( STATUS "MPI_CXX_LINK_FLAGS = ${MPI_CXX_LINK_FLAGS}" )
    ##  message ( STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}" )

    set ( _index 0 )
    foreach ( _mpilib ${MPI_CXX_LIBRARIES} )
	set ( MPI_CXX_LIB${_index} ${_mpilib} PARENT_SCOPE )
        message ( STATUS "MPI_CXX_LIB${_index} = ${_mpilib}" )
	math ( EXPR _index "${_index} + 1" )
    endforeach ()
    set ( _num_MPI_libs ${_index} )
    message ( STATUS "Number of MPI Libraries = ${_num_MPI_libs}" )

    set ( _index 0 )
    foreach ( _mpiinc ${MPI_CXX_INCLUDE_DIRS} )
	set ( MPI_CXX_INCL${_index} ${_mpiinc} PARENT_SCOPE )
        message ( STATUS "MPI_CXX_INCL${_index} = ${_mpiinc}" )
	math ( EXPR _index "${_index} + 1" )
    endforeach ()
    set ( _num_MPI_incls ${_index} )
    message ( STATUS "Number of MPI Include Dirs = ${_num_MPI_incls}" )
    
endfunction ()


##  FFTX_find_libraries() is a function called to locate the various libraries
##  of transforms built by FFTX.  It relies on FFTX_HOME being set (typically an
##  external application would need this in order to include
##  $FFTX_HOME/CMakeInclude/FFTXCmakeFunctions.cmake in a CMake file anyway).
##  The libraries must be in the 'lib' folder (under $FFTX_HOME).  All libraries
##  found are noted and the include path directive for the library is set to
##  $FFTX_HOME/include (all FFTX public headers are installed there).

function ( FFTX_find_libraries )

    ##  Start by finding FFTX home...

    if ( DEFINED ENV{FFTX_HOME} )
	##  message ( STATUS "FFTX_HOME = $ENV{FFTX_HOME}" )
	set ( FFTX_SOURCE_DIR $ENV{FFTX_HOME} )
    else ()
	if ( "x${FFTX_HOME}" STREQUAL "x" )
            message ( FATAL_ERROR "FFTX_HOME environment variable undefined and not specified on command line" )
	endif ()
	set ( FFTX_SOURCE_DIR ${FFTX_HOME} )
    endif ()

    ##  Find the 'installed' directory containing the libraries
    set (_root_folder "${FFTX_SOURCE_DIR}" )
    if ( NOT IS_DIRECTORY ${_root_folder}/lib )
	message ( SEND_ERROR "${_root_folder}/lib is not a directory -- no libraries found, CANNOT build" )
    endif ()
    
    set ( _add_link_directory )
    set ( _libraries_added )
    set ( _includes_added ${FFTX_SOURCE_DIR}/include )
    
    set ( _lib_found FALSE )
    file ( GLOB _libs RELATIVE ${_root_folder}/lib ${_root_folder}/lib/*fftx_* )
    ##  message ( STATUS "Check for libs in: ${_libs}" )
    foreach ( _lib ${_libs} )
	string ( REGEX REPLACE "^lib"  "" _lib ${_lib} )	## strip leading 'lib' if present
	string ( REGEX REPLACE ".so.*$" "" _lib ${_lib} )	## strip trailing stuff - Linux
	string ( REGEX REPLACE ".dll.*$" "" _lib ${_lib} )	## strip trailing stuff - Windows
	string ( REGEX REPLACE ".dylib.*$" "" _lib ${_lib} )	## strip trailing stuff - MAC
	list ( FIND _libraries_added "${_lib}" _posnlist )
	if ( ${_posnlist} EQUAL -1 )
	    ##  message ( STATUS "${_lib} not in list -- adding" )
	    list ( APPEND _libraries_added "${_lib}" )
	    ##  list ( APPEND _includes_added  "${_root_folder}/examples/library/lib_${_lib}_srcs" )
	    set ( _lib_found TRUE )
	endif ()
    endforeach ()
    if ( ${_lib_found} )
	list ( APPEND _add_link_directory ${_root_folder}/lib )
	message ( STATUS "Add linker dir: ${_add_link_directory}" )
    endif ()

    ##  message ( STATUS "Include paths: ${_includes_added}" )
    ##  message ( STATUS "Libraries found: ${_libraries_added}" )
    ##  message ( STATUS "Library path is: ${_add_link_directory}" )
    
    ##  setup FFTX variables in parent scope for include dirs, library path, and library names
    set ( FFTX_LIB_INCLUDE_PATHS ${_includes_added}     PARENT_SCOPE )
    set ( FFTX_LIB_NAMES         ${_libraries_added}    PARENT_SCOPE )
    set ( FFTX_LIB_LIBRARY_PATH  ${_add_link_directory} PARENT_SCOPE )

endfunction ()


##  FFTX_add_includes_libs_to_target() is a function called to add FFTX include
##  directory paths, FFTX library paths, and FFTX libraries to an executable
##  target.  This will add the required paths and settings to an external target
##  (calls FFTX_find_libraries() first).

function ( FFTX_add_includes_libs_to_target _target )
    
    FFTX_find_libraries ()
    
    target_compile_options     ( ${_target} PRIVATE ${ADDL_COMPILE_FLAGS} )
    target_include_directories ( ${_target} PRIVATE ${FFTX_LIB_INCLUDE_PATHS} )
    target_link_directories    ( ${_target} PRIVATE ${FFTX_LIB_LIBRARY_PATH} )
    target_link_libraries      ( ${_target} PRIVATE ${FFTX_LIB_NAMES} )

endfunction ()
