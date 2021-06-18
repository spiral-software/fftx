##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

include ( ../../ExamplesCommon.cmake )

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  ===== For most examples you should not need to modify anything ABOVE this line =====

if ( ${_codegen} STREQUAL "GPU" )
    set ( _lang_add LANGUAGES CUDA CXX )
    include ("${SPIRAL_SOURCE_DIR}/config/CMakeIncludes/SetupCudaDefines.cmake")
    setup_cuda_variables_for_spiral ()
    message ( STATUS "SPIRAL_CUDA_INCLUDE_DIRS = ${SPIRAL_CUDA_INCLUDE_DIRS}" )
endif ()

##  Set the project name.  Preferred name is just the *name* of the example folder 
project ( 3DFFT ${_lang_add} )

##  File naming convention is: <prefix>.<stem>.xxxxx; set _stem and _prefix(es)
##  Test program name is:  test<project name>.{cpp|cu}

set ( _stem fftx )
set ( _prefixes mddft3d )
set ( BUILD_PROGRAM testmddft3d )

##  ===== For most examples you should not need to modify anything below this line =====

set ( _libname ${PROJECT_NAME}mpi )
message ( STATUS "Project = '${PROJECT_NAME}', Local library name = '${_libname}'" )
set ( SRCS ../common/gpu.cu ../common/mddft3d.cpp )

set ( _generator_script COMPLETE )		## generator script is complete, do not add preamble

manage_deps_codegen ( ${_codegen} ${_stem} "${_prefixes}" )
##  list ( APPEND _all_build_srcs ${BUILD_PROGRAM}.cpp )		## ${_suffix}

add_library                ( ${_libname} STATIC ${SRCS} ${_all_build_srcs} )
target_compile_options     ( ${_libname} PRIVATE ${ADDL_COMPILE_FLAGS} )
target_include_directories ( ${_libname}
                             PUBLIC
			         ${${PROJECT_NAME}_BINARY_DIR}
				 ${${PROJECT_NAME}_SOURCE_DIR}
				 ${${PROJECT_NAME}_SOURCE_DIR}/../common
				 ${SPIRAL_CUDA_INCLUDE_DIRS}
				 ${MPI_CXX_INCLUDE_DIRS}
				 ${CMAKE_BINARY_DIR} )
add_dependencies           ( ${_libname} ${_all_build_deps} )

##  Add custom command to run mpicxx to build the test program...

if ( WIN32 )
    add_custom_command ( OUTPUT ${BUILD_PROGRAM}
	COMMAND IF EXIST ${CMAKE_CURRENT_BINARY_DIR}/${BUILD_PROGRAM} ( DEL /F ${CMAKE_CURRENT_BINARY_DIR}/${BUILD_PROGRAM} )
	COMMAND ${MPI_CXX_COMPILER} -D__FILEROOT__=${_prefixes}	-I${${PROJECT_NAME}_BINARY_DIR}	-I${${PROJECT_NAME}_SOURCE_DIR}	-I${${PROJECT_NAME}_SOURCE_DIR}/../common -I${SPIRAL_CUDA_INCLUDE_DIRS}	-I${MPI_CXX_INCLUDE_DIRS} -I${CMAKE_BINARY_DIR}	-std=c++11 -o ${CMAKE_CURRENT_BINARY_DIR}/${BUILD_PROGRAM} ${${PROJECT_NAME}_SOURCE_DIR}/${BUILD_PROGRAM}.cpp ${MPI_CXX_LIBRARIES} -L ${SPIRAL_CUDA_LIBRARIES} -lcudart	-L${${PROJECT_NAME}_BINARY_DIR} -l${_libname}
	DEPENDS ${_libname}
	VERBATIM
	COMMENT "Generating ${BUILD_PROGRAM}" )

else ()
    message ( STATUS "Run custom command to build '${BUILD_PROGRAM}'" )
    include ( FindUnixCommands )
    add_custom_command ( OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${BUILD_PROGRAM}
	COMMAND ${BASH} -c "${MPI_CXX_COMPILER} -D__FILEROOT__=${_prefixes} -I${${PROJECT_NAME}_BINARY_DIR} -I${${PROJECT_NAME}_SOURCE_DIR} -I${${PROJECT_NAME}_SOURCE_DIR}/../common -I${SPIRAL_CUDA_INCLUDE_DIRS} -I${MPI_CXX_INCLUDE_DIRS} -I${CMAKE_BINARY_DIR} -std=c++11 -o ${CMAKE_CURRENT_BINARY_DIR}/${BUILD_PROGRAM} ${${PROJECT_NAME}_SOURCE_DIR}/${BUILD_PROGRAM}.cpp ${MPI_CXX_LIBRARIES} -L ${SPIRAL_CUDA_LIBRARIES} -lcudart -L${${PROJECT_NAME}_BINARY_DIR} -l${_libname}" 
	DEPENDS ${_libname}
	VERBATIM
	COMMENT "Generating ${BUILD_PROGRAM}" )

endif ()

add_custom_target ( build_mpiexample ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BUILD_PROGRAM} )

set ( INSTALL_DIR_TARGET ${CMAKE_BINARY_DIR}/bin )

install ( PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${BUILD_PROGRAM} DESTINATION ${INSTALL_DIR_TARGET} )

