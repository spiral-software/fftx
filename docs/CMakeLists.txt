##
## Copyright (c) 2018-2023, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

set (CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

##  Find all the public headers
##  get_target_property ( FFTX_PROJECT_PUBLIC_HEADER_DIR FFTX_PROJECT INTERFACE_INCLUDE_DIRECTORIES )
##  set ( FFTX_PROJECT_PUBLIC_HEADER_DIR ${FFTX_INCLUDE} )
##  Use the <proj>/src/include directory (instead of <proj>/include) in case files not installed
set ( FFTX_PROJECT_PUBLIC_HEADER_DIR ${FFTX_PROJECT_SOURCE_DIR}/src/include )

file ( GLOB_RECURSE FFTX_PROJECT_PUBLIC_HEADERS 
       ${FFTX_PROJECT_PUBLIC_HEADER_DIR}/*.hpp
       ${FFTX_PROJECT_PUBLIC_HEADER_DIR}/*.H
       ${FFTX_PROJECT_PUBLIC_HEADER_DIR}/*.h)

message ( STATUS "Found public header files: ${FFTX_PROJECT_PUBLIC_HEADERS}" )

##  Setup locations for Doxygen
set ( DOXYGEN_INPUT_DIR ${FFTX_PROJECT_PUBLIC_HEADER_DIR} )
set ( DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen )
set ( DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml )
set ( DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in )
set ( DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile )

##  Set the Doxygen input and output directories in the Doxyfile
configure_file ( ${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY )

##  Doxygen won't create this for us
file ( MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR} ) 

##  Regenerate Doxygen when the Doxyfile or public headers change
add_custom_command ( OUTPUT ${DOXYGEN_INDEX_FILE}
		DEPENDS ${FFTX_PROJECT_PUBLIC_HEADERS}
        	COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
		MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
        	COMMENT "Generating FFTX docs"
		VERBATIM )

##  Add a command to generate README.rst; converts main FFTX README.md (Github markdown to rst)
include ( FindUnixCommands )
add_custom_command ( OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/README.rst
                     DEPENDS ${FFTX_PROJECT_SOURCE_DIR}/README.md
                     COMMAND pandoc --from=gfm --to=rst --output=${CMAKE_CURRENT_SOURCE_DIR}/README.rst ${FFTX_PROJECT_SOURCE_DIR}/README.md
                     COMMENT "Generate FFTX Readme in RST format" )

##  Named target so we can run the job easily
add_custom_target ( Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE} )

set ( SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} )
set ( SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx )
set ( SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html )

set ( DOC_RST_FILES apis.rst bibliography.rst contribute.rst distributed.rst extending.rst
                    faq.rst gethelp.rst index.rst install.rst introduction.rst team.rst )

##  Regenerate Sphinx when:
##    - Doxygen has rerun 
##    - Our doc files have been updated
##    - The Sphinx config has been updated

add_custom_command ( OUTPUT ${SPHINX_INDEX_FILE}
	COMMAND 
		${SPHINX_EXECUTABLE} -b html
		# Tell Breathe where to find the Doxygen output
		-Dbreathe_projects.FFTX=${DOXYGEN_OUTPUT_DIR}/xml
		${SPHINX_SOURCE} ${SPHINX_BUILD}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	DEPENDS 
		${DOC_RST_FILES}                ##  RST doc files (listed above)
		${CMAKE_CURRENT_SOURCE_DIR}/README.rst
		${DOXYGEN_INDEX_FILE}
	MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
	COMMENT "Generating documentation with Sphinx")

##  Named target so we can run the job easily
add_custom_target ( Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE}
                    COMMENT "Documentation is available in ${SPHINX_BUILD}" )

##  include ( GNUInstallDirs )
##  install ( DIRECTORY ${SPHINX_BUILD}
##  	  DESTINATION ${CMAKE_INSTALL_DOCDIR} )

