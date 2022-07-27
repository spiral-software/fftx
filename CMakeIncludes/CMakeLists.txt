##
## Copyright (c) 2018-2022, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

set ( _incl_files FFTXCmakeFunctions.cmake FFTXSetup.cmake FindSphinx.cmake )

install ( FILES ${_incl_files}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/CMakeIncludes )

