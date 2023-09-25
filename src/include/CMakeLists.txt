##
## Copyright (c) 2018-2022, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

set ( _incl_files fftx3.hpp fftx3utilities.h doxygen.config )
list ( APPEND _incl_files cpubackend.hpp cudabackend.hpp dftbatlib.hpp fftxfft.hpp
                          hipbackend.hpp interface.hpp mddftlib.hpp mdprdftlib.hpp
                          transformlib.hpp )
list ( APPEND _incl_files batch1ddftObj.hpp ibatch1ddftObj.hpp batch2ddftObj.hpp ibatch2ddftObj.hpp)
list ( APPEND _incl_files batch1dprdftObj.hpp ibatch1dprdftObj.hpp batch2dprdftObj.hpp ibatch2dprdftObj.hpp)
list ( APPEND _incl_files mddftObj.hpp imddftObj.hpp mdprdftObj.hpp imdprdftObj.hpp)

install ( FILES ${_incl_files}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
