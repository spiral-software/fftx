##
## Copyright (c) 2018-2024, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

set ( _incl_files fftx3.hpp fftxutilities.hpp doxygen.config )
list ( APPEND _incl_files fftxcpubackend.hpp fftxcudabackend.hpp fftxdftbatlib.hpp fftxfft.hpp
                          fftxhipbackend.hpp fftxinterface.hpp fftxmddftlib.hpp fftxmdprdftlib.hpp
                          fftxtransformlib.hpp fftxsyclbackend.hpp )
list ( APPEND _incl_files fftxbatch1ddftObj.hpp fftxibatch1ddftObj.hpp fftxbatch2ddftObj.hpp fftxibatch2ddftObj.hpp)
list ( APPEND _incl_files fftxbatch1dprdftObj.hpp fftxibatch1dprdftObj.hpp fftxbatch2dprdftObj.hpp fftxibatch2dprdftObj.hpp)
list ( APPEND _incl_files fftxmddftObj.hpp fftximddftObj.hpp fftxmdprdftObj.hpp fftximdprdftObj.hpp)

install ( FILES ${_incl_files}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
