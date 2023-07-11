#!/bin/bash

##
## Copyright (c) 2018-2022, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

##  Set options for building the various FFTX libraries.  These are used to drive what is
##  built for FFTX -- libraries are generated first, then an options configuration CMake
##  file is built to drive what CMake builds (e.g., some examples cannot be built unless
##  all the pre-requisite libraries are built).

##  Build the libraries but only add a few [typically small] sizes to the library to
##  demonstrate the ability to run with both fixed sizes libraries and to generate code
##  (RTC) as needed.  Ultimately, create all libraries but with no fixed transforms in
##  them -- this will allow complete on the fly RTC but create the necessary APIs for
##  library access thus supporting fboth fixed-sixed libraries and RTC as the user may
##  prefer. 

##  Whenever a library is built both the forward and inverse transforms (when applicable)
##  are built
##  Build the batch 1D DFT (complex to complex) library
DFTBAT_LIB=true

##  Build the batch 1D packed real DFT (real to complex, complex to real) library
PRDFTBAT_LIB=false

##  Build the 3D DFT (complex to complex) library
MDDFT_LIB=true

##  Build the 3D DFT (real to complex, complex to real) library
MDPRDFT_LIB=true

##  Build the Real Convolution library
RCONV_LIB=true

##  Deprecated - do not build -- will be removed in next release
##  Build the MPI distributed FFT library (MPI is required to build examples)
##  DISTDFT_LIB=false

##  Build the PSATD fixed sizes library
PSATD_LIB=false

##  File containing the sizes to build for the CPU version of MDDFT, MDPRDFT, and RCONV
CPU_SIZES_FILE="cube-sizes-cpu.txt"

##  File containing the sizes to build for the GPU version of MDDFT, MDPRDFT, and RCONV
GPU_SIZES_FILE="cube-sizes-gpu.txt"
##  GPU_SIZES_FILE="cube-sizes.txt"

##  File containing the sizes to build for the CPU version of batch 1D DFT and batch 1D PRDFT
DFTBAT_SIZES_FILE="dftbatch-sizes.txt"

##  Deprecated - do not build -- will be removed in next release
##  File containing the sizes to build for the distributed FFT library
##  DISTDFT_SIZES_FILE="distdft-sizes.txt"

##  File containing the sizes to build for the PSATD library
PSATD_SIZES_FILE="cube-psatd.txt"

##  Build all examples: If set true the FFTX example programs will be built.  NOTE: Only
##  those example programs which have their dependent libraries will be built.  Some
##  examples do not depend on any of the libraries and these will also be built.  When the
##  flag is set to false, NO example programs will be built.
BUILD_EXAMPLES=true

##Build FFTX for CPU
BUILD_FOR_CPU=false

##  Build FFTX for CUDA
BUILD_FOR_CUDA=false

##  Build FFTX for HIP
BUILD_FOR_HIP=true

#############################################################################################
##
##  DO NOT make changes below this line when configuring the FFTX build options
##
#############################################################################################

##  Write the library build options and default cube size filenames
pushd src/library
rm -f build-lib-code-options.sh
touch build-lib-code-options.sh

echo "#!/bin/bash" >> build-lib-code-options.sh

echo "DFTBAT_LIB=$DFTBAT_LIB" >> build-lib-code-options.sh
echo "PRDFTBAT_LIB=$PRDFTBAT_LIB" >> build-lib-code-options.sh
echo "MDDFT_LIB=$MDDFT_LIB" >> build-lib-code-options.sh
echo "MDPRDFT_LIB=$MDPRDFT_LIB" >> build-lib-code-options.sh
echo "RCONV_LIB=$RCONV_LIB" >> build-lib-code-options.sh
##  Deprecated - do not build -- will be removed in next release
##  echo "DISTDFT_LIB=$DISTDFT_LIB" >> build-lib-code-options.sh
echo "PSATD_LIB=$PSATD_LIB" >> build-lib-code-options.sh
echo "CPU_SIZES_FILE=$CPU_SIZES_FILE" >> build-lib-code-options.sh
echo "GPU_SIZES_FILE=$GPU_SIZES_FILE" >> build-lib-code-options.sh
echo "DFTBAT_SIZES_FILE=$DFTBAT_SIZES_FILE" >> build-lib-code-options.sh
##  Deprecated - do not build -- will be removed in next release
##  echo "DISTDFT_SIZES_FILE=$DISTDFT_SIZES_FILE" >> build-lib-code-options.sh
echo "PSATD_SIZES_FILE=$PSATD_SIZES_FILE" >> build-lib-code-options.sh

popd

##  Make a directory for cached JIT files (may be written by code gen from build-lib-code or
##  at run time to cache files created during the RTC process
if [ -v FFTX_HOME ]; then
    ##  FFTX_HOME is set
    echo "FFTX_HOME = $FFTX_HOME"
else
    echo "FFTX_HOME is not set; set it to `pwd`"
    export FFTX_HOME=`pwd`
fi
mkdir ${FFTX_HOME}/cache_jit_files

echo "Build for CPU = $BUILD_FOR_CPU"
if [ "$BUILD_FOR_CPU" = true ]; then
    ##  Build the libraries for CPU
    pushd src/library
    ./build-lib-code.sh "CPU"
    popd
fi

echo "Build for CUDA = $BUILD_FOR_CUDA"
if [ "$BUILD_FOR_CUDA" = true ]; then
    ##  Build the libraries for CUDA
    pushd src/library
    ./build-lib-code.sh "CUDA"
    popd
fi

echo "Build for HIP = $BUILD_FOR_HIP"
if [ "$BUILD_FOR_HIP" = true ]; then
    ##  Build the libraries for HIP
    pushd src/library
    ./build-lib-code.sh "HIP"
    popd
fi

rm -f options.cmake
touch options.cmake

echo "##" >> options.cmake
echo "## Copyright (c) 2018-2022, Carnegie Mellon University" >> options.cmake
echo "## All rights reserved." >> options.cmake
echo "##" >> options.cmake
echo "## See LICENSE file for full information" >> options.cmake
echo "##" >> options.cmake

if [ "$DFTBAT_LIB" = true ]; then
    setopt="ON"
else
    setopt="OFF" 
fi
echo "option ( DFTLIB_BAT \"Build the batch 1D DFT (complex to complex) library\" $setopt )" >> options.cmake 

if [ "$PRDFTBAT_LIB" = true ]; then
    setopt="ON"
else
    setopt="OFF"
fi
echo "option ( PRDFTBAT_LIB \"Build the batch 1D packed real DFT (real to complex, complex to real) library\" $setopt )" >> options.cmake

if [ "$MDDFT_LIB" = true ]; then
    setopt="ON"
else
    setopt="OFF"
fi
echo "option ( MDDFT_LIB \"Build the 3D DFT (complex to complex) library\" $setopt )" >> options.cmake

if [ "$MDPRDFT_LIB" = true ]; then
    setopt="ON"
else
    setopt="OFF"
fi
echo "option ( MDPRDFT_LIB \"Build the 3D DFT (real to complex, complex to real) library\" $setopt )" >> options.cmake

if [ "$RCONV_LIB" = true ]; then
    setopt="ON"
else
    setopt="OFF"
fi
echo "option ( RCONV_LIB \"Build the Real Convolution library\" $setopt )" >> options.cmake

##  Deprecated - do not build -- will be removed in next release
##  if [ "$DISTDFT_LIB" = true ]; then
##      setopt="ON"
##  else
##      setopt="OFF"
##  fi
##  echo "option ( DISTDFT_LIB \"Build the MPI distributed FFT library (MPI is required to build examples)\" $setopt )" >> options.cmake

if [ "$PSATD_LIB" = true ]; then
    setopt="ON"
else
    setopt="OFF"
fi
echo "option ( PSATD_LIB \"Build the PSATD library\" $setopt )" >> options.cmake

if [ "$BUILD_EXAMPLES" = true ]; then
    setopt="ON"
else
    setopt="OFF"
fi
echo "option ( BUILD_EXAMPLES \"Build the FFTX example programs\" $setopt )" >> options.cmake
