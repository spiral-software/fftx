#!/bin/bash

##
## Copyright (c) 2018-2023, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

##  Set options for building the various FFTX libraries.  These are used to drive what is
##  built for FFTX -- libraries are generated first, then an options configuration CMake
##  file is built to drive what CMake builds (e.g., building examples can be enabled or
##  disabled).

##  Build *all* libraries but only add as many sizes to the library as are needed (perhaps
##  few, if any) to demonstrate the ability to run with both fixed sizes libraries and to
##  generate code (RTC) as needed.  Each library *must* be built to ensure that library's
##  API (including header files) is available.  This will allow complete on-the-fly RTC
##  but also create the necessary APIs for library access thus supporting both fixed-sized
##  libraries and RTC as the user may prefer.

##  Build all examples: If set true the FFTX example programs will be built.  When the
##  flag is set to false, NO example programs will be built.
BUILD_EXAMPLES=true

##  This script accecpts an argument passed on the command line indicating which
##  architecture is to be built.  Choices are { CPU | CUDA | HIP }.  The default is CPU
##  (if nothing is specified).  NOTE: This mechanism is used to facilitate building with
##  spack.

BUILD_FOR_CPU=true
BUILD_FOR_CUDA=false
BUILD_FOR_HIP=false
BUILD_FOR_SYCL=false

if [ $# -gt 0 ]; then
    ##  Command line argument is present
    targ=$1
    if [ "$targ" == "CUDA" ]; then
        BUILD_FOR_CPU=false
        BUILD_FOR_CUDA=true
    elif [ "$targ" == "HIP" ]; then
        BUILD_FOR_CPU=false
        BUILD_FOR_HIP=true
    elif [ "$targ" == "SYCL" ]; then
        BUILD_FOR_CPU=false
        BUILD_FOR_SYCL=true
    ##  else just build for CPU when none of CUDA, HIP, or SYCL are specified
    fi
fi

#############################################################################################
##
##  DO NOT make changes below this line when configuring the FFTX build options
##
#############################################################################################

##  Whenever a library is built both the forward and inverse transforms (when applicable)
##  are built.  The true/false flags associated with each library do not imply building it
##  is optional; rather they are used to manage control logic later. 
##  Build the batch 1D DFT (complex to complex) library
DFTBAT_LIB=true

##  Build the batch 1D packed real DFT (real to complex, complex to real) library
PRDFTBAT_LIB=true

##  Build the 3D DFT (complex to complex) library
MDDFT_LIB=true

##  Build the 3D DFT (real to complex, complex to real) library
MDPRDFT_LIB=true

##  Build the Real Convolution library
RCONV_LIB=true

##  Build the PSATD fixed sizes library
PSATD_LIB=false

##  File containing the sizes to build for the CPU version of MDDFT, MDPRDFT, and RCONV
CPU_SIZES_FILE="cube-sizes-cpu.txt"

##  File containing the sizes to build for the GPU version of MDDFT, MDPRDFT, and RCONV
GPU_SIZES_FILE="cube-sizes-gpu.txt"
##  GPU_SIZES_FILE="cube-sizes.txt"

##  File containing the sizes to build for the CPU version of batch 1D DFT and batch 1D PRDFT
DFTBAT_SIZES_FILE="dftbatch-sizes.txt"

##  File containing the sizes to build for the PSATD library
PSATD_SIZES_FILE="cube-psatd.txt"

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
echo "PSATD_LIB=$PSATD_LIB" >> build-lib-code-options.sh
echo "CPU_SIZES_FILE=$CPU_SIZES_FILE" >> build-lib-code-options.sh
echo "GPU_SIZES_FILE=$GPU_SIZES_FILE" >> build-lib-code-options.sh
echo "DFTBAT_SIZES_FILE=$DFTBAT_SIZES_FILE" >> build-lib-code-options.sh
echo "PSATD_SIZES_FILE=$PSATD_SIZES_FILE" >> build-lib-code-options.sh

popd

##  Make a directory for cached JIT files (may be written by code gen from build-lib-code or
##  at run time to cache files created during the RTC process
if [ -n "$FFTX_HOME" ]; then
    ##  FFTX_HOME is set
    echo "FFTX_HOME = $FFTX_HOME"
else
    echo "FFTX_HOME is not set; set it to `pwd`"
    export FFTX_HOME=`pwd`
fi
mkdir -p -v ${FFTX_HOME}/cache_jit_files

echo "Build for CPU = $BUILD_FOR_CPU"
if [ "$BUILD_FOR_CPU" = true ]; then
    ##  Build the libraries for CPU
    pushd src/library
    ./build-lib-code.sh "CPU"
    ##  echo "Run ./build-lib-code.sh CPU"
    popd
fi

echo "Build for CUDA = $BUILD_FOR_CUDA"
if [ "$BUILD_FOR_CUDA" = true ]; then
    ##  Build the libraries for CUDA
    pushd src/library
    ./build-lib-code.sh "CUDA"
    ##  echo " Run ./build-lib-code.sh CUDA"
    popd
fi

echo "Build for HIP = $BUILD_FOR_HIP"
if [ "$BUILD_FOR_HIP" = true ]; then
    ##  Build the libraries for HIP
    pushd src/library
    ./build-lib-code.sh "HIP"
    ##  echo "Run ./build-lib-code.sh HIP"
    popd
fi

echo "Build for SYCL = $BUILD_FOR_SYCL"
if [ "$BUILD_FOR_SYCL" = true ]; then
    ##  Build the libraries for SYCL
    pushd src/library
    ./build-lib-code.sh "SYCL"
    ##  echo "Run ./build-lib-code.sh SYCL"
    popd
fi

rm -f options.cmake
touch options.cmake

echo "##" >> options.cmake
echo "## Copyright (c) 2018-2023, Carnegie Mellon University" >> options.cmake
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
