#! /bin/bash

##  This script will build one of the examples for HIP

##  Assumptions:
##  Using hipify-perl to convert CUDA source to Hip compatible source
##  Steps:
##      Build CUDA version (e.g., on Summit since CUDA/nvcc required)
##      cd to the example in the build tree, e.g., cd buildGPU/examples/mddft
##      Run this script
##  This script will:
##      use the basename to determine the example being built (e.g., mddft)
##      create a sub folder (hip) to hold the hipify'ed code
##      build the executable (e.g., testmddft) and install it in the bin folder with a '-hip' suffix
##

##  Determine the name of the example we're building

_lpwd=`pwd`
_examp=`(basename ${_lpwd})`
_exec="test"${_examp}

if [ "spock" == `hostname -d | sed -e 's/\..*$//'` ]; then
    ##  We can test for the rocm module on spock
    if ! module is-loaded rocm ; 
    then 
	echo "Hip/rocm module is not loaded; please setup your module environment first"
	exit 1
    fi
fi

##  Translate the known sources from CUDA to HIP
if [ -d "hip" ];
then
    rm -rf hip
fi
mkdir hip

for xx in *.hpp *.cu
do 
    hipify-perl $xx > hip/$xx
done

echo "#include <hip/hip_runtime.h>" > hip/${_exec}.cu
hipify-perl ../../../examples/${_examp}/${_exec}.cu >> hip/${_exec}.cu

hipcc -I ${ROCM_PATH}/hipfft/include -I ../../../include -I ../../../examples/${_examp} -L ${ROCM_PATH}/lib -lhipfft -lrocfft -std=c++11 hip/*.cu -O3 -o hip/${_exec}

if [ -f hip/${_exec} ];
then
    cp -p hip/${_exec} ../../bin/${_exec}-hip
fi

exit 0

