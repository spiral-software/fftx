#! /bin/bash

##  This script will build one of the examples for HIP

##  Assumptions:
##  Using hipify-perl to convert CUDA source to Hip compatible source
##  Steps:
##      Build CUDA version
##      cd to the example in the build tree, e.g., cd buildGPU/examples/test_plan_dft
##      Run this script
##  This script will:
##      use the basename to determine the example being built (e.g., test_plan_dft)
##      create a sub folder (hip) to hold the hipify'ed code
##      build the executable (e.g., testmddft) and install it in the bin folder with a '-hip' suffix
##

##  Determine the name of the example we're building

_lpwd=`pwd`
_examp=`(basename ${_lpwd})`
_exec="test"${_examp}
if [ ${_examp} == "test_plan_dft" ];
then
    _exec="testmddft"
fi

if ! module is-loaded hip; 
then 
    echo "Hip module is not loaded; please setup your module environment first"
    exit 1
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

hipcc -I ../../../include -I ../../../examples/${_examp} -std=c++11 hip/*.cu -O3 -o hip/${_exec}
if [ -f hip/${_exec} ];
then
    cp -p hip/${_exec} ../../bin/${_exec}-hip
fi

exit 0
