
#! /bin/sh

##  Script to generate the source code for the libraries

##  Return code from this script indicate if CPU, CUDA, or HIP code generated
##    0 ==> Only CPU code generated
##    1 ==> CUDA code generated
##    2 ==> HIP code generated

retcode=0

echo "Always create the CPU code ... commands:"

echo "python gen_files.py fftx_mddft cpu true nogen"
echo "python gen_files.py fftx_mddft cpu false nogen"
echo "python gen_files.py fftx_mdprdft cpu true nogen"
echo "python gen_files.py fftx_mdprdft cpu false nogen"
echo "python gen_files.py fftx_rconv cpu true nogen"

##  Build DFT batch for CPU

echo "python gen_dftbat.py fftx_dftbat cpu true nogen"
echo "python gen_dftbat.py fftx_dftbat cpu false nogen"
echo "python gen_dftbat.py fftx_prdftbat cpu true nogen"
echo "python gen_dftbat.py fftx_prdftbat cpu false nogen"

##  Create code for a GPU if we find an appropriate compiler ...

which nvcc
if [ $? -eq 0 ]; then
    ##  nvcc found -- generate Nvidia GPU (CUDA) code
    echo "nvcc found ... generate CUDA code ... commands:"
    echo "python gen_files.py fftx_mddft cuda true nogen"
    echo "python gen_files.py fftx_mddft cuda false nogen"
    echo "python gen_files.py fftx_mdprdft cuda true nogen"
    echo "python gen_files.py fftx_mdprdft cuda false nogen"
    echo "python gen_files.py fftx_rconv cuda true nogen"
    retcode=1
else
    echo "NVCC was not found -- DO NOT generate CUDA code"
fi

which hipcc
if [ $? -eq 0 ]; then
    ##  hipcc found -- generate AMD GPU (HIP) code
    echo "hipcc found ... generate HIP code ... commands:"
    echo "python gen_files.py fftx_mddft hip true nogen"
    echo "python gen_files.py fftx_mddft hip false nogen"
    echo "python gen_files.py fftx_mdprdft hip true nogen"
    echo "python gen_files.py fftx_mdprdft hip false nogen"
    echo "python gen_files.py fftx_rconv hip true nogen"
    retcode=2
else
    echo "HIPCC was not found -- DO NOT generate HIP code"
fi

exit $retcode
