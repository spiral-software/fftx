
#! /bin/sh

##  Script to generate the source code for the libraries

##  Exit code indicates the appropriate code gen flag for cmake:
##  0 ==> -D_codegen=CPU | 1 ==> -D_codegen=GPU | 2 ==> -D_codegen=HIP

retc=0

echo "Always create the CPU code ... commands:"

python gen_files.py fftx_mddft cpu true nogen
python gen_files.py fftx_mddft cpu false nogen
python gen_files.py fftx_mdprdft cpu true nogen
python gen_files.py fftx_mdprdft cpu false nogen
python gen_files.py fftx_rconv cpu true nogen

##  Build DFT batch for CPU

python gen_dftbat.py fftx_dftbat cpu true nogen
python gen_dftbat.py fftx_dftbat cpu false nogen
python gen_dftbat.py fftx_prdftbat cpu true nogen
python gen_dftbat.py fftx_prdftbat cpu false nogen

##  Create code for a GPU if we find an appropriate compiler ...

which nvcc
if [ $? -eq 0 ]; then
    ##  nvcc found -- generate Nvidia GPU (CUDA) code
    echo "nvcc found ... generate CUDA code ... commands:"
    python gen_files.py fftx_mddft cuda true nogen
    python gen_files.py fftx_mddft cuda false nogen
    python gen_files.py fftx_mdprdft cuda true nogen
    python gen_files.py fftx_mdprdft cuda false nogen
    python gen_files.py fftx_rconv cuda true nogen
    retc=1
else
    echo "NVCC was not found -- DO NOT generate CUDA code"
fi

which hipcc
if [ $? -eq 0 ]; then
    ##  hipcc found -- generate AMD GPU (HIP) code
    echo "hipcc found ... generate HIP code ... commands:"
    python gen_files.py fftx_mddft hip true nogen
    python gen_files.py fftx_mddft hip false nogen
    python gen_files.py fftx_mdprdft hip true nogen
    python gen_files.py fftx_mdprdft hip false nogen
    python gen_files.py fftx_rconv hip true nogen
    retc=2
else
    echo "HIPCC was not found -- DO NOT generate HIP code"
fi

exit $retc
