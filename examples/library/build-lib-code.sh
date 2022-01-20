
#! /bin/sh

##  Script to generate the source code for the libraries

##  Exit code indicates the appropriate code gen flag for cmake:
##  0 ==> -D_codegen=CPU | 1 ==> -D_codegen=CUDA | 2 ==> -D_codegen=HIP

retc=0

echo "Always create the CPU code ... commands:"

python gen_files.py fftx_mddft cpu true &
python gen_files.py fftx_mddft cpu false &
python gen_files.py fftx_mdprdft cpu true &
python gen_files.py fftx_mdprdft cpu false &
python gen_files.py fftx_rconv cpu true &

wait		##  wait for the chile processes to complete

##  Build DFT batch for CPU

python gen_dftbat.py fftx_dftbat cpu true &
python gen_dftbat.py fftx_dftbat cpu false &
python gen_dftbat.py fftx_prdftbat cpu true &
python gen_dftbat.py fftx_prdftbat cpu false &

wait

##  Create code for a GPU if we find an appropriate compiler ...

which nvcc
if [ $? -eq 0 ]; then
    ##  nvcc found -- generate Nvidia GPU (CUDA) code
    echo "nvcc found ... generate CUDA code ... commands:"
    python gen_files.py fftx_mddft cuda true &
    python gen_files.py fftx_mddft cuda false &
    python gen_files.py fftx_mdprdft cuda true &
    python gen_files.py fftx_mdprdft cuda false &
    python gen_files.py fftx_rconv cuda true &
    wait
    retc=1
else
    echo "NVCC was not found -- DO NOT generate CUDA code"
fi

which hipcc
if [ $? -eq 0 ]; then
    ##  hipcc found -- generate AMD GPU (HIP) code
    echo "hipcc found ... generate HIP code ... commands:"
    python gen_files.py fftx_mddft hip true &
    python gen_files.py fftx_mddft hip false &
    python gen_files.py fftx_mdprdft hip true &
    python gen_files.py fftx_mdprdft hip false &
    python gen_files.py fftx_rconv hip true &
    wait
    retc=2
else
    echo "HIPCC was not found -- DO NOT generate HIP code"
fi

exit $retc
