
#! /bin/sh

##  Script to generate the source code for the libraries

##  Returns the appropriate code gen flag for cmake:
##    -D_codegen=CPU | -D_codegen=GPU | -D_codegen=HIP

FIL=precmake$$.txt

echo "Always create the CPU code ... commands:" > $FIL

echo "python gen_files.py fftx_mddft cpu true nogen" >> $FIL
echo "python gen_files.py fftx_mddft cpu false nogen" >> $FIL
echo "python gen_files.py fftx_mdprdft cpu true nogen" >> $FIL
echo "python gen_files.py fftx_mdprdft cpu false nogen" >> $FIL
echo "python gen_files.py fftx_rconv cpu true nogen" >> $FIL

##  Build DFT batch for CPU

echo "python gen_dftbat.py fftx_dftbat cpu true nogen" >> $FIL
echo "python gen_dftbat.py fftx_dftbat cpu false nogen" >> $FIL
echo "python gen_dftbat.py fftx_prdftbat cpu true nogen" >> $FIL
echo "python gen_dftbat.py fftx_prdftbat cpu false nogen" >> $FIL

CODEGEN="-D_codegen=CPU"

##  Create code for a GPU if we find an appropriate compiler ...

which nvcc >> $FIL
if [ $? -eq 0 ]; then
    ##  nvcc found -- generate Nvidia GPU (CUDA) code
    echo "nvcc found ... generate CUDA code ... commands:" >> $FIL
    echo "python gen_files.py fftx_mddft cuda true nogen" >> $FIL
    echo "python gen_files.py fftx_mddft cuda false nogen" >> $FIL
    echo "python gen_files.py fftx_mdprdft cuda true nogen" >> $FIL
    echo "python gen_files.py fftx_mdprdft cuda false nogen" >> $FIL
    echo "python gen_files.py fftx_rconv cuda true nogen" >> $FIL
    CODEGEN="-D_codegen=GPU"
else
    echo "NVCC was not found -- DO NOT generate CUDA code" >> $FIL
fi

which hipcc >> $FIL
if [ $? -eq 0 ]; then
    ##  hipcc found -- generate AMD GPU (HIP) code
    echo "hipcc found ... generate HIP code ... commands:" >> $FIL
    echo "python gen_files.py fftx_mddft hip true nogen" >> $FIL
    echo "python gen_files.py fftx_mddft hip false nogen" >> $FIL
    echo "python gen_files.py fftx_mdprdft hip true nogen" >> $FIL
    echo "python gen_files.py fftx_mdprdft hip false nogen" >> $FIL
    echo "python gen_files.py fftx_rconv hip true nogen" >> $FIL
    CODEGEN="-D_codegen=HIP"
else
    echo "HIPCC was not found -- DO NOT generate HIP code" >> $FIL
fi

echo $CODEGEN
