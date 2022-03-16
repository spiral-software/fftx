
#! /bin/sh

##  Script to generate the source code for the libraries

##  Exit code indicates the appropriate code gen flag for cmake:
##  0 ==> -D_codegen=CPU | 1 ==> -D_codegen=CUDA | 2 ==> -D_codegen=HIP

retc=0

##  A user may have both python & python3 in their PATH; try to find a python version 3...

pyexe="not_found"
trypy="python"
which $trypy > /dev/null

if [ $? -eq 0 ]; then
    ##  Found $trypy, test its version
    vpy=`$trypy --version 2>&1 | sed -e 's/Python //' | sed -e 's/\..*//'`
    if [ $vpy -eq "2" ]; then
	##  python is version 2, look for python3 executable
	trypy="python3"
    elif [ $vpy -eq "3" ]; then
	##  found python is version 3...
	pyexe = $trypy
    fi
else
    ##  $trypy NOT FOUND, look for python3...
    trypy="python3"
fi
if [ $pyexe == "not_found" ]; then
    which $trypy > /dev/null
    if [ $? -ne 0 ]; then
	echo "NO suitable python executable found ... exiting"
	exit 9
    fi
    ##  Found $trypy, use it...
    pyexe=$trypy
fi

echo "Python executable is $pyexe"

##  Use a file named build-lib-code-failures.txt to note any files that fail to
##  generate -- failures are noted & written by the python drivers: gen_{files|dftbat}.py

rm -f build-lib-code-failures.txt
touch build-lib-code-failures.txt

echo "Always create the CPU code ... commands:"

$pyexe gen_files.py fftx_mddft cpu true &
$pyexe gen_files.py fftx_mddft cpu false &
$pyexe gen_files.py fftx_mdprdft cpu true &
$pyexe gen_files.py fftx_mdprdft cpu false &
$pyexe gen_files.py fftx_rconv cpu true &

wait		##  wait for the chile processes to complete

##  Build DFT batch for CPU

$pyexe gen_dftbat.py fftx_dftbat cpu true &
$pyexe gen_dftbat.py fftx_dftbat cpu false &
$pyexe gen_dftbat.py fftx_prdftbat cpu true &
$pyexe gen_dftbat.py fftx_prdftbat cpu false &

wait

##  Create code for a GPU if we find an appropriate compiler ...

which nvcc
if [ $? -eq 0 ]; then
    ##  nvcc found -- generate Nvidia GPU (CUDA) code
    echo "nvcc found ... generate CUDA code ... commands:"
    $pyexe gen_files.py fftx_mddft cuda true &
    $pyexe gen_files.py fftx_mddft cuda false &
    $pyexe gen_files.py fftx_mdprdft cuda true &
    $pyexe gen_files.py fftx_mdprdft cuda false &
    $pyexe gen_files.py fftx_rconv cuda true &
    wait
    retc=1
else
    echo "NVCC was not found -- DO NOT generate CUDA code"
fi

which hipcc
if [ $? -eq 0 ]; then
    ##  hipcc found -- generate AMD GPU (HIP) code
    echo "hipcc found ... generate HIP code ... commands:"
    $pyexe gen_files.py fftx_mddft hip true &
    $pyexe gen_files.py fftx_mddft hip false &
    $pyexe gen_files.py fftx_mdprdft hip true &
    $pyexe gen_files.py fftx_mdprdft hip false &
    $pyexe gen_files.py fftx_rconv hip true &
    wait
    retc=2
else
    echo "HIPCC was not found -- DO NOT generate HIP code"
fi

exit $retc
