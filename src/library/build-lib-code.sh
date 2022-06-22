#! /bin/bash

##  Script to generate the source code for the libraries

##  Expects either 0 or 1 argument:  no arg ==> build CPU code only
##  arg1 = { CUDA | HIP } ==> build code for the respective GPU type

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
	pyexe=$trypy
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

if [ $# -eq 0 ]; then
    build_type="CPU"
else
    if [ $1 = "CPU" ] || [ $1 = "CUDA" ] || [ $1 = "HIP" ]; then
	build_type=$1
    else
	echo "$1 -- build type parameter not recognized, terminating"
	exit 9
    fi
fi

echo -n "Build library for CPU "
if [ $build_type != 'CPU' ]; then
    echo -n "and $build_type"
fi
echo ""

##  Use a file named build-lib-code-failures.txt to note any files that fail to
##  generate -- failures are noted & written by the python drivers: gen_{files|dftbat}.py

rm -f build-lib-code-failures.txt
touch build-lib-code-failures.txt

echo "Always create the CPU code ... commands:"

$pyexe gen_files.py fftx_mddft cube-sizes-cpu.txt cpu true &
$pyexe gen_files.py fftx_mddft cube-sizes-cpu.txt cpu false &
$pyexe gen_files.py fftx_mdprdft cube-sizes-cpu.txt cpu true &
$pyexe gen_files.py fftx_mdprdft cube-sizes-cpu.txt cpu false &
$pyexe gen_files.py fftx_rconv cube-sizes-cpu.txt cpu true &

wait		##  wait for the chile processes to complete

##  Build DFT batch for CPU

$pyexe gen_dftbat.py fftx_dftbat dftbatch-sizes.txt cpu true &
$pyexe gen_dftbat.py fftx_dftbat dftbatch-sizes.txt cpu false &
$pyexe gen_dftbat.py fftx_prdftbat dftbatch-sizes.txt cpu true &
$pyexe gen_dftbat.py fftx_prdftbat dftbatch-sizes.txt cpu false &

wait

##  Create code for a GPU if requested (parameter 1 is { CUDA | HIP })

if [ $build_type = "CUDA" ]; then
    ##  Generate Nvidia GPU (CUDA) code
    echo "Generate CUDA code ..."
    $pyexe gen_files.py fftx_mddft cube-sizes-gpu.txt cuda true &
    $pyexe gen_files.py fftx_mddft cube-sizes-gpu.txt cuda false &
    $pyexe gen_files.py fftx_mdprdft cube-sizes-gpu.txt cuda true &
    $pyexe gen_files.py fftx_mdprdft cube-sizes-gpu.txt cuda false &
    $pyexe gen_files.py fftx_rconv cube-sizes-gpu.txt cuda true &
    wait
fi

if [ $build_type = "HIP" ]; then
    ##  Generate AMD GPU (HIP) code
    echo "Generate HIP code ..."
    $pyexe gen_files.py fftx_mddft cube-sizes-gpu.txt hip true &
    $pyexe gen_files.py fftx_mddft cube-sizes-gpu.txt hip false &
    $pyexe gen_files.py fftx_mdprdft cube-sizes-gpu.txt hip true &
    $pyexe gen_files.py fftx_mdprdft cube-sizes-gpu.txt hip false &
    $pyexe gen_files.py fftx_rconv cube-sizes-gpu.txt hip true &
    wait
fi

exit 0
