#! /bin/bash

##  Script to generate the source code for the libraries

##  Expects either 0 or 1 argument:  no arg ==> build CPU code only
##  arg1 = { CPU | CUDA | HIP } ==> build code for the respective target

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

echo "Build library for $build_type "

##  Use a file named build-lib-code-failures.txt to note any files that fail to
##  generate -- failures are noted & written by the python drivers: gen_{files|dftbat}.py

rm -f build-lib-code-failures.txt
touch build-lib-code-failures.txt

##  We expect a file called build-lib-code-options.sh to be created (automatically created
##  by top level config library script in root of tree).  If its available, source it for
##  the variable values; if its missing just provide defaults

if [ -f ./build-lib-code-options.sh ]; then
    echo "./build-lib-code-options.sh file exists - load it"
    . ./build-lib-code-options.sh
else
    ##  file not found -- set defaults
    echo "./build-lib-code-options.sh file does not exist - assign default values"
    DFTBAT_LIB=true
    PRDFTBAT_LIB=true
    MDDFT_LIB=true
    MDPRDFT_LIB=true
    RCONV_LIB=true
    DISTDFT_LIB=true
    CPU_SIZES_FILE="cube-sizes-cpu.txt"
    GPU_SIZES_FILE="cube-sizes-gpu.txt"
    DFTBAT_SIZES_FILE="dftbatch-sizes.txt"
    DISTDFT_SIZES_FILE="distdft-sizes.txt"
fi

if [ $build_type = "CPU" ]; then
    ##  Generate code for CPU
    echo "Generate CPU code ..."
    waitspiral=false

    if [ "$DFTBAT_LIB" = true ]; then
	##  Build DFT batch for CPU
	waitspiral=true
	$pyexe gen_dftbat.py fftx_dftbat $DFTBAT_SIZES_FILE $build_type true &
	$pyexe gen_dftbat.py fftx_dftbat $DFTBAT_SIZES_FILE $build_type false &
    fi
    if [ "$PRDFTBAT_LIB" = true ]; then
	##  Build PRDFT batch for CPU
	waitspiral=true
	$pyexe gen_dftbat.py fftx_prdftbat $DFTBAT_SIZES_FILE $build_type true &
	$pyexe gen_dftbat.py fftx_prdftbat $DFTBAT_SIZES_FILE $build_type false &
    fi
    if [ "$waitspiral" = true ]; then
	wait		##  wait for the child processes to complete
    fi

    waitspiral=false
    ##  Build the remaining libraries for the specified target
    if [ "$MDDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py fftx_mddft $CPU_SIZES_FILE $build_type true &
	$pyexe gen_files.py fftx_mddft $CPU_SIZES_FILE $build_type false &
    fi
    if [ "$MDPRDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py fftx_mdprdft $CPU_SIZES_FILE $build_type true &
	$pyexe gen_files.py fftx_mdprdft $CPU_SIZES_FILE $build_type false &
    fi
    if [ "$RCONV_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py fftx_rconv $CPU_SIZES_FILE $build_type true &
    fi
    if [ "$waitspiral" = true ]; then
	wait		##  wait for the child processes to complete
    fi
    ##  Not attempting to build Distributed DFT (MPI) for CPU
fi

if [[ $build_type = "CUDA" || $build_type = "HIP" ]]; then
    ##  Create code for a GPU
    echo "Generate code for $build_type ..."
    waitspiral=false

    if [ "$MDDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py fftx_mddft $GPU_SIZES_FILE $build_type true &
	$pyexe gen_files.py fftx_mddft $GPU_SIZES_FILE $build_type false &
    fi
    if [ "$MDPRDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py fftx_mdprdft $GPU_SIZES_FILE $build_type true &
	$pyexe gen_files.py fftx_mdprdft $GPU_SIZES_FILE $build_type false &
    fi
    if [ "$RCONV_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py fftx_rconv $GPU_SIZES_FILE $build_type true &
    fi
    if [ "$waitspiral" = true ]; then
	wait		##  wait for the child processes to complete
    fi    
    if [ "$DISTDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_distdft.py fftx_distdft $DISTDFT_SIZES_FILE $build_type true false &
	$pyexe gen_distdft.py fftx_distdft $DISTDFT_SIZES_FILE $build_type true true &
    fi
    if [ "$waitspiral" = true ]; then
	wait		##  wait for the child processes to complete
    fi    
fi

exit 0
