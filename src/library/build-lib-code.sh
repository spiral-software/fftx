#! /bin/bash

##  Script to generate the source code for the libraries

##  Expects either 0 or 1 argument:  no arg ==> build CPU code only
##  arg1 = { CPU | CUDA | HIP | SYCL } ==> build code for the respective target

##  A user may have both python & python3 in their PATH; try to find a python version 3...

pyexe="not_found"
trypy="python"
echo "See if $trypy is in the PATH"
which $trypy > /dev/null 2>&1

if [ $? -eq 0 ]; then
    ##  Found $trypy, test its version
    echo "$trypy found in PATH, check version"
    vpy="$($trypy --version 2>&1 | sed -e 's/Python //' | sed -e 's/\..*//')"
    if [ $vpy -eq "2" ]; then
	##  python is version 2, look for python3 executable
	trypy="python3"
    elif [ $vpy -eq "3" ]; then
	##  found python is version 3...
	pyexe=$trypy
    fi
else
    ##  $trypy NOT FOUND, look for python3...
    echo "$trypy not found, look for python3"
    trypy="python3"
fi
if [ $pyexe == "not_found" ]; then
    which $trypy > /dev/null 2>&1
    if [ $? -ne 0 ]; then
	echo "NO suitable python executable found ... exiting"
	exit 9
    fi
    ##  Found $trypy, use it...
    pyexe=$trypy
fi

vpy="$($pyexe --version 2>&1 | awk '{print $2}')"
echo "Python executable is $pyexe (version $vpy)"

if [ $# -eq 0 ]; then
    build_type="CPU"
else
    if [ $1 = "CPU" ] || [ $1 = "CUDA" ] || [ $1 = "HIP" ] || [ $1 = "SYCL" ]; then
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
    PSATD_LIB=false
    CPU_SIZES_FILE="cube-sizes-cpu.txt"
    GPU_SIZES_FILE="cube-sizes-gpu.txt"
    DFTBAT_SIZES_FILE="dftbatch-sizes.txt"
    PSATD_SIZES_FILE="cube-psatd.txt"
fi

##  We will run concurrent processes to build the libraries ... will use up to number of cpus
##  as reported by `nproc` (sysctl -n hw.ncpu on mac).  Don't start more parralel jobs than this;
##  When the max jobs is reached just wait for them to finish and continue.
##  NOTE: nproc is used on Windows as it should be available with Mingw or cygwin.

if [ "$(uname)" == "Linux" ]; then
    max_jobs=$(nproc)
    echo "Linux: max_jobs set to: $max_jobs"
elif [ "$(uname)" == "Darwin" ]; then
    max_jobs=$(sysctl -n hw.ncpu)
    echo "MAC: max_jobs set to: $max_jobs"
else            ## Assume windows with mingw or cygwin or similar
    max_jobs=$(nproc)
    echo "Other OS (windows?): max_jobs set to: $max_jobs"
fi

curr_jobs=0

##  Function to check and wait if the number of running jobs is at the max
check_and_wait() {
    if [ $curr_jobs -ge $max_jobs ]; then
        echo "Running $curr_jobs concurrent jobs; >= $max_jobs; **wait**" 
        wait            ##  Wait for all current jobs to finish
        curr_jobs=0
    fi
}

if [ $build_type = "CPU" ]; then
    ##  Generate code for CPU
    echo "Generate CPU code ..."
    waitspiral=false

    if [ "$DFTBAT_LIB" = true ]; then
	##  Build DFT batch for CPU
	waitspiral=true
	$pyexe gen_dftbat.py -t fftx_dftbat -s $DFTBAT_SIZES_FILE -p $build_type &
        ((curr_jobs++))
	$pyexe gen_dftbat.py -t fftx_dftbat -s $DFTBAT_SIZES_FILE -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait
    
    if [ "$PRDFTBAT_LIB" = true ]; then
	##  Build PRDFT batch for CPU
	waitspiral=true
	$pyexe gen_dftbat.py -t fftx_prdftbat -s $DFTBAT_SIZES_FILE -p $build_type &
        ((curr_jobs++))
	$pyexe gen_dftbat.py -t fftx_prdftbat -s $DFTBAT_SIZES_FILE -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$MDDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_mddft -s $CPU_SIZES_FILE -p $build_type &
        ((curr_jobs++))
	$pyexe gen_files.py -t fftx_mddft -s $CPU_SIZES_FILE -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$MDPRDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_mdprdft -s $CPU_SIZES_FILE -p $build_type &
        ((curr_jobs++))
	$pyexe gen_files.py -t fftx_mdprdft -s $CPU_SIZES_FILE -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$RCONV_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_rconv -s $CPU_SIZES_FILE -p $build_type &
        ((curr_jobs++))
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

    if [ "$DFTBAT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_dftbat.py -t fftx_dftbat -s $DFTBAT_SIZES_FILE -p $build_type &
        ((curr_jobs++))
	$pyexe gen_dftbat.py -t fftx_dftbat -s $DFTBAT_SIZES_FILE -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$PRDFTBAT_LIB" = true ]; then
        waitspiral=true
        $pyexe gen_dftbat.py -t fftx_prdftbat -s $DFTBAT_SIZES_FILE -p $build_type &
        ((curr_jobs++))
        $pyexe gen_dftbat.py -t fftx_prdftbat -s $DFTBAT_SIZES_FILE -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$MDDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_mddft -s $GPU_SIZES_FILE -p $build_type &
        ((curr_jobs++))
	$pyexe gen_files.py -t fftx_mddft -s $GPU_SIZES_FILE -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$MDPRDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_mdprdft -s $GPU_SIZES_FILE -p $build_type &
        ((curr_jobs++))
	$pyexe gen_files.py -t fftx_mdprdft -s $GPU_SIZES_FILE -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$RCONV_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_rconv -s $GPU_SIZES_FILE -p $build_type &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$PSATD_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_psatd -s $PSATD_SIZES_FILE -p $build_type &
        ((curr_jobs++))
    fi

    if [ "$waitspiral" = true ]; then
	wait		##  wait for the child processes to complete
    fi    
fi

if [[ $build_type = "SYCL" ]]; then
    ##  Create empty libraries for SYCL
    echo "Generating empty libraries for $build_type ..."
    waitspiral=false

    if [ "$DFTBAT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_dftbat.py -t fftx_dftbat -s empty-sizes.txt -p $build_type &
        ((curr_jobs++))
	$pyexe gen_dftbat.py -t fftx_dftbat -s empty-sizes.txt -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$PRDFTBAT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_dftbat.py -t fftx_prdftbat -s empty-sizes.txt -p $build_type &
        ((curr_jobs++))
	$pyexe gen_dftbat.py -t fftx_prdftbat -s empty-sizes.txt -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$MDDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_mddft -s empty-sizes.txt -p $build_type &
        ((curr_jobs++))
	$pyexe gen_files.py -t fftx_mddft -s empty-sizes.txt -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$MDPRDFT_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_mdprdft -s empty-sizes.txt -p $build_type &
        ((curr_jobs++))
	$pyexe gen_files.py -t fftx_mdprdft -s empty-sizes.txt -p $build_type -i &
        ((curr_jobs++))
    fi
    check_and_wait

    if [ "$RCONV_LIB" = true ]; then
	waitspiral=true
	$pyexe gen_files.py -t fftx_rconv -s empty-sizes.txt -p $build_type &
        ((curr_jobs++))
    fi
    check_and_wait

    # if [ "$PSATD_LIB" = true ]; then
    # 	waitspiral=true
    # 	$pyexe gen_files.py -t fftx_psatd -s $PSATD_SIZES_FILE -p $build_type &
    #   ((curr_jobs++))
    # fi

    if [ "$waitspiral" = true ]; then
	wait		##  wait for the child processes to complete
    fi    
fi
exit 0
