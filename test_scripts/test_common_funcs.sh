#!/bin/bash

##
##  Copyright (c) 2018-2025, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information.
##

##  Test suite for FFTX
##  To run: either: ./test_suite.sh or source test_suite.sh
##
##  If FFTX has been built for a GPU (any of CUDA, HIP, or SYCL),
##  then you may need to be logged in to the device, not the host,
##  in order to run the script.
##

##  Function to time a run and report pass or fail
time_test() {
    start_time=$(date +%s.%N)
    eval "$1"
    rc=$?
    end_time=$(date +%s.%N)

    ##  Windows [Mingw] doesn't have the basic calculator, "bc".  Try python instead:
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        runtime=$(python -c "print(round($end_time - $start_time, 4))")
    else
        runtime=$( echo "$end_time - $start_time" | bc -l )
    fi
    if [[ $rc -eq 0 ]]; then                    ##  Compare as integers; disambiguates '=' or '=='
        echo "PASSED $1 in $runtime sec"
    else
        echo "FAILED $1 in $runtime sec"
    fi
}

##  Function to run tests for an executable
run_tests() {
    local base=$1
    local itns=$2
    shift 2                             ##  Remove first two arguments
    local sizes=("$@")                  ##  Capture all remaining arguments as sizes

    ##  Determine executable name based on environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        exen="${FFTX_HOME}/bin/${base}.exe"
    else
        exen="${FFTX_HOME}/bin/${base}"
    fi

    ## Set itnspec to "-i $itns" if itns is nonzero, otherwise blank
    if [[ "$itns" -eq 0 ]]; then
        itnspec=""
    else
        itnspec="-i $itns"
    fi

    ##  Check if executable exists and run tests
    if [ -f "$exen" ]; then
        echo "Running tests for $base..."
        if [ "${#sizes[@]}" -eq 0 ]; then
            ##  No sizes argument, run the program with just the iteration switch
           time_test "$exen $itnspec"
        else
            ##  Iterate runs of the program with each specified size
            for xx in "${sizes[@]}"; do
                echo "Run $base size = $xx"
                time_test "$exen $itnspec -s $xx"
            done
        fi
    else
        echo "Executable $exen not found, skipping tests."
    fi
}
