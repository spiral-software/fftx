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
    start_time=`date +%s.%N`
    eval $1
    rc=$?
    end_time=`date +%s.%N`
    runtime=$( echo "$end_time - $start_time" | bc -l )
    if [[ $rc = 0 ]]; then
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

##  Test definitions
run_tests "testbatch1ddft"      5   256x64 512x32 1024x16
run_tests "testbatch1dprdft"    5   256x64 512x32 1024x16
run_tests "testmddft"           5   40x40x40 64x64x64 72x72x72 128x128x128
run_tests "testmdprdft"         5   40x40x40 64x64x64 72x72x72 128x128x128
run_tests "testhockneyconv"     5   32x32x128 32x128x32 128x32x32 64x64x64

run_tests "testrconv_lib"       5           ##  No sizes
run_tests "testrconv"           5   40x40x40 64x64x64 72x72x72 128x128x128
run_tests "testverify_lib"      5           ##  No sizes
run_tests "testverify"          5   40x40x40 64x64x64 72x72x72 128x128x128

run_tests "fortran_main"        5   32x32x32 64x64x64
