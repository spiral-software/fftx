#!/bin/bash

##  Test suite for FFTX
##  To run: either: ./test_suite.sh or source test_suite.sh
##
##  If FFTX has been built for a GPU (any of CUDA, HIP, or SYCL),
##  then you may need to be logged in to the device, not the host,
##  in order to run the script.
##

##  Function to run tests for an executable
run_tests() {
    local base=$1
    local is_fortran=$2
    shift 2                             ##  Remove first two arguments
    local sizes=("$@")                  ##  Capture all remaining arguments as sizes

    ##  Determine executable name based on environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        exen="./bin/${base}.exe"
    else
        exen="./bin/${base}"
    fi

    ##  Check if executable exists and run tests
    if [ -f "$exen" ]; then
        echo "Running tests for $base..."
        if [ "$is_fortran" = true ]; then
            ##  For Fortran-specific arguments
            for xx in "${sizes[@]}"; do
                echo "Run $base size = $xx"
                $exen $xx             ##  No additional quotes needed
            done
        elif [ "${#sizes[@]}" -eq 0 ]; then
            ##  No sizes argument, run the program with just the iteration switch
            $exen -i 5
        else
            ##  Iterate runs of the program with each specified size
            for xx in "${sizes[@]}"; do
                echo "Run $base size = $xx"
                $exen -i 5 -s "$xx"
            done
        fi
    else
        echo "Executable $exen not found, skipping tests."
    fi
}

##  Test definitions
run_tests "testbatch1ddft"      false   256x64 512x32 1024x16
run_tests "testbatch1dprdft"    false   256x64 512x32 1024x16
run_tests "testmddft"           false   40x40x40 64x64x64 72x72x72 128x128x128
run_tests "testmdprdft"         false   40x40x40 64x64x64 72x72x72 128x128x128
run_tests "testhockneyconv"     false   32x32x128 32x128x32 128x32x32 64x64x64

run_tests "testrconv_lib"       false           ##  No sizes
run_tests "testrconv"           false   40x40x40 64x64x64 72x72x72 128x128x128
run_tests "testverify_lib"      false           ##  No sizes
run_tests "testverify"          false   40x40x40 64x64x64 72x72x72 128x128x128

##  We need to specify the fortran sizes differently because we need embedded spaces...
run_tests "fortran_main"        true    "32 32 32 " "64 64 64"
