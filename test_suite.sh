#!/bin/bash

#
# Test suite for FFTX that can be run on command line.
# To run from bash:
# source test_suite.sh
#
# If FFTX has been built for GPU (either CUDA or HIP or SYCL),
# then you may need to be logged in to the device, not the host,
# in order to run the script.
#

./bin/testbatch1ddft -s 256x64
./bin/testbatch1ddft -s 512x32
./bin/testbatch1ddft -s 1024x16

./bin/testbatch1dprdft -s 256x64
./bin/testbatch1dprdft -s 512x32
./bin/testbatch1dprdft -s 1024x16

./bin/testmddft -s 40x40x40
./bin/testmddft -s 72x72x72
./bin/testmddft -s 128x128x128

./bin/testmdprdft -s 40x40x40
./bin/testmdprdft -s 72x72x72
./bin/testmdprdft -s 128x128x128

./bin/testrconv_lib
./bin/testrconv -s 40x40x40
./bin/testrconv -s 72x72x72
./bin/testrconv -s 128x128x128

./bin/testverify_lib
./bin/testverify -s 40x40x40
./bin/testverify -s 72x72x72
./bin/testverify -s 128x128x128

./bin/fortran_main 32 32 32
./bin/fortran_main 64 64 64
