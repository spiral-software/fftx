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

./bin/testbatch1ddft -i 5 -s 256x64
./bin/testbatch1ddft -i 5 -s 512x32
./bin/testbatch1ddft -i 5 -s 1024x16

./bin/testbatch1dprdft -i 5 -s 256x64
./bin/testbatch1dprdft -i 5 -s 512x32
./bin/testbatch1dprdft -i 5 -s 1024x16

./bin/testmddft -i 5 -s 40x40x40
./bin/testmddft -i 5 -s 72x72x72
./bin/testmddft -i 5 -s 128x128x128

./bin/testmdprdft -i 5 -s 40x40x40
./bin/testmdprdft -i 5 -s 72x72x72
./bin/testmdprdft -i 5 -s 128x128x128

./bin/testhockneyconv -i 5 -s 32x32x128
./bin/testhockneyconv -i 5 -s 32x128x32
./bin/testhockneyconv -i 5 -s 128x32x32
./bin/testhockneyconv -i 5 -s 64x64x64

./bin/testrconv_lib
./bin/testrconv -i 5 -s 40x40x40
./bin/testrconv -i 5 -s 72x72x72
./bin/testrconv -i 5 -s 128x128x128

./bin/testverify_lib
./bin/testverify -i 5 -s 40x40x40
./bin/testverify -i 5 -s 72x72x72
./bin/testverify -i 5 -s 128x128x128

./bin/fortran_main 32 32 32
./bin/fortran_main 64 64 64
