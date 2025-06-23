#!/bin/bash

##
##  Copyright (c) 2018-2025, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information.
##

source "$(dirname "$0")/test_common_funcs.sh"

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
