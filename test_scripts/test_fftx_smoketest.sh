#!/bin/bash

##
##  Copyright (c) 2018-2025, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information.
##

source "$(dirname "$0")/test_common_funcs.sh"

##  Test definitions
run_tests "testbatch1ddft"      5   256x64
run_tests "testmddft"           5   64x64x64
run_tests "testmdprdft"         5   64x64x64
run_tests "testrconv_lib"       5           ##  No sizes
