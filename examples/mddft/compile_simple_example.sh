#!/bin/bash
module load rocm/5.4.0
hipcc -I ../../src/include/ -I ../../src/library/ -DFFTX_HIP simple_example.cpp