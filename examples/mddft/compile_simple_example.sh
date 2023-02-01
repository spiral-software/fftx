#!/bin/bash
module load rocm/5.4.0
hipcc -I $FFTX_HOME/include/ -DFFTX_HIP simple_example.cpp
