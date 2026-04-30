#!/bin/bash

##
##  Copyright (c) 2018-2025, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information.
##

### Need these settings:
### export SBATCH_ACCOUNT=(your account name)
### export SPIRAL_HOME=(home directory for SPIRAL)
### export FFTX_HOME=(home directory for FFTX)
### Then to submit:
### sbatch test_batch_perlmutter.sh
#SBATCH --time=0:10:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=FFTX_test_batch
#SBATCH --output=test_batch_perlmutter_out.%J
#SBATCH --error=test_batch_perlmutter_err.%J

### Should already have cmake, and these modules loaded: cudatoolkit, PrgEnv-gnu, cray-mpich.

### Compile failure on default gcc-native/14, so load older version.
module load gcc-native/13.2
### Need to load this module to get Python 3
module load python
### Need this for #include <mpi.h>
export CPATH=$CRAY_MPICH_DIR/include:$CPATH
### Always need this
export FFTX_HOME=$PWD

source $FFTX_HOME/test_scripts/test_batch_script.sh
