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
### sbatch test_batch_frontier.sh
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=FFTX_test_batch
#SBATCH --output=test_batch_frontier_out.%J
#SBATCH --error=test_batch_frontier_err.%J

export FABRIC_PATH=/opt/cray/libfabric/1.22.0
export ROCM_PATH=/opt/rocm-6.2.4
export CRAY_MPICH_DIR=/opt/cray/pe/mpich/8.1.31/ofi/gnu/12.3
export PYTHON_PATH=/opt/cray/pe/python/3.11.7
export LD_LIBRARY_PATH=$PYTHON_PATH/lib:$FABRIC_PATH/lib64:$CRAY_MPICH_DIR/lib:$ROCM_PATH/lib:$FABRIC_PATH/lib64

cd $FFTX_HOME
source test_batch_script.sh
