#!/bin/bash
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

module purge
module load cmake cudatoolkit PrgEnv-gnu
export LIBRARY_PATH=$CUDATOOLKIT_HOME/../../math_libs/lib64
export CPATH=$CUDATOOLKIT_HOME/../../math_libs/include
module load openmpi
module load python

cd $FFTX_HOME
source test_batch_script.sh
