#!/bin/bash
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

module purge
module load rocm
module load PrgEnv-gnu
module load python

cd $FFTX_HOME
source test_batch_script.sh
