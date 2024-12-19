#!/bin/bash
### fill in with your account name
#SBATCH --account=
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

### export SPIRAL_HOME=(home directory for SPIRAL)

export FFTX_HOME=$PWD

### dimensions of distributed FFT
M=128
N=128
K=128

srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testmddft -s $M"x"$N"x"$K

srun --nodes=1 --ntasks=4 --gpus=4 bin/fortran_main $M $N $K

M=64
N=64
K=64

srun --nodes=1 --ntasks=4 --gpus=4 bin/fortran_main $M $N $K

batch=1

embedded=1
unembedded=0

### forward or inverse transform
forward=1
inverse=0

### complex or real transform
complex=1
real=0

trials=3
check=2

srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_1D $M $N $K $batch $unembedded $forward $complex $trials $check
srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_1D $M $N $K $batch $unembedded $inverse $complex $trials $check
srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_1D $M $N $K $batch $unembedded $forward $real    $trials $check
srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_1D $M $N $K $batch $unembedded $inverse $real    $trials $check

rows=2
cols=2

srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_2D $M $N $K $batch $rows $cols $unembedded $forward $complex
srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_2D $M $N $K $batch $rows $cols $unembedded $inverse $complex
srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_2D $M $N $K $batch $rows $cols $unembedded $forward $real
srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_2D $M $N $K $batch $rows $cols $unembedded $inverse $real
