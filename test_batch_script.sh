#!/bin/bash

##  Function to time a run and report pass or fail
time_test() {
    start_time=`date +%s.%N`
    eval $1
    rc=$?
    end_time=`date +%s.%N`
    runtime=$( echo "$end_time - $start_time" | bc -l )
    if [[ $rc = 0 ]]; then
        echo "PASSED $1 in $runtime sec"
    else
        echo "FAILED $1 in $runtime sec"
    fi
}

N=256
B=64
NxB=$N"x"$B

time_test "srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testbatch1ddft -s $NxB -i 5 -r 0x0"
time_test "srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testbatch1ddft -s $NxB -i 5 -r 0x1"
time_test "srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testbatch1ddft -s $NxB -i 5 -r 1x0"
time_test "srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testbatch1ddft -s $NxB -i 5 -r 1x1"
time_test "srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testbatch1dprdft -s $NxB -i 5 -r 0x0"
time_test "srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testbatch1dprdft -s $NxB -i 5 -r 0x1"
time_test "srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testbatch1dprdft -s $NxB -i 5 -r 1x0"
time_test "srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testbatch1dprdft -s $NxB -i 5 -r 1x1"

### dimensions of distributed FFT
M=128
N=128
K=128
MxNxK=$M"x"$N"x"$K

time_test "srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 bin/testmddft -s $MxNxK -i 5"

time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/fortran_main -s $MxNxK -i 5"

M=64
N=64
K=64
MxNxK=$M"x"$N"x"$K

time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/fortran_main -s $MxNxK -i 5"

batch=1

### embedded or unembedded
embed=1
noembed=0

### forward or inverse transform
fwd=1
inv=0

### complex or real transform
complex=1
real=0

trials=3
check=2

time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_1D $M $N $K $batch $noembed $fwd $complex $trials $check"
time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_1D $M $N $K $batch $noembed $inv $complex $trials $check"
time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_1D $M $N $K $batch $noembed $fwd $real    $trials $check"
time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_1D $M $N $K $batch $noembed $inv $real    $trials $check"

rows=2
cols=2

time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_2D $M $N $K $batch $rows $cols $noembed $fwd $complex"
time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_2D $M $N $K $batch $rows $cols $noembed $inv $complex"
time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_2D $M $N $K $batch $rows $cols $noembed $fwd $real"
time_test "srun --nodes=1 --ntasks=4 --gpus=4 bin/test3DDFT_mpi_2D $M $N $K $batch $rows $cols $noembed $inv $real"
