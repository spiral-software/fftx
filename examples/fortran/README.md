Convolution in FFTX in Fortran
==============================

Before building,
set environment variable FFTX_HOME to the directory where FFTX
has been installed.
Example:
```
export FFTX_HOME=~/fftx_develop
```

This package has been tested on CPUs in serial on Linux and macOS
as well as on GPUs on three different supercomputers:
CUDA on Perlmutter at NERSC, HIP on Frontier at OLCF,
and SYCL on Sunspot at ALCF.
If building on one of these supercomputers, you will need to
follow the instructions at `$FFTX_HOME/supercomputer-README.md`
to load the appropriate modules.

### To build:

Once FFTX_HOME is set and modules are loaded, if necessary, then
you can just do
```
make
```
which will build for CUDA, HIP, or SYCL, if running on one of those
platforms, or otherwise on a CPU platform.

You can also pick your backend, with:
```
make CUDA
make HIP
make SYCL
make CPU
```
In any case, the executable will be in `main`.

### To run in serial:

The command arguments are the dimensions.
For example, to run convolution tests in serial on 32x40x48, do:
```
./main 32 40 48
```

So far, serial transforms work that are complex-to-complex forward
(MDDFT) and inverse (IMDDFT),
real-to-complex (MDPRDFT), and complex-to-real (IMDPRDFT).
Serial and distributed, real and complex convolution tests are all passing.
These tests take a forward transform on random input, multiply by
the symbol for a discrete laplacian, then take an inverse transform,
and finally apply a discrete laplacian to the result, which should give
the original input back.

### To run in parallel:

Distributed transforms work on HIP and CUDA platforms only.

To run convolution tests with 4 MPI ranks on 32x40x48, do one of:
```
mpirun -np 4 ./main 32 40 48

srun -n 4 ./main 32 40 48
```
