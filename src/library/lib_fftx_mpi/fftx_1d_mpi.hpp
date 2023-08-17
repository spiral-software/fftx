#ifndef __FFTX_1D_MPI__
#define __FFTX_1D_MPI__

#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>

#include "device_macros.h"
#include "fftx_gpu.h"
#include "fftx_util.h"
#include "fftx_mpi.hpp"

using namespace std;

void init_1d_comms(fftx_plan plan, int pp, int M, int N, int K);
void destroy_1d_comms(fftx_plan plan);

fftx_plan  fftx_plan_distributed_1d(int p, int M, int N, int K, int batch, bool is_embedded, bool is_complex);
void fftx_execute_1d(fftx_plan plan, double* out_buffer, double*in_buffer, int direction);

void fftx_mpi_rcperm_1d(fftx_plan plan, double * Y, double *X, int stage, bool is_embedded);

#endif
