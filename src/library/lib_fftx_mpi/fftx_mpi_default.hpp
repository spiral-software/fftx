#ifndef __FFTX_MPI_DEFAULT__
#define __FFTX_MPI_DEFAULT__

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

fftx_plan  fftx_plan_distributed_default(int r, int c, int M, int N, int K, int batch, bool is_embedded, bool is_complex);
void fftx_execute_default(fftx_plan plan, double* out_buffer, double*in_buffer,int direction);
void fftx_plan_destroy_default(fftx_plan plan);

#endif
