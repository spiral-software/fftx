#include <complex>
#include <vector>
#include <mpi.h>
#include <iostream>
#include "gpu.h"
#include "util.h"

void init_2d_comms(int r, int c, int M, int N, int K);
void destroy_2d_comms();
void fftx_mpi_rcperm(double* Y, double *X, int sizes, int stage, int dim, int M, int N, int K);
