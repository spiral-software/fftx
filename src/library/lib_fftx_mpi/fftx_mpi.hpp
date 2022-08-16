// #include <__clang_cuda_builtin_vars.h>
#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>
#include "gpu.h"
#include "util.h"

#define FFTX_MPI_EMBED_1 1
#define FFTX_MPI_EMBED_2 2

#define FFTX_CUDA 1
#include "device_macros.h"


using namespace std;

#define CPU_PERMUTE 1
#define CUDA_AWARE_MPI 0
// implement on GPU.
// [A, B, C] -> [B, A, C]
// launch with c thread blocks? can change parallelism if that's too much
// work for a single thread block.

static complex<double> *recv_buffer, *send_buffer;
static vector<int> shape;
static int r, c;
static MPI_Comm row_comm, col_comm;

void init_2d_comms(int rr, int cc, int M, int N, int K, bool is_embedded);
void destroy_2d_comms();

// perm: [a, b, c] -> [a, c, b]
void pack_embed(complex<double> *dst, complex<double> *src, int a, int b, int c, bool is_embedded);
void fftx_mpi_rcperm(double * _Y, double *_X, int stage, bool is_embedded);

