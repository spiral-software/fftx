// #include <__clang_cuda_builtin_vars.h>
#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>

#include "device_macros.h"
#include "fftx_gpu.h"
#include "fftx_util.h"

#define FFTX_MPI_EMBED_1 1
#define FFTX_MPI_EMBED_2 2
#define FFTX_MPI_EMBED_3 3
#define FFTX_MPI_EMBED_4 4

#define FFTX_MPI_3D_CUFFT_STAGE1 1
#define FFTX_MPI_3D_CUFFT_STAGE2 2

#define FFTX_FORWARD  1
#define FFTX_BACKWARD 2


using namespace std;

#define CPU_PERMUTE 0     //Todo: Fix CPU PERMUTE to work with batch + embedded
#define CUDA_AWARE_MPI 0  //Todo: CUDA_AWARE_MPI not working

// implement on GPU.
// [A, B, C] -> [B, A, C]
// launch with c thread blocks? can change parallelism if that's too much
// work for a single thread block.

// static complex<double> *recv_buffer, *send_buffer;
struct fftx_plan_t
{
  complex<double> *recv_buffer, *send_buffer;
  vector<int> shape;
  int r, c, b;
  MPI_Comm row_comm, col_comm;
  DEVICE_FFT_HANDLE stg3, stg2, stg1;
  DEVICE_FFT_HANDLE stg2i, stg1i;  
  double  *Q3, *Q4;
  bool is_embed;
  bool is_forward;
};

typedef fftx_plan_t* fftx_plan;


fftx_plan  fftx_plan_distributed(int r, int c, int M, int N, int K, int batch, bool is_embedded);
void fftx_execute(fftx_plan plan, double* out_buffer, double*in_buffer,int direction);
void fftx_plan_destroy(fftx_plan plan);

void init_2d_comms(fftx_plan plan, int rr, int cc, int M, int N, int K, bool is_embedded);
void destroy_2d_comms(fftx_plan plan);

// perm: [a, b, c] -> [a, c, b]
void pack_embed(complex<double> *dst, complex<double> *src, int a, int b, int c, int batch, bool is_embedded);
void fftx_mpi_rcperm(fftx_plan plan, double * _Y, double *_X, int stage, bool is_embedded);
