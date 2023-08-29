#ifndef __FFTX_MPI_SPIRAL__
#define __FFTX_MPI_SPIRAL__

#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>

#include "device_macros.h"
#include "fftx_gpu.h"
#include "fftx_util.h"

// #define FFTX_MPI_EMBED_1 1
// #define FFTX_MPI_EMBED_2 2

// #define FFTX_MPI_EMBED_3 3
// #define FFTX_MPI_EMBED_4 4

// #define FFTX_FORWARD  1
// #define FFTX_BACKWARD 2

using namespace std;

// #define CPU_PERMUTE 0     //Todo: Fix CPU PERMUTE to work with batch + embedded
// #define CUDA_AWARE_MPI 0

// implement on GPU.
// [A, B, C] -> [B, A, C]
// launch with c thread blocks? can change parallelism if that's too much
// work for a single thread block.

// // static complex<double> *recv_buffer, *send_buffer;
// struct fftx_plan_t {
//   complex<double> *recv_buffer, *send_buffer;
//   int r, c, b;
//   double  *Q3, *Q4;
//   bool is_embed;
//   bool is_forward;
//   bool is_complex;
//   MPI_Comm row_comm, col_comm;
//   size_t shape[6]; // used for buffers for A2A.
//   int M, N, K; // used for FFT sizes.
//   DEVICE_FFT_HANDLE stg3, stg2, stg1;
//   DEVICE_FFT_HANDLE stg2i, stg1i;
// };

// typedef fftx_plan_t* fftx_plan;

// //fftx_plan  fftx_plan_distributed_1d(int p, int M, int N, int K, int batch, bool is_embedded, bool is_complex);
// fftx_plan  fftx_plan_distributed(int r, int c, int M, int N, int K, int batch, bool is_embedded, bool is_complex);
// void fftx_execute(fftx_plan plan, double* out_buffer, double*in_buffer,int direction, bool use_fftx);
// void fftx_plan_destroy(fftx_plan plan);

// // perm: [a, b, c] -> [a, c, b]
// void pack_embed(fftx_plan plan, complex<double> *dst, complex<double> *src, int a, int b, int c, bool is_embedded);
// void fftx_mpi_rcperm(fftx_plan plan, double * _Y, double *_X, int stage, bool is_embedded);


fftx_plan  fftx_plan_distributed_spiral(int r, int c, int M, int N, int K, int batch, bool is_embedded, bool is_complex);
void fftx_execute_spiral(fftx_plan plan, double* out_buffer, double*in_buffer,int direction);
void fftx_plan_destroy_spiral(fftx_plan plan);

#endif
