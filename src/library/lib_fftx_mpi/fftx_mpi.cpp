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

#define CPU_PERMUTE 0
#define CUDA_AWARE_MPI 0
// implement on GPU.
// [A, B, C] -> [B, A, C]
// launch with c thread blocks? can change parallelism if that's too much
// work for a single thread block.


static complex<double> *recv_buffer, *send_buffer;
static vector<int> shape;
static int r, c;
static MPI_Comm row_comm, col_comm;

void init_2d_comms(int rr, int cc, int M, int N, int K, bool is_embedded) {
  // pass in the dft size. if embedded, double dims when necessary.
  r = rr; c = cc;
  size_t max_size = M*N*K*(is_embedded ? 8 : 1)/(r*c);

#if CUDA_AWARE_MPI
  DEVICE_MALLOC(&recv_buffer, max_size * sizeof(complex<double>));
#else
  send_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
  recv_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
#endif

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  int col_color = world_rank % r;
  int row_color = world_rank / r;

  MPI_Comm_split(MPI_COMM_WORLD, row_color, world_rank, &row_comm);
  MPI_Comm_split(MPI_COMM_WORLD, col_color, world_rank, &col_comm);  
    
  shape.resize(6);
  shape[0] = M/r;
  shape[1] = r;  
  shape[2] = N/r;
  shape[3] = r;  
  shape[4] = K/c;
  shape[5] = c;
}

void destroy_2d_comms() {
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);      
#if CUDA_AWARE_MPI
  DEVICE_FREE(recv_buffer);
#else
  free(send_buffer);
  free(recv_buffer);
#endif
}


// perm: [a, b, c] -> [a, 2c, b]
void pack_embed(complex<double> *dst, complex<double> *src, int a, int b, int c, bool is_embedded) {
  size_t buffer_size = a * b * c * (is_embedded ? 2 : 1); // assume embedded
#if CPU_PERMUTE
  if (is_embedded) {
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c/2; ic++) {
        for (int ia = 0; ia < a; ia++) {
          send_buffer[ib * 2*c*a + ic * a + ia] = complex<double>(0,0);
        }
      }
      for (int icd = c/2, ics = 0; icd < 3*c/2; icd++, ics++) {
        for (int ia = 0; ia < a; ia++) {
          send_buffer[ib * 2*c*a + icd * a + ia] = recv_buffer[ics * b*a + ib * a + ia];
        }
      }
      for (int ic = 3*c/2; ic < 2*c; ic++) {
        for (int ia = 0; ia < a; ia++) {
          send_buffer[ib * 2*c*a + ic * a + ia] = complex<double>(0,0);
        }
      }
    }
  } else {
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c; ic++) {
        for (int ia = 0; ia < a; ia++) {
          send_buffer[ib * c*a + ic * a + ia] = recv_buffer[ic * b*a + ib * a + ia];
        }
      }
    }
  }
  DEVICE_MEM_COPY(dst, send_buffer, buffer_size * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);
#else
#if (!CUDA_AWARE_MPI)
  DEVICE_MEM_COPY(src, recv_buffer, buffer_size * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);
#endif
  
  DEVICE_ERROR_T err;
  if (is_embedded) {
    err = pack_embedded(
      dst, src,
      c, b, a
    );
  } else {
    err = pack(
      dst, src,
      b,   a, c*a,
      c, b*a,   a,
      a, 1
    );
  }
  if (err != DEVICE_SUCCESS) {
    fprintf(stderr, "pack failed Y <- St1_Comm!\n");
    exit(-1);
  }

#endif
}



void fftx_mpi_rcperm(double * _Y, double *_X, int stage, bool is_embedded) {
  complex<double> *X = (complex<double> *) _X;
  complex<double> *Y = (complex<double> *) _Y;
  switch (stage) {
    case FFTX_MPI_EMBED_1:
      {
        // after first 1D FFT on K dim.
        shape[0] *= (is_embedded ? 2 : 1);
        size_t buffer_size = shape[2] * shape[4] * shape[0] * shape[1];
        int sendSize = shape[2] * shape[4] * shape[0];
        int recvSize = sendSize;
#if CUDA_AWARE_MPI
        send_buffer = X;
        X = recv_buffer;
#else
        DEVICE_MEM_COPY(send_buffer, X, buffer_size * sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);
#endif
        // [yl, zl, xl, xr] -> [yl, zl, xl, yr]
        MPI_Alltoall(
          send_buffer, sendSize,
          MPI_DOUBLE_COMPLEX,
          recv_buffer, recvSize,
          MPI_DOUBLE_COMPLEX,
          row_comm
        ); // assume N dim is initially distributed along col comm.

        // [yl, (zl, xl), yr] -> [yl, yr, (zl, xl)]
        pack_embed(Y, X, shape[2], shape[4] * shape[0], shape[3], is_embedded);
      } // end FFTX_MPI_EMBED_1
      break;

    case FFTX_MPI_EMBED_2:
      {
        shape[2] *= (is_embedded ? 2 : 1);
        size_t buffer_size = shape[4] * shape[0] * shape[2] * shape[3];
        int sendSize = shape[4] * shape[0] * shape[2];
        int recvSize = sendSize;
#if CUDA_AWARE_MPI
        send_buffer = X;
        X = recv_buffer;
#else
        DEVICE_MEM_COPY(send_buffer, X, buffer_size * sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);
#endif
        // [zl, xl, yl, yr] -> [zl, xl, yl, zr]
        MPI_Alltoall(
          send_buffer, sendSize,
          MPI_DOUBLE_COMPLEX,
          recv_buffer, recvSize,
          MPI_DOUBLE_COMPLEX,
          col_comm
        ); // assume K dim is initially distributed along row comm.

        // [zl, (xl, yl), zr] -> [zl, zr, (xl, yl)]
        pack_embed(Y, X, shape[4], shape[0] * shape[2], shape[5], is_embedded);
      } // end FFTX_MPI_EMBED_2
      break;
    default:
      break;
  }
}

