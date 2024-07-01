#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>

#include "device_macros.h"
#include "fftx_gpu.h"
#include "fftx_util.h"
#include "fftx_mpi.hpp"

#include "fftx_mpi_default.hpp"
#include "fftx_mpi_spiral.hpp"
#include "fftx_1d_mpi.hpp"
#include "fftx_1d_mpi_default.hpp"
#include "fftx_1d_mpi_spiral.hpp"

using namespace std;

void init_2d_comms(fftx_plan plan, int rr, int cc, int M, int N, int K) {
  // pass in the dft size. if embedded, double dims when necessary.
  plan->r = rr;
  plan->c = cc;
  uint64_t m = M;
  uint64_t n = N;
  uint64_t k = K;

  size_t max_size = m*n*k*(plan->is_embed ? 8 : 1)/(plan->r * plan->c) * plan->b;

#if CUDA_AWARE_MPI
  DEVICE_MALLOC(&(plan->send_buffer), max_size * sizeof(complex<double>));
  DEVICE_MALLOC(&(plan->recv_buffer), max_size * sizeof(complex<double>));
#else
  plan->send_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
  plan->recv_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
#endif

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int col_color = world_rank % plan->r;
  int row_color = world_rank / plan->r;

  MPI_Comm_split(MPI_COMM_WORLD, row_color, world_rank, &(plan->row_comm));
  MPI_Comm_split(MPI_COMM_WORLD, col_color, world_rank, &(plan->col_comm));

  size_t kDim = K;
  if (!(plan->is_complex))
    {
      kDim = kDim / 2 + 1;
      if (kDim % (rr*cc)) //not a multiple of the number of processor
	{
	  size_t rem = kDim % (rr*cc);
	  kDim = kDim + (rr*cc - rem);
	}
    }

  // initial layout is [4, 5, 0, 1, 2, 3], after first fft [0, 1, 2, 3, 4, 5]
  plan->shape[0] = M/plan->r;
  plan->shape[1] = plan->r;
  plan->shape[2] = N/plan->r;
  plan->shape[3] = plan->r;
  plan->shape[4] = kDim/plan->c;
  plan->shape[5] = plan->c;

}

void destroy_2d_comms(fftx_plan plan) {
  if (plan) {
    MPI_Comm_free(&(plan->row_comm));
    MPI_Comm_free(&(plan->col_comm));

#if CUDA_AWARE_MPI
    DEVICE_FREE(plan->send_buffer);
    DEVICE_FREE(plan->recv_buffer);
#else
    free(plan->send_buffer);
    free(plan->recv_buffer);
#endif
  }
}

fftx_plan fftx_plan_distributed(int r, int c, int M, int N, int K, int batch, bool is_embedded, bool is_complex) {
  fftx_plan plan;
#if FORCE_VENDOR_LIB
  {
#else
   if(is_complex || (!is_complex && batch == 1)) {
    plan = fftx_plan_distributed_spiral(r, c, M, N, K, batch, is_embedded, is_complex);
    plan->use_fftx = true;
  } else {
    std::cout << "configuration not supported, using vendor backend" << std::endl;
#endif
    plan = fftx_plan_distributed_default(r, c, M, N, K, batch, is_embedded, is_complex);
    plan->use_fftx = false;
  }
  return plan;
}

void fftx_execute(fftx_plan plan, double* out_buffer, double*in_buffer, int direction) {
#if FORCE_VENDOR_LIB
  {
#else
  if(plan->use_fftx) {
    fftx_execute_spiral(plan, out_buffer, in_buffer, direction);
  } else {
#endif
    fftx_execute_default(plan, out_buffer, in_buffer, direction);
  }
}

void fftx_plan_destroy(fftx_plan plan) {
#if FORCE_VENDOR_LIB
  {
#else
  if(plan->use_fftx) {
    fftx_plan_destroy_spiral(plan);
  } else {
#endif
    fftx_plan_destroy_default(plan);
  }
}

// perm: [a, b, c] -> [a, 2c, b]
void pack_embed(fftx_plan plan, complex<double> *dst, complex<double> *src, size_t a, size_t b, size_t c, bool is_embedded) {
  // size_t buffer_size = a * b * c * (is_embedded ? 2 : 1); // assume embedded
  size_t buffer_size = a * b * c;
#if CPU_PERMUTE
  if (is_embedded) {
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c/2; ic++) {
        for (int ia = 0; ia < a; ia++) {
          plan->send_buffer[ib * 2*c*a + ic * a + ia] = complex<double>(0,0);
        }
      }
      for (int icd = c/2, ics = 0; icd < 3*c/2; icd++, ics++) {
        for (int ia = 0; ia < a; ia++) {
          plan->send_buffer[ib * 2*c*a + icd * a + ia] = plan->recv_buffer[ics * b*a + ib * a + ia];
        }
      }
      for (int ic = 3*c/2; ic < 2*c; ic++) {
        for (int ia = 0; ia < a; ia++) {
          plan->send_buffer[ib * 2*c*a + ic * a + ia] = complex<double>(0,0);
        }
      }
    }
  } else {
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c; ic++) {
        for (int ia = 0; ia < a; ia++) {
            plan->send_buffer[ib * c*a + ic * a + ia] =
            plan->recv_buffer[ic * b*a + ib * a + ia];
        }
      }
    }
  }
  DEVICE_MEM_COPY(dst, plan->send_buffer, buffer_size * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);
#else
  //this part of the code does unpacking on the GPU
#if (!CUDA_AWARE_MPI)  //this copies data to the GPU to perform packing
  DEVICE_MEM_COPY(src, plan->recv_buffer, buffer_size * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);
#endif

  DEVICE_ERROR_T err;
  if (is_embedded) {
    err = pack_embedded(
      dst, src,
      c, b, a
    );
  } else {
    // [a, b, c] -> [a, c, b]
    err = pack(
      dst, src,
      b,   a, c*a,
      c, b*a,   a,
      a
    );
  }
  if (err != DEVICE_SUCCESS) {
    fprintf(stderr, "pack failed Y <- St1_Comm!\n");
    exit(-1);
  }

#endif
}

// perm: [a, b, c] -> [a, 2c, b]
void unpack_embed(fftx_plan plan, complex<double> *dst, complex<double> *src, int a, int b, int c, bool is_embedded) {
  size_t buffer_size = a * b * c;
#if CPU_PERMUTE
  //copy data to recv buffer on host in order to unpack into the send_buffer
  DEVICE_MEM_COPY(plan->recv_buffer, src, buffer_size * sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);

  //the CPU code needs to be updated. It is currently the packing code.
  if (is_embedded) {
    // TODO:
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c/2; ic++) {
        for (int ia = 0; ia < a; ia++) {
          plan->send_buffer[ib * 2*c*a + ic * a + ia] = complex<double>(0,0);
        }
      }
      for (int icd = c/2, ics = 0; icd < 3*c/2; icd++, ics++) {
        for (int ia = 0; ia < a; ia++) {
          plan->send_buffer[ib * 2*c*a + icd * a + ia] = plan->recv_buffer[ics * b*a + ib * a + ia];
        }
      }
      for (int ic = 3*c/2; ic < 2*c; ic++) {
        for (int ia = 0; ia < a; ia++) {
          plan->send_buffer[ib * 2*c*a + ic * a + ia] = complex<double>(0,0);
        }
      }
    }
  } else {
    // [a, b, c] <- [a, c, b]
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c; ic++) {
        for (int ia = 0; ia < a; ia++) {
          plan->send_buffer[ic * b*a + ib * a + ia] =
          plan->recv_buffer[ib * c*a + ic * a + ia];
        }
      }
    }
  }
#else

  //this part of the code does unpacking on the GPU
  DEVICE_ERROR_T err;
  if (is_embedded) {
    //embedded unpack GPU code needs to be updated
    err = unpack_embedded(
      dst, src,
      c, b, a
    );
  } else {
    err = unpack(
      dst, src,
      b,   a, c*a,
      c, b*a,   a,
      a
    );
  }
  if (err != DEVICE_SUCCESS) {
    fprintf(stderr, "pack failed Y <- St1_Comm!\n");
    exit(-1);
  }
#if (!CUDA_AWARE_MPI)  //this copies data to the GPU to perform packing
  DEVICE_MEM_COPY(plan->send_buffer, dst, buffer_size * sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);
#endif
#endif
}

void fftx_mpi_rcperm(fftx_plan plan, double * _Y, double *_X, int stage, bool is_embedded) {
  // int rank = -1;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // double start, stop, max_time;
  // if (rank == 0) { printf("%f,", -2.0); }

  complex<double> *X = (complex<double> *) _X;
  complex<double> *Y = (complex<double> *) _Y;

  switch (stage) {
    case FFTX_MPI_EMBED_1:
      {
        // after first 1D FFT on K dim.
        // [xl, yl, zl, zr]
        size_t buffer_size = plan->shape[0] * plan->shape[2] * plan->shape[4] * (is_embedded ? 2 : 1) * plan->shape[5];
        int       sendSize = plan->shape[0] * plan->shape[2] * plan->shape[4] * (is_embedded ? 2 : 1);
        int recvSize = sendSize;

        // [xl, yl, zl, zr] -> [xl, yl, zl, xr]
        // [xl, (yl, zl), xr] -> [xl, xr, (yl, zl)]
#if CUDA_AWARE_MPI
        // start = MPI_Wtime();
        DEVICE_MEM_COPY(plan->send_buffer, X, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_DEVICE_TO_DEVICE);
        MPI_Alltoall(
          // X, sendSize*plan->b,
          plan->send_buffer, sendSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->recv_buffer, recvSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->row_comm // TODO: Make sure this is the correct communicator
        ); // assume N dim is initially distributed along col comm.
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }
        // start = MPI_Wtime();
        pack_embed(plan, Y, plan->recv_buffer, plan->b * plan->shape[0], plan->shape[2] * plan->shape[4] * (is_embedded ? 2 : 1), plan->shape[1], is_embedded);
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }
        // start = MPI_Wtime();
#else
        // start = MPI_Wtime();
        DEVICE_MEM_COPY(plan->send_buffer, X, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_DEVICE_TO_HOST);
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }

        // start = MPI_Wtime();
        MPI_Alltoall(
          plan->send_buffer, sendSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->recv_buffer, recvSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->row_comm // TODO: Make sure this is the correct communicator
        ); // assume N dim is initially distributed along col comm.
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }

        // start = MPI_Wtime();
        pack_embed(plan, Y,                 X, plan->b * plan->shape[0], plan->shape[2] * plan->shape[4] * (is_embedded ? 2 : 1), plan->shape[1], is_embedded);
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }
#endif
      } // end FFTX_MPI_EMBED_1
      break;

    case FFTX_MPI_EMBED_2:
      {
        // [yl, zl, xl, xr]
        size_t buffer_size = plan->shape[2] * plan->shape[4] * (is_embedded ? 2 : 1) * plan->shape[0] * (is_embedded ? 2 : 1) * plan->shape[1];
        int sendSize       = plan->shape[2] * plan->shape[4] * (is_embedded ? 2 : 1) * plan->shape[0] * (is_embedded ? 2 : 1);
        int recvSize = sendSize;

        // [yl, zl, xl, xr] -> [yl, zl, xl, yr]
        // [yl, (zl, xl), yr] -> [yl, yr, (zl, xl)]
#if CUDA_AWARE_MPI
        // start = MPI_Wtime();
        // TODO: this copy shouldn't be necessary, but provides incorrect results without it.
        // should be able to just use X as input to MPI.
        // it could be a synchronization issue?
        DEVICE_MEM_COPY(plan->send_buffer, X, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_DEVICE_TO_DEVICE);
        MPI_Alltoall(
	        plan->send_buffer, sendSize*plan->b,
	        // X, sendSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->recv_buffer, recvSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->col_comm // TODO: make sure this is the right communicator to support non-square grid
        );
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }

        // start = MPI_Wtime();
        pack_embed(plan, Y, plan->recv_buffer, plan->b * plan->shape[2], plan->shape[4] * (is_embedded ? 2 : 1) * plan->shape[0] * (is_embedded ? 2 : 1), plan->shape[3], is_embedded);
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }
#else
        // start = MPI_Wtime();
        DEVICE_MEM_COPY(plan->send_buffer, X, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_DEVICE_TO_HOST);
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }

        // start = MPI_Wtime();
        MPI_Alltoall(
	        plan->send_buffer, sendSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->recv_buffer, recvSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->col_comm // TODO: make sure this is the right communicator to support non-square grid
        );
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }

        // start = MPI_Wtime();
        pack_embed(plan, Y, X, plan->b * plan->shape[2], plan->shape[4] * (is_embedded ? 2 : 1) * plan->shape[0] * (is_embedded ? 2 : 1), plan->shape[3], is_embedded);
        // stop = MPI_Wtime();
        // max_time = max_diff(start, stop, MPI_COMM_WORLD);
        // if (rank == 0) { printf("%f,", max_time); }
#endif
      } // end FFTX_MPI_EMBED_2
      break;

    case FFTX_MPI_EMBED_3:
      {
        // TODO: add embedded for inverse.
        // [yl, yr, zl, xl]
        size_t buffer_size = plan->shape[2] * plan->shape[3] * plan->shape[4] * plan->shape[0];
        int sendSize = plan->shape[2] * plan->shape[4] * plan->shape[0];
        int recvSize = sendSize;
        // [yl, yr, (zl, xl)] -> [yl, (zl, xl), yr]
        // [yl, zl, xl, yr] -> [yl, zl, xl, xr]
#if CUDA_AWARE_MPI
        unpack_embed(plan, plan->send_buffer, X, plan->b * plan->shape[2], plan->shape[4] * plan->shape[0], plan->shape[3], is_embedded);
        MPI_Alltoall(
	        plan->send_buffer, sendSize*plan->b,
          MPI_DOUBLE_COMPLEX,
	        plan->recv_buffer, recvSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->col_comm
        ); // assume K dim is initially distributed along row comm.
        DEVICE_MEM_COPY(Y, plan->recv_buffer, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_DEVICE_TO_DEVICE);
#else
        unpack_embed(plan, Y, X, plan->b * plan->shape[2], plan->shape[4] * plan->shape[0], plan->shape[3], is_embedded);
        MPI_Alltoall(
	        plan->send_buffer, sendSize*plan->b,
          MPI_DOUBLE_COMPLEX,
	        plan->recv_buffer, recvSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->col_comm
        ); // assume K dim is initially distributed along row comm.
        DEVICE_MEM_COPY(Y, plan->recv_buffer, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_HOST_TO_DEVICE);
#endif
      } // end FFTX_MPI_EMBED_3
      break;

    case FFTX_MPI_EMBED_4:
      {
        // [xl, xr, yl, zl]
        size_t buffer_size = plan->shape[0] * plan->shape[1] * plan->shape[2] * plan->shape[4];
        int sendSize = plan->shape[0] * plan->shape[2] * plan->shape[4];
        int recvSize = sendSize;

        // [xl, xr, (yl, zl)] -> [xl, (yl, zl), xr]
        // [xl, yl, zl, xr] -> [xl, yl, zl, zr]
#if CUDA_AWARE_MPI
        unpack_embed(plan, plan->send_buffer, X, plan->b * plan->shape[0], plan->shape[2] * plan->shape[4], plan->shape[1], is_embedded);
        MPI_Alltoall(
          plan->send_buffer, sendSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->recv_buffer, recvSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->row_comm
        ); // assume N dim is initially distributed along col comm.
        DEVICE_MEM_COPY(Y, plan->recv_buffer, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_DEVICE_TO_DEVICE);
#else
        unpack_embed(plan, Y, X, plan->b * plan->shape[0], plan->shape[2] * plan->shape[4], plan->shape[1], is_embedded);
        MPI_Alltoall(
          plan->send_buffer, sendSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->recv_buffer, recvSize*plan->b,
          MPI_DOUBLE_COMPLEX,
          plan->row_comm
        ); // assume N dim is initially distributed along col comm.
        DEVICE_MEM_COPY(Y, plan->recv_buffer, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_HOST_TO_DEVICE);
#endif
      } // end FFTX_MPI_EMBED_4
      break;
  default:
      break;
  }
}