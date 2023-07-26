#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>

#include "device_macros.h"
#include "fftx_gpu.h"
#include "fftx_1d_gpu.h"
#include "fftx_util.h"
#include "fftx_mpi.hpp"

using namespace std;

void init_1d_comms(fftx_plan plan, int pp, int M, int N, int K) {
  size_t max_size = M*N*K*(plan->is_embed ? 8 : 1)/(plan->r) * plan->b;
#if CUDA_AWARE_MPI
  DEVICE_MALLOC(&(plan->send_buffer), max_size * sizeof(complex<double>));
  DEVICE_MALLOC(&(plan->recv_buffer), max_size * sizeof(complex<double>));
#else
  plan->send_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
  plan->recv_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
#endif
}

void destroy_1d_comms(fftx_plan plan) {
  if (plan) {
#if CUDA_AWARE_MPI
    DEVICE_FREE(plan->send_buffer);
    DEVICE_FREE(plan->recv_buffer);
#else
    free(plan->send_buffer);
    free(plan->recv_buffer);
#endif
  }
}


fftx_plan fftx_plan_distributed_1d(
  int p, int M, int N, int K,
  int batch, bool is_embedded, bool is_complex
) {
  fftx_plan plan   = (fftx_plan) malloc(sizeof(fftx_plan_t));
  plan->M = M;
  plan->N = N;
  plan->K = K;
  plan->r = p;
  plan->c = 0;
  plan->b          = batch;
  plan->is_complex = is_complex;
  plan->is_embed   = is_embedded;
  int e            = is_embedded ? 2 : 1;

  init_1d_comms(plan, p, M, N, K);   //embedding uses the input sizes

  DEVICE_MALLOC(&(plan->Q3), M*e*N*e*K*e / p * sizeof(complex<double>) * batch);
  DEVICE_MALLOC(&(plan->Q4), M*e*N*e*K*e / p * sizeof(complex<double>) * batch);

  /*
    R2C is
    [K, N,       M]
    [K, N, M/2 + 1]

    C2R is
    [K, N, M/2 + 1]
    [K, N,       M]
  */

  // DFT sizes.
  int inM = M * e;
  int inN = N * e;
  int inK = K * e;

  int M0 = -1;
  if (plan->is_complex) {
    M0 = (M*e)/p;
  } else {
    M0 = (M*e/2 + 1)/p;
  }
  // pad up if M' not divisible by p.
  M0 += M0*p < M*e;
  int M1 = p;

  // set shape
  plan->shape[0] = M0; // this overwrites M/p if r2c or c2r, right?
  plan->shape[1] = M1;
  plan->shape[2] = N;
  plan->shape[3] = 1;
  plan->shape[4] = K/p;
  plan->shape[5] = p;

  if (plan->is_complex) {
    //only correct if not embedded
    int batch_sizeX = N * K/p;  // stage 1, dist Z
    // int batch_sizeX = inN*inK/p;
    DEVICE_FFT_PLAN_MANY(
        &(plan->stg1),  1, &inM,
        &inM, plan->b, inM*plan->b,
        &inM, batch_sizeX*plan->b, plan->b,
        DEVICE_FFT_Z2Z, batch_sizeX
    );
    int batch_sizeY = K * M0;   // stage 2, dist X
    // int batch_sizeY = inM*inK/p;
    DEVICE_FFT_PLAN_MANY(
        &(plan->stg2),  1, &inN,
        &inN, plan->b, inN*plan->b,
        &inN, batch_sizeY*plan->b, plan->b,
        DEVICE_FFT_Z2Z, batch_sizeY
    );
    int batch_sizeZ = M0 * N*e; // stage 3, dist X
    // int batch_sizeZ = inN*inM/p;
    DEVICE_FFT_PLAN_MANY(
        &(plan->stg3), 1, &inK,
        &inK, plan->b, inK*plan->b,
        &inK, plan->b, inK*plan->b,
        DEVICE_FFT_Z2Z, batch_sizeZ
    );

    DEVICE_FFT_PLAN_MANY(
        &(plan->stg2i),  1, &inN,
        &inN, batch_sizeY*plan->b, plan->b,
        &inN, plan->b, inN*plan->b,
        DEVICE_FFT_Z2Z, batch_sizeY
    );

    DEVICE_FFT_PLAN_MANY(
        &(plan->stg1i),  1, &inM,
        &inM, batch_sizeX*plan->b, plan->b,
        &inM, plan->b, inM*plan->b,
        DEVICE_FFT_Z2Z, batch_sizeX
    );
  } else {
    int xr = inM;
    int xc = inM/2 + 1;
    // slowest to fastest
    // [X', Z/pz, Y, b] <= [Z/pz, Y, X, b] (read seq, write strided)
    // X' is complex size M/2 + 1
    int batch_sizeX = N * K/p;  // stage 1, dist Z
    DEVICE_FFT_PLAN_MANY(
      &(plan->stg1), 1, &xr,
      &xr, plan->b, xr*plan->b,
      &xc, batch_sizeX*plan->b, plan->b,
      DEVICE_FFT_D2Z, batch_sizeX
    );
    // [Y, X'/px, Z] <= [X'/px, Z, Y] (read seq, write strided)
    // if embedded, put output in
    // [Y, X'/px, 2Z]
    {
      int batch_sizeY = K * M0;   // stage 2, dist X
      DEVICE_FFT_PLAN_MANY(
        &(plan->stg2),  1, &inN,
        &inN, plan->b, inN*plan->b,
        &inN, batch_sizeY*plan->b, plan->b,
        DEVICE_FFT_Z2Z, batch_sizeY
      );

      int batch_sizeZ = M0 * N*e; // stage 3, dist X
      // [Y, X'/px, Z] <= [Y, X'/px, Z] (read seq, write seq)
      DEVICE_FFT_PLAN_MANY(
        &(plan->stg3), 1, &inK,
        &inK, plan->b, inK*plan->b,
        &inK, plan->b, inK*plan->b,
        DEVICE_FFT_Z2Z, batch_sizeZ
      );
    }

    // [X'/px, Z, Y] <= [Y, X'/px, Z] (read strided, write seq)
    // TODO: update for embedded
    {
      // int M0 = (M*e)/p;
      int M0 = ((M*e)/2 + 1)/p;
      M0 += (M0*p < M*e);
      int batch_sizeY = K * M0;   // stage 2, dist X
      DEVICE_FFT_PLAN_MANY(
        &(plan->stg2i), 1, &inN,
        &inN, batch_sizeY*plan->b, plan->b,
        &inN, plan->b, inN*plan->b,
        DEVICE_FFT_Z2Z, batch_sizeY
      );

    // [Z/pz, Y, X] <= [X', Z/pz, Y] (read strided, write seq)
    // X' is complex size M, X is (M-1)*2
    // TODO: update for embedded
      DEVICE_FFT_PLAN_MANY(
        &(plan->stg1i), 1, &xr,
        &xc, batch_sizeX*plan->b, plan->b,
        &xr, plan->b, xr*plan->b,
        DEVICE_FFT_Z2D, batch_sizeX
      );
    }
  }
  return plan;
}

void fftx_mpi_rcperm_1d(
  fftx_plan plan, double * Y, double *X, int stage, bool is_embedded
) {
  // int e = is_embedded ? 2 : 1;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  switch (stage) {
    case FFTX_MPI_EMBED_1:
      {
          // TODO: support different MPI CPU/HOST devices.
          // after first 1D FFT on X dim (slowest to fastest)
          // [X'   , Z/p, Y] <= [Z/p, Y, X]
          // [M/2+1, K/p, N] <= [K/p, N, M]
          // or for C2C
          // [M    , K/p, N] <= [K/p, N, M]

          // TODO: check these sizes.
          // TODO: change this to be acceptable for C2C, not just R2C.
          // size_t buffer_size = (plan->M*e/2+1) * plan->shape[4] * plan->shape[2] * plan->b;
          size_t buffer_size = plan->shape[1] * plan->shape[0] * plan->shape[4] * plan->shape[3] * plan->shape[2] * plan->b;
          size_t sendSize    =                  plan->shape[0] * plan->shape[4] * plan->shape[3] * plan->shape[2] * plan->b;
          size_t recvSize    = sendSize;

          DEVICE_MEM_COPY(
            plan->send_buffer, X,
            buffer_size * sizeof(complex<double>),
            MEM_COPY_DEVICE_TO_HOST
          );

          // [pz, X'/px, Z/pz, Y] <= [X', Z/pz, Y]
          MPI_Alltoall(
            plan->send_buffer, sendSize,
            MPI_DOUBLE_COMPLEX,
            plan->recv_buffer, recvSize,
            MPI_DOUBLE_COMPLEX,
            MPI_COMM_WORLD
          );
          //      [X'/px, pz, Z/pz, Y] <= [pz, X'/px, Z/pz, Y]
          // i.e. [X'/px,        Z, Y]
          if (is_embedded) {
            DEVICE_MEM_COPY(
              Y, plan->recv_buffer,
              plan->shape[5] * plan->shape[0] * plan->shape[4] * plan->shape[2] * sizeof(complex<double>) * plan->b,
              MEM_COPY_HOST_TO_DEVICE
            );
            pack_embed(
              plan,
              (complex<double> *) X, (complex<double> *) Y,
              plan->shape[4] * plan->shape[2] * plan->b,
              plan->shape[0],
              plan->shape[5],
              false
            );
            embed(
              (complex<double> *) Y, (complex<double> *) X,
              plan->shape[2],
              plan->shape[0] * plan->shape[5] * plan->shape[4] * plan->b
            );
          } else {
            DEVICE_MEM_COPY(
              X, plan->recv_buffer,
              plan->shape[5] * plan->shape[0] * plan->shape[4] * plan->shape[2] * sizeof(complex<double>) * plan->b,
              MEM_COPY_HOST_TO_DEVICE
            );
            pack_embed(
              plan,
              (complex<double> *) Y, (complex<double> *) X,
              plan->shape[4] * plan->shape[2] * plan->b,
              plan->shape[0],
              plan->shape[5],
              false
            );
          }
      } // end stage 1 case.
      break;

    case FFTX_MPI_EMBED_2:
      {
        if (is_embedded) {
          // TODO: copy buffer into embedded matrix
          // [Y, X'/px, 2Z] <= [Y, X'/px, Z]
          embed(
            (complex<double> *) Y, (complex<double> *) X,
            plan->shape[4] * plan->shape[5], // fastest dim, to be doubled and embedded
            plan->shape[2] * plan->shape[0] * plan->b // slower dim
          );

        } else {
          // NOTE: this case handled outside of this function.
          // swap pointers.
          // void *tmp = (void *) X;
          // X = Y;
          // Y = (complex<double> *) tmp;
        }
      }
      break;

    case FFTX_MPI_EMBED_3:
      {
        // TODO: resume here.

      }
      break;

    case FFTX_MPI_EMBED_4:
      {
        // if inverse,
        // permute such that
        // [X'/px, pz, Z/pz, Y] <= [X'/px,        Z, Y] (reshape)
        // [pz, X'/px, Z/pz, Y] <= [X'/px, pz, Z/pz, Y] (permute)
        if (is_embedded) {
          // don't care at the moment.
        } else {
          // TODO: fix this so we don't copy back and forth for inverse.
          // b/c pack_embed assumes data is in recv_buffer.
          DEVICE_MEM_COPY(
            plan->recv_buffer, X,
            sizeof(complex<double>) * plan->shape[5] * plan->shape[0] * plan->shape[4] * plan->shape[3] * plan->shape[2] * plan->b,
            MEM_COPY_DEVICE_TO_HOST
          );

          // send <- recv
          pack_embed(
            plan,
            (complex<double> *) Y, (complex<double> *) X,
            plan->shape[4] * plan->shape[3] * plan->shape[2] * plan->b,
            plan->shape[5],
            plan->shape[0],
            false
          );
        }
        size_t sendSize = plan->shape[0] * plan->shape[4] * plan->shape[3] * plan->shape[2] * plan->b;
        size_t recvSize = sendSize;
        DEVICE_MEM_COPY(
          plan->send_buffer, Y,
          sizeof(complex<double>) * plan->shape[5] * sendSize,
          MEM_COPY_DEVICE_TO_HOST
        );

        // [px, X'/px, Z/pz, Y] <= [pz, X'/px, Z/pz, Y] (all2all)
        // [       X', Z/pz, Y] <=                      (reshape)
        MPI_Alltoall(
          plan->send_buffer,  sendSize,
          MPI_DOUBLE_COMPLEX,
          plan->recv_buffer, recvSize,
          MPI_DOUBLE_COMPLEX,
          MPI_COMM_WORLD
        );

        DEVICE_MEM_COPY(
          Y, plan->recv_buffer,
          sizeof(complex<double>) * plan->shape[5] * recvSize,
          MEM_COPY_HOST_TO_DEVICE
        );
      }
      break;
    } // end switch/case.
}

void fftx_execute_1d(
  fftx_plan plan,
  double * out_buffer, double * in_buffer,
  int direction
) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (direction == DEVICE_FFT_FORWARD) {
    if (plan->is_complex) {
      // [X', Z/p, Y, b] <= [Z/p, Y, X, b]
      for (int i = 0; i < plan->b; i++) {
        DEVICE_FFT_EXECZ2Z(
          plan->stg1,
          ((DEVICE_FFT_DOUBLECOMPLEX *) in_buffer) + i,
          ((DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3)  + i,
          direction
        );
      }

      // [X'/px, pz, b, Z/pz, Y] <= [px, X'/px, b, Z/pz, Y] // is this right? should batch be inner?
      fftx_mpi_rcperm_1d(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);

      for (int i = 0; i < plan->b; ++i) {
        DEVICE_FFT_EXECZ2Z(
          plan->stg2,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4) + i,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3) + i,
          direction
        );
      }

      double *stg2_output = (double *) plan->Q3;
      double *stg3_input  = (double *) plan->Q4;
      if (plan->is_embed) {
        fftx_mpi_rcperm_1d(plan, stg3_input, stg2_output, FFTX_MPI_EMBED_2, plan->is_embed);
      } else {
        // no permutation necessary, use previous output as input.
        stg3_input = stg2_output;
      }
      // [Y, X'/px, Z] (no permutation on last stage)
      for (int i = 0; i < plan->b; ++i) {
        DEVICE_FFT_EXECZ2Z(
          plan->stg3,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) stg3_input) + i,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer) + i,
          direction
        );
      }
    } else {
      //forward real
      // [X', Z/p, Y, b] <= [Z/p, Y, X, b]
      for (int i = 0; i < plan->b; i++) {
        DEVICE_FFT_EXECD2Z(
          plan->stg1,
          ((DEVICE_FFT_DOUBLEREAL    *) in_buffer) + i,
          ((DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3)  + i
        );
      }
      // [X'/px, pz, b, Z/pz, Y] <= [px, X'/px, b, Z/pz, Y]
      fftx_mpi_rcperm_1d(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);

      // [Y, X'/px, Z] <= [X'/px, Z, Y]
      // TODO: change plan to put output in embedded space?
      // [Y, X'/px, 2Z]
      for (int i = 0; i < plan->b; ++i) {
        DEVICE_FFT_EXECZ2Z(
          plan->stg2,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4) + i,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3) + i,
          direction
        );
      }

      double *stg2_output = (double *) plan->Q3;
      double *stg3_input  = (double *) plan->Q4;
      if (plan->is_embed) {
        fftx_mpi_rcperm_1d(plan, stg3_input, stg2_output, FFTX_MPI_EMBED_2, plan->is_embed);
      } else {
        // no permutation necessary, use previous output as input.
        stg3_input = stg2_output;
      }

      // [Y, X'/px, Z] (no permutation on last stage)
      for (int i = 0; i < plan->b; ++i) {
        DEVICE_FFT_EXECZ2Z(
          plan->stg3,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) stg3_input) + i,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer) + i,
          direction
        );
      }
    }
  } else if (direction == DEVICE_FFT_INVERSE) { // backward
    DEVICE_FFT_DOUBLECOMPLEX *stg3i_input  = (DEVICE_FFT_DOUBLECOMPLEX *) in_buffer;
    DEVICE_FFT_DOUBLECOMPLEX *stg3i_output = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3;
    // [Y, X'/px, Z] <= [Y, X'/px, Z] (read seq, write seq)
    for (int i = 0; i < plan->b; i++) {
      DEVICE_FFT_EXECZ2Z(plan->stg3, stg3i_input + i, stg3i_output + i, direction);
    }
    // no permutation necessary, use previous output as input.
    DEVICE_FFT_DOUBLECOMPLEX *stg2i_input  = stg3i_output;
    DEVICE_FFT_DOUBLECOMPLEX *stg2i_output = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q4;
    // TODO: add code here if we expect embedded.

    //stage 2i
    // [X'/px, Z, Y] <= [Y, X'/px, Z] (read strided, write seq)
    for (int i = 0; i < plan->b; ++i) {
      DEVICE_FFT_EXECZ2Z(plan->stg2i, stg2i_input + i, stg2i_output + i, direction);
    }

    DEVICE_FFT_DOUBLECOMPLEX *stg1i_input = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3;

    // permute such that
    // [X'/px, pz, Z/pz, Y] <= [X'/px         Z, Y] (reshape)
    // [pz, X'/px, Z/pz, Y] <= [X'/px, pz, Z/pz, Y] (permute)
    // [px, X'/px, Z/pz, Y] <= [pz, X'/px, Z/pz, Y] (all2all)
    // [       X', Z/pz, Y] <= [px, X'/px, Z/pz, Y] (reshape)
    fftx_mpi_rcperm_1d(plan, (double *) stg1i_input, (double *) stg2i_output, FFTX_MPI_EMBED_4, plan->is_embed);

    DEVICE_FFT_DOUBLECOMPLEX *stg1i_output = (DEVICE_FFT_DOUBLECOMPLEX *) out_buffer;

    //stage 1i
    for (int i = 0; i < plan->b; ++i) {
      if (plan->is_complex) {
        //backward complex
        DEVICE_FFT_EXECZ2Z(plan->stg1i, stg1i_input + i, stg1i_output + i, direction);
      } else {
        //backward real
        DEVICE_FFT_EXECZ2D(plan->stg1i, stg1i_input + i, ((DEVICE_FFT_DOUBLEREAL *) stg1i_output) + i);
      }
    }
  } // end backward.
}
