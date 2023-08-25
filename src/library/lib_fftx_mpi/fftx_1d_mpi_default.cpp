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

#include "fftx_1d_mpi.hpp"
#include "fftx_1d_mpi_default.hpp"

using namespace std;

fftx_plan fftx_plan_distributed_1d_default(
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
  size_t e            = is_embedded ? 2 : 1;

  init_1d_comms(plan, p, M, N, K);   //embedding uses the input sizes

  DEVICE_MALLOC(&(plan->Q3), sizeof(complex<double>) * (((size_t)M)*e*N*e*K*e / p) * batch);
  DEVICE_MALLOC(&(plan->Q4), sizeof(complex<double>) * (((size_t)M)*e*N*e*K*e / p) * batch);
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
  plan->shape[4] = K/p; // TODO: round up?
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

    batch_sizeY = K*e * M0;   // stage 2i, dist X
    DEVICE_FFT_PLAN_MANY(
        &(plan->stg2i),  1, &inN,
        &inN, batch_sizeY*plan->b, plan->b,
        &inN, plan->b, inN*plan->b,
        DEVICE_FFT_Z2Z, batch_sizeY
    );
    // TODO: fix rounding here.
    batch_sizeX = N*e * K*e/p;  // stage 1i, dist Z
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
      int batch_sizeY = K*e * M0;   // stage 2i, dist X
      DEVICE_FFT_PLAN_MANY(
        &(plan->stg2i), 1, &inN,
        &inN, batch_sizeY*plan->b, plan->b,
        &inN, plan->b, inN*plan->b,
        DEVICE_FFT_Z2Z, batch_sizeY
      );

    // [Z/pz, Y, X] <= [px * ceil(X'/px), Z/pz, Y] (read strided, write seq)
    // X' is complex size M, X is (M-1)*2
    // TODO: update for embedded
      // [Z/pz, Y, X] <= [px * ceil(X'/px), Z/pz, Y]
      int batch_sizeX = N*e * K*e/p;  // stage 1i, dist Z
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


void fftx_execute_1d_default(
  fftx_plan plan,
  double * out_buffer, double * in_buffer,
  int direction
) {

  size_t e = plan->is_embed ? 2 : 1;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (direction == DEVICE_FFT_FORWARD) {
    if (plan->is_complex) {
      // [X', Z/p, Y, b] <= [Z/p, Y, X, b]
      for (int b = 0; b < plan->b; b++) {
        DEVICE_FFT_EXECZ2Z(
          plan->stg1,
          ((DEVICE_FFT_DOUBLECOMPLEX *) in_buffer) + b,
          ((DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3)  + b,
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
      for (int b = 0; b < plan->b; b++) {
        DEVICE_FFT_EXECD2Z(
          plan->stg1,
          ((DEVICE_FFT_DOUBLEREAL    *) in_buffer) + b,
          ((DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3)  + b
        );
      }
      // [X'/px, pz, b, Z/pz, Y] <= [px, X'/px, b, Z/pz, Y]
      fftx_mpi_rcperm_1d(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);

      // [Y, ceil(X'/px), Z] <= [ceil(X'/px), Z, Y]
      // TODO: change plan to put output in embedded space?
      // [Y, ceil(X'/px), 2Z]
      for (int b = 0; b < plan->b; ++b) {
        DEVICE_FFT_EXECZ2Z(
          plan->stg2,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4) + b,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3) + b,
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

      // [Y, ceil(X'/px), Z] (no permutation on last stage)
      for (int b = 0; b < plan->b; ++b) {
        DEVICE_FFT_EXECZ2Z(
          plan->stg3,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) stg3_input) + b,
          ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer) + b,
          direction
        );
      }
    }
  } else if (direction == DEVICE_FFT_INVERSE) { // backward
    DEVICE_FFT_DOUBLECOMPLEX *stg3i_input  = (DEVICE_FFT_DOUBLECOMPLEX *) in_buffer;
    DEVICE_FFT_DOUBLECOMPLEX *stg3i_output = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3;
    // [Y, ceil(X'/px), Z] <= [Y, ceil(X'/px), Z] (read seq, write seq)
    for (int b = 0; b < plan->b; b++) {
      DEVICE_FFT_EXECZ2Z(plan->stg3, stg3i_input + b, stg3i_output + b, direction);
    }
    // no permutation necessary, use previous output as input.
    DEVICE_FFT_DOUBLECOMPLEX *stg2i_input  = stg3i_output;
    DEVICE_FFT_DOUBLECOMPLEX *stg2i_output = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q4;
    // TODO: add code here if we expect embedded.

    //stage 2i
    // [ceil(X'/px), Z, Y] <= [Y, ceil(X'/px), Z] (read strided, write seq)
    for (int b = 0; b < plan->b; ++b) {
      DEVICE_FFT_EXECZ2Z(plan->stg2i, stg2i_input + b, stg2i_output + b, direction);
    }

    DEVICE_FFT_DOUBLECOMPLEX *stg1i_input = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3;

    // permute such that
    // [ceil(X'/px),          pz, Z/pz, Y] <= [ceil(X'/px),                 Z, Y] (reshape)
    // [         pz, ceil(X'/px), Z/pz, Y] <= [ceil(X'/px),          pz, Z/pz, Y] (permute)
    // [         px, ceil(X'/px), Z/pz, Y] <= [         pz, ceil(X'/px), Z/pz, Y] (all2all)
    // [        px * ceil(X'/px), Z/pz, Y] <= [px, ceil(X'/px), Z/pz, Y] (reshape)
    fftx_mpi_rcperm_1d(plan, (double *) stg1i_input, (double *) stg2i_output, FFTX_MPI_EMBED_4, plan->is_embed);

    // Transpose as part of FFT
    // [Z/pz, Y, X] <= [px * ceil(X'/px), Z/pz, Y]
    DEVICE_FFT_DOUBLECOMPLEX *stg1i_output = (DEVICE_FFT_DOUBLECOMPLEX *) out_buffer;

    //stage 1i
    for (int b = 0; b < plan->b; ++b) {
      if (plan->is_complex) {
        //backward complex
        DEVICE_FFT_EXECZ2Z(plan->stg1i, stg1i_input + b, stg1i_output + b, direction);
      } else {
        //backward real
        DEVICE_FFT_EXECZ2D(plan->stg1i, stg1i_input + b, ((DEVICE_FFT_DOUBLEREAL *) stg1i_output) + b);
      }
    }
  } // end backward.
}