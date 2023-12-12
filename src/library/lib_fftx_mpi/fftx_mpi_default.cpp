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


fftx_plan fftx_plan_distributed_default(int r, int c, int M, int N, int K, int batch, bool is_embedded, bool is_complex) {

  fftx_plan plan = (fftx_plan) malloc(sizeof(fftx_plan_t));

  plan->b = batch;
  plan->is_embed = is_embedded;
  plan->is_complex = is_complex;

  init_2d_comms(plan, r, c,  M,  N, K);   //embedding uses the input sizes

  uint64_t m = M;
  uint64_t n = N;
  uint64_t k = K;

  DEVICE_MALLOC(&(plan->Q3), m*n*k*(is_embedded ? 8 : 1) / (r * c) * sizeof(complex<double>) * batch);
  DEVICE_MALLOC(&(plan->Q4), m*n*k*(is_embedded ? 8 : 1) / (r * c) * sizeof(complex<double>) * batch);

  int batch_sizeZ = m/r * n/c;
  int batch_sizeX = n/c * k/r;
  int batch_sizeY = k/r * m/c;

  int inK = K * (is_embedded ? 2 : 1);
  int inM = M * (is_embedded ? 2 : 1);
  int inN = N * (is_embedded ? 2 : 1);


  // int outK = K * (is_embedded ? 2 : 1);
  // int outM = M * (is_embedded ? 2 : 1);
  // int outN = N * (is_embedded ? 2 : 1);


  batch_sizeX *= (is_embedded ? 2 : 1);
  batch_sizeY *= (is_embedded ? 4 : 1);


  if ((plan->is_complex))
    {
      //read seq write strided
      DEVICE_FFT_PLAN_MANY(&(plan->stg1), 1, &inK,
			   &inK,             plan->b, inK*plan->b,
			   &inK, batch_sizeZ*plan->b, plan->b,
			   DEVICE_FFT_Z2Z, batch_sizeZ);

      //inverse plan -> read strided write seq
      DEVICE_FFT_PLAN_MANY(&(plan->stg1i), 1, &inK,
			   &inK, batch_sizeZ*plan->b, plan->b,
			   &inK,             plan->b, inK*plan->b,
			   DEVICE_FFT_Z2Z, batch_sizeZ);

    }
  else
    {
      //read seq write strided
      DEVICE_FFT_PLAN_MANY(&(plan->stg1), 1, &inK,
			   &inK,             plan->b, inK*plan->b,
			   &inK, batch_sizeZ*plan->b, plan->b,
			   DEVICE_FFT_D2Z, batch_sizeZ);

      //inverse plan -> read strided write seq
      DEVICE_FFT_PLAN_MANY(&(plan->stg1i), 1, &inK,
			   &inK, batch_sizeZ*plan->b, plan->b,
			   &inK,             plan->b, inK*plan->b,
			   DEVICE_FFT_Z2D, batch_sizeZ);
    }

  //read seq write strided
  DEVICE_FFT_PLAN_MANY(&(plan->stg2), 1, &inM,
		       &inM,           plan->b, inM*plan->b,
		       &inM, batch_sizeX*plan->b, plan->b,
		       DEVICE_FFT_Z2Z, batch_sizeX);

  //read seq write seq
  DEVICE_FFT_PLAN_MANY(&(plan->stg3), 1, &inN,
		       &inN, plan->b, inN*plan->b,
		       &inN, plan->b, inN*plan->b,
		       DEVICE_FFT_Z2Z, batch_sizeY);

  //read strided write seq
  DEVICE_FFT_PLAN_MANY(&(plan->stg2i), 1, &inM,
		       &inM, batch_sizeX*plan->b, plan->b,
		       &inM,           plan->b, inM*plan->b,
		       DEVICE_FFT_Z2Z, batch_sizeX);

  return plan;
}

void fftx_execute_default(fftx_plan plan, double* out_buffer, double*in_buffer, int direction) {
  // int rank = -1;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // double start, stop, max_time;
  // if (rank == 0) { printf("%f,", -1.0); }

  if (direction == DEVICE_FFT_FORWARD) {
    // start = MPI_Wtime();
    if (plan->is_complex) {
      for (int i = 0; i != plan->b; ++i) {
        DEVICE_FFT_EXECZ2Z(plan->stg1, ((DEVICE_FFT_DOUBLECOMPLEX  *) in_buffer + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i), direction);
      }
    } else {
      for (int i = 0; i != plan->b; ++i) {
        DEVICE_FFT_EXECD2Z(plan->stg1, ((DEVICE_FFT_DOUBLEREAL  *) in_buffer + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i));
      }
    }
    // stop = MPI_Wtime();
    // max_time = max_diff(start, stop, MPI_COMM_WORLD);
    // if (rank == 0) { printf("%f,", max_time); }

    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);

    // start = MPI_Wtime();
    for (int i = 0; i != plan->b; ++i) {
      DEVICE_FFT_EXECZ2Z(plan->stg2, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i), direction);
    }
    // stop = MPI_Wtime();
    // max_time = max_diff(start, stop, MPI_COMM_WORLD);
    // if (rank == 0) { printf("%f,", max_time); }

    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_2, plan->is_embed);

    // start = MPI_Wtime();
    for (int i = 0; i != plan->b; ++i) {
      DEVICE_FFT_EXECZ2Z(plan->stg3, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer + i), direction);
    }
    // stop = MPI_Wtime();
    // max_time = max_diff(start, stop, MPI_COMM_WORLD);
    // if (rank == 0) { printf("%f,", max_time); }

  } else if (direction == DEVICE_FFT_INVERSE) {
    for (int i = 0; i != plan->b; ++i) {
      DEVICE_FFT_EXECZ2Z(
        plan->stg3,
        ((DEVICE_FFT_DOUBLECOMPLEX  *) in_buffer + i),
        ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i),
        direction
      );
    }

    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_3, plan->is_embed);

    for (int i = 0; i != plan->b; ++i){
      DEVICE_FFT_EXECZ2Z(plan->stg2i, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i), direction);
    }

    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_4, plan->is_embed);

    if (plan->is_complex) {
      for (int i = 0; i != plan->b; ++i) {
        DEVICE_FFT_EXECZ2Z(plan->stg1i, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer + i), direction);
      }
    } else { // untested
      for (int i = 0; i != plan->b; ++i) {
        DEVICE_FFT_EXECZ2D(plan->stg1i, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i), ((DEVICE_FFT_DOUBLEREAL  *) out_buffer + i));
      }
    }
  }
}


void fftx_plan_destroy_default(fftx_plan plan) {
  if (plan) {
    if (plan->c == 0)
      destroy_1d_comms(plan);
    else
      destroy_2d_comms(plan);

    DEVICE_FREE(plan->Q3);
    DEVICE_FREE(plan->Q4);

    free(plan);
  }
}