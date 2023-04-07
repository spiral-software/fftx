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
  plan->M = M;
  plan->N = N;
  plan->K = K;

  plan->r = pp;
  plan->c = 0;

  size_t max_size = M*N*K*(plan->is_embed ? 8 : 1)/(plan->r) * plan->b;
  
#if CUDA_AWARE_MPI
  DEVICE_MALLOC(&(plan->send_buffer), max_size * sizeof(complex<double>));
  DEVICE_MALLOC(&(plan->recv_buffer), max_size * sizeof(complex<double>));
#else
  plan->send_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
  plan->recv_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
#endif
  
  //compute shape
  // TODO: update this for real to complex. M/2+1 etc.
  // currently being done elsewhere.
  plan->shape[0] = M/pp;
  plan->shape[1] = pp;
  plan->shape[2] = N;
  plan->shape[3] = 1;  
  plan->shape[4] = K/pp;
  plan->shape[5] = pp;
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


fftx_plan  fftx_plan_distributed_1d(int p, int M, int N, int K, int batch, bool is_embedded, bool is_complex) {
  fftx_plan plan = (fftx_plan) malloc(sizeof(fftx_plan_t));
  plan->b = batch;
  plan->is_embed = is_embedded;
  plan->is_complex = is_complex;

  init_1d_comms(plan, p,  M,  N, K);   //embedding uses the input sizes

  int e = is_embedded ? 2 : 1;

  DEVICE_MALLOC(&(plan->Q3), M*e*N*e*K*e / p * sizeof(complex<double>) * batch);
  DEVICE_MALLOC(&(plan->Q4), M*e*N*e*K*e / p * sizeof(complex<double>) * batch);

  int M0 = (M*e)/p;
  if (!(plan->is_complex)) {
    M0 = (M*e/2 + 1)/p;
    if (M0*p < M*e) { // pad up if M' not divisible by p.
      M0 += 1;
    }
    plan->shape[0] = M0;
  }
  int M1 = p;

  // DFT sizes.  
  int inM = M * e;
  int inN = N * e;
  int inK = K * e;

  // DFT batch sizes.
  int batch_sizeX = N * K/p;  // stage 1, dist Z
  int batch_sizeY = K * M0;   // stage 2, dist X
  int batch_sizeZ = M0 * N*e; // stage 3, dist X
  
  if (plan->is_complex) {
      //only correct if not embedded
      /*
      DEVICE_FFT_PLAN_MANY(&(plan->stg1), 2, sizes, 
        sizes, plan->b, inM*inN*batch,
        sizes, plan->b, inM*inN*batch,
        CUFFT_Z2Z, inK/plan->r
      );

      DEVICE_FFT_PLAN_MANY(&(plan->stg2), 1, &inK,
        &inM, inM*inN/p*batch, batch,
        &inM, inM*inN/p*batch, batch, 
        CUFFT_Z2Z, inM*inN/plan->r*batch);
      */
  } else {
      // slowest to fastest
      // [X', Z/pz, Y] <= [Z/pz, Y, X] (read seq, write strided)
      // X' is complex size M/2 + 1
      DEVICE_FFT_PLAN_MANY(
        &(plan->stg1), 1, &inM,
        &inM, plan->b, inM*plan->b,
        &inM, batch_sizeX*plan->b, plan->b,
        DEVICE_FFT_D2Z, batch_sizeX
			);
      
      // [Y, X'/px, Z] <= [X'/px, Z, Y] (read seq, write strided)
      // if embedded, put output in
      // [Y, X'/px, 2Z]
      DEVICE_FFT_PLAN_MANY(
        &(plan->stg2),  1,            &inN,
        &inN,     plan->b,            inN*plan->b,	       
        &inN,    batch_sizeY*plan->b, plan->b,
        DEVICE_FFT_Z2Z, batch_sizeY
			);
      
      // [Y, X'/px, Z] <= [Y, X'/px, Z] (read seq, write seq)
      DEVICE_FFT_PLAN_MANY(
        &(plan->stg3), 1, &inK,
        &inK, plan->b, inK*plan->b,
        &inK, plan->b, inK*plan->b,
        DEVICE_FFT_Z2Z, batch_sizeZ
			);
  }
  return plan;  
}


void fftx_mpi_rcperm_1d(fftx_plan plan, double * _Y, double *_X, int stage, bool is_embedded) {
  complex<double> *X = (complex<double> *) _X;
  complex<double> *Y = (complex<double> *) _Y;
  
  int e = is_embedded ? 2 : 1;

  switch (stage) {
    case FFTX_MPI_EMBED_1:
      {
        // TODO: support different MPI CPU/HOST devices.
        // after first 1D FFT on X dim (slowest to fastest)
        // [X'   , Z/p, Y]    <= [Z/p, Y, X]
        // [M/2+1, K/p, N] <= [K/p, N, M]

        //#if CUDA_AWARE_MPI
        //      DEVICE_MEM_COPY(plan->send_buffer, X, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_DEVICE_TO_DEVICE);
        //#else
        // TODO: check these sizes.
        // TODO: change this to be acceptable for C2C, not just R2C.
        size_t buffer_size = (plan->M*e/2+1) * plan->shape[4] * plan->shape[2] * plan->b;
        size_t sendSize    = plan->shape[0]  * plan->shape[4] * plan->shape[2] * plan->b;
        size_t recvSize = sendSize;

        DEVICE_MEM_COPY(
          plan->send_buffer, X,
          buffer_size * sizeof(complex<double>),
          MEM_COPY_DEVICE_TO_HOST
        );
        // int rank = -1;
        // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // if (rank == 0) {
        //   for (int i = 0; i < plan->M * e; i++) {
        //     printf("%f \t", plan->send_buffer[i].real());
        //   }
        //   printf("\n");
        // }
        // TODO: fill out send buffer with zeros if R2C size not divisible by p.


        // [pz, X'/px, Z/pz, Y] <= [X', Z/pz, Y]
        MPI_Alltoall(
          plan->send_buffer,  sendSize,
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
            (complex<double> *) X, Y,
            plan->shape[4] * plan->shape[2] * plan->b,
            plan->shape[0],
            plan->shape[5],
            false
          );
          embed(
            Y, X,
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
            (complex<double> *) Y, X,
            plan->shape[4] * plan->shape[2] * plan->b,
            plan->shape[0],
            plan->shape[5],
            false
          );
        }
        //#endif
      }
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
    }
}

void fftx_execute_1d(fftx_plan plan, double* out_buffer, double*in_buffer, int direction) {
  if (direction == DEVICE_FFT_FORWARD) {
    if (plan->is_complex) {
      // TODO: write complex.
    } else {
      //forward real	
      for (int i = 0; i != plan->b; ++i) {
        DEVICE_FFT_EXECD2Z(
          plan->stg1,
          ((DEVICE_FFT_DOUBLEREAL    *) in_buffer + i),
          ((DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3  + i)
        );
      }
    }
    
    fftx_mpi_rcperm_1d(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);
    
    // [Y, X'/px, Z] <= [X'/px, Z, Y]
    // TODO: change plan to put output in embedded space?
    // [Y, X'/px, 2Z]
    for (int i = 0; i != plan->b; ++i) {
      DEVICE_FFT_EXECZ2Z(
        plan->stg2,
        ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i),
			  ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i),
        direction
      );
    }

    double *st2_output = (double *) plan->Q3;
    double *st3_input  = (double *) plan->Q4;
    if (plan->is_embed) {
      // st3_input = st2_output;
      fftx_mpi_rcperm_1d(plan, st3_input, st2_output, FFTX_MPI_EMBED_2, plan->is_embed);

    } else {
      // use previous output as input.
      st3_input = st2_output;
    }
    
    for (int i = 0; i != plan->b; ++i) {
      DEVICE_FFT_EXECZ2Z(
        plan->stg3,
        ((DEVICE_FFT_DOUBLECOMPLEX  *) st3_input  + i),
			  ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer + i),
        direction
      );
    }      
  } else { // backward
    
    //stage 2i
    for (int i = 0; i < plan->b; ++i){
      if (plan->is_complex) {
        //backward complex	  
      } else {
        //backward real
      }
    }

    //COMMS


    //stage 1i
    for (int i = 0; i < plan->b; ++i){
      if (plan->is_complex) {
        //backward complex
      } else {
        //backward real
      }
    }
  } // end backward.
}
