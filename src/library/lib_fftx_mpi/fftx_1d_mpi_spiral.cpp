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

#include "interface.hpp"
#include "batch1ddftObj.hpp"
#include "ibatch1ddftObj.hpp"
#include "batch2ddftObj.hpp"
#include "ibatch2ddftObj.hpp"
#if defined FFTX_CUDA
#include "cudabackend.hpp"
#elif defined FFTX_HIP
#include "hipbackend.hpp"
#else
#include "cpubackend.hpp"
#endif

using namespace std;

fftx_plan fftx_plan_distributed_1d_spiral(
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
  // int inM = M * e;
  // int inN = N * e;
  // int inK = K * e;

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

  return plan;
}

void fftx_execute_1d_spiral(
  fftx_plan plan,
  double * out_buffer, double * in_buffer,
  int direction )
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int inM = plan->M * (plan->is_embed ? 2 : 1);
  int inN = plan->N * (plan->is_embed ? 2 : 1);
  int inK = plan->K * (plan->is_embed ? 2 : 1);
  int batch_sizeX = plan->N * plan->K/(int)plan->shape[5];
  int batch_sizeY = plan->K * (int)plan->shape[0];
  int batch_sizeZ = (int)plan->shape[0] * plan->N*(plan->is_embed ? 2 : 1); 

  BATCH1DDFTProblem bdstg1;
  BATCH1DDFTProblem bdstg2;
  BATCH1DDFTProblem bdstg3;
  IBATCH1DDFTProblem ibdstg1;
  IBATCH1DDFTProblem ibdstg2;

  BATCH2DDFTProblem b2dstg1;
  BATCH2DDFTProblem b2dstg2;
  BATCH2DDFTProblem b2dstg3;
  IBATCH2DDFTProblem ib2dstg1;
  IBATCH2DDFTProblem ib2dstg2;

  std::vector<int> size_stg1;  
  std::vector<int> size_stg2;  
  std::vector<int> size_stg3;  
  std::vector<int> size_istg2;  
  std::vector<int> size_istg1; 

  if(plan->b == 1) {
    std::vector<int> size_stg1 = {inM, batch_sizeX, 0, 1};  
    std::vector<int> size_stg2 = {inN, batch_sizeY, 0, 1};  
    std::vector<int> size_stg3 = {inK, batch_sizeZ, 0, 0};  
    std::vector<int> size_istg2 = {inN, batch_sizeY, 1, 0};  
    std::vector<int> size_istg1 = {inM, batch_sizeX, 1, 0}; 
    bdstg1.setSizes(size_stg1);
    bdstg2.setSizes(size_stg2);
    bdstg3.setSizes(size_stg3);
    ibdstg1.setSizes(size_istg1);
    ibdstg2.setSizes(size_istg2);
    bdstg1.setName("b1dft");
    bdstg2.setName("b1dft");
    bdstg3.setName("b1dft");
    ibdstg1.setName("ib1dft");
    ibdstg2.setName("ib1dft");
  } else {
    std::vector<int> size_stg1 = {inM, plan->b, batch_sizeX, 0, 1};  
    std::vector<int> size_stg2 = {inN, plan->b, batch_sizeY, 0, 1};  
    std::vector<int> size_stg3 = {inK, plan->b, batch_sizeZ, 0, 0};  
    std::vector<int> size_istg2 = {inN, plan->b, batch_sizeY, 1, 0};  
    std::vector<int> size_istg1 = {inM, plan->b, batch_sizeX, 1, 0};  
    b2dstg1.setSizes(size_stg1);
    b2dstg2.setSizes(size_stg2);
    b2dstg3.setSizes(size_stg3);
    ib2dstg1.setSizes(size_istg1);
    ib2dstg2.setSizes(size_istg2);
    b2dstg1.setName("b2dft");
    b2dstg2.setName("b2dft");
    b2dstg3.setName("b2dft");
    ib2dstg1.setName("ib2dft");
    ib2dstg2.setName("ib2dft");
  }

  if (direction == DEVICE_FFT_FORWARD) {
    if (plan->is_complex) {
      // [X', Z/p, Y, b] <= [Z/p, Y, X, b]
      if(plan->b  == 1){
        for (int i = 0; i < plan->b; i++) {
          std::vector<void*> args{plan->Q3 + i, in_buffer+i};
          bdstg1.setArgs(args);
          bdstg1.transform();
        }
      } else {
        std::vector<void*> args{plan->Q3, in_buffer};
        b2dstg1.setArgs(args);
        b2dstg1.transform();
      }

      // [X'/px, pz, b, Z/pz, Y] <= [px, X'/px, b, Z/pz, Y] // is this right? should batch be inner?
      fftx_mpi_rcperm_1d(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);
      if(plan->b == 1) {
        for (int i = 0; i < plan->b; ++i) {
          std::vector<void*> args{plan->Q3 + i, plan->Q4 + i};
          bdstg2.setArgs(args);
          bdstg2.transform();
        }
      } else {
        std::vector<void*> args{plan->Q3, plan->Q4};
        b2dstg2.setArgs(args);
        b2dstg2.transform();
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
      if(plan->b == 1) {
        for (int i = 0; i < plan->b; ++i) {
          std::vector<void*> args{out_buffer + i, stg3_input + i};
          bdstg3.setArgs(args);
          bdstg3.transform();
        }
      } else {
        std::vector<void*> args{out_buffer, stg3_input};
        b2dstg3.setArgs(args);
        b2dstg3.transform();
      }
    }
  } else if (direction == DEVICE_FFT_INVERSE) { // backward
    DEVICE_FFT_DOUBLECOMPLEX *stg3i_input  = (DEVICE_FFT_DOUBLECOMPLEX *) in_buffer;
    DEVICE_FFT_DOUBLECOMPLEX *stg3i_output = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3;
    // [Y, X'/px, Z] <= [Y, X'/px, Z] (read seq, write seq)
    if(plan->b == 1) {
      for (int i = 0; i < plan->b; i++) {
        std::vector<void*> args{stg3i_output + i, stg3i_input + i};
        bdstg3.setArgs(args);
        bdstg3.transform();
      }
    } else {
      std::vector<void*> args{stg3i_output, stg3i_input};
      b2dstg3.setArgs(args);
      b2dstg3.transform();
    }
    // no permutation necessary, use previous output as input.
    DEVICE_FFT_DOUBLECOMPLEX *stg2i_input  = stg3i_output;
    DEVICE_FFT_DOUBLECOMPLEX *stg2i_output = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q4;
    // TODO: add code here if we expect embedded.

    //stage 2i
    // [X'/px, Z, Y] <= [Y, X'/px, Z] (read strided, write seq)
    if(plan->b == 1) {
      for (int i = 0; i < plan->b; ++i) {
        std::vector<void*> args{stg2i_output + i, stg2i_input + i};
        ibdstg2.setArgs(args);
        ibdstg2.transform();
      }
    } else {
      std::vector<void*> args{stg2i_output, stg2i_input};
      ib2dstg2.setArgs(args);
      ib2dstg2.transform();
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
    if(plan->b == 1) {
      for (int i = 0; i < plan->b; ++i) {
        if (plan->is_complex) {
          std::vector<void*> args{stg1i_output + i,  stg1i_input + i};
          ibdstg1.setArgs(args);
          ibdstg1.transform();
        }
      }
    } else {
       std::vector<void*> args{stg1i_output,  stg1i_input};
       ib2dstg1.setArgs(args);
       ib2dstg1.transform();
    }
  } // end backward.
}
