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
#include "batch1dprdftObj.hpp"
#include "ibatch1dprdftObj.hpp"
// #include "batch2dprdftObj.hpp"
// #include "ibatch2dprdftObj.hpp"
#if defined FFTX_CUDA
#include "cudabackend.hpp"
#elif defined FFTX_HIP
#include "hipbackend.hpp"
#else
#include "cpubackend.hpp"
#endif

using namespace std;

BATCH1DDFTProblem bdstg1_1d;
BATCH1DDFTProblem bdstg2_1d;
BATCH1DDFTProblem bdstg3_1d;
IBATCH1DDFTProblem ibdstg1_1d;
IBATCH1DDFTProblem ibdstg2_1d;

BATCH2DDFTProblem b2dstg1_1d;
BATCH2DDFTProblem b2dstg2_1d;
BATCH2DDFTProblem b2dstg3_1d;
IBATCH2DDFTProblem ib2dstg1_1d;
IBATCH2DDFTProblem ib2dstg2_1d;

BATCH1DPRDFTProblem bprdstg1_1d;
IBATCH1DPRDFTProblem ibprdstg1_1d;



inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

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

  /*
    R2C is
    [K, N,       M]
    [K, N, M/2 + 1]

    C2R is
    [K, N, M/2 + 1]
    [K, N,       M]
  */

  // DFT sizes.
  //  int inM = M * e;
  //  int inN = N * e;
  //  int inK = K * e;

  int M0 = plan->is_complex ? ceil_div(M*e, p) : ceil_div(M*e/2+1, p);
  int M1 = p;

  int K0 = ceil_div(K, p);
  int K1 = p;

  // set shape
  plan->shape[0] = M0;
  plan->shape[1] = M1;
  plan->shape[2] = N;
  plan->shape[3] = 1;
  plan->shape[4] = K0;
  plan->shape[5] = K1;

  int invK0 = ceil_div(K*e, p);

  size_t buff_size = ((size_t) M0) * ((size_t) M1) * ((size_t) N*e) * 1 * ((size_t) invK0) * ((size_t) batch); // can either omit M1 or K1. arbit omit K1.
  DEVICE_MALLOC(&(plan->Q3), sizeof(complex<double>) * buff_size * batch);
  DEVICE_MALLOC(&(plan->Q4), sizeof(complex<double>) * buff_size * batch);

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
  int batch_sizeX = plan->N * (int)plan->shape[4];
  int batch_sizeY = plan->K * (int)plan->shape[0];
  int batch_sizeZ = (int)plan->shape[0] * plan->N*(plan->is_embed ? 2 : 1);
  int batch_sizeY_inv = (int)plan->shape[0] * plan->K*(plan->is_embed ? 2 : 1);
  int K0 = ceil_div(plan->K*(plan->is_embed ? 2 : 1), plan->r); 
  int batch_sizeX_inv = plan->N * (plan->is_embed ? 2 : 1) * K0;

  // BATCH2DPRDFTProblem b2prdstg1;
  // IBATCH2DPRDFTProblem ib2prdstg1;

  std::vector<int> size_stg1;  
  std::vector<int> size_stg2;  
  std::vector<int> size_stg3;  
  std::vector<int> size_istg2;  
  std::vector<int> size_istg1; 

  if(plan->is_complex) {
    if(plan->b == 1) {
      std::vector<int> size_stg1 = {inM, batch_sizeX, 0, 1};  
      std::vector<int> size_stg2 = {inN, batch_sizeY, 0, 1};  
      std::vector<int> size_stg3 = {inK, batch_sizeZ, 0, 0};  
      std::vector<int> size_istg2 = {inN, batch_sizeY_inv, 1, 0};  
      std::vector<int> size_istg1 = {inM, batch_sizeX_inv, 1, 0}; 
      bdstg1_1d.setSizes(size_stg1);
      bdstg2_1d.setSizes(size_stg2);
      bdstg3_1d.setSizes(size_stg3);
      ibdstg1_1d.setSizes(size_istg1);
      ibdstg2_1d.setSizes(size_istg2);
      bdstg1_1d.setName("b1dft");
      bdstg2_1d.setName("b1dft");
      bdstg3_1d.setName("b1dft");
      ibdstg1_1d.setName("ib1dft");
      ibdstg2_1d.setName("ib1dft");
    } else {
      std::vector<int> size_stg1 = {inM, plan->b, batch_sizeX, 0, 1};  
      std::vector<int> size_stg2 = {inN, plan->b, batch_sizeY, 0, 1};  
      std::vector<int> size_stg3 = {inK, plan->b, batch_sizeZ, 0, 0};  
      std::vector<int> size_istg2 = {inN, plan->b, batch_sizeY_inv, 1, 0};  
      std::vector<int> size_istg1 = {inM, plan->b, batch_sizeX_inv, 1, 0};  
      b2dstg1_1d.setSizes(size_stg1);
      b2dstg2_1d.setSizes(size_stg2);
      b2dstg3_1d.setSizes(size_stg3);
      ib2dstg1_1d.setSizes(size_istg1);
      ib2dstg2_1d.setSizes(size_istg2);
      b2dstg1_1d.setName("b2dft");
      b2dstg2_1d.setName("b2dft");
      b2dstg3_1d.setName("b2dft");
      ib2dstg1_1d.setName("ib2dft");
      ib2dstg2_1d.setName("ib2dft");
    }
  } else {
    if(plan->b == 1) {
      std::vector<int> size_stg1 = {inM, batch_sizeX, 0, 1};  
      std::vector<int> size_stg2 = {inN, batch_sizeY, 0, 1};  
      std::vector<int> size_stg3 = {inK, batch_sizeZ, 0, 0};  
      std::vector<int> size_istg2 = {inN, batch_sizeY_inv, 1, 0};  
      std::vector<int> size_istg1 = {inM, batch_sizeX_inv, 1, 0}; 
      bprdstg1_1d.setSizes(size_stg1);
      bdstg2_1d.setSizes(size_stg2);
      bdstg3_1d.setSizes(size_stg3);
      ibprdstg1_1d.setSizes(size_istg1);
      ibdstg2_1d.setSizes(size_istg2);
      bprdstg1_1d.setName("b1prdft");
      bdstg2_1d.setName("b1dft");
      bdstg3_1d.setName("b1dft");
      ibprdstg1_1d.setName("ib1prdft");
      ibdstg2_1d.setName("ib1dft");
    }
    // } else {
    //   std::vector<int> size_stg1 = {inM, plan->b, batch_sizeX, 0, 1};  
    //   std::vector<int> size_stg2 = {inN, plan->b, batch_sizeY, 0, 1};  
    //   std::vector<int> size_stg3 = {inK, plan->b, batch_sizeZ, 0, 0};  
    //   std::vector<int> size_istg2 = {inN, plan->b, batch_sizeY_inv, 1, 0};  
    //   std::vector<int> size_istg1 = {inM, plan->b, batch_sizeX_inv, 1, 0};  
    //   b2prdstg1.setSizes(size_stg1);
    //   b2dstg2.setSizes(size_stg2);
    //   b2dstg3.setSizes(size_stg3);
    //   ib2prdstg1.setSizes(size_istg1);
    //   ib2dstg2.setSizes(size_istg2);
    //   b2prdstg1.setName("b2prdft");
    //   b2dstg2.setName("b2dft");
    //   b2dstg3.setName("b2dft");
    //   ib2prdstg1.setName("ib2prdft");
    //   ib2dstg2.setName("ib2dft");
    // }
  }

  if (direction == DEVICE_FFT_FORWARD) {
    if (plan->is_complex) {
      // [X', Z/p, Y, b] <= [Z/p, Y, X, b]
      if(plan->b  == 1){
        #if defined FFTX_CUDA
        std::vector<void*> args{&plan->Q3, &in_buffer};
        #else 
        std::vector<void*> args{plan->Q3, in_buffer};
        #endif
        bdstg1_1d.setArgs(args);
        bdstg1_1d.transform();
      } else {
        #if defined FFTX_CUDA
          std::vector<void*> args{&plan->Q3, &in_buffer};
        #else 
          std::vector<void*> args{plan->Q3, in_buffer};
        #endif
        b2dstg1_1d.setArgs(args);
        b2dstg1_1d.transform();
      }

      // [X'/px, pz, b, Z/pz, Y] <= [px, X'/px, b, Z/pz, Y] // is this right? should batch be inner?
      fftx_mpi_rcperm_1d(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);
      if(plan->b == 1) {
        #if defined FFTX_CUDA
        std::vector<void*> args{&(plan->Q3), &(plan->Q4)};
        #else 
        std::vector<void*> args{plan->Q3, plan->Q4};
        #endif
        bdstg2_1d.setArgs(args);
        bdstg2_1d.transform();
      } else {
        #if defined FFTX_CUDA
          std::vector<void*> args{&plan->Q3, &plan->Q4};
        #else 
          std::vector<void*> args{plan->Q3, plan->Q4};
        #endif
        b2dstg2_1d.setArgs(args);
        b2dstg2_1d.transform();
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
        #if defined FFTX_CUDA
        std::vector<void*> args{&(out_buffer), &(stg3_input)};
        #else 
        std::vector<void*> args{out_buffer, stg3_input};
        #endif
        bdstg3_1d.setArgs(args);
        bdstg3_1d.transform();
      } else {
        #if defined FFTX_CUDA
          std::vector<void*> args{&out_buffer, &stg3_input};
        #else 
          std::vector<void*> args{out_buffer, stg3_input};
        #endif
        b2dstg3_1d.setArgs(args);
        b2dstg3_1d.transform();
      }
    } else {
      // [X', Z/p, Y, b] <= [Z/p, Y, X, b]
      if(plan->b  == 1){
        #if defined FFTX_CUDA
        std::vector<void*> args{&plan->Q3, &in_buffer};
        #else 
        std::vector<void*> args{plan->Q3, in_buffer};
        #endif
        bprdstg1_1d.setArgs(args);
        bprdstg1_1d.transform();
      }
      // } else {
      //   #if defined FFTX_CUDA
      //     std::vector<void*> args{&plan->Q3, &in_buffer};
      //   #else 
      //     std::vector<void*> args{plan->Q3, in_buffer};
      //   #endif
      //   b2prdstg1.setArgs(args);
      //   b2prdstg1.transform();
      // }

      // [X'/px, pz, b, Z/pz, Y] <= [px, X'/px, b, Z/pz, Y] // is this right? should batch be inner?
      fftx_mpi_rcperm_1d(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);
      if(plan->b == 1) {
        #if defined FFTX_CUDA
        std::vector<void*> args{&(plan->Q3), &(plan->Q4)};
        #else 
        std::vector<void*> args{plan->Q3, plan->Q4};
        #endif
        bdstg2_1d.setArgs(args);
        bdstg2_1d.transform();
      } 
      // else {
      //   #if defined FFTX_CUDA
      //     std::vector<void*> args{&plan->Q3, &plan->Q4};
      //   #else 
      //     std::vector<void*> args{plan->Q3, plan->Q4};
      //   #endif
      //   b2dstg2.setArgs(args);
      //   b2dstg2.transform();
      // }

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
        #if defined FFTX_CUDA
        std::vector<void*> args{&(out_buffer), &(stg3_input)};
        #else 
        std::vector<void*> args{out_buffer, stg3_input};
        #endif
        bdstg3_1d.setArgs(args);
        bdstg3_1d.transform();
      } 
      // else {
      //   #if defined FFTX_CUDA
      //     std::vector<void*> args{&out_buffer, &stg3_input};
      //   #else 
      //     std::vector<void*> args{out_buffer, stg3_input};
      //   #endif
      //   b2dstg3.setArgs(args);
      //   b2dstg3.transform();
      // }
    }
  } else if (direction == DEVICE_FFT_INVERSE) { // backward
    DEVICE_FFT_DOUBLECOMPLEX *stg3i_input  = (DEVICE_FFT_DOUBLECOMPLEX *) in_buffer;
    DEVICE_FFT_DOUBLECOMPLEX *stg3i_output = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q3;
    // [Y, X'/px, Z] <= [Y, X'/px, Z] (read seq, write seq)
    if(plan->b == 1) {
      #if defined FFTX_CUDA
      std::vector<void*> args{&stg3i_output, &stg3i_input};
      #else 
      std::vector<void*> args{stg3i_output, stg3i_input};
      #endif
      bdstg3_1d.setArgs(args);
      bdstg3_1d.transform();
    } else {
      #if defined FFTX_CUDA
      std::vector<void*> args{&stg3i_output, &stg3i_input};
      #else 
      std::vector<void*> args{stg3i_output, stg3i_input};
      #endif
      b2dstg3_1d.setArgs(args);
      b2dstg3_1d.transform();
    }
    // no permutation necessary, use previous output as input.
    DEVICE_FFT_DOUBLECOMPLEX *stg2i_input  = stg3i_output;
    DEVICE_FFT_DOUBLECOMPLEX *stg2i_output = (DEVICE_FFT_DOUBLECOMPLEX *) plan->Q4;
    // TODO: add code here if we expect embedded.

    //stage 2i
    // [X'/px, Z, Y] <= [Y, X'/px, Z] (read strided, write seq)
    if(plan->b == 1) {
      #if defined FFTX_CUDA
      std::vector<void*> args{&stg2i_output, &stg2i_input};
      #else 
      std::vector<void*> args{stg2i_output, stg2i_input};
      #endif
      ibdstg2_1d.setArgs(args);
      ibdstg2_1d.transform();
    } else {
      #if defined FFTX_CUDA
      std::vector<void*> args{&stg2i_output, &stg2i_input};
      #else 
      std::vector<void*> args{stg2i_output, stg2i_input};
      #endif
      ib2dstg2_1d.setArgs(args);
      ib2dstg2_1d.transform();
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
    if(plan->is_complex) {
      if(plan->b == 1) {
          #if defined FFTX_CUDA
          std::vector<void*> args{&stg1i_output,  &stg1i_input};
          #else 
          std::vector<void*> args{stg1i_output,  stg1i_input};
          #endif
          ibdstg1_1d.setArgs(args);
          ibdstg1_1d.transform();
      } else {
        #if defined FFTX_CUDA
        std::vector<void*> args{&stg1i_output,  &stg1i_input};
        #else 
        std::vector<void*> args{stg1i_output,  stg1i_input};
        #endif
        ib2dstg1_1d.setArgs(args);
        ib2dstg1_1d.transform();
      }
    } else {
      if(plan->b == 1) {
          #if defined FFTX_CUDA
          std::vector<void*> args{&stg1i_output,  &stg1i_input};
          #else 
          std::vector<void*> args{stg1i_output,  stg1i_input};
          #endif
          ibprdstg1_1d.setArgs(args);
          ibprdstg1_1d.transform();
      } 
      // else {
      //   #if defined FFTX_CUDA
      //   std::vector<void*> args{&stg1i_output,  &stg1i_input};
      //   #else 
      //   std::vector<void*> args{stg1i_output,  stg1i_input};
      //   #endif
      //   ib2prdstg1.setArgs(args);
      //   ib2prdstg1.transform();
      // }
    }
  } // end backward.
}
