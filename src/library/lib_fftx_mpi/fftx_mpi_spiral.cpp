#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>

#include "device_macros.h"
#include "fftx_gpu.h"
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

BATCH1DPRDFTProblem bprdstg1;
IBATCH1DPRDFTProblem ibprdstg1;


fftx_plan fftx_plan_distributed_spiral(int r, int c, int M, int N, int K, int batch, bool is_embedded, bool is_complex) {

  fftx_plan plan = (fftx_plan) malloc(sizeof(fftx_plan_t));

  plan->b = batch;
  plan->is_embed = is_embedded;
  plan->is_complex = is_complex;
  plan->M = M;
  plan->N = N;
  plan->K = K;

  init_2d_comms(plan, r, c,  M,  N, K);   //embedding uses the input sizes

  DEVICE_MALLOC(&(plan->Q3), M*N*K*(is_embedded ? 8 : 1) / (r * c) * sizeof(complex<double>) * batch);
  DEVICE_MALLOC(&(plan->Q4), M*N*K*(is_embedded ? 8 : 1) / (r * c) * sizeof(complex<double>) * batch);

  // int batch_sizeZ = M/r * N/c;
  int batch_sizeX = N/c * K/r;
  int batch_sizeY = K/r * M/c;

  // int inK = K * (is_embedded ? 2 : 1);
  // int inM = M * (is_embedded ? 2 : 1);
  // int inN = N * (is_embedded ? 2 : 1);


  // int outK = K * (is_embedded ? 2 : 1);
  // int outM = M * (is_embedded ? 2 : 1);
  // int outN = N * (is_embedded ? 2 : 1);


  batch_sizeX *= (is_embedded ? 2 : 1);
  batch_sizeY *= (is_embedded ? 4 : 1);

  return plan;
}

void fftx_execute_spiral(fftx_plan plan, double* out_buffer, double*in_buffer, int direction)
{
  int batch_sizeZ = plan->M/plan->r * plan->N/plan->c;
  int batch_sizeX = plan->N/plan->c * plan->K/plan->r;
  int batch_sizeY = plan->K/plan->r * plan->M/plan->c;

  int inK = plan->K * (plan->is_embed ? 2 : 1);
  int inM = plan->M * (plan->is_embed ? 2 : 1);
  int inN = plan->N * (plan->is_embed ? 2 : 1);

  batch_sizeX *= (plan->is_embed ? 2 : 1);
  batch_sizeY *= (plan->is_embed ? 4 : 1);

  // BATCH2DPRDFTProblem b2prdstg1;
  // IBATCH2DPRDFTProblem ib2prdstg1;

  std::vector<int> size_stg1; 
  std::vector<int> size_stg2;
  std::vector<int> size_stg3;  
  std::vector<int> size_istg1;
  std::vector<int> size_istg2;
  if(plan->is_complex) {
    if(plan->b == 1) {
      size_stg1 = {inK, batch_sizeZ, 0, 1}; 
      size_stg2 = {inM, batch_sizeX, 0, 1};
      size_stg3 = {inN, batch_sizeY, 0, 0};  
      size_istg1 = {inK, batch_sizeZ, 1, 0};
      size_istg2 = {inM, batch_sizeX, 1, 0}; 
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
      size_stg1 = {inK,  plan->b, batch_sizeZ, 0, 1}; 
      size_stg2 = {inM, plan->b, batch_sizeX, 0, 1};
      size_stg3 = {inN, plan->b, batch_sizeY, 0, 0};  
      size_istg1 = {inK, plan->b, batch_sizeZ, 1, 0};
      size_istg2 = {inM,plan->b, batch_sizeX, 1, 0}; 
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
  } else {
    if(plan->b == 1) {
      size_stg1 = {inK, batch_sizeZ, 0, 1}; 
      size_stg2 = {inM, batch_sizeX, 0, 1};
      size_stg3 = {inN, batch_sizeY, 0, 0};  
      size_istg1 = {inK, batch_sizeZ, 1, 0};
      size_istg2 = {inM, batch_sizeX, 1, 0}; 
      bprdstg1.setSizes(size_stg1);
      bdstg2.setSizes(size_stg2);
      bdstg3.setSizes(size_stg3);
      ibprdstg1.setSizes(size_istg1);
      ibdstg2.setSizes(size_istg2);
      bprdstg1.setName("b1dft");
      bdstg2.setName("b1dft");
      bdstg3.setName("b1dft");
      ibprdstg1.setName("ib1prdft");
      ibdstg2.setName("ib1dft");
    } 
    // else {
    //   size_stg1 = {inK,  plan->b, batch_sizeZ, 0, 1}; 
    //   size_stg2 = {inM, plan->b, batch_sizeX, 0, 1};
    //   size_stg3 = {inN, plan->b, batch_sizeY, 0, 0};  
    //   size_istg1 = {inK, plan->b, batch_sizeZ, 1, 0};
    //   size_istg2 = {inM,plan->b, batch_sizeX, 1, 0}; 
    //   b2prdstg1.setSizes(size_stg1);
    //   b2dstg2.setSizes(size_stg2);
    //   b2dstg3.setSizes(size_stg3);
    //   ib2prdstg1.setSizes(size_istg1);
    //   ib2dstg2.setSizes(size_istg2);
    //   b2prdstg1.setName("b2dft");
    //   b2dstg2.setName("b2dft");
    //   b2dstg3.setName("b2dft");
    //   ib2prdstg1.setName("ib2dft");
    //   ib2dstg2.setName("ib2dft");
    // }
  }
  if (direction == DEVICE_FFT_FORWARD) {
    if (plan->is_complex) {
      if(plan->b == 1) {
        #if defined FFTX_CUDA
        std::vector<void*> args{&plan->Q3, &in_buffer};
        #else
        std::vector<void*> args{plan->Q3, in_buffer};
        #endif
        bdstg1.setArgs(args);
        bdstg1.transform();
      } else{
        #if defined FFTX_CUDA
        std::vector<void*> args{&plan->Q3, &in_buffer};
        #else
        std::vector<void*> args{plan->Q3, in_buffer};
        #endif
        b2dstg1.setArgs(args);
        b2dstg1.transform();
      }
    } else {
      if(plan->b == 1) {
        #if defined FFTX_CUDA
        std::vector<void*> args{&plan->Q3, &in_buffer};
        #else
        std::vector<void*> args{plan->Q3, in_buffer};
        #endif
        bprdstg1.setArgs(args);
        bprdstg1.transform();
      } 
      // else{
      //   #if defined FFTX_CUDA
      //   std::vector<void*> args{&plan->Q3, &in_buffer};
      //   #else
      //   std::vector<void*> args{plan->Q3, in_buffer};
      //   #endif
      //   b2prdstg1.setArgs(args);
      //   b2prdstg1.transform();
      // }
    }

    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);
    
    if(plan->b == 1) {
      #if defined FFTX_CUDA
      std::vector<void*> args{&plan->Q3, &plan->Q4};
      #else
      std::vector<void*> args{plan->Q3, plan->Q4};
      #endif
      bdstg2.setArgs(args);
      bdstg2.transform();
    } else {
      #if defined FFTX_CUDA
      std::vector<void*> args{&plan->Q3, &plan->Q4};
      #else
      std::vector<void*> args{plan->Q3, plan->Q4};
      #endif
      b2dstg2.setArgs(args);
      b2dstg2.transform();
    }

    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_2, plan->is_embed);
    if(plan->b == 1) {
      #if defined FFTX_CUDA
      std::vector<void*> args{&out_buffer, &plan->Q4};
      #else
      std::vector<void*> args{out_buffer, plan->Q4};
      #endif
      bdstg3.setArgs(args);
      bdstg3.transform();
    } else {
      #if defined FFTX_CUDA
      std::vector<void*> args{&out_buffer, &plan->Q4};
      #else
      std::vector<void*> args{out_buffer, plan->Q4};
      #endif
      b2dstg3.setArgs(args);
      b2dstg3.transform();
    }
  } else if (direction == DEVICE_FFT_INVERSE) {
    if(plan->b == 1) {
      #if defined FFTX_CUDA
      std::vector<void*> args{&plan->Q3, &in_buffer};
      #else
      std::vector<void*> args{plan->Q3, in_buffer};
      #endif
      bdstg3.setArgs(args);
      bdstg3.transform();
    } else {
      #if defined FFTX_CUDA
      std::vector<void*> args{&plan->Q3, &in_buffer};
      #else
      std::vector<void*> args{plan->Q3, in_buffer};
      #endif
      b2dstg3.setArgs(args);
      b2dstg3.transform();
    }
    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_3, plan->is_embed);
    if(plan->b == 1) {
      #if defined FFTX_CUDA
      std::vector<void*> args{&plan->Q3, &plan->Q4};
      #else
      std::vector<void*> args{plan->Q3, plan->Q4};
      #endif
      ibdstg2.setArgs(args);
      ibdstg2.transform();
    } else {
      #if defined FFTX_CUDA
      std::vector<void*> args{&plan->Q3, &plan->Q4};
      #else
      std::vector<void*> args{plan->Q3, plan->Q4};
      #endif
      ib2dstg2.setArgs(args);
      ib2dstg2.transform();
    }
    fftx_mpi_rcperm(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_4, plan->is_embed);

    if (plan->is_complex) {
      if(plan->b == 1) {
        #if defined FFTX_CUDA
        std::vector<void*> args{&out_buffer, &plan->Q4};
        #else
        std::vector<void*> args{out_buffer, plan->Q4};
        #endif
        ibdstg1.setArgs(args);
        ibdstg1.transform();
      } else {
        #if defined FFTX_CUDA
        std::vector<void*> args{&out_buffer, &plan->Q4};
        #else
        std::vector<void*> args{out_buffer, plan->Q4};
        #endif
        ib2dstg1.setArgs(args);
        ib2dstg1.transform();
      }
    } else {
      if(plan->b == 1) {
        #if defined FFTX_CUDA
        std::vector<void*> args{&out_buffer, &plan->Q4};
        #else
        std::vector<void*> args{out_buffer, plan->Q4};
        #endif
        ibprdstg1.setArgs(args);
        ibprdstg1.transform();
      } 
      // else {
      //   #if defined FFTX_CUDA
      //   std::vector<void*> args{&out_buffer, &plan->Q4};
      //   #else
      //   std::vector<void*> args{out_buffer, plan->Q4};
      //   #endif
      //   ib2prdstg1.setArgs(args);
      //   ib2prdstg1.transform();
      // }
    }
  }
}

void fftx_plan_destroy_spiral(fftx_plan plan) {
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
