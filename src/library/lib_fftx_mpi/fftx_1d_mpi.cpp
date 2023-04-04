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


void init_1d_comms(fftx_plan plan, int pp, int M, int N, int K)
{

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
  plan->shape[0] = M;
  plan->shape[1] = 1;  
  plan->shape[2] = N;
  plan->shape[3] = 1;  
  plan->shape[4] = K/pp;
  plan->shape[5] = pp;
}

void destroy_1d_comms(fftx_plan plan)
{
  if (plan){
    
#if CUDA_AWARE_MPI
    DEVICE_FREE(plan->send_buffer);
    DEVICE_FREE(plan->recv_buffer);
#else
    free(plan->send_buffer);
    free(plan->recv_buffer);
#endif
  }
}


fftx_plan  fftx_plan_distributed_1d(int p, int M, int N, int K, int batch, bool is_embedded, bool is_complex)
{
  fftx_plan plan = (fftx_plan) malloc(sizeof(fftx_plan_t));
  plan->b = batch;
  plan->is_embed = is_embedded;
  plan->is_complex = is_complex;

  init_1d_comms(plan, p,  M,  N, K);   //embedding uses the input sizes

  int inK = K * (is_embedded ? 2 : 1);
  int inM = M * (is_embedded ? 2 : 1);
  int inN = N * (is_embedded ? 2 : 1);
  
  int outK = K * (is_embedded ? 2 : 1);
  int outM = M * (is_embedded ? 2 : 1);
  int outN = N * (is_embedded ? 2 : 1);


  DEVICE_MALLOC(&(plan->Q3), M*N*K*(is_embedded ? 8 : 1) / p * sizeof(complex<double>) * batch);
  DEVICE_MALLOC(&(plan->Q4), M*N*K*(is_embedded ? 8 : 1) / p * sizeof(complex<double>) * batch);

  int Mdim = M/p;
  if (!(plan->is_complex))
    {
      Mdim = (M/2+1)/p;      
      if ((M/2 + 1) % p) {
	Mdim += 1;
      }

      plan->shape[1]=Mdim;
    }
  
  int batch_sizeZ = plan->shape[1] * N;
  int batch_sizeX = N * K/p;
  int batch_sizeY = K * plan->shape[1];

  
  if (plan->is_complex)
    {
      //only correct if not embedded
      /*
      DEVICE_FFT_PLAN_MANY(&(plan->stg1), 2, sizes, 
			   sizes, plan->b, inM*inN*batch,
			   sizes, plan->b, inM*inN*batch,
			   CUFFT_Z2Z, inK/plan->r);

      DEVICE_FFT_PLAN_MANY(&(plan->stg2), 1, &inK,
			   &inM, inM*inN/p*batch, batch,
			   &inM, inM*inN/p*batch, batch, 
			   CUFFT_Z2Z, inM*inN/plan->r*batch);
      */
    }
  else
    {
      
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
      DEVICE_FFT_PLAN_MANY(
			   &(plan->stg2),  1,            &inN,
			   &inN,     plan->b,            inN*plan->b,	       
			   &inN,    batch_sizeY*plan->b, plan->b,
			   DEVICE_FFT_Z2Z, batch_sizeY
			   );
      
      // [Y, X'/px, Z] <= [Y, X'/px, Z] (read seq, write seq)
      DEVICE_FFT_PLAN_MANY(
			   &(plan->stg3), 1, &K,
			   &K, plan->b, K*plan->b,
			   &K, plan->b, K*plan->b,
			   DEVICE_FFT_Z2Z, batch_sizeZ
			   );
      
    }
  
  return plan;  
}


void fftx_mpi_rcperm_1d(fftx_plan plan, double * _Y, double *_X, int stage, bool is_embedded) {
  complex<double> *X = (complex<double> *) _X;
  complex<double> *Y = (complex<double> *) _Y;
  
  switch (stage)
    {
    case FFTX_MPI_EMBED_1 :
      {
      //#if CUDA_AWARE_MPI
      //      DEVICE_MEM_COPY(plan->send_buffer, X, buffer_size * sizeof(complex<double>) * plan->b, MEM_COPY_DEVICE_TO_DEVICE);
      //#else      
      DEVICE_MEM_COPY(plan->send_buffer, X,
		      plan->shape[0] * plan->shape[2] * plan->shape[4]* sizeof(complex<double>) * plan->b, MEM_COPY_DEVICE_TO_HOST);
      
      size_t sendSize = plan->shape[1] * plan->shape[2] * plan->shape[4];//N  * K/p;
      size_t recvSize = plan->shape[1] * plan->shape[2] * plan->shape[4];
      MPI_Alltoall(
		   plan->send_buffer,  sendSize * plan->b,
		   MPI_DOUBLE_COMPLEX,
		   plan->recv_buffer, recvSize * plan->b,
		   MPI_DOUBLE_COMPLEX,
		   MPI_COMM_WORLD
		   );

      DEVICE_MEM_COPY(X, plan->recv_buffer, 
		      plan->shape[0] * plan->shape[2] * plan->shape[4]* sizeof(complex<double>) * plan->b, MEM_COPY_HOST_TO_DEVICE);
            
      pack_embed(plan, (complex<double> *) Y, X, plan->shape[2] * plan->shape[4]* plan->b, plan->shape[1], plan->r, is_embedded);
      //#endif
    }
      break;
      
    case FFTX_MPI_EMBED_2 :
      {  
      // swap pointers.

	void *tmp = (void *) X;
	X = Y;
	Y = (complex<double> *) tmp;
      }
      break;
    }
}

void fftx_execute_1d(fftx_plan plan, double* out_buffer, double*in_buffer, int direction)
{
  if (direction == DEVICE_FFT_FORWARD) {

    if (plan->is_complex)
      {
      }
    else
      {
	//forward real	
	for (int i = 0; i != plan->b; ++i)
	  {	    
	  DEVICE_FFT_EXECD2Z(plan->stg1, ((DEVICE_FFT_DOUBLEREAL*)in_buffer + i),
			                 ((DEVICE_FFT_DOUBLECOMPLEX*)plan->Q3 + i));
	  }
      }

    
    fftx_mpi_rcperm_1d(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_1, plan->is_embed);
    
    for (int i = 0; i != plan->b; ++i) {
      DEVICE_FFT_EXECZ2Z(plan->stg2, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q4 + i),
			             ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i), direction);
    }
      
    fftx_mpi_rcperm_1d(plan, plan->Q4, plan->Q3, FFTX_MPI_EMBED_2, plan->is_embed);
    
    for (int i = 0; i != plan->b; ++i) {
      DEVICE_FFT_EXECZ2Z(plan->stg3, ((DEVICE_FFT_DOUBLECOMPLEX  *) plan->Q3 + i),
			             ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer + i), direction);
    }      
    
  }
  else{   // Backward
    
    //stage 2i
    for (int i = 0; i < plan->b; ++i){
      if (plan->is_complex)
	{
	  //backward complex	  
	}
      else
	{
	  //backward real
	}
    }

    //COMMS

    
    //stage 1i
    for (int i = 0; i < plan->b; ++i){
      if (plan->is_complex)
	{
	  //backward complex
	}
      else
	{
	  //backward real
	}
    }
  }
}
