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

void init_2d_comms(int rr, int cc, int M, int N, int K, bool is_embedded) {
  // pass in the dft size. if embedded, double dims when necessary.
  r = rr; c = cc; 
  size_t max_size = M*N*K*(is_embedded ? 8 : 1)/(r*c) * b;
  
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

fftx_plan fftx_plan_distributed(int r, int c, int M, int N, int K, int batch, bool is_embedded)
{
  is_embed = is_embedded;
  b = batch;

  fftx_plan plan = (fftx_plan) malloc(sizeof(fftx_plan_t));
  
  init_2d_comms(r, c,  M,  N, K, is_embed);   //embedding uses the input sizes
  
  DEVICE_MALLOC(&Q3, M*N*K*(is_embed ? 8 : 1) / (r * c) * sizeof(complex<double>) * batch);
  DEVICE_MALLOC(&Q4, M*N*K*(is_embed ? 8 : 1) / (r * c) * sizeof(complex<double>) * batch);

  int batch_sizeZ = M/r * N/c;
  int batch_sizeX = K/r * N/c;
  int batch_sizeY = K/r * M/c;

  int inK = K * (is_embed ? 2 : 1);
  int inM = M * (is_embed ? 2 : 1);
  int inN = N * (is_embed ? 2 : 1);

  batch_sizeX *= (is_embed ? 2: 1);
  batch_sizeY *= (is_embed ? 4: 1);  

  //read seq write strided
  DEVICE_FFT_PLAN_MANY(&(plan->stg1), 1, &inK,
		       &inK,             b, inK*b,
		       &inK, batch_sizeZ*b, b,
		       DEVICE_FFT_Z2Z, batch_sizeZ);

  //read seq write strided  
  DEVICE_FFT_PLAN_MANY(&(plan->stg2), 1, &inM,
		       &inM,           b, inM*b,
		       &inM, batch_sizeX*b, b,
		       DEVICE_FFT_Z2Z, batch_sizeX);

  //read seq write seq
  DEVICE_FFT_PLAN_MANY(&(plan->stg3), 1, &inN,
		       &inN, b, inN*b,
		       &inN, b, inN*b,
		       DEVICE_FFT_Z2Z, batch_sizeY);

  //read strided write seq
  DEVICE_FFT_PLAN_MANY(&(plan59i), 1, &inM,
		       &inM, batch_sizeX*b, b,
		       &inM,           b, inM*b,		       
		       DEVICE_FFT_Z2Z, batch_sizeX);

  //read strided write seq  
  DEVICE_FFT_PLAN_MANY(&(plan60i), 1, &inK,
		       &inK, batch_sizeZ*b, b,
		       &inK,             b, inK*b,		       
		       DEVICE_FFT_Z2Z, batch_sizeZ);

  return plan;
}

void fftx_execute(fftx_plan plan, double* out_buffer, double*in_buffer,int direction)
{
  if (direction == DEVICE_FFT_FORWARD){
  
  for (int i = 0; i != b; ++i)
    {
      DEVICE_FFT_EXECZ2Z(plan->stg1, ((DEVICE_FFT_DOUBLECOMPLEX  *) in_buffer + i),
			             ((DEVICE_FFT_DOUBLECOMPLEX  *) Q3 + i), direction);
    }
  
  fftx_mpi_rcperm(Q4, Q3, FFTX_MPI_EMBED_1, is_embed);
  
  for (int i = 0; i != b; ++i)
    {
      DEVICE_FFT_EXECZ2Z(plan->stg2, ((DEVICE_FFT_DOUBLECOMPLEX  *) Q4 + i),
			             ((DEVICE_FFT_DOUBLECOMPLEX  *) Q3 + i), direction);
    }
  
  fftx_mpi_rcperm(Q4, Q3, FFTX_MPI_EMBED_2, is_embed);
  
  for (int i = 0; i != b; ++i)
    {
      DEVICE_FFT_EXECZ2Z(plan->stg3, ((DEVICE_FFT_DOUBLECOMPLEX  *) Q4 + i),
			             ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer + i), direction);
    }  
  }
  
  if (direction == DEVICE_FFT_INVERSE)
    {
      
      for (int i = 0; i != b; ++i)
	{
	  DEVICE_FFT_EXECZ2Z(plan58, ((DEVICE_FFT_DOUBLECOMPLEX  *) in_buffer + i),
			             ((DEVICE_FFT_DOUBLECOMPLEX  *) Q3 + i), direction);
	}      
                
      fftx_mpi_rcperm(Q4, Q3, FFTX_MPI_EMBED_3, is_embed);
      
      
      for (int i = 0; i != b; ++i)
	{
	  DEVICE_FFT_EXECZ2Z(plan59i, ((DEVICE_FFT_DOUBLECOMPLEX  *) Q4 + i),
			             ((DEVICE_FFT_DOUBLECOMPLEX  *) Q3 + i), direction);
	}
      
      fftx_mpi_rcperm(Q4, Q3, FFTX_MPI_EMBED_4, is_embed);
      
      
      for (int i = 0; i != b; ++i)
	{
	  DEVICE_FFT_EXECZ2Z(plan60i, ((DEVICE_FFT_DOUBLECOMPLEX  *) Q4 + i),
		 	             ((DEVICE_FFT_DOUBLECOMPLEX  *) out_buffer + i), direction);
	}      
    }      
}

void fftx_plan_destroy(fftx_plan plan)
{
  destroy_2d_comms();
  
  DEVICE_FREE(Q3);
  DEVICE_FREE(Q4);

  if (plan != NULL)
    {      
      free(plan);
    }
}

// perm: [a, b, c] -> [a, 2c, b]
void pack_embed(complex<double> *dst, complex<double> *src, int a, int b, int c, int batch, bool is_embedded) {
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
	  for (int i = 0; i < batch; ++i){
	    send_buffer[(ib * c*a + ic * a + ia)*batch + i] = recv_buffer[(ic * b*a + ib * a + ia)*batch + i];
	  }
        }
      }
    }
  }
  DEVICE_MEM_COPY(dst, send_buffer, buffer_size * sizeof(complex<double>) * batch, MEM_COPY_HOST_TO_DEVICE);
#else
  //this part of the code does unpacking on the GPU
#if (!CUDA_AWARE_MPI)  //this copies data to the GPU to perform packing
  DEVICE_MEM_COPY(src, recv_buffer, buffer_size * sizeof(complex<double>) * batch, MEM_COPY_HOST_TO_DEVICE);
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
      a, batch
    );
  }
  if (err != DEVICE_SUCCESS) {
    fprintf(stderr, "pack failed Y <- St1_Comm!\n");
    exit(-1);
  }

#endif
}

// perm: [a, b, c] -> [a, 2c, b]
void unpack_embed(complex<double> *dst, complex<double> *src, int a, int b, int c, int batch, bool is_embedded) {
  //returns
  
  size_t buffer_size = a * b * c * (is_embedded ? 2 : 1); // assume embedded
  
#if CPU_PERMUTE
  //copy data to recv buffer on host in order to unpack into the send_buffer
  DEVICE_MEM_COPY(recv_buffer, dst, buffer_size * sizeof(complex<double>) * batch, MEM_COPY_DEVICE_TO_HOST);
  
  //the CPU code needs to be updated. It is currently the packing code.
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
	  for (int i = 0; i < batch; ++i){
	    send_buffer[(ib * c*a + ic * a + ia)*batch + i] = recv_buffer[(ic * b*a + ib * a + ia)*batch + i];
	  }
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
      a, batch
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
        DEVICE_MEM_COPY(send_buffer, X, buffer_size * sizeof(complex<double>) * b, MEM_COPY_DEVICE_TO_HOST);
#endif
        // [yl, zl, xl, xr] -> [yl, zl, xl, yr]
        MPI_Alltoall(
          send_buffer, sendSize*b,
          MPI_DOUBLE_COMPLEX,
          recv_buffer, recvSize*b,
          MPI_DOUBLE_COMPLEX,
          row_comm
        ); // assume N dim is initially distributed along col comm.

        // [yl, (zl, xl), yr] -> [yl, yr, (zl, xl)]
        pack_embed(Y, X, shape[2], shape[4] * shape[0], shape[3], b, is_embedded);
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
        DEVICE_MEM_COPY(send_buffer, X, buffer_size * sizeof(complex<double>) * b, MEM_COPY_DEVICE_TO_HOST);
#endif
        // [zl, xl, yl, yr] -> [zl, xl, yl, zr]
        MPI_Alltoall(
	  send_buffer, sendSize*b,
          MPI_DOUBLE_COMPLEX,
          recv_buffer, recvSize*b,
          MPI_DOUBLE_COMPLEX,
          col_comm
        ); // assume K dim is initially distributed along row comm.

        // [zl, (xl, yl), zr] -> [zl, zr, (xl, yl)]
        pack_embed(Y, X, shape[4], shape[0] * shape[2], shape[5], b, is_embedded);
      } // end FFTX_MPI_EMBED_2
      break;
      
    case FFTX_MPI_EMBED_3:
      {
        shape[2] *= (is_embedded ? 2 : 1);
        size_t buffer_size = shape[4] * shape[0] * shape[2] * shape[3];
        int sendSize = shape[4] * shape[0] * shape[2];
        int recvSize = sendSize;
	
        // [zl, (xl, yl), zr] <- [zl, zr, (xl, yl)]
        unpack_embed(Y, X, shape[4], shape[0] * shape[2], shape[5], b, is_embedded);
		
#if CUDA_AWARE_MPI
	//is this correct?
        send_buffer = Y;
        Y = recv_buffer;
	//#endif		  
#else	//#if (!CUDA_AWARE_MPI)
	DEVICE_MEM_COPY(send_buffer, Y, buffer_size * sizeof(complex<double>) * b, MEM_COPY_DEVICE_TO_HOST);
#endif

        // [zl, xl, yl, yr] <- [zl, xl, yl, zr]	
        MPI_Alltoall(
	  send_buffer, sendSize*b,
          MPI_DOUBLE_COMPLEX,
	  recv_buffer, recvSize*b,
          MPI_DOUBLE_COMPLEX,
          col_comm
        ); // assume K dim is initially distributed along row comm.
	
#if (!CUDA_AWARE_MPI)
	DEVICE_MEM_COPY(Y, recv_buffer, buffer_size * sizeof(complex<double>) * b, MEM_COPY_HOST_TO_DEVICE);
#endif
	
      } // end FFTX_MPI_EMBED_3
      break;
      
    case FFTX_MPI_EMBED_4:
      {
        // after first 1D FFT on K dim.
        shape[0] *= (is_embedded ? 2 : 1);
        size_t buffer_size = shape[2] * shape[4] * shape[0] * shape[1];
        int sendSize = shape[2] * shape[4] * shape[0];
        int recvSize = sendSize;

	// [yl, (zl, xl), yr] <- [yl, yr, (zl, xl)]
	unpack_embed(Y, X, shape[2], shape[4] * shape[0], shape[3], b, is_embedded);
	
#if CUDA_AWARE_MPI
        send_buffer = Y;
        Y = recv_buffer;
#else
        DEVICE_MEM_COPY(send_buffer, Y, buffer_size * sizeof(complex<double>) * b, MEM_COPY_DEVICE_TO_HOST);
#endif
        // [yl, zl, xl, xr] -> [yl, zl, xl, yr]
        MPI_Alltoall(
          send_buffer, sendSize*b,
          MPI_DOUBLE_COMPLEX,
          recv_buffer, recvSize*b,
          MPI_DOUBLE_COMPLEX,
          row_comm
        ); // assume N dim is initially distributed along col comm.
	
#if (!CUDA_AWARE_MPI)
	DEVICE_MEM_COPY(Y, recv_buffer, buffer_size * sizeof(complex<double>) * b, MEM_COPY_HOST_TO_DEVICE);
#endif	      	
      } // end FFTX_MPI_EMBED_4
      break;

  default:
      break;
  }
}

