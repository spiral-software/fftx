#include <complex>
#include <vector>
#include <mpi.h>
#include <iostream>
#include "gpu.h"
#include "util.h"

using namespace std;

complex<double> *recv_buffer, *send_buffer;
vector<int> shape;
int r, c;
MPI_Comm row_comm, col_comm;

void init_2d_comms(int rr, int cc, int M, int N, int K)
{
  r = rr; c= cc;
  size_t max_size = M*N*K/(r*c);

  cudaError_t err = cudaMalloc(&send_buffer,
			       max_size * sizeof(complex<double>));
  if (err != cudaSuccess) {
    cout << "Failed to create send buffer\n" << endl;
    exit(-1);
  }
  err = cudaMalloc(&recv_buffer, max_size * sizeof(complex<double>));
  if (err != cudaSuccess) {
    cout << "Failed to create recv buffer\n"<<endl;
    exit(-1);
  }
  cudaDeviceSynchronize();

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  int row_color = world_rank / r;
  int col_color = world_rank % r;

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

void destroy_2d_comms()
{

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);      
  
  cudaFree(recv_buffer);
  cudaFree(send_buffer);
}

void fftx_mpi_rcperm(double* Y, double *X, int sizes, int stage, int dim, int M, int N, int K)
{
  if (stage == 2) {

#ifdef NOTCUFFT    
    {
      //pre communicate packing
        size_t cp_size = 1;
        size_t a_dim   = shape[1] * shape[0];
        size_t b_dim   = shape[2] * shape[4];

        size_t a_i_stride =     1 * cp_size;
        size_t a_o_stride = b_dim * cp_size;

        size_t b_i_stride = a_dim * cp_size;
        size_t b_o_stride =     1 * cp_size;

        cudaError_t err = pack(
          send_buffer,
          (complex<double>*)X,
          a_dim, a_i_stride, a_o_stride,
          b_dim, b_i_stride, b_o_stride,
          cp_size,
          1
        );
        if (err != cudaSuccess) {
          fprintf(stderr, "pack failed St1_Output <- X!\n");
          exit(-1);
        }
    }
#endif  

    int sendSize = (shape[0]*shape[1]/r) * shape[2] * shape[4];
    int recvSize = (shape[0]*shape[1]/r) * shape[2] * shape[4];    
    
    //    St1_Comm.AllToAllRedistFrom_Simple(St1_Output, s0);
    
    MPI_Alltoall(send_buffer, sendSize,
		 MPI_DOUBLE_COMPLEX , recv_buffer, recvSize,
		 MPI_DOUBLE_COMPLEX , row_comm);
    
    {
        size_t cp_size = shape[2];
        size_t a_dim   = shape[4] * shape[0];
        size_t b_dim   = shape[3]           ;

        size_t a_i_stride =     1 * cp_size;
        size_t a_o_stride = b_dim * cp_size;

        size_t b_i_stride = a_dim * cp_size;
        size_t b_o_stride =     1 * cp_size;

        cudaError_t err = pack(
          (complex<double>*) Y,
          recv_buffer,
          a_dim, a_i_stride, a_o_stride,
          b_dim, b_i_stride, b_o_stride,
          cp_size,
          1
        );
        if (err != cudaSuccess) {
          fprintf(stderr, "pack failed Y <- St1_Comm!\n");
          exit(-1);
        }
      }
  } // end if stage 2.
  if (stage == 1) {

#ifdef NOTCUFFT    
    {  //pre communicate packing
      size_t cp_size = 1;
      size_t a_dim   = shape[2] * shape[3];
      size_t b_dim   = shape[4] * shape[0];

      size_t a_i_stride =     1 * cp_size;
      size_t a_o_stride = b_dim * cp_size;

      size_t b_i_stride = a_dim * cp_size;
      size_t b_o_stride =     1 * cp_size;

      cudaError_t err = pack(
        send_buffer,
        (complex<double>*)X,
        a_dim, a_i_stride, a_o_stride,
        b_dim, b_i_stride, b_o_stride,
        cp_size,
        1
      );
      if (err != cudaSuccess) {
        fprintf(stderr, "pack failed St2_Output <- X!\n");
        exit(-1);
      }
    }
#endif
    
    int sendSize = (shape[2]*shape[3])/r * shape[0] * shape[4];
    int recvSize = (shape[2]*shape[3])/r * shape[0] * shape[4];
      
    
    //    St2_Comm.AllToAllRedistFrom_Simple(St2_Output, s0);
    MPI_Alltoall(send_buffer, sendSize,
		 MPI_DOUBLE_COMPLEX , recv_buffer, recvSize,
		 MPI_DOUBLE_COMPLEX , col_comm);
      
    {
      size_t batch   = 1;

      size_t cp_size = shape[4];               

      size_t a_dim   = shape[0] * shape[2];    
      size_t b_dim   = shape[5];               

      size_t a_i_stride =     1 * cp_size;
      size_t a_o_stride = b_dim * cp_size;

      size_t b_i_stride = a_dim * cp_size;
      size_t b_o_stride =     1 * cp_size;

      cudaError_t err = pack(
        (complex<double>*)Y,
        recv_buffer,
        a_dim, a_i_stride, a_o_stride,
        b_dim, b_i_stride, b_o_stride,
        cp_size,
        2
      );

      if (err != cudaSuccess) {
        fprintf(stderr, "pack failed Y <- St2_Comm!\n");
        exit(-1);
      }
    }
  } // end if stage 1.

}
