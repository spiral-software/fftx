#include <complex>
#include "gpu.h"
#include "util.h"

using namespace std;

complex<double> *recv_buffer, *send_buffer;
vector<int> shape;

MPI_Comm row_comm, col_comm;

void init_2d_comms(int p, int M, int N, int K)
{
  size_t max_size = M*N*K/p;

  cudaError_t err = cudaMalloc(&send_buffer, max_size * sizeof(complex<double>));
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

  //assumed power of 2 grid sizes, and square 
  int r = 0; int c = 0;
  int bitIdx = __builtin_ctzll(p);
  r = (p >> bitIdx/2);
  c = p/r;
   
  //  ObjShape gridShape(2);
  //  gridShape[0] = r;
  //  gridShape[1] = c;

  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  
  int row_color = commRank / r;
  int col_color = commRank % r;

  MPI_Comm_split(MPI_COMM_WORLD, row_color, world_rank, &row_comm);
  MPI_Comm_split(MPI_COMM_WORLD, col_color, world_rank, &col_comm);  
    
  shape.resize(6);
  shape[0] = M/_g->Dimension(0);
  shape[1] = _g->Dimension(0);
  shape[2] = N/_g->Dimension(0);
  shape[3] = _g->Dimension(0);
  shape[4] = K/_g->Dimension(1);
  shape[5] = _g->Dimension(1);
  
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

    DistTensor< complex<double> > St1_Output      ("[()()()(0)()(1)]", *_g);
    DistTensor< complex<double> > St1_Comm        ("[()(0)()()()(1)]", *_g);
    {
      ObjShape localStrides(6), alignments(6);
      computeLocalStrides(localStrides, shape, St1_Output.GridViewShape(), 6);
      computeAlignments(alignments, 6);
      St1_Output.Attach(shape, alignments, send_buffer, localStrides, *_g);
    }
    {
      ObjShape localStrides(6), alignments(6);
      computeLocalStrides(localStrides, shape, St1_Comm.GridViewShape(), 6);
      computeAlignments(alignments, 6);
      St1_Comm.Attach(shape, alignments, recv_buffer, localStrides, *_g);
    }

    St1_Output.auxMemory_.Require(prod(St1_Output.LocalShape()) + prod(St1_Comm.LocalShape()));

    {  //pre communicate packing
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

    St1_Comm.AllToAllRedistFrom_Simple(St1_Output, s0);
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
    DistTensor< complex<double> > St2_Output      ("[()(0)()()()(1)]", *_g);
    DistTensor< complex<double> > St2_Comm        ("[()(0)()(1)()()]", *_g);

    {
      ObjShape localStrides(6), alignments(6);
      computeLocalStrides(localStrides, shape, St2_Output.GridViewShape(), 6);
      computeAlignments(alignments, 6);
      St2_Output.Attach(shape, alignments, send_buffer, localStrides, *_g);
    }
    {
      ObjShape localStrides(6), alignments(6);
      computeLocalStrides(localStrides, shape, St2_Comm.GridViewShape(), 6);
      computeAlignments(alignments, 6);
      St2_Comm.Attach(shape, alignments, recv_buffer, localStrides, *_g);
    }

    St2_Output.auxMemory_.Require(prod(St2_Output.LocalShape()) + prod(St2_Comm.LocalShape()));

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

    St2_Comm.AllToAllRedistFrom_Simple(St2_Output, s0);
    {
      size_t batch   = 1;

      size_t cp_size = shape[4];               //(8/2)

      size_t a_dim   = shape[0] * shape[2];    //(8/4) * (8/4)
      size_t b_dim   = shape[5];               //2

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
