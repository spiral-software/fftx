#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>
#include "gpu.h"
#include "util.h"

#define FFTX_MPI_EMBED_1 1
#define FFTX_MPI_EMBED_2 2

#define FFTX_CUDA 1
#include "../common/device_macros.h"


using namespace std;

static complex<double> *recv_buffer, *send_buffer;
static vector<int> shape;
static int r, c;
static MPI_Comm row_comm, col_comm;

void init_2d_comms(int rr, int cc, int M, int N, int K, bool is_embedded)
{
  // pass in the dft size. if embedded, double dims when necessary.
  r = rr; c = cc;
  size_t max_size = M*N*K*(is_embedded ? 8 : 1)/(r*c);

  send_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
  recv_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));

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
  
  free(send_buffer);
  free(recv_buffer);
}

void fftx_mpi_rcperm(double * _Y, double *_X, int stage, bool is_embedded)
{
  complex<double> *X = (complex<double> *) _X;
  complex<double> *Y = (complex<double> *) _Y;
  switch (stage) {
    case FFTX_MPI_EMBED_1:
      {
        // after first 1D FFT on K dim.
        shape[4] *= (is_embedded ? 2 : 1);
        size_t buffer_size = shape[2] * shape[4] * shape[0] * shape[1];
        int sendSize = shape[0] * shape[2] * shape[4];
        int recvSize = sendSize;
        DEVICE_MEM_COPY(send_buffer, X, buffer_size * sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);
        // layout is ml, nl, 2(kl, kr)
        MPI_Alltoall(
          send_buffer, sendSize,
          MPI_DOUBLE_COMPLEX,
          recv_buffer, recvSize,
          MPI_DOUBLE_COMPLEX,
          row_comm
        ); // assume M dim is initially distributed along col comm.
        int jr, kr;
        MPI_Comm_rank(col_comm, &jr);
        MPI_Comm_rank(row_comm, &kr);

        // [xl, yl, zl, xr] -> [yl, zl, xl, xr]
        for (int ir = 0; ir < shape[1]; ir++) {
          for (int kl = 0; kl < shape[4]; kl++) {
            for (int jl = 0; jl < shape[2]; jl++) {
              for (int il = 0; il < shape[0]; il++) {
                complex<double> tmp =
                send_buffer[
                  ir * shape[0]*shape[4]*shape[2] + 
                  il *          shape[4]*shape[2] + 
                  kl *                   shape[2] +
                  jl                  
                ] = recv_buffer[
                  ir * shape[4]*shape[2]*shape[0] + 
                  kl *          shape[2]*shape[0] +
                  jl *                   shape[0] +
                  il
                ];
                // printf(
                //   "[%3d,%3d,%3d]: %f+%fi\n", 
                //   ir * shape[0] + il,
                //   jr * shape[2] + jl,
                //   kr * shape[4] + kl,
                //   tmp.real(), tmp.imag()
                // );

              }
            }
          }
        }
        if (is_embedded) {
          // TODO: device_macros.h doesn't have memset
          DEVICE_MEM_SET(Y + 0*buffer_size/2, 0, buffer_size/2 * sizeof(complex<double>));
          DEVICE_MEM_SET(Y + 3*buffer_size/2, 0, buffer_size/2 * sizeof(complex<double>));
          Y += 1*buffer_size/2; 
        }
        DEVICE_MEM_COPY(Y, send_buffer, buffer_size * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);

        // DEVICE_MEM_COPY(X, recv_buffer, buffer_size * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);
        if (false) {
          // layout is now      ml, (nl, 2kl), mr
          // want output to be  (nl, 2kl), ml, mr
          // size_t buffer_size = shape[2] * 2*shape[4] * shape[0] * shape[1];
          // DEVICE_MEM_SET(Y+(0*buffer_size/2), 0, buffer_size/2 * sizeof(complex<double>)); // embed packed buffer in twice as many zeros.
          // DEVICE_MEM_SET(Y+(3*buffer_size/2), 0, buffer_size/2 * sizeof(complex<double>));

          size_t batch   = shape[1];

          size_t cp_size = 1;               

          size_t a_dim   = shape[0];    
          size_t b_dim   = shape[2] * shape[4];               
          // size_t b_dim   = shape[2] * 2*shape[4];               

          size_t a_i_stride =     1 * cp_size;
          size_t a_o_stride = b_dim * cp_size;

          size_t b_i_stride = a_dim * cp_size;
          size_t b_o_stride =     1 * cp_size;
          DEVICE_ERROR_T err = pack(
            (complex<double>*) Y,
            // (complex<double>*) Y + buffer_size/2,
            (complex<double>*) X,
            a_dim, a_i_stride, a_o_stride,
            b_dim, b_i_stride, b_o_stride,
            cp_size,
            batch
          );
          if (err != DEVICE_SUCCESS) {
            fprintf(stderr, "pack failed Y <- St2_Comm!\n");
            exit(-1);
          }
        }
      }
      break;

    case FFTX_MPI_EMBED_2:
      {
        shape[0] *= (is_embedded ? 2 : 1);
        size_t buffer_size = shape[4] * shape[0] * shape[2] * shape[3];
        int sendSize = shape[2] * shape[4] * shape[0];
        int recvSize = sendSize;
        DEVICE_MEM_COPY(send_buffer, X, buffer_size * sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);
        // layout is nl, 2kl, 2(ml, mr)
        MPI_Alltoall(
          send_buffer, sendSize,
          MPI_DOUBLE_COMPLEX,
          recv_buffer, recvSize,
          MPI_DOUBLE_COMPLEX,
          col_comm
        ); // assume N dim is initially distributed along row comm.
        int ir, kr;
        MPI_Comm_rank(col_comm, &ir);
        MPI_Comm_rank(row_comm, &kr);
        // [yl, zl, xl, yr] -> [zl, xl, yl, yr]
        for (int jr = 0; jr < shape[3]; jr++) {
          for (int il = 0; il < shape[0]; il++) {
            for (int kl = 0; kl < shape[4]; kl++) {
              for (int jl = 0; jl < shape[2]; jl++) {
                complex<double> tmp = 
                send_buffer[
                  jr * shape[2]*shape[0]*shape[4] +
                  jl *          shape[0]*shape[4] +
                  il *                   shape[4] +
                  kl
                ] = recv_buffer[
                  jr * shape[0]*shape[4]*shape[2] +
                  il *          shape[4]*shape[2] +
                  kl *                   shape[2] +
                  jl
                ];
                // printf(
                //   "[%3d,%3d,%3d]: %f+%fi\n", 
                //   ir * shape[0] + il,
                //   jr * shape[2] + jl,
                //   kr * shape[4] + kl,
                //   tmp.real(), tmp.imag()
                // );
              }
            }
          }
        }
        if (is_embedded) {
          DEVICE_MEM_SET(Y + 0*buffer_size/2, 0, buffer_size/2 * sizeof(complex<double>));
          DEVICE_MEM_SET(Y + 3*buffer_size/2, 0, buffer_size/2 * sizeof(complex<double>));
          Y += 1*buffer_size/2; 
        }
        DEVICE_MEM_COPY(Y, send_buffer, buffer_size * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);
        // DEVICE_MEM_COPY(X, recv_buffer, buffer_size * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);
        if (false) {
          // layout is now      nl, (2kl, 2ml), nr
          // want output to be  (2kl, 2ml), nl, nr
          // size_t buffer_size = 2*shape[4] * 2*shape[0] * shape[2] * shape[3];
          // DEVICE_MEM_SET(Y+(0*buffer_size/2), 0, buffer_size/2 * sizeof(complex<double>)); // embed packed buffer in twice as many zeros.
          // DEVICE_MEM_SET(Y+(3*buffer_size/2), 0, buffer_size/2 * sizeof(complex<double>));

          size_t batch   = shape[3];
          size_t cp_size = 1;

          size_t a_dim   = shape[2];
          size_t b_dim   = shape[4] * shape[0];
          // size_t b_dim   = 2*shape[4] * 2*shape[0];

          size_t a_i_stride =     1 * cp_size;
          size_t a_o_stride = b_dim * cp_size;

          size_t b_i_stride = a_dim * cp_size;
          size_t b_o_stride =     1 * cp_size;

          DEVICE_ERROR_T err = pack(
            (complex<double>*) Y,
            // (complex<double>*) Y + buffer_size/2,
            (complex<double>*) X,
            a_dim, a_i_stride, a_o_stride,
            b_dim, b_i_stride, b_o_stride,
            cp_size,
            batch
          );
          if (err != DEVICE_SUCCESS) {
            fprintf(stderr, "pack failed Y <- St1_Comm!\n");
            exit(-1);
          }
        }
      } // end FFTX_MPI_EMBED_2
      break;
    default:
      break;
  }
}

