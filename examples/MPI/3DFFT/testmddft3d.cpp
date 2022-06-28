#include <mpi.h>
#include <limits.h>
#include <complex>
#include <float.h>
#include <unistd.h>
#include <string>
#include "gpu.h"
#include "util.h"
#include "mpi.h"

#include <stdlib.h>     /* srand, rand */

#include "include/fftx_mpi_int.h"

#define FFTX_CUDA 1
#include "../common/device_macros.h"

#define CHECK_WITH_CUFFT 1
#if CHECK_WITH_CUFFT
// #include ""
#endif

// dims are fastest to slowest
//        [Local         ] [Distri]
// Init:  [xl, yl, zl, zr] [xr, yr]
// A2A:   [xl, yl, zl, xr] [zr, yr]
// Perm:  [yl, zl, xl, xr] [zr, yr]
// A2A:   [yl, zl, xl, yr] [zr, xr]
// Perm:  [zl, xl, yl, yr] [zr, xr]
// Gather:
//  In:   [xl, yl, zl, zr, xr, yr]
//  Out:  [zl, xl, yl, yr, zr, xr]

using namespace std;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int commRank;
  int p;
  
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  const int root = 0;

  // 3d fft sizes
  int M = MM;
  int N = NN;
  int K = KK;

  bool is_embedded = true; // TODO: get embedded testing working
  int r = PP;
  int c = PP;

  int Mi = M;
  int Ni = N;
  int Ki = K;
  int Mo = M * (is_embedded ? 2 : 1);
  int No = N * (is_embedded ? 2 : 1);
  int Ko = K * (is_embedded ? 2 : 1);

  int check = 1;

  complex<double> *in_buffer = NULL;
  DEVICE_ERROR_T err = DEVICE_MALLOC(&in_buffer, Mi*Ni*Ko/p * sizeof(complex<double>));
  if (err != DEVICE_SUCCESS) {
    cout << "DEVICE_MALLOC failed\n" << endl;
    exit(-1);
  }
  complex<double> *out_buffer = NULL;
  err = DEVICE_MALLOC(&out_buffer, Mo*No*Ko/p * sizeof(complex<double>));
  if (err != DEVICE_SUCCESS) {
    cout << "DEVICE_MALLOC failed\n" << endl;
    exit(-1);
  }

  // initialize data to random values in the range (-1, 1).
  complex<double> *in_buff = new complex<double>[Mi*Ni*Ko/p];
  
  for (int k = 0; k < Ko; k++) {
    for (int j = 0; j < Ni/r; j++) {
      for (int i = 0; i < Mi/c; i++) {
        in_buff[
          k * (Ni/r)*(Mi/c) +
          j *        (Mi/c) +
          i
        ] = ( // embedded
          // Mi/4 <= i && i < 3*Mi/4 &&
          // Ni/4 <= j && j < 3*Ni/4 &&
          Ki/2 <= k && k < 3*Ki/2 &&
          true
        ) ?
          complex<double>(
            1 - ((double) rand()) / (double) (RAND_MAX/2),
            1 - ((double) rand()) / (double) (RAND_MAX/2)
            // 1,
            // 0
          )
        :
          complex<double>(0, 0);
      }
    }
  }

  err = DEVICE_MEM_COPY(
    in_buffer,
    in_buff,
	  Mi*Ni*Ko/p * sizeof(complex<double>),
    MEM_COPY_HOST_TO_DEVICE
  );
  if (err != DEVICE_SUCCESS) {
    cout << "DEVICE_MEM_COPY failed\n" << endl;
    exit(-1);
  }

  DEVICE_SYNCHRONIZE();

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  INIT_FN_NAME();

  __FILEROOT__((double*)out_buffer, (double*)in_buffer);

  DESTROY_FN_NAME();

  double end_time = MPI_Wtime();

  double min_time    = min_diff(start_time, end_time, MPI_COMM_WORLD);
  double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);

  if (commRank == 0) {
    printf("%lf %lf\n", min_time, max_time);
  }
  

  // Check correctness against local compute.
#if CHECK_WITH_CUFFT
  {
    // can rewrite this bit..
    // int M = shape[0] * shape[1];
    // int N = shape[2] * shape[3];
    // int K = shape[4] * shape[5];
    // int p = r*c;
    // int r = 2;
    // int c = 2;
    
    complex<double> *in_root, *out_root, *recv_buff;
    DEVICE_FFT_DOUBLECOMPLEX *device_fft_in_root, *device_fft_out_root;
    complex<double> *fftx_out, *fftx_out_root;
    fftx_out = (complex<double> *) malloc((Mo*No*Ko/p) * sizeof(complex<double>));
    DEVICE_MEM_COPY(fftx_out, out_buffer, (Mo*No*Ko/p) * sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);
    if (commRank == root) {
      in_root       = (complex<double> *) malloc(Mo*No*Ko * sizeof(complex<double>));
      if (is_embedded) {
        for (size_t i = 0; i < Mo*No*Ko; i++) {
          in_root[i] = complex<double>(0.0, 0.0);
        }
      }
      out_root      = (complex<double> *) malloc(Mo*No*Ko * sizeof(complex<double>));
      fftx_out_root = (complex<double> *) malloc(Mo*No*Ko * sizeof(complex<double>));
      recv_buff     = (complex<double> *) malloc(Mo*No*Ko * sizeof(complex<double>));
      DEVICE_MALLOC(&device_fft_in_root,  Mo*No*Ko * sizeof(complex<double>));
      DEVICE_MALLOC(&device_fft_out_root, Mo*No*Ko * sizeof(complex<double>));
    }
    MPI_Gather(
      in_buff + (is_embedded ? Ki/2*Mi*Ni/p: 0),
      Mi*Ni*Ki/p,
      MPI_DOUBLE_COMPLEX,
      recv_buff,
      Mi*Ni*Ki/p,
      MPI_DOUBLE_COMPLEX,
      root,
      MPI_COMM_WORLD
    ); // root layout will be [xl, yl, zl, zr, xr, yr]

    if (commRank == root) {
      // repack to XYZ layout. col major, M fastest, then N, K.
      // r divides M, c divides N.
      for (int jr = 0; jr < c; jr++) {
        for (int ir = 0; ir < r; ir++) {
          for (int k = 0; k < Ki; k++) {
            for (int jl = 0; jl < Ni/c; jl++) {
              for (int il = 0; il < Mi/r; il++) {
                in_root[
                  ((is_embedded ? Ki/2 : 0) + k               ) * No*Mo +
                  ((is_embedded ? Ni/2 : 0) + jr * (Ni/c) + jl) *    Mo +
                  ((is_embedded ? Mi/2 : 0) + ir * (Mi/r) + il)
                ] = recv_buff[
                  jr * r*Ki*(Ni/c)*(Mi/r) +
                  ir *   Ki*(Ni/c)*(Mi/r) +
                  k  *      (Ni/c)*(Mi/r) +
                  jl *             (Mi/r) +
                  il
                ];
              }
            }
          }
        }
      }
      DEVICE_MEM_COPY(device_fft_in_root, in_root, Mo*No*Ko * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);

      DEVICE_FFT_HANDLE plan;
      DEVICE_FFT_RESULT res;

      // TODO: device_macros.h doesn't have cufftCreate or cufftMakePlan3d.
      // https://hipfft.readthedocs.io/en/rocm-5.1.3/api.html
      res = cufftCreate(&plan);
      if (res != DEVICE_FFT_SUCCESS) { printf ("*Create failed\n"); }
      size_t worksize[1]; // 1 GPU.
      res = cufftMakePlan3d(
        plan,
        Ko, No, Mo,
        DEVICE_FFT_Z2Z,
        worksize
      );
      if (res != DEVICE_FFT_SUCCESS) { printf ("*MakePlan* failed\n"); }
      res = DEVICE_FFT_EXECZ2Z(
        plan,
        device_fft_in_root,
        device_fft_out_root,
        DEVICE_FFT_FORWARD
      );
      if (res != DEVICE_FFT_SUCCESS) { printf ("*DEVICE_FFT_EXECZ2Z* failed\n"); }
      DEVICE_MEM_COPY(out_root, device_fft_out_root, Mo*No*Ko * sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);
    }
    MPI_Gather(
      fftx_out,
      Mo*No*Ko/p,
      MPI_DOUBLE_COMPLEX,
      recv_buff,
      Mo*No*Ko/p,
      MPI_DOUBLE_COMPLEX,
      root,
      MPI_COMM_WORLD
    ); // root layout will be [zl, xl, yl, yr, zr, xr]
    if (commRank == root) {
      // repack to XYZ layout.
      for (int ir = 0; ir < c; ir++) {
        for (int kr = 0; kr < r; kr++) {
          for (int j = 0; j < No; j++) {
            for (int il = 0; il < Mo/c; il++) {
              for (int kl = 0; kl < Ko/r; kl++) {
                fftx_out_root[
                  (kr * (Ko/r) + kl) * No*Mo +
                  (j               ) *    Mo +
                  (ir * (Mo/c) + il)
                ] = recv_buff[
                  ir * r*No*(Mo/c)*(Ko/r) +
                  kr *   No*(Mo/c)*(Ko/r) +
                  j  *      (Mo/c)*(Ko/r) +
                  il *             (Ko/r) +
                  kl
                ];
              }
            }
          }
        }
      }
      size_t miscompares = 0;
      for (int k = 0; k < Ko; k++) {
        for (int j = 0; j < No; j++) {
          for (int i = 0; i < Mo; i++) {
            complex<double> cfft, fftx;
            cfft = out_root     [k * No*Mo + j * Mo + i];
            fftx = fftx_out_root[k * No*Mo + j * Mo + i];
            // compare outputs.
            if (
              abs(cfft.real() - fftx.real()) >= 1e-7 ||
              abs(cfft.imag() - fftx.imag()) >= 1e-7 ||
              false
            ) {
              // if (j == 0 && k == 0) {
              // if (i == 0 && k == 0) {
              if (i == 0 && j == 0) {
                printf(
                  "[%3d,%3d,%3d]: %11f %c%11fi\t%11f %c%11fi\n",
                  i, j, k,
                  fftx.real(), fftx.imag() < 0 ? '-' : '+', fftx.imag() < 0 ? -fftx.imag() : fftx.imag(),
                  cfft.real(), cfft.imag() < 0 ? '-' : '+', cfft.imag() < 0 ? -cfft.imag() : cfft.imag()
                );
              }
              miscompares += 1;
            }
          }
        }
      }
      printf("%lu miscompares, %f%% correct\n", miscompares, ((double ) Mo*No*Ko - miscompares) / (Mo*No*Ko) * 100.0);
    }
  }
#endif



  destroy_2d_comms();
  
  MPI_Finalize();
  
  delete in_buff;
  return 0;
}


/*

array([[[ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j],
        [ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j],
        [ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j],
        [ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j]],

       [[ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j],
        [ 2.+0.j, -1.-1.j,  0.+0.j, -1.+1.j],
        [ 2.+0.j, -1.-1.j,  0.+0.j, -1.+1.j],
        [ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j]],

       [[ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j],
        [ 2.+0.j, -1.-1.j,  0.+0.j, -1.+1.j],
        [ 2.+0.j, -1.-1.j,  0.+0.j, -1.+1.j],
        [ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j]],

       [[ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j],
        [ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j],
        [ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j],
        [ 0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j]]])
>>> np.fft.fftn(x)

array([[[ 8.+0.j, -4.-4.j,  0.+0.j, -4.+4.j],
        [-4.-4.j,  0.+4.j,  0.+0.j,  4.-0.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-0.j],
        [-4.+4.j,  4.+0.j,  0.+0.j,  0.-4.j]],

       [[-4.-4.j,  0.+4.j,  0.+0.j,  4.-0.j],
        [ 0.+4.j,  2.-2.j,  0.+0.j, -2.-2.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-0.j],
        [ 4.+0.j, -2.-2.j,  0.+0.j, -2.+2.j]],

       [[ 0.+0.j,  0.+0.j,  0.+0.j,  0.-0.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-0.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-0.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-0.j]],

       [[-4.+4.j,  4.+0.j,  0.+0.j,  0.-4.j],
        [ 4.+0.j, -2.-2.j,  0.+0.j, -2.+2.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-0.j],
        [ 0.-4.j, -2.+2.j,  0.+0.j,  2.+2.j]]])
*/