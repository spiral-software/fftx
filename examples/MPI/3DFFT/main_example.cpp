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

using namespace std;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int commRank;
  int p;
  
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  
  // 3d fft sizes
  int M = 32;
  int N = 32;
  int K = 32;

  int check = 1;

  complex<double> *in_buffer = NULL;
  cudaError_t err = cudaMalloc(&in_buffer, M*N*K/p * sizeof(complex<double>));
  if (err != cudaSuccess) {
    cout << "cudaMalloc failed\n" << endl;
    exit(-1);
  }
  complex<double> *out_buffer = NULL;
  err = cudaMalloc(&out_buffer, M*N*K/p * sizeof(complex<double>));
  if (err != cudaSuccess) {
    cout << "cudaMalloc failed\n" << endl;
    exit(-1);
  }

  // initialize data to random values in the range (-1, 1).
  complex<double> *in_buff = new complex<double>[M*N*K/p];
  
  for (int i = 0; i < M*N*K/p; i++) {
    in_buff[i] = complex<double>(
	 1 - ((double) rand()) / (double) (RAND_MAX/2),
	 1 - ((double) rand()) / (double) (RAND_MAX/2)
	 );
  }
  
  cudaMemcpy(in_buffer, in_buff,
	     M*N*K/p * sizeof(complex<double>), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  delete in_buff;

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  INIT_FN_NAME();

  __FILEROOT__((double*)out_buffer, (double*)in_buffer);

  DESTROY_FN_NAME();

  double end_time = MPI_Wtime();

  double min_time    = min_diff(start_time, end_time, MPI_COMM_WORLD);
  double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);

  if (commRank == 0)
    printf("%lf %lf\n", min_time, max_time);
  
  //  destroy_2d_comms();
  
  MPI_Finalize();
  
  return 0;
}
