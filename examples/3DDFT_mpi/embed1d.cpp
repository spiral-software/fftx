#include <mpi.h>
#include <complex>
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include "device_macros.h"

#include "fftx_mpi.hpp"

using namespace std;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int commRank;
  int p;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

  if (argc != 6) {
    printf("usage: %s <M> <N> <K> <batch> <embedded>\n", argv[0]);
    exit(-1);
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int batch = atoi(argv[4]);
  bool is_embedded = 0 < atoi(argv[5]);

  if (commRank == 0) printf("%d, %d, %d, %d\n", M, N, K, batch);

  bool is_forward = false;
  bool is_complex = false;

  int Mi, Ni, Ki;
  int Mo, No, Ko;
  Mi = M;
  Ni = N;
  Ki = K;
  Mo = M * (is_embedded ? 2 : 1);
  No = N * (is_embedded ? 2 : 1);
  Ko = K * (is_embedded ? 2 : 1);

  double          *host_in , *dev_in;
  complex<double> *host_out, *dev_out, *Q3;

  // X is first dim, so embed in dim of size Mo.
  host_in  = (double *         ) malloc(sizeof(complex<double>) * (Ki/p) * Ni * Mo * batch);
  host_out = (complex<double> *) malloc(sizeof(complex<double>) * (Ko/p) * No * Mo * batch);

  DEVICE_MALLOC(&dev_in , sizeof(complex<double>) * (Ki/p) * Ni * Mo * batch);
  DEVICE_MALLOC(&dev_out, sizeof(complex<double>) * (Ko/p) * No * Mo * batch);

  for (int l = 0; l != Ki/p; ++l) {
    for (int j = 0; j != Ni; ++j) {
      for (int i = 0; i != Mo; ++i) {
        for (int k = 0; k != batch; ++k) {
          host_in[l * Ni*Mo*batch + j * Mo*batch + i * batch + k] = (
            is_embedded && (i < Mi/2 || 3 * Mi/2 <= i) ||
            !is_forward
            ) ?
              0.0:
              1.0;
          // complex<double>(l*M*N*K  + k+1.0, 0.0);
        }
      }
    }
  }
  if (!is_forward) {
    if (commRank == 0) {
      for (int b = 0; b < batch; b++) {
        host_in[b*2 + 0] = (M * N * K * (b+1));
        if (is_complex) {
          host_in[b*2 + 1] = (M * N * K * (b+1));
        }
      }
    }
  }

  DEVICE_MEM_COPY(dev_in, host_in, sizeof(double) * (Ki/p) * Ni * Mo * batch, MEM_COPY_HOST_TO_DEVICE);
  
  int Mdim = (Mo/2+1)/p;
  if ((Mo/2 + 1) % p) {
    Mdim += 1;
  }

  fftx_plan  plan = fftx_plan_distributed_1d(p, M, N, K, batch, is_embedded, is_complex);
  
  for (int t = 0; t < 1; t++) {
    double start_time = MPI_Wtime();
    fftx_execute_1d(plan, (double*)dev_out, dev_in, (is_forward ? DEVICE_FFT_FORWARD : DEVICE_FFT_INVERSE));
    double end_time = MPI_Wtime();

    double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);
    
    if (commRank == 0) {
      cout << "end_to_end," << max_time<<endl;
    }
  }

  // layout is [Y, X'/px, Z]
  DEVICE_MEM_COPY(host_out, dev_out, sizeof(complex<double>) * No * Mdim * Ko * batch, MEM_COPY_DEVICE_TO_HOST);

  if (commRank == 0) {
    printf("diff: %f\n", (double) (M * N * K) - host_out[0].real());
  }

  fftx_plan_destroy(plan);
  
  DEVICE_FREE(dev_in);
  DEVICE_FREE(dev_out);  
  
  free(host_in);
  free(host_out);

  MPI_Finalize();
  
  return 0;
}
