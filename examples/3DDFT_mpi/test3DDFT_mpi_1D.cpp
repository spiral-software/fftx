#include <mpi.h>
#include <complex>
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include "device_macros.h"

#include "fftx_mpi.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  if (argc != 8) {
    printf("usage: %s <M> <N> <K> <batch> <embedded> <forward> <complex>\n", argv[0]);
    exit(-1);
  }

  MPI_Init(&argc, &argv);

  int rank;
  int p;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // X dim is size M,
  // Y dim is size N,
  // Z dim is size K.
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  int batch        = atoi(argv[4]);
  bool is_embedded = 0 < atoi(argv[5]);
  bool is_forward  = 0 < atoi(argv[6]);
  bool is_complex  = 0 < atoi(argv[7]);
  // (slowest to fastest)
  // R2C input is [K,       N, M]         doubles, block distributed Z.
  // C2R input is [N, M/2 + 1, K] complex doubles, block distributed X.
  // C2C input is [K,       N, M] complex doubles, block distributed Z.
  // TODO: check this
  // C2C inv   is [N,       M, K] complex doubles, block distributed Z.
  bool R2C = !is_complex &&  is_forward;
  bool C2R = !is_complex && !is_forward;
  bool C2C =  is_complex;

  int Mi, Ni, Ki;
  int Mo, No, Ko;

  Mi = C2R ? (M/2) + 1 : M;
  Ni = N;
  Ki = K;

  Mo = M * (is_embedded ? 2 : 1); // TODO: change for R2C?
  No = N * (is_embedded ? 2 : 1);
  Ko = K * (is_embedded ? 2 : 1);

  double *host_in, *dev_in;
  double *host_out, *dev_out, *Q3;
  int CI = C2C || C2R ? 2 : 1; // complex input.
  int CO = C2C || R2C ? 2 : 1; // complex output.

  // TODO: update C2R test to assume X is distributed initially.

  // X is first dim, so embed in dim of size Mo.
  if (is_forward) {
    host_in  = (double *) malloc(sizeof(complex<double>) * (Ki/p) * Ni * Mo * batch);
    host_out = (double *) malloc(sizeof(complex<double>) * (Ko/p) * No * Mo * batch);

    DEVICE_MALLOC(&dev_in , sizeof(complex<double>) * (Ki/p) * Ni * Mo * batch);
    DEVICE_MALLOC(&dev_out, sizeof(complex<double>) * (Ko/p) * No * Mo * batch);

    // assume layout is [Z, Y, X, b] (slowest to fastest)
    for (int l = 0; l < Ki/p; l++) {
      for (int j = 0; j < Ni; j++) {
        for (int i = 0; i < Mo; i++) {
          for (int b = 0; b < batch; b++) {
            host_in[((l * Ni*Mo + j * Mo + i)*batch + b)*CI + 0] = (
                is_embedded && (i < Mi/2 || 3 * Mi/2 <= i)
              ) ?
                0.0 :
                // 1.0 * (b + 1) * (rank*1 + 1) * (i*1 + 1) * (j*1 + 1) * (l*1 + 1);
                1.0 * rand() / RAND_MAX;
            if (CI == 2) {
              host_in[((l * Ni*Mo + j * Mo + i)*batch + b)*CI + 1] = 0;
            }
          }
        }
      }
    }

    DEVICE_MEM_COPY(dev_in, host_in, sizeof(double) * (Ki/p) * Ni * Mo * batch * CI, MEM_COPY_HOST_TO_DEVICE);
  } else {
    // TODO: fix for embedded.
    int M0 = Mi / p;
    M0 += M0*p < Mi;
    int M1 = p;

    host_in  = (double *) malloc(sizeof(complex<double>) * Ni * M0 * Ki * batch);
    host_out = (double *) malloc(sizeof(complex<double>) * Ko * No * M0 * batch);

    DEVICE_MALLOC(&dev_in , sizeof(complex<double>) * Ni * M0 * Ki * batch);
    DEVICE_MALLOC(&dev_out, sizeof(complex<double>) * Ko * No * M0 * batch);

    // assume layout is [Y, X'/px, Z] (slowest to fastest)
    for (int j = 0; j < Ni; j++) {
      for (int i = 0; i < M0; i++) {
        for (int l = 0; l < Ki; l++) {
          for (int b = 0; b < batch; b++) {
            for (int c = 0; c < CI; c++) {
              host_in[((j * M0*Ki + i * Ki + l)*batch + b) * CI + c] = {};
            }
          }
        }
      }
    }
    if (rank == 0) {
      for (int b = 0; b < batch; b++) {
        host_in[b * CI + 0] = (M * N * K * (b+1));
        if (CI == 2) {
          host_in[b * CI + 1] = 0;
        }
      }
    }
    DEVICE_MEM_COPY(dev_in, host_in, sizeof(complex<double>) * Ki * Ni * M0 * batch, MEM_COPY_HOST_TO_DEVICE);
  } // end forward/inverse check.

  // TODO: resume conversion of forward to inverse from here.
  fftx_plan plan = fftx_plan_distributed_1d(p, M, N, K, batch, is_embedded, is_complex);
  for (int t = 0; t < 1; t++) {
    double start_time = MPI_Wtime();
    fftx_execute_1d(plan, (double*)dev_out, (double*)dev_in, (is_forward ? DEVICE_FFT_FORWARD : DEVICE_FFT_INVERSE));

    // for (int b = 0; b < batch; b++) {
    //   fftx_execute_1d(plan, (double*)dev_out + b, dev_in + b, (is_forward ? DEVICE_FFT_FORWARD : DEVICE_FFT_INVERSE));
    // }
    double end_time = MPI_Wtime();
    double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);
    if (rank == 0) {
      cout << M << "," << N << "," << K  << "," << batch  << "," << is_embedded << "," << is_forward << "," << max_time << endl;
    }
  }

  {
    // TODO: initially only support non-embedded C2C.
    if (!(is_embedded == false && is_forward == true && C2C == true)) {
      cout << "Initially only support local check for forward non-embedded C2C" << endl;
    } else {
      double *href_in, *href_out, *htest_out;
      double *dref_in, *dref_out;
      int root = 0;
      if (rank == root) {
        // TODO: fix o vs i here. embedded doesn't exist locally.?
        href_in  = (double *) malloc(sizeof(double) * Ko * No * Mo * batch * CI);
        href_out = (double *) malloc(sizeof(double) * Ko * No * Mo * batch * CO);
        htest_out = (double *) malloc(sizeof(double) * Ko * No * Mo * batch * CO);
        DEVICE_MALLOC(&dref_in , sizeof(double) * Ko * No * Mo * batch * CI);
        DEVICE_MALLOC(&dref_out, sizeof(double) * Ko * No * Mo * batch * CO);
      }
      // gather all ins
      int count = Ki/p * Ni * Mo * batch * CI;
      MPI_Gather(host_in, count, MPI_DOUBLE, href_in, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
      // gather all outs
      // TODO update count
      MPI_Gather(host_out, count, MPI_DOUBLE, htest_out, count, MPI_DOUBLE, root, MPI_COMM_WORLD);

      // TODO: embed.?
      if (rank == root) {
        DEVICE_MEM_COPY(dref_in, href_in, sizeof(double) * Ko * No * Mo * batch * CI, MEM_COPY_HOST_TO_DEVICE);
        // create cuFFT plan 3d
        cufftHandle plan;
        // slowest to fastest.
        cufftPlan3d(&plan, Ko, No, Mo, CUFFT_C2C);
        cufftExecZ2Z(plan, (cufftDoubleComplex *) dref_in, (cufftDoubleComplex *) dref_out, CUFFT_FORWARD);
        DEVICE_MEM_COPY(href_out, dref_out, sizeof(double) * Ko * No * Mo * batch * CO, MEM_COPY_DEVICE_TO_HOST);

        // check href_out against htest_out.
        for (int i = 0; i < Ko * No * Mo * batch * CO; i++) {
          if (abs(href_out[i] - htest_out[i]) > 1e-8) {
            cout << "Error: " << i << " " << href_out[i] << " != " << htest_out[i] << endl;
          }
        }
        free(href_in);
        free(href_out);
        free(htest_out);
        DEVICE_FREE(dref_in);
        DEVICE_FREE(dref_out);
      }
    }
  }

  // Check first element.
  if (is_forward) {
    // TODO: copy more for C2R?
    int Mdim = R2C ? (Mo/2+1)/p : Mo/p;
    Mdim += Mdim * p < Mo;

    DEVICE_MEM_COPY(host_out, dev_out, sizeof(complex<double>) * No * Mdim * Ko * batch, MEM_COPY_DEVICE_TO_HOST);

    double *first_elems = (double *) malloc(sizeof(double) * batch);
    for (int b = 0; b < batch; b++) {
      first_elems[b] = {};
    }

    // initial distribution is [Z/p, Y, X, b]
    for (int l = 0; l < Ki/p; l++) {
      for (int j = 0; j < Ni; j++) {
        for (int i = 0; i < Mo; i++) {
          for (int b = 0; b < batch; b++) {
            first_elems[b] += host_in[((l * Ni*Mo + j * Mo + i)*batch + b)*CI + 0];
            // skip imaginary elements.
          }
        }
      }
    }

    for (int b = 0; b < batch; b++) {
      MPI_Allreduce(MPI_IN_PLACE, first_elems + b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // distribution is [Y, X'/p, Z, b]
    if (rank == 0) {
      printf("diff:\n");
      for (int b = 0; b < batch; b++) {
        printf(
          "%16f %16f %16f\n",
          first_elems[b],
          host_out[b*CI + 0],
          first_elems[b] - host_out[b*CI + 0]
        );
      }
      printf("\n");
    }
    free(first_elems);
  } else { // C2R
    // TODO: fix this
    int Mdim = C2R ? (Mo/2+1)/p : Mo/p;
    Mdim += Mdim * p < Mo;

    DEVICE_MEM_COPY(host_out, dev_out, sizeof(complex<double>) * No * Mdim * Ko * batch, MEM_COPY_DEVICE_TO_HOST);
    if (rank == 0) {
      printf("diff:\n");
      for (int b = 0; b < batch; b++) {
        printf(
          "%16f %16f %16f\t",
          (double) (M * N * K * (b+1)),
          ((complex<double> *)host_out)[b].real(),
          (double) (M * N * K * (b+1)) - ((complex<double> *)host_out)[b].real()
        );
        printf("\n");
      }
    }
  }

  fftx_plan_destroy(plan);

  DEVICE_FREE(dev_in);
  DEVICE_FREE(dev_out);

  free(host_in);
  free(host_out);

  MPI_Finalize();

  return 0;
}
