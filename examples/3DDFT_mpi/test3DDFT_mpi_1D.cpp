#include <mpi.h>
#include <complex>
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include "device_macros.h"

#include "fftx_mpi.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  if (argc != 9) {
    printf("usage: %s <M> <N> <K> <batch> <embedded> <forward> <complex> <check>\n", argv[0]);
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
  bool check       = 0 < atoi(argv[8]);
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
  double *host_out, *dev_out;
  int CI = C2C || C2R ? 2 : 1; // complex input.
  int CO = C2C || R2C ? 2 : 1; // complex output.

  // Used only for inv? and inv checking.
  // TODO: should we do the same for Ki on fwd?
  int M0 = Mi / p;
  M0 += M0*p < Mi;
  int M1 = p;


  // TODO: update C2R test to assume X is distributed initially.

  // X is first dim, so embed in dim of size Mo.
  if (is_forward) {
    // TODO: what about when Ki % p != 0?
    host_in  = (double *) malloc(sizeof(double) * (Ki/p) * Ni * Mo * batch * CI);
    host_out = (double *) malloc(sizeof(double) * (Ko/p) * No * Mo * batch * CO);

    DEVICE_MALLOC(&dev_in , sizeof(double) * (Ki/p) * Ni * Mo * batch * CI);
    DEVICE_MALLOC(&dev_out, sizeof(double) * (Ko/p) * No * Mo * batch * CO);

    // () is distributed
    // assume layout is [(pz), Z/pz, Y, X, b] (slowest to fastest).
    // embed X in the middle of dimension of twice the size, pad with zeros.
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
              host_in[((l * Ni*Mo + j * Mo + i)*batch + b)*CI + 1] = 0; // TODO: also randomize this.
            }
          }
        }
      }
    }

    DEVICE_MEM_COPY(dev_in, host_in, sizeof(double) * (Ki/p) * Ni * Mo * batch * CI, MEM_COPY_HOST_TO_DEVICE);
  } else {
    // TODO: fix for embedded.

    host_in  = (double *) malloc(sizeof(double) * Ni * M0 * Ki * batch * CI);
    host_out = (double *) malloc(sizeof(double) * Ko * No * M0 * batch * CO);

    DEVICE_MALLOC(&dev_in , sizeof(double) * Ni * M0 * Ki * batch * CI);
    DEVICE_MALLOC(&dev_out, sizeof(double) * Ko * No * M0 * batch * CO);

    // assume layout is [(px), Y, X'/px, Z] (slowest to fastest)
    for (int j = 0; j < Ni; j++) {
      for (int i = 0; i < M0; i++) {
        for (int l = 0; l < Ki; l++) {
          for (int b = 0; b < batch; b++) {
            // for (int c = 0; c < CI; c++) {
              int c = 0;
              // TODO: try with random data.
              // host_in[((j * M0*Ki + i * Ki + l)*batch + b) * CI + c] = {};
              // host_in[((j * M0*Ki + i * Ki + l)*batch + b) * CI + c] = 1.0 * (b + 1) * (rank*1 + 1) * (i*1 + 1) * (j*1 + 1) * (l*1 + 1);

              host_in[((j * M0*Ki + i * Ki + l)*batch + b) * CI + c] = 1.0 * rand() / RAND_MAX;
            // }
          }
        }
      }
    }
    // TODO: get rid of this when we have random data.
    // if (rank == 0) {
    //   for (int b = 0; b < batch; b++) {
    //     host_in[b * CI + 0] = (M * N * K * (b+1));
    //     if (CI == 2) {
    //       host_in[b * CI + 1] = 0;
    //     }
    //   }
    // }
    DEVICE_MEM_COPY(dev_in, host_in, sizeof(double) * Ki * Ni * M0 * batch * CI, MEM_COPY_HOST_TO_DEVICE);
  } // end forward/inverse check.

  // TODO: resume conversion of forward to inverse from here.
  // fftx_plan plan = fftx_plan_distributed_1d(p, M, N, K, 1, is_embedded, is_complex);
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
      cout << M << "," << N << "," << K  << "," << batch  << "," << is_embedded << "," << is_forward << "," << max_time;
    }
  }

  if (check) {
    // TODO: initially only support non-embedded C2C.
    // TODO: guard on volume of cube? maybe limit to 1 GB local mem or something reasonable.
    if (!(is_embedded == false && C2C == true)) {
      if (rank == 0) {
        cout << "Initially only support local check for forward non-embedded C2C" << endl;
      }
    } else {
      double *href_in, *href_in_perm, *href_out, *htest_out;
      double *dref_in, *dref_out;
      int root = 0;
      if (rank == root) {
        // TODO: fix o vs i here. embedded doesn't exist locally.?
        size_t in_size, out_size;
        if (is_forward) {
          in_size  = Ko * No * Mo * batch * CI;
          out_size = Ko * No * Mo * batch * CO;
        } else {
          in_size  = Ko * No * Mo * batch * CO;
          out_size = Ko * No * Mo * batch * CI;
        }
          href_in   = (double *) malloc(sizeof(double) * in_size);
          if (!is_forward) {
            href_in_perm = (double *) malloc(sizeof(double) * in_size);
          }
          DEVICE_MALLOC(&dref_in , sizeof(double) * in_size);

          href_out  = (double *) malloc(sizeof(double) * out_size);
          htest_out = (double *) malloc(sizeof(double) * out_size);
          DEVICE_MALLOC(&dref_out, sizeof(double) * out_size);
      }
      // gather all ins
      int gather_in_count, gather_out_count;
      if (is_forward) {
        // TODO: change to K0?
        // TODO: fix these?
        // TODO: gather_in_count should match the size of input?
        gather_in_count  = (Ki/p) * Ni * Mo * batch * CI;
        gather_out_count = (Ki/p) * Ni * Mo * batch * CO;
      } else {
        // TODO: fix these?
        gather_in_count  = (Ko  ) * No * M0 * batch * CO;
        gather_out_count = (Ko  ) * No * M0 * batch * CI;
      }
      // fwd [Z, Y, X, b] <= Gather pz on [(pz), Z/pz, Y, X, b]
      // inv [px, Y, X'/px, Z] <= Gather px on [(px), Y, X'/px, Z]
      MPI_Gather(host_in, gather_in_count, MPI_DOUBLE, href_in, gather_in_count, MPI_DOUBLE, root, MPI_COMM_WORLD);
      // gather all outs
      // TODO update count for embedded.
      // fwd [px, Y, X'/px, Z] <= Gather px on [(px), Y, X'/px, Z]
      // inv [Z, Y, X, b] Gather pz on [(pz), Z/pz, Y, X, b]
      DEVICE_MEM_COPY(host_out, dev_out, sizeof(double) * gather_out_count, MEM_COPY_DEVICE_TO_HOST);
      MPI_Gather(host_out, gather_out_count, MPI_DOUBLE, htest_out, gather_out_count, MPI_DOUBLE, root, MPI_COMM_WORLD);
      // TODO: permute to [Z, Y, X, b] for comparing against reference.

      // TODO: embed.?
      if (rank == root) {
        // if inverse, permute data before calling cuFFT.
        if (!is_forward) {
          // inverse input  layout is [px, Y, X'/px, Z]
          // inverse output layout is [Z, Y, X, b]
          // permute data to [Z, Y, X, b] from [px, Y, X'/px, Z] before calling cuFFT.
          for (int i1 = 0; i1 < M1; i1++) {
            for (int j = 0; j < No; j++) {
              for (int i0 = 0; i0 < M0; i0++) {
                for (int k = 0; k < Ko; k++) {
                  for (int b = 0; b < batch; b++) {
                    for (int c = 0; c < CI; c++) {
                    href_in_perm[((k * No*M1*M0 + j * M1*M0 + i1 * M0 + i0) * batch + b) * CI + c] = href_in[((i1 * No*M0*Ko + j * M0*Ko + i0 * Ko + k) * batch + b) * CI + c];
                    }
                  }
                }
              }
            }
          }
          // swap pointers.
          double *tmp = href_in_perm;
          href_in_perm = href_in;
          href_in = tmp;
        }

        DEVICE_MEM_COPY(dref_in, href_in, sizeof(double) * Ko * No * Mo * batch * CI, MEM_COPY_HOST_TO_DEVICE);
        // TODO: change to vendor agnostic names.
        // create cuFFT plan 3d
        DEVICE_FFT_HANDLE plan;
        // slowest to fastest.
        DEVICE_FFT_PLAN3D(&plan, Ko, No, Mo, DEVICE_FFT_Z2Z);
        DEVICE_FFT_EXECZ2Z(plan, (DEVICE_FFT_DOUBLECOMPLEX *) dref_in, (DEVICE_FFT_DOUBLECOMPLEX *) dref_out, is_forward ? DEVICE_FFT_FORWARD : DEVICE_FFT_INVERSE);
        DEVICE_MEM_COPY(href_out, dref_out, sizeof(double) * Ko * No * Mo * batch * CO, MEM_COPY_DEVICE_TO_HOST);


        // check href_out against htest_out.
        bool correct = true;
        if (is_forward) {
          for (int i = 0; i < M0; i++) {
            for (int ii = 0; ii < M1; ii++) {
              for (int j = 0; j < No; j++) {
                for (int k = 0; k < Ko; k++) {
                  for (int b = 0; b < batch; b++) {
                    for (int c = 0; c < CO; c++) {
                      int ref_idx = ((k * No*M0*M1 + j * M0*M1 + ii * M0 + i)*batch + b) * CO + c;
                      int tst_idx = ((ii * No*M0*Ko + j * M0*Ko + i * Ko + k)*batch + b) * CO + c;
                      if (abs(href_out[ref_idx] - htest_out[tst_idx]) > 1e-8) {
                        correct = false;
                        // cout << "Error: (" << k << "," << j << "," << i*M1 + ii << ")\t"<< "\t" << href_out[ref_idx] << " != " << htest_out[tst_idx] << endl;
                      }
                    }
                  }
                }
              }
            }
          }
        } else { // inverse
          for (int k = 0; k < Ko; k++) {
            for (int j = 0; j < No; j++) {
              for (int i = 0; i < Mo; i++) {
                for (int b = 0; b < batch; b++) {
                  for (int c = 0; c < CO; c++) {
                    int ref_idx = ((k * No*Mo + j * Mo + i)*batch + b) * CO + c;
                    int tst_idx = ((k * No*Mo + j * Mo + i)*batch + b) * CO + c;
                    if (abs(href_out[ref_idx] - htest_out[tst_idx]) > 1e-8) {
                      correct = false;
                      // cout << "Error: (" << k+1 << "," << j+1 << "," << i+1 << ")\t"<< "\t" << href_in[ref_idx] << "\t" << href_out[ref_idx] << " != " << htest_out[tst_idx] << endl;
                    }
                  }
                }
              }
            }
          }
        }
        if (correct) {
          cout << ",1";
        } else {
          cout << ",0";
        }

        // for (int i = 0; i < Ko * No * Mo * batch * CO; i++) {
        //   cout << i << " " << href_in[i] << " " << href_out[i] << " " << htest_out[i] << endl;
        //   if (abs(href_out[i] - htest_out[i]) > 1e-8) {
        //     cout << "Error: " << i << "\t" << href_out[i] << " != " << htest_out[i] << endl;
        //   }
        // }
        free(href_in);
        free(href_out);
        free(htest_out);
        DEVICE_FREE(dref_in);
        DEVICE_FREE(dref_out);
      }
    }
  } else {
    // not checking.
    if (rank == 0) {
      cout << ",-";
    }
  }

  // // Check first element.
  // if (is_forward) {
  //   // TODO: copy more for C2R?
  //   int Mdim = R2C ? (Mo/2+1)/p : Mo/p;
  //   Mdim += Mdim * p < Mo;

  //   DEVICE_MEM_COPY(host_out, dev_out, sizeof(complex<double>) * No * Mdim * Ko * batch, MEM_COPY_DEVICE_TO_HOST);

  //   double *first_elems = (double *) malloc(sizeof(double) * batch);
  //   for (int b = 0; b < batch; b++) {
  //     first_elems[b] = {};
  //   }

  //   // initial distribution is [Z/p, Y, X, b]
  //   for (int l = 0; l < Ki/p; l++) {
  //     for (int j = 0; j < Ni; j++) {
  //       for (int i = 0; i < Mo; i++) {
  //         for (int b = 0; b < batch; b++) {
  //           first_elems[b] += host_in[((l * Ni*Mo + j * Mo + i)*batch + b)*CI + 0];
  //           // skip imaginary elements.
  //         }
  //       }
  //     }
  //   }

  //   for (int b = 0; b < batch; b++) {
  //     MPI_Allreduce(MPI_IN_PLACE, first_elems + b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  //   }

  //   // distribution is [Y, X'/p, Z, b]
  //   if (rank == 0) {
  //     printf("diff:\n");
  //     for (int b = 0; b < batch; b++) {
  //       printf(
  //         "%16f %16f %16f\n",
  //         first_elems[b],
  //         host_out[b*CI + 0],
  //         first_elems[b] - host_out[b*CI + 0]
  //       );
  //     }
  //     printf("\n");
  //   }
  //   free(first_elems);
  // } else { // C2R
  //   // TODO: fix this
  //   int Mdim = C2R ? (Mo/2+1)/p : Mo/p;
  //   Mdim += Mdim * p < Mo;

  //   DEVICE_MEM_COPY(host_out, dev_out, sizeof(complex<double>) * No * Mdim * Ko * batch, MEM_COPY_DEVICE_TO_HOST);
  //   if (rank == 0) {
  //     printf("diff:\n");
  //     for (int b = 0; b < batch; b++) {
  //       printf(
  //         "%16f %16f %16f\t",
  //         (double) (M * N * K * (b+1)),
  //         ((complex<double> *)host_out)[b].real(),
  //         (double) (M * N * K * (b+1)) - ((complex<double> *)host_out)[b].real()
  //       );
  //       printf("\n");
  //     }
  //   }
  // }

  if (rank == 0) {
    cout << endl;
  }

  fftx_plan_destroy(plan);

  DEVICE_FREE(dev_in);
  DEVICE_FREE(dev_out);

  free(host_in);
  free(host_out);

  MPI_Finalize();

  return 0;
}
