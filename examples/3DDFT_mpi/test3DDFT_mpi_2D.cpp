#include <mpi.h>
#include <complex>
#include <iostream>
#include <stdlib.h>     /* srand, rand */

#include "fftx_mpi.hpp"

using namespace std;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int commRank;
  int p;

  // ==== for timing, set by argument ====================
  if (argc != 9) {
    printf("usage: %s <M> <N> <K> <batch> <grid dim> <embedded> <forward> <complex>\n", argv[0]);
    exit(-1);
  }
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int batch = atoi(argv[4]);
  int grid = atoi(argv[5]);
  bool is_embedded = 0 < atoi(argv[6]);
  bool is_forward = 0 < atoi(argv[7]);
  bool is_complex = 0 < atoi(argv[8]);
  // -----------------------------------------------------

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

  //define grid
  int r = grid;
  int c = grid;

  // 3d fft sizes
  uint64_t Mi, Ni, Ki;
  uint64_t Mo, No, Ko;

  Mi = M;
  Ni = N;
  Ki = K;
  Mo = M * (is_embedded ? 2 : 1);
  No = N * (is_embedded ? 2 : 1);
  Ko = K * (is_embedded ? 2 : 1);

  //define device buffers
  double *in_buffer = NULL;
  complex<double> *out_buffer = NULL;

  //define host buffers
  double *fftx_in;
  complex<double> *fftx_out;


  //embedded requires dim Z to be padded to full size (Ko instead of Ki)
  if (is_complex) {
    fftx_in = new double[Ko * Mi * Ni/p * batch * 2];
    fftx_out = new complex<double>[Mo * No * Ko/p * batch];
  } else {
    fftx_in = new double[Ko * Mi * Ni/p * batch];
    fftx_out = new complex<double>[Mo * No * Ko/p * batch];  //does this need to be padded further?
  }

  //allocate buffers
  DEVICE_ERROR_T err = DEVICE_MALLOC(&in_buffer, Ko*Mi*Ni/p * (is_complex ? sizeof(complex<double>): sizeof(double))  * batch);
  if (err != DEVICE_SUCCESS) {
    cout << "DEVICE_MALLOC failed\n" << endl;
    exit(-1);
  }

  err = DEVICE_MALLOC(&out_buffer, Mo*No*Ko/p * sizeof(complex<double>) * batch);
  if (err != DEVICE_SUCCESS) {
    cout << "DEVICE_MALLOC failed\n" << endl;
    exit(-1);
  }

  // initialize data to random values in the range (-1, 1).
  // assume data is padded in one dim as input.
  uint64_t cmplx = is_complex ? 2 : 1;
  for (uint64_t n = 0; n < Ni/r; n++) {
    for (uint64_t m = 0; m < Mi/c; m++) {
      for (uint64_t k = 0; k < Ko; k++) {
        for (uint64_t b = 0; b < batch; b++) {
          double *in = fftx_in +
            n * (Mi/c) * Ko * batch * cmplx +
            m          * Ko * batch * cmplx +
            k               * batch * cmplx +
            b                       * cmplx +
            0
          ;
          in[0] = (
            (Ki/2 <= k && k < 3*Ki/2 ) ||
            !is_embedded
          ) ?
          //1 - ((double) rand()) / (double) (RAND_MAX/2),
          (is_forward ? (b+1) : 0)
          :
          0;
          if (is_complex) { // set real component.
            in[1] = (
              (Ki/2 <= k && k < 3*Ki/2 ) ||
              !is_embedded
            ) ?
              //1 - ((double) rand()) / (double) (RAND_MAX/2),
              (is_forward ? (b+1) : 0)
              :
              0;
          }
        }
      }
    }
  }

  if (!is_forward) {
    if (commRank == 0) {
      for (int b = 0; b < batch; b++) {
        fftx_in[b*2 + 0] = (M*N*K*(b+1));
        if (is_complex) {
          fftx_in[b*2 + 1] = (M*N*K*(b+1));
        }
      }
    }
  }
  //end init

  err = DEVICE_MEM_COPY( in_buffer, fftx_in, Ko*Mi*Ni/p *  (is_complex ? sizeof(complex<double>): sizeof(double)) * batch, MEM_COPY_HOST_TO_DEVICE );
  if (err != DEVICE_SUCCESS) {
    cout << "DEVICE_MEM_COPY failed\n" << endl;
    exit(-1);
  }

  fftx_plan  plan = fftx_plan_distributed(r, c, M, N, K, batch, is_embedded, is_complex);

  DEVICE_SYNCHRONIZE();
  MPI_Barrier(MPI_COMM_WORLD);

  if (commRank == 0) {
    cout << "Problem size    : " << M << " x " << N << " x " << K << endl;
    cout << "Batch size      : " << batch << endl;
    cout << "Complex         : " << (is_complex ? "Yes" : "No") << endl;
    cout << "Embedded        : " << (is_embedded ? "Yes": "No") << endl;
    cout << "Direction       : " << (is_forward ? "Forward": "Inverse") << endl;
    cout << "Grid size       : " << r << " x " << c << endl;
  }

  for (int t = 0; t < 3; t++) {

    double start_time = MPI_Wtime();

    fftx_execute(plan, (double*)out_buffer, (double*)in_buffer, (is_forward ? DEVICE_FFT_FORWARD: DEVICE_FFT_INVERSE));

    double end_time = MPI_Wtime();

    // double min_time    = min_diff(start_time, end_time, MPI_COMM_WORLD);
    double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);

    DEVICE_MEM_COPY(fftx_out, out_buffer, (Mo*No*Ko/p) * sizeof(complex<double>)*batch, MEM_COPY_DEVICE_TO_HOST);
    DEVICE_SYNCHRONIZE();

    if (commRank == 0) {
      // cout<<endl<<"end_to_end," << max_time<<endl;
      cout<<endl;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // for (int rank = 0; rank < p; ++rank){
  for (int rank = 0; rank < 1; ++rank){
    if (rank == commRank){
      cout<<commRank<<": ";

      /*
      for (int i = 0; i != Mo*No*Ko/p; ++i)
	cout<<fftx_out[i].real()<<" ";
      cout<<endl;
      */
      for (int b = 0; b < batch; b++) {
	cout<<fftx_out[b].real()<<" ";
      }
      cout<<endl;

    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

 fftx_plan_destroy(plan);

 MPI_Finalize();

 DEVICE_FREE(in_buffer);
 DEVICE_FREE(out_buffer);
 delete[] fftx_in;
 delete[] fftx_out;
 return 0;
}
