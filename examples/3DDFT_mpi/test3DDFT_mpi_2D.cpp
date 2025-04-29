#include <mpi.h>
#include <complex>
#include <iostream>
#include <stdlib.h>     /* srand, rand */

#include "fftx_mpi.hpp"

// using namespace std;

using cx = std::complex<double>;

int main(int argc, char* argv[])
{
  int status = 0;

  int ntrials = 3;
  
  MPI_Init(&argc, &argv);

  int commRank;
  int p;

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

  // ==== for timing, set by argument ====================
  if (argc != 10) {
    if (commRank == 0) {
      // printf("usage: %s <M> <N> <K> <batch> <grid rows> <grid columns> <embedded> <forward> <complex>\n", argv[0]);
      fftx::OutStream() << "usage: " << argv[0]
                        << " <M> <N> <K> <batch> <grid rows> <grid columns> <embedded> <forward> <complex>"
                        << std::endl;
    }
    MPI_Finalize();
    exit(-1);
  }
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int batch = atoi(argv[4]);
  int r = atoi(argv[5]);
  int c = atoi(argv[6]);
  bool is_embedded = 0 < atoi(argv[7]);
  bool is_forward = 0 < atoi(argv[8]);
  bool is_complex = 0 < atoi(argv[9]);
  // -----------------------------------------------------

  if (c * r != p) {
    if (commRank == 0) {
      // printf("error: product of splits %d and %d is %d, but there are %d MPI ranks\n", c, r, c*r, p);
      fftx::OutStream() << "error: product of splits "
                        << c << " and " << r << " is " << (c*r)
                        << ", but there are " << p << " MPI ranks"
                        << std::endl;
    }
    MPI_Finalize();
    exit(-1);
  }
  //define grid
  // int r = grid;
  // int c = grid;

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
  cx *out_buffer = NULL;

  //define host buffers
  double *fftx_in;
  cx *fftx_out;

  //embedded requires dim Z to be padded to full size (Ko instead of Ki)
  if (is_complex) {
    fftx_in = new double[(Ko * Mi * Ni)/p * batch * 2];
    fftx_out = new cx[(Mo * No * Ko)/p * batch];
  } else {
    fftx_in = new double[(Ko * Mi * Ni)/p * batch];
    fftx_out = new cx[(Mo * No * Ko)/p * batch];  //does this need to be padded further?
  }

  //allocate buffers
  FFTX_DEVICE_ERROR_T err = FFTX_DEVICE_MALLOC(&in_buffer, (Ko*Mi*Ni)/p * (is_complex ? sizeof(cx): sizeof(double))  * batch);
  if (err != FFTX_DEVICE_SUCCESS) {
    fftx::OutStream() << "FFTX_DEVICE_MALLOC failed\n" << std::endl;
    exit(-1);
  }

  err = FFTX_DEVICE_MALLOC(&out_buffer, (Mo*No*Ko)/p * sizeof(cx) * batch);
  if (err != FFTX_DEVICE_SUCCESS) {
    fftx::OutStream() << "FFTX_DEVICE_MALLOC failed\n" << std::endl;
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
            m               * Ko * batch * cmplx +
            k                    * batch * cmplx +
            b                            * cmplx +
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

  err = FFTX_DEVICE_MEM_COPY( in_buffer, fftx_in, (Ko*Mi*Ni)/p *  (is_complex ? sizeof(cx): sizeof(double)) * batch, FFTX_MEM_COPY_HOST_TO_DEVICE );
  if (err != FFTX_DEVICE_SUCCESS) {
    fftx::OutStream() << "FFTX_DEVICE_MEM_COPY failed\n" << std::endl;
    exit(-1);
  }

  fftx_plan  plan = fftx_plan_distributed(MPI_COMM_WORLD, r, c, M, N, K, batch, is_embedded, is_complex);

  FFTX_DEVICE_SYNCHRONIZE();
  MPI_Barrier(MPI_COMM_WORLD);

  if (commRank == 0) {
    fftx::OutStream() << "Problem size    : " << M << " x " << N << " x " << K << std::endl;
    fftx::OutStream() << "Batch size      : " << batch << std::endl;
    fftx::OutStream() << "Grid rows       : " << r << std::endl;
    fftx::OutStream() << "Grid columns    : " << c << std::endl;
    fftx::OutStream() << "Embedded        : " << (is_embedded ? "Yes": "No") << std::endl;
    fftx::OutStream() << "Direction       : " << (is_forward ? "Forward": "Inverse") << std::endl;
    fftx::OutStream() << "Complex         : " << (is_complex ? "Yes" : "No") << std::endl;
  }

  FFTX_DEVICE_EVENT_T custart, custop;
  FFTX_DEVICE_EVENT_CREATE ( &custart );
  FFTX_DEVICE_EVENT_CREATE ( &custop );
  for (int t = 1; t <= ntrials; t++) {

    FFTX_DEVICE_EVENT_RECORD ( custart );

    fftx_execute(plan, (double*)out_buffer, (double*)in_buffer,
                 (is_forward ? FFTX_DEVICE_FFT_FORWARD:
                  FFTX_DEVICE_FFT_INVERSE));

    FFTX_DEVICE_EVENT_RECORD ( custop );
    FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
    float millisec;
    FFTX_DEVICE_EVENT_ELAPSED_TIME ( &millisec, custart, custop );
    float max_time;
    MPI_Reduce(&millisec, &max_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    FFTX_DEVICE_MEM_COPY(fftx_out, out_buffer, ((Mo*No*Ko)/p) * sizeof(cx)*batch, FFTX_MEM_COPY_DEVICE_TO_HOST);
    FFTX_DEVICE_SYNCHRONIZE();

    if (commRank == 0) {
      // fftx::OutStream()<<std::endl<<"end_to_end," << max_time<<std::endl;
      fftx::OutStream()<<std::endl;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // for (int rank = 0; rank < p; ++rank){
  for (int rank = 0; rank < 1; ++rank){
    if (rank == commRank){
      fftx::OutStream()<<commRank<<": ";

      /*
      for (int i = 0; i != Mo*No*Ko/p; ++i)
	fftx::OutStream()<<fftx_out[i].real()<<" ";
      fftx::OutStream()<<std::endl;
      */
      for (int b = 0; b < batch; b++) {
	fftx::OutStream()<<fftx_out[b].real()<<" ";
      }
      fftx::OutStream()<<std::endl;

    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

 fftx_plan_destroy(plan);

 MPI_Finalize();

 FFTX_DEVICE_FREE(in_buffer);
 FFTX_DEVICE_FREE(out_buffer);
 delete[] fftx_in;
 delete[] fftx_out;
 return status;
}
