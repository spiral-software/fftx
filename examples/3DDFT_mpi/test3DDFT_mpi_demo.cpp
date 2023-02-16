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
  if (argc != 8) {
    printf("usage: %s <M> <N> <K> <batch> <grid dim> <embedded> <forward>\n", argv[0]);
    exit(-1);
  }
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int batch = atoi(argv[4]);
  int grid = atoi(argv[5]);
  bool is_embedded = 0 < atoi(argv[6]);
  bool is_forward = 0 < atoi(argv[7]);
  // -----------------------------------------------------

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);


  // 3d fft sizes 
  M = M / (is_embedded ? 2 : 1);
  N = N / (is_embedded ? 2 : 1);
  K = K / (is_embedded ? 2 : 1);

  //define grid
  int r = grid;
  int c = grid;

  
  int Mi = M;
  int Ni = N;
  int Ki = K;
  int Mo = M * (is_embedded ? 2 : 1);
  int No = N * (is_embedded ? 2 : 1);
  int Ko = K * (is_embedded ? 2 : 1);

  //define device buffers
  complex<double> *in_buffer = NULL;
  complex<double> *out_buffer = NULL;

  //define host buffers
  // complex<double> *fftx_in  = new complex<double>[ Mo * Ni * Ki/p * batch];
  complex<double> *fftx_in  = new complex<double>[ Ko * Mi * Ni/p * batch];
  complex<double> *fftx_out = new complex<double>[ Mo * No * Ko/p * batch];  

  //allocate buffers
  // DEVICE_ERROR_T err = DEVICE_MALLOC(&in_buffer, Mo*Ni*Ki/p * sizeof(complex<double>) * batch);
  DEVICE_ERROR_T err = DEVICE_MALLOC(&in_buffer, Ko*Mi*Ni/p * sizeof(complex<double>) * batch);
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
  // for (int k = 0; k < Ki/r; k++) {
  //   for (int j = 0; j < Ni/c; j++) {
  //     for (int i = 0; i < Mo; i++) {
  //       for (int b = 0; b < batch; b++) {
  //         fftx_in[
  //           (k * (Ni/c)*Mo +
  //           j *        Mo +
  //           i)* batch     +
  //           b
  //           ] = ( // embedded
  //             (Mi/2 <= i && i < 3*Mi/2 ) ||
  //             !is_embedded
  //           ) ?
  //           complex<double>(
  //               //1 - ((double) rand()) / (double) (RAND_MAX/2),
  //               //1 - ((double) rand()) / (double) (RAND_MAX/2)
  //               (is_forward ? (b+1) : 0),
  //               0
  //               )
  //           :
  //           complex<double>(0, 0);
  //       }
  //     }
  //     }
  // }

  for (int n = 0; n < Ni/r; n++) {
    for (int m = 0; m < Mi/c; m++) {
      for (int k = 0; k < Ko; k++) {
        for (int b = 0; b < batch; b++) {
          fftx_in[
            (n * (Mi/c)*Ko +
             m *        Ko +
             k)* batch     +
            b
            ] = ( // embedded
              (Ki/2 <= k && k < 3*Ki/2 ) ||
              !is_embedded
            ) ?
            complex<double>(
                //1 - ((double) rand()) / (double) (RAND_MAX/2),
                //1 - ((double) rand()) / (double) (RAND_MAX/2)
                (is_forward ? (b+1) : 0),
                0
                )
            :
            complex<double>(0, 0);
        }
      }
      }
  }


  if (!is_forward) {
    if (commRank == 0) {
      for (int b = 0; b < batch; b++)
        fftx_in[b] = complex<double>(M*N*K*(b+1), 0);
    }
  }
  //end init

  err = DEVICE_MEM_COPY( in_buffer, fftx_in, Ko*Mi*Ni/p * sizeof(complex<double>) * batch, MEM_COPY_HOST_TO_DEVICE );
  if (err != DEVICE_SUCCESS) {
    cout << "DEVICE_MEM_COPY failed\n" << endl;
    exit(-1);
  }

  fftx_plan  plan = fftx_plan_distributed(r, c, M, N, K, batch, is_embedded);

  DEVICE_SYNCHRONIZE();
  MPI_Barrier(MPI_COMM_WORLD); 
  
  if (commRank == 0) {
    cout<<"Problem size: "<<M<<" x "<<N<<" x "<<K<<endl;
    cout<<"Batch size  : "<<batch<<endl;
    cout<<"Embedded    : "<<(is_embedded ? "Yes": "No")<<endl;
    cout<<"Direction   : "<<(is_forward ? "Forward": "Inverse")<<endl;
    cout<<"Grid size   : "<<r<<" x "<<c<<endl;
  }

  for (int t = 0; t < 10; t++) {

    double start_time = MPI_Wtime();
    
    fftx_execute(plan, (double*)out_buffer, (double*)in_buffer, (is_forward ? DEVICE_FFT_FORWARD: DEVICE_FFT_INVERSE));
    
    double end_time = MPI_Wtime();
    
    // double min_time    = min_diff(start_time, end_time, MPI_COMM_WORLD);
    double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);

    DEVICE_MEM_COPY(fftx_out, out_buffer, (Mo*No*Ko/p) * sizeof(complex<double>)*batch, MEM_COPY_DEVICE_TO_HOST);
    DEVICE_SYNCHRONIZE();
        
    if (commRank == 0) {
      cout<<"end_to_end," << max_time<<endl;
    }    
  }
  MPI_Barrier(MPI_COMM_WORLD); 

  for (int b = 0; b < batch; b++) {
	  cout<<fftx_out[b].real()<<endl;
  }
  
 fftx_plan_destroy(plan);

 MPI_Finalize();

 DEVICE_FREE(in_buffer);
 DEVICE_FREE(out_buffer);
 delete[] fftx_in;
 delete[] fftx_out;
 return 0;
}
