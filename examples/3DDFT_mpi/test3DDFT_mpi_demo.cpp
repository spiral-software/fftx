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

  int size = 8;
  
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

  bool is_embedded = false;
  bool is_forward = true;
  int batch = 1;

  // 3d fft sizes 
  int M = size / (is_embedded ? 2 : 1);
  int N = size / (is_embedded ? 2 : 1);
  int K = size / (is_embedded ? 2 : 1);

  //define grid
  int r = 2;
  int c = 2;

  
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
  complex<double> *fftx_in  = new complex<double>[ Mo * Ni * Ki/p * batch];
  complex<double> *fftx_out = new complex<double>[ Mo * No * Ko/p * batch];  

  //allocate buffers
  DEVICE_ERROR_T err = DEVICE_MALLOC(&in_buffer, Mo*Ni*Ki/p * sizeof(complex<double>) * batch);
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
  for (int k = 0; k < Ki/r; k++) {
    for (int j = 0; j < Ni/c; j++) {
      for (int i = 0; i < Mo; i++) {
	for (int b = 0; b < batch; b++) {	    
	    fftx_in[
		    (k * (Ni/c)*Mo +
		     j *        Mo +
		     i)* batch     +
		     b
		    ] = ( // embedded
			 (Mi/2 <= i &&
			  i < 3*Mi/2 )||
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

  if (!is_forward)
    {
      if (commRank == 0){
	for (int b = 0; b < batch; b++)
	  fftx_in[b] = complex<double>(M*N*K*(b+1),0);
      }
    }
  //end init

  
  err = DEVICE_MEM_COPY( in_buffer, fftx_in, Mo*Ni*Ki/p * sizeof(complex<double>) * batch, MEM_COPY_HOST_TO_DEVICE );
  if (err != DEVICE_SUCCESS) {
    cout << "DEVICE_MEM_COPY failed\n" << endl;
    exit(-1);
  }

  fftx_plan  plan = fftx_plan_distributed(r, c, M, N, K, batch, is_embedded);

  DEVICE_SYNCHRONIZE();
  MPI_Barrier(MPI_COMM_WORLD); 
  
  for (int t = 0; t < 1; t++) {

    double start_time = MPI_Wtime();
    
    fftx_execute(plan, (double*)out_buffer, (double*)in_buffer, (is_forward ? DEVICE_FFT_FORWARD: DEVICE_FFT_INVERSE));
    
    double end_time = MPI_Wtime();
    
    double min_time    = min_diff(start_time, end_time, MPI_COMM_WORLD);
    double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);

    DEVICE_MEM_COPY(fftx_out, out_buffer, (Mo*No*Ko/p) * sizeof(complex<double>)*batch, MEM_COPY_DEVICE_TO_HOST);
    DEVICE_SYNCHRONIZE();
        
    if (commRank == 0) {

      cout<<"Problem size: "<<M<<" x "<<N<<" x "<<K<<endl;
      cout<<"Batch size  : "<<batch<<endl;
      cout<<"Embedded    : "<<(is_embed ? "Yes": "No")<<endl;
      cout<<"Grid size   : "<<r<<" x "<<c<<endl;

      cout<<min_time<<" "<<max_time<<endl;
      cout<<endl;
      for (int i = 0; i != batch; ++i)
	cout<<fftx_out[i].real()<<endl;
    }    
  }
  
 fftx_plan_destroy(plan);

 MPI_Finalize();

 DEVICE_FREE(in_buffer);
 DEVICE_FREE(out_buffer);
 delete[] fftx_in;
 delete[] fftx_out;
 return 0;
}
