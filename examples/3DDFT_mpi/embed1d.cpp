#include <mpi.h>
#include <complex>
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include "device_macros.h"

#include "fftx_mpi.hpp"

using namespace std;

int main(int argc, char* argv[])
{

  MPI_Init(&argc, &argv);

  int commRank;
  int p = 2;
  
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

  bool is_embedded = false;
  bool is_forward = true;
  int batch = 1;
  bool is_complex = false;


  double *dev_in, *host_in;
  complex<double> *host_out, *dev_out, *send_buf, *recv_buf, *Q3;

  int M = 6;
  int N = 4;
  int K = 6;
  int b = 1;

  
  host_in = (double*)malloc(sizeof(complex<double>) * M * N * b * K / p);
  host_out = (complex<double>*)malloc(sizeof(complex<double>) * M * N * b * K / p);

  //  Q3 = (complex<double>*)malloc(sizeof(complex<double>) * M * N * b * K / p);

  //  send_buf = (complex<double>*)malloc(sizeof(complex<double>) * M * N * b * K / p);
  //  recv_buf = (complex<double>*)malloc(sizeof(complex<double>) * M * N * b * K / p);     

  // replace plan init with init_1d_comms?
  // define any fields here that are used by pack functions.
  //  fftx_plan plan = (fftx_plan) malloc(sizeof(fftx_plan_t));
  //  plan->send_buffer = send_buf;
  //  plan->recv_buffer = recv_buf;
  // end plan init.

  DEVICE_MALLOC(&dev_in, sizeof(complex<double>) * M * N * b * K/p);
  DEVICE_MALLOC(&dev_out, sizeof(complex<double>) * M * N * b * K/p);  

  for (int l = 0; l != K/p; ++l)
    for (int i = 0; i != M; ++i)
      for (int j = 0; j != N; ++j)
	for (int k = 0; k != b; ++k)
	  host_in[l*M*N*b + i*N*b + j*b + k] = 1.0;//complex<double>(l*M*N*K  + k+1.0, 0.0);

  /*
  if (commRank == 0){
    cout<<"In"<<endl;
    for (int i = 0; i != K/p; ++i){
      for (int j = 0; j != M*N*b; ++j)
	cout<<host_in[i*N*M*b + j]<<" ";
      cout<<endl;
    }
  }
  */
  
  DEVICE_MEM_COPY(dev_in, host_in, N * M * b * K/p * sizeof(double), MEM_COPY_HOST_TO_DEVICE);

  
  int Mdim = (M/2+1)/p;
  if ((M/2 + 1) % p) {
    Mdim += 1;
  }

  /*
  DEVICE_FFT_HANDLE plan1, plan2, plan3;

  // slowest to fastest
  // [X', Z/pz, Y] <= [Z/pz, Y, X] (read seq, write strided)
  // X' is complex size M/2 + 1
  DEVICE_FFT_PLAN_MANY(
    &plan1, 1, &M,
    &M, b, M*b,
    &M, N*K/p*b, b,
    DEVICE_FFT_D2Z, N*K/p
  );
  
  // [Y, X'/px, Z] <= [X'/px, Z, Y] (read seq, write strided)
  DEVICE_FFT_PLAN_MANY(
    &plan2, 1, &N,
    &N,     b, N*b,	       
    &N, Mdim*K*b, b,
    DEVICE_FFT_Z2Z, Mdim*K
		       );
  
  // [Y, X'/px, Z] <= [Y, X'/px, Z] (read seq, write seq)
  DEVICE_FFT_PLAN_MANY(
    &plan3, 1, &K,
    &K, b, K*b,
    &K, b, K*b,
    DEVICE_FFT_Z2Z, N*Mdim
  );
*/
  fftx_plan  plan = fftx_plan_distributed_1d(p, M, N, K, b, is_embedded, is_complex);
  fftx_execute_1d(plan, (double*)dev_out, dev_in, DEVICE_FFT_FORWARD);
  
  send_buf = plan->send_buffer;
  recv_buf = plan->recv_buffer;


  //begin execute
  //  for (int i = 0; i != b; ++i)
  //    DEVICE_FFT_EXECD2Z(plan->stg1, ((DEVICE_FFT_DOUBLEREAL*)dev_in + i),
  //		              ((DEVICE_FFT_DOUBLECOMPLEX*)dev_out + i));
  /*
  DEVICE_MEM_COPY(send_buf, dev_out, N * M * b * K/p* sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);

  //RCPERM
  
  //pack for 1D MPI

  // slowest to fastest
  // tile X' by X'/p,
  // [(pz), px, X'/px, Z/pz, Y] <= [(pz), X', Z/pz, Y]
  // then gather pz - scatter px
  // [(px), pz, X'/px, Z/pz, Y] <= [(pz), px, X'/px, Z/pz, Y]
  size_t sendSize = Mdim * N  * K/p;
  size_t recvSize = Mdim * N  * K/p;
  MPI_Alltoall(
    send_buf, sendSize * b,
    MPI_DOUBLE_COMPLEX,
    recv_buf, recvSize * b,
    MPI_DOUBLE_COMPLEX,
    MPI_COMM_WORLD
  );

  //      [X'/px, pz, Z/pz, Y] <= [pz, X'/px, Z/pz, Y]
  // i.e. [X'/px,        Z, Y]
  pack_embed(plan, (complex<double> *) dev_in, recv_buf, N * K/p * b, Mdim, p, is_embedded);
  // endRCPERM

  // [Y, X'/px, Z] <= [X'/px, Z, Y]
  for (int i = 0; i != b; ++i)
    DEVICE_FFT_EXECZ2Z(plan->stg2, ((DEVICE_FFT_DOUBLECOMPLEX*)dev_in + i) ,
		              ((DEVICE_FFT_DOUBLECOMPLEX*)dev_out + i), DEVICE_FFT_FORWARD);
  
  { // swap pointers.
    void *tmp = (void *) dev_in;
    dev_in = (double *) dev_out;
    dev_out = (complex<double> *) dev_in;
  }

  for (int i = 0; i != b; ++i)
    DEVICE_FFT_EXECZ2Z(plan->stg3, ((DEVICE_FFT_DOUBLECOMPLEX*)dev_in + i) ,
		       ((DEVICE_FFT_DOUBLECOMPLEX*)dev_out + i), DEVICE_FFT_FORWARD);
  //end execute

  */
  DEVICE_MEM_COPY(host_out, dev_out, N * Mdim * b * K* sizeof(complex<double>), MEM_COPY_DEVICE_TO_HOST);

  if (commRank == 0)
    {
      cout<<M<<"x"<<N<<"x"<<K<<"x"<<b<<endl;    

      for (int i = 0; i != Mdim*N*K/p; ++i)
	cout<<host_out[i]<<" ";
      cout<<endl;
    }


  fftx_plan_destroy(plan);
  
  DEVICE_FREE(dev_in);
  DEVICE_FREE(dev_out);  
  
  free(host_in);
  free(host_out);
  //  free(send_buf);
  //  free(recv_buf);
  //  free(Q3);

  MPI_Finalize();
  
  return 0;
}
