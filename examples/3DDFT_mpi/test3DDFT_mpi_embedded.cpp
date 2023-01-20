#include <mpi.h>
#include <limits.h>
#include <complex>
#include <float.h>
#include <string>
#include "fftx_gpu.h"
#include "fftx_util.h"
#include "fftx3.hpp"
#include "fftx_mpi.hpp"

#include "fftx_distdft_embed_gpu_public.h"

#include <stdlib.h>     /* srand, rand */

#include "device_macros.h"

#define CHECK_WITH_CUFFT 1
#if CHECK_WITH_CUFFT
#endif

using namespace std;

int main(int argc, char* argv[]) {

	int row = 0, col = 0;
	int baz = 0;
	char *prog = argv[0];

	if ( argc == 1 ) {
		printf ( "Usage: %s: -g ROWxCOL \n", argv[0] );
		printf ( "Grid size to use must be specified\n" );
		exit (0);
	}
		
	while ( argc > 1 && argv[1][0] == '-' ) {
		switch ( argv[1][1] ) {
		case 'g':
			argv++, argc--;
			row = atoi ( argv[1] );
			while ( argv[1][baz] != 'x' ) baz++;
			baz++ ;
			col = atoi ( & argv[1][baz] );
			break;

		case 'h':
			printf ( "Usage: %s: -g ROWxCOL \n", argv[0] );
			exit (0);

		default:
			printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
		}
		argv++, argc--;
	}

	printf ( "%s: Run distributed DFT for all sizes using grid = %dx%d\n", prog, row, col );
	
  MPI_Init(&argc, &argv);

  int commRank;
  int p;
  
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  const int root = 0;

  bool is_embedded = true; // TODO: get embedded testing working

  fftx::point_t<5> *wcube = fftx_distdft_embed_gpu_QuerySizes();
  if (wcube == NULL) {
    printf ( "Failed to get list of available sizes\n");
    exit (-1);
  }
  
  for (fftx::point_t<5> *req = wcube;
       req->x[0] != 0 && req->x[1] != 0 && req->x[2] != 0 && req->x[3] != 0 && req->x[4] != 0;
       req++)
  {
	  int r = req->x[0], c = req->x[1];							//  grid (r x c) for this entry is first 2 values from point_t<>
	  int Mo = req->x[2], No = req->x[3], Ko = req->x[4];		//  Output cube size is remaining values
	  int Mi = Mo / 2, Ni = No / 2, Ki = Ko / 2;				//  Input cube dimensions are half output dims

	  if ( ( r != row ) || ( c != col ) ) continue;				//  Different grid, skip it

	  printf ( "Test cube: input size = [ %d, %d, %d ], output size = [ %d, %d, %d ]\n", Mi, Ni, Ki, Mo, No, Ko );

	int check = 1;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //    fftx::point_t<5> req = ({r, c, Mo, No, Ko}) ;
    transformTuple_t *tptr = fftx_distdft_embed_gpu_Tuple(*req);  
    
    if (tptr != NULL){
      //init_mddft3d();
      (* tptr->initfp)();

      // initialize data to random values in the range (-1, 1).
      complex<double> *in_buff = new complex<double>[Mo*Ni*Ki/p];
      
      // assume data is padded on x as input.
      for (int k = 0; k < Ki/r; k++) {
	for (int j = 0; j < Ni/c; j++) {
	  for (int i = 0; i < Mo; i++) {
	    in_buff[
		    k * (Ni/c)*Mo +
		    j *        Mo +
		    i
		    ] = ( // embedded
			 (Mi/2 <= i &&
			  i < 3*Mi/2 )||
			 !is_embedded
			  ) ?
	      complex<double>(
			      1 - ((double) rand()) / (double) (RAND_MAX/2),
			      1 - ((double) rand()) / (double) (RAND_MAX/2)
			      )
	      :
	      complex<double>(0, 0);
	  }
	}
      }           
      
      complex<double> *in_buffer = NULL;
      DEVICE_ERROR_T err = DEVICE_MALLOC(&in_buffer, Mo*Ni*Ki/p * sizeof(complex<double>));
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
      
      err = DEVICE_MEM_COPY(
			    in_buffer,
			    in_buff,
			    Mo*Ni*Ki/p * sizeof(complex<double>),
			    MEM_COPY_HOST_TO_DEVICE
			    );
      if (err != DEVICE_SUCCESS) {
	cout << "DEVICE_MEM_COPY failed\n" << endl;
	exit(-1);
      }
      
      DEVICE_SYNCHRONIZE();      
      
      for (int t = 0; t < 1; t++) {
	double start_time = MPI_Wtime();
	
	//  mddft3d((double*)out_buffer, (double*)in_buffer);
	(* tptr->runfp)((double*)out_buffer, (double*)in_buffer);
	
	double end_time = MPI_Wtime();
	
	double min_time    = min_diff(start_time, end_time, MPI_COMM_WORLD);
	double max_time    = max_diff(start_time, end_time, MPI_COMM_WORLD);
	
	if (commRank == 0) {
	  printf("%lf %lf\n", min_time, max_time);
	}
      }
      
  // Check correctness against local compute.
#if CHECK_WITH_CUFFT
  {
    
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
	       in_buff +  0, // (is_embedded ? Mi/2*Ni*Ki/p: 0),
	       Mo*Ni*Ki/p,
	       MPI_DOUBLE_COMPLEX,
	       recv_buff,
	       Mo*Ni*Ki/p,
	       MPI_DOUBLE_COMPLEX,
	       root,
	       MPI_COMM_WORLD
	       );
    // old root layout was     [xl, yl, zl, zr, xr, yr]
    // new root layout will be [xl, xr, yl, zl, yr, zr]
    if (commRank == root) {
      // repack to XYZ layout. col major, M fastest, then N, K.
      // r divides M, c divides N.
      for (int jr = 0; jr < c; jr++) {
        for (int ir = 0; ir < r; ir++) {
          for (int kr = 0; kr < c; kr++) {
            for (int kl = 0; kl < Ki/c; kl++) {
              for (int jl = 0; jl < Ni/c; jl++) {
                for (int il = 0; il < Mo/r; il++) {
                  in_root[
			  ((is_embedded ? Ki/2 : 0) + kr * (Ki/c) + kl) * No*Mo +
			  ((is_embedded ? Ni/2 : 0) + jr * (Ni/c) + jl) *    Mo +
			  (                         + ir * (Mo/r) + il)
			  ] = recv_buff[
					kr * c*(Ki/c)*(Ni/c)*Mo +
					jr *   (Ki/c)*(Ni/c)*Mo +
					kl *          (Ni/c)*Mo +
					jl *                 Mo +
					ir * (Mo/r) + il
					];
                }
              }
            }
          }
        }
      }

      DEVICE_MEM_COPY(device_fft_in_root, in_root, Mo*No*Ko * sizeof(complex<double>), MEM_COPY_HOST_TO_DEVICE);

      DEVICE_FFT_HANDLE plan;
      DEVICE_FFT_RESULT res;

      res = DEVICE_FFT_CREATE(&plan);
      if (res != DEVICE_FFT_SUCCESS) { printf ("*Create failed\n"); }
      size_t worksize[1]; // 1 GPU.
      res = DEVICE_FFT_MAKE_PLAN_3D(
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
    );
    // old root layout was     [zl, xl, yl, yr, zr, xr]
    // new root layout will be [zl, zr, xl, yl, xr, yr]

    if (commRank == root) {
      // repack to XYZ layout.
      for (int ir = 0; ir < c; ir++) {
        for (int kr = 0; kr < r; kr++) {
          for (int jr = 0; jr < c; jr++) {
            for (int jl = 0; jl < No/c; jl++) {
              for (int il = 0; il < Mo/c; il++) {
                for (int kl = 0; kl < Ko/r; kl++) {
                  fftx_out_root[
                    (kr * (Ko/r) + kl) * No*Mo +
                    (jr * (No/c) + jl) *    Mo +
                    (ir * (Mo/c) + il)
                  ] = recv_buff[
                    jr * r*(No/c)*(Mo/r)*Ko +
                    ir *   (No/c)*(Mo/r)*Ko +
                    jl *          (Mo/r)*Ko +
                    il *                 Ko +
                    kr * (Ko/r) + kl

                  ];
                }
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

  DEVICE_FREE(in_buffer);
  DEVICE_FREE(out_buffer);  
  delete[] in_buff;
  
  //  destroy_mddft3d();
  (* tptr->destroyfp)();
    }
  else {
  	  printf ( "Distributed library entry for req = { %d, %d, %d, %d, %d } not found ... skipping\n",
  			   req->x[0], req->x[1], req->x[2], req->x[3], req->x[4] );
    }

  }
  
  MPI_Finalize();
  
  return 0;
}
