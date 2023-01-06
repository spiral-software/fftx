#include "fftx3.hpp"
#include "interface.hpp"
#include "mddftObj.hpp"
#include "imddftObj.hpp"
#include <string>
#include <fstream>

#if defined FFTX_CUDA
#include "cudabackend.hpp"
#endif
#if defined FFTX_HIP
#include "hipbackend.hpp"
#endif
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#include "device_macros.h"
#endif

int main(int argc, char* argv[])
{
  // std::cout <<"this is my program X3\n";
  // printf("%s: Entered test program\n call mddft::init()\n", argv[0]);
  int iterations = 2;
  int mm = 24, nn = 32, kk = 40; // cube dimensions
  char *prog = argv[0];
  int baz = 0;
  while ( argc > 1 && argv[1][0] == '-' ) {
    switch ( argv[1][1] ) {
    case 'i':
    argv++, argc--;
    iterations = atoi ( argv[1] );
    break;
    case 's':
    argv++, argc--;
    mm = atoi ( argv[1] );
    while ( argv[1][baz] != 'x' ) baz++;
    baz++ ;
    nn = atoi ( & argv[1][baz] );
    while ( argv[1][baz] != 'x' ) baz++;
    baz++ ;
    kk = atoi ( & argv[1][baz] );
    break;
    case 'h':
      printf ( "Usage: %s: [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]\n", argv[0] );
      exit (0);
    default:
      printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
    }
    argv++, argc--;
  }
  // std::cout << "left while loop\n";
  std::cout << mm << " " << nn << " " << kk << std::endl;
  std::vector<int> sizes{mm,nn,kk};
  fftx::box_t<3> domain ( point_t<3> ( { { 1, 1, 1 } } ),
  point_t<3> ( { { mm, nn, kk } } ));

	fftx::array_t<3,std::complex<double>> inputHost(domain);
	fftx::array_t<3,std::complex<double>> outputHost(domain);

	forall([](std::complex<double>(&v), const fftx::point_t<3>& p) {
			v=std::complex<double>(2.0,0.0);
		},inputHost);

    //initDevice();
#if defined FFTX_CUDA
    CUdevice cuDevice;
    CUcontext context;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    //  CUdeviceptr  dX, dY, dsym;
	std::complex<double> *dX, *dY, *dsym;
#endif
#if defined FFTX_HIP
    hipDeviceptr_t  dX, dY, dsym;
#endif

// std::cout << "allocating memory\n";
// CUDA_SAFE_CALL(cuMemAlloc(&dX, inputHost.m_domain.size() * sizeof(std::complex<double>)));
// std::cout << "allocated X\n";
// CUDA_SAFE_CALL(cuMemcpyHtoD(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(std::complex<double>)));
// std::cout << "copied X\n";
// CUDA_SAFE_CALL(cuMemAlloc(&dY, outputHost.m_domain.size() * sizeof(std::complex<double>)));
// std::cout << "allocated Y\n";
// // //CUDA_SAFE_CALL(cuMemcpyHtoD(dY, Y, 64* sizeof(double)));
// CUDA_SAFE_CALL(cuMemAlloc(&dsym, outputHost.m_domain.size() * sizeof(std::complex<double>)));
// std::cout << "allocating memory\n" << 30720 << "\n";
// CUDA_SAFE_CALL(hipMalloc((void **)&dX, inputHost.m_domain.size() * sizeof(std::complex<double>)));
// std::cout << "allocated X\n";
// CUDA_SAFE_CALL(hipMemcpy(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(std::complex<double>), hipMemcpyHostToDevice));
// std::cout << "copied X\n";
// CUDA_SAFE_CALL(hipMalloc((void **)&dY, outputHost.m_domain.size() * sizeof(std::complex<double>)));
// std::cout << "allocated Y\n";
// // //HIP_SAFE_CALL(cuMemcpyHtoD(dY, Y, 64* sizeof(double)));
// CUDA_SAFE_CALL(hipMalloc((void **)&dsym,  outputHost.m_domain.size()*  sizeof(std::complex<double>)));

	std::cout << "allocating memory\n";
	DEVICE_MALLOC((void **)&dX, inputHost.m_domain.size() * sizeof(std::complex<double>));
	std::cout << "allocated X\n";
	DEVICE_MEM_COPY(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(std::complex<double>), MEM_COPY_HOST_TO_DEVICE);
	std::cout << "copied X\n";
	DEVICE_MALLOC((void **)&dY, outputHost.m_domain.size() * sizeof(std::complex<double>));
	std::cout << "allocated Y\n";
	//  HIP_SAFE_CALL(cuMemcpyHtoD(dY, Y, 64* sizeof(double)));
	DEVICE_MALLOC((void **)&dsym,  outputHost.m_domain.size()*  sizeof(std::complex<double>));

	double* mddft_cpu = new double[iterations];
	double* imddft_cpu = new double[iterations];
	// #if defined (FFTX_CUDA) || defined(FFTX_HIP)
	// additional code for GPU programs
	float* mddft_gpu = new float[iterations];
	float* imddft_gpu = new float[iterations];
	std::string descrip = "CPU and GPU";
#if defined FFTX_CUDA
	std::vector<void*> args{&dY,&dX,&dsym};
#endif
#if defined FFTX_HIP
	std::vector<void*> args{dY,dX,dsym};
#endif
//   std::complex<double> * bufferDevicePtr;
//   std::complex<double> * inputDevicePtr;
//   std::complex<double> * outputDevicePtr;
//   DEVICE_MALLOC(&bufferDevicePtr,
//                 test_plan::domain.size()*sizeof(std::complex<double>)*2);
//   inputDevicePtr = bufferDevicePtr;
//   outputDevicePtr = bufferDevicePtr + test_plan::domain.size();
//   DEVICE_MEM_COPY(inputDevicePtr, inputHost.m_data.local(),
//                   test_plan::domain.size()*sizeof(std::complex<double>),
//                   MEM_COPY_HOST_TO_DEVICE);
//   fftx::array_t<3,std::complex<double>> inputDevice(fftx::global_ptr<std::complex<double>>(inputDevicePtr,0,1), test_plan::domain);
//   fftx::array_t<3,std::complex<double>> outputDevice(fftx::global_ptr<std::complex<double>>(outputDevicePtr,0,1), test_plan::domain);

//   fftx::array_t<3,std::complex<double>>& input = inputDevice;
//   fftx::array_t<3,std::complex<double>>& output = outputDevice;
//   // end special code for GPU
// #else
  // std::string descrip = "CPU";
  // fftx::array_t<3,std::complex<double>>& input = inputHost;
  // fftx::array_t<3,std::complex<double>>& output = outputHost;
//#endif  

  // std::vector<fftx::array_t<3,std::complex<double>>> inList;
  // std::vector<fftx::array_t<3,std::complex<double>>> outList;

  // inList.push_back(inputHost);
  // outList.push_back(outputHost);


  //MDDFTProblem mdp(inList, outList);
  //std::cout << *((int*)args.at(3)) << std::endl;
  MDDFTProblem mdp(args, sizes);


  printf("call mddft::init()\n");
  // mddft::init();

  printf("call mddft::transform()\n");

  for (int itn = 0; itn < iterations; itn++)
    {
      mdp.transform();
      //gatherOutput(outputHost, args);
	  DEVICE_MEM_COPY ( outputHost.m_data.local(), &dY,
						outputHost.m_domain.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
	  
// #ifdef FFTX_HIP
//       mddft_gpu[itn] = mddft::GPU_milliseconds;
// #endif
      mddft_cpu[itn] = mdp.getTime();
    }
printf("finished the code\n");
  // mddft::destroy();

  printf("call imddft::init()\n");
  // imddft::init();

  IMDDFTProblem imdp(args, sizes);

  printf("call imddft::transform()\n");
  for (int itn = 0; itn < iterations; itn++)
    {
      imdp.transform();
      //gatherOutput(outputHost, args);
	  DEVICE_MEM_COPY ( outputHost.m_data.local(), &dY,
						outputHost.m_domain.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );

      //imddft::transform(input, output);
// #ifdef FFTX_HIP
//       imddft_gpu[itn] = imddft::GPU_milliseconds;
// #endif
      imddft_cpu[itn] = imdp.getTime();
    }

//   imddft::destroy();

  printf("Times in milliseconds for %s on mddft on %d trials of size %d %d %d:\n",
         descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
  for (int itn = 0; itn < iterations; itn++)
    {
// #ifdef FFTX_HIP
//         printf("%.7e  %.7e\n", mddft_cpu[itn], mddft_gpu[itn]);
// #else
      printf("%.7e\n", mddft_cpu[itn]);
// #endif
    }

  printf("Times in milliseconds for %s on imddft on %d trials of size %d %d %d:\n",
         descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
  for (int itn = 0; itn < iterations; itn++)
    {
// #ifdef FFTX_HIP
//       printf("%.7e  %.7e\n", imddft_cpu[itn], imddft_gpu[itn]);
// #else
      printf("%.7e\n", imddft_cpu[itn]);
// #endif
    }

  delete[] mddft_cpu;
  delete[] imddft_cpu;
#ifdef FFTX_MPI
  delete[] mddft_gpu;
  delete[] imddft_gpu;
#endif

  printf("%s: All done, exiting\n", argv[0]);
  
  return 0;
}
