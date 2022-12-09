#include "mddft.fftx.codegen.hpp"
#include "imddft.fftx.codegen.hpp"
#include "test_plan.h"
#include "interface.hpp"
#include "mddftObj.hpp"
#include "imddftObj.hpp"
//#include "data_interaction.hpp"
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
  std::cout <<"this is my program X3\n";
  printf("%s: Entered test program\n call mddft::init()\n", argv[0]);

  int iterations = 2;
  if (argc > 1)
    {
      iterations = atoi(argv[1]);
    }
  // Does not work:
  //  fftx::box_t<3> domain(lo, hi);

  fftx::array_t<3,std::complex<double>> inputHost(test_plan::domain);
  fftx::array_t<3,std::complex<double>> outputHost(test_plan::domain);

  forall([](std::complex<double>(&v), const fftx::point_t<3>& p)
         {
           v=std::complex<double>(2.0,0.0);
         },inputHost);

//initDevice();
  #if defined FFTX_CUDA
    CUdevice cuDevice;
    CUcontext context;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    CUdeviceptr  dX, dY, dsym;
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
std::cout << "allocating memory\n" << 30720 << "\n";
DEVICE_MALLOC((void **)&dX, inputHost.m_domain.size() * sizeof(std::complex<double>));
std::cout << "allocated X\n";
DEVICE_MEM_COPY(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(std::complex<double>), MEM_COPY_HOST_TO_DEVICE);
std::cout << "copied X\n";
DEVICE_MALLOC((void **)&dY, outputHost.m_domain.size() * sizeof(std::complex<double>));
std::cout << "allocated Y\n";
// //HIP_SAFE_CALL(cuMemcpyHtoD(dY, Y, 64* sizeof(double)));
DEVICE_MALLOC((void **)&dsym,  outputHost.m_domain.size()*  sizeof(std::complex<double>));

  double* mddft_cpu = new double[iterations];
  double* imddft_cpu = new double[iterations];
// #if defined (FFTX_CUDA) || defined(FFTX_HIP)
//   // additional code for GPU programs
  float* mddft_gpu = new float[iterations];
  float* imddft_gpu = new float[iterations];
  std::string descrip = "CPU and GPU";
#if defined FFTX_CUDA
std::vector<void*> args{&dY,&dX,&dsym}
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
  MDDFTProblem mdp(args);


  printf("call mddft::init()\n");
  mddft::init();

  printf("call mddft::transform()\n");

  for (int itn = 0; itn < iterations; itn++)
    {
      mdp.transform();
      //gatherOutput(outputHost, args);
      hipMemcpy(outputHost.m_data.local(), &dY,  outputHost.m_domain.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost);

// #ifdef FFTX_HIP
//       mddft_gpu[itn] = mddft::GPU_milliseconds;
// #endif
      mddft_cpu[itn] = mdp.getTime();
    }
printf("finished the code\n");
  mddft::destroy();

  printf("call imddft::init()\n");
  // imddft::init();

  IMDDFTProblem imdp(args);

  printf("call imddft::transform()\n");
  for (int itn = 0; itn < iterations; itn++)
    {
      imdp.transform();
      //gatherOutput(outputHost, args);
      hipMemcpy(outputHost.m_data.local(), &dY,  outputHost.m_domain.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost);

      //imddft::transform(input, output);
// #ifdef FFTX_HIP
//       imddft_gpu[itn] = imddft::GPU_milliseconds;
// #endif
      imddft_cpu[itn] = imdp.getTime();
    }

//   imddft::destroy();

  printf("Times in milliseconds for %s on mddft on %d trials of size %d %d %d:\n",
         descrip.c_str(), iterations, fftx_nx, fftx_ny, fftx_nz);
  for (int itn = 0; itn < iterations; itn++)
    {
// #ifdef FFTX_HIP
//         printf("%.7e  %.7e\n", mddft_cpu[itn], mddft_gpu[itn]);
// #else
      printf("%.7e\n", mddft_cpu[itn]);
// #endif
    }

  printf("Times in milliseconds for %s on imddft on %d trials of size %d %d %d:\n",
         descrip.c_str(), iterations, fftx_nx, fftx_ny, fftx_nz);
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
