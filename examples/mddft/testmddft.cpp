#include "mddft.fftx.codegen.hpp"
#include "imddft.fftx.codegen.hpp"
#include "test_plan.h"
#include <string>
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#include "device_macros.h"
#endif

int main(int argc, char* argv[])
{
  printf("%s: Entered test program\n call mddft::init()\n", argv[0]);

  int iterations = 20;
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

  double* mddft_cpu = new double[iterations];
  double* imddft_cpu = new double[iterations];
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  // additional code for GPU programs
  float* mddft_gpu = new float[iterations];
  float* imddft_gpu = new float[iterations];
  std::string descrip = "CPU and GPU";

  std::complex<double> * bufferDevicePtr;
  std::complex<double> * inputDevicePtr;
  std::complex<double> * outputDevicePtr;
  DEVICE_MALLOC(&bufferDevicePtr,
                test_plan::domain.size()*sizeof(std::complex<double>)*2);
  inputDevicePtr = bufferDevicePtr;
  outputDevicePtr = bufferDevicePtr + test_plan::domain.size();
  DEVICE_MEM_COPY(inputDevicePtr, inputHost.m_data.local(),
                  test_plan::domain.size()*sizeof(std::complex<double>),
                  MEM_COPY_HOST_TO_DEVICE);
  fftx::array_t<3,std::complex<double>> inputDevice(fftx::global_ptr<std::complex<double>>(inputDevicePtr,0,1), test_plan::domain);
  fftx::array_t<3,std::complex<double>> outputDevice(fftx::global_ptr<std::complex<double>>(outputDevicePtr,0,1), test_plan::domain);

  fftx::array_t<3,std::complex<double>>& input = inputDevice;
  fftx::array_t<3,std::complex<double>>& output = outputDevice;
  // end special code for GPU
#else
  std::string descrip = "CPU";
  fftx::array_t<3,std::complex<double>>& input = inputHost;
  fftx::array_t<3,std::complex<double>>& output = outputHost;
#endif  

  printf("call mddft::init()\n");
  mddft::init();

  printf("call mddft::transform()\n");

  for (int itn = 0; itn < iterations; itn++)
    {
      mddft::transform(input, output);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      mddft_gpu[itn] = mddft::GPU_milliseconds;
#endif
      mddft_cpu[itn] = mddft::CPU_milliseconds;
    }

  mddft::destroy();

  printf("call imddft::init()\n");
  imddft::init();

  printf("call imddft::transform()\n");
  for (int itn = 0; itn < iterations; itn++)
    {
      imddft::transform(input, output);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      imddft_gpu[itn] = imddft::GPU_milliseconds;
#endif
      imddft_cpu[itn] = imddft::CPU_milliseconds;
    }

  imddft::destroy();

  printf("Times in milliseconds for %s on mddft on %d trials of size %d %d %d:\n",
         descrip.c_str(), iterations, fftx_nx, fftx_ny, fftx_nz);
  for (int itn = 0; itn < iterations; itn++)
    {
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        printf("%.7e  %.7e\n", mddft_cpu[itn], mddft_gpu[itn]);
#else
      printf("%.7e\n", mddft_cpu[itn]);
#endif
    }

  printf("Times in milliseconds for %s on imddft on %d trials of size %d %d %d:\n",
         descrip.c_str(), iterations, fftx_nx, fftx_ny, fftx_nz);
  for (int itn = 0; itn < iterations; itn++)
    {
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      printf("%.7e  %.7e\n", imddft_cpu[itn], imddft_gpu[itn]);
#else
      printf("%.7e\n", imddft_cpu[itn]);
#endif
    }

  delete[] mddft_cpu;
  delete[] imddft_cpu;
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  delete[] mddft_gpu;
  delete[] imddft_gpu;
#endif

  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
