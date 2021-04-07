

#include "mddft.fftx.codegen.hpp"
#include "imddft.fftx.codegen.hpp"
#include "test_plan.h"

#include <chrono>

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
  
  fftx::array_t<3,std::complex<double>> inputH(test_plan::domain);
  fftx::array_t<3,std::complex<double>> outputH(test_plan::domain);

  forall([](std::complex<double>(&v), const fftx::point_t<3>& p)
         {
           v=std::complex<double>(2.0,0.0);
         },inputH);
  // additional code for GPU programs
  std::complex<double> * bufferPtr;
  std::complex<double> * inputPtr;
  std::complex<double> * outputPtr;
  cudaMalloc(&bufferPtr, test_plan::domain.size()*sizeof(std::complex<double>)*2);
  inputPtr = bufferPtr;
  outputPtr = bufferPtr + test_plan::domain.size();
  cudaMemcpy(inputPtr, inputH.m_data.local(), test_plan::domain.size()*sizeof(std::complex<double>),
             cudaMemcpyHostToDevice);
  fftx::array_t<3,std::complex<double>> input(fftx::global_ptr<std::complex<double>>(inputPtr,0,1), test_plan::domain);
  fftx::array_t<3,std::complex<double>> output(fftx::global_ptr<std::complex<double>>(outputPtr,0,1), test_plan::domain);
  //  end special code for GPU
  
  double* mddft_cpu = new double[iterations];
  float* mddft_gpu = new float[iterations];
  printf("call mddft::init()\n");
  mddft::init();

  printf("call mddft::transform()\n");

  for (int itn = 0; itn < iterations; itn++)
    {
      mddft::transform(input, output);
      mddft_gpu[itn] = mddft::GPU_milliseconds;
      mddft_cpu[itn] = mddft::CPU_milliseconds;
    }

  // printf("mddft for size 32 32 32 took  %.7e milliseconds wrt CPU\n", mddft::CPU_milliseconds);
  // printf("mddft for size %d %d %d took  %.7e milliseconds wrt CPU\n",
  // nx, ny, nz, mddft::CPU_milliseconds);
  // printf("mddft for size 32 32 32 took  %.7e milliseconds wrt GPU\n", mddft::GPU_milliseconds);
  // printf("mddft for size %d %d %d took  %.7e milliseconds wrt CPU\n",
  // nx, ny, nz, mddft::GPU_milliseconds);

  mddft::destroy();

  double* imddft_cpu = new double[iterations];
  float* imddft_gpu = new float[iterations];
  printf("call imddft::init()\n");
  imddft::init();

  printf("call imddft::transform()\n");
  for (int itn = 0; itn < iterations; itn++)
    {
      imddft::transform(input, output);
      imddft_gpu[itn] = imddft::GPU_milliseconds;
      imddft_cpu[itn] = imddft::CPU_milliseconds;
    }

  imddft::destroy();

  printf("Times in milliseconds for CPU and GPU on mddft on %d trials of size %d %d %d:\n",
         iterations, test_plan::nx, test_plan::ny, test_plan::nz);
  for (int itn = 0; itn < iterations; itn++)
    {
      printf("%.7e  %.7e\n", mddft_cpu[itn], mddft_gpu[itn]);
    }
  delete[] mddft_cpu;
  delete[] mddft_gpu;

  printf("Times in milliseconds for CPU and GPU on imddft on %d trials of size %d %d %d:\n",
         iterations, test_plan::nx, test_plan::ny, test_plan::nz);
  for (int itn = 0; itn < iterations; itn++)
    {
      printf("%.7e  %.7e\n", imddft_cpu[itn], imddft_gpu[itn]);
    }

  delete[] imddft_cpu;
  delete[] imddft_gpu;

  printf("%s: All done, exiting\n", argv[0]);

  return 0;
}
