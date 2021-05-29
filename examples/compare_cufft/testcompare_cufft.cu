#include <stdio.h>

#ifdef FFTX_HIP
#include <hip/hip_runtime.h>
#include <hipfft.h>
#include "rocfft.h"
#endif

#include <cufft.h>
#include "spiralmddft.fftx.codegen.hpp"
#include "test_comp.h"
#include "fftx3utilities.h"

#include <chrono>

template<typename T>
T minSubarray(const T* arr, int lo, int hi)
{
  T val = arr[lo];
  for (int i = lo+1; i <= hi; i++)
    {
      if (arr[i] < val)
        {
          val = arr[i];
        }
    }
  return val;
}


template<typename T>
T maxSubarray(const T* arr, int lo, int hi)
{
  T val = arr[lo];
  for (int i = lo+1; i <= hi; i++)
    {
      if (arr[i] > val)
        {
          val = arr[i];
        }
    }
  return val;
}


template<typename T>
T avgSubarray(const T* arr, int lo, int hi)
{
  T tot = 0.;
  int len = 0;
  for (int i = lo; i <= hi; i++)
    {
      tot += arr[i];
      len++;
    }
  T avg = tot / (len * 1.);
  return avg;
}


int main(int argc, char* argv[])
{
  printf("Usage:  %s [verbosity=0] [iterations=20]\n", argv[0]);
  printf("verbosity 0 for avg times, 1 for min/max, 2 for all iterations, 3 for errors\n");
  int verbosity = 0;
  int iterations = 20;
  if (argc > 1)
    {
      verbosity = atoi(argv[1]);
      if (argc > 2)
        {
          iterations = atoi(argv[2]);
        }
    }
  printf("Running with verbosity %d and %d iterations\n", verbosity, iterations);

  /*
    Allocate space for arrays, and set input array.
  */
  fftx::array_t<3,std::complex<double>> inputArrayHost(test_comp::domain);
  size_t npts = test_comp::domain.size();
  forall([](std::complex<double>(&v), const fftx::point_t<3>& p)
         {
           v = std::complex<double>(1. - ((double) rand()) / (double) (RAND_MAX/2),
                                    1. - ((double) rand()) / (double) (RAND_MAX/2));                                 
         }, inputArrayHost);
  std::complex<double>* inputHostPtr = inputArrayHost.m_data.local();
  // additional code for GPU programs
  std::complex<double> * bufferDevicePtr;
  std::complex<double> * inputDevicePtr;
  std::complex<double> * outputSpiralDevicePtr;
  std::complex<double> * outputCufftDevicePtr;
  cudaMalloc(&bufferDevicePtr, 3 * npts*sizeof(std::complex<double>));
  std::complex<double> * thisDevicePtr = bufferDevicePtr;
  inputDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  outputSpiralDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  outputCufftDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  cudaMemcpy(inputDevicePtr, inputHostPtr, // dest, source
             npts*sizeof(std::complex<double>), // bytes
             cudaMemcpyHostToDevice); // type
  
  fftx::array_t<3,std::complex<double>> inputArrayDevice(fftx::global_ptr<std::complex<double>>(inputDevicePtr,0,1), test_comp::domain);
  fftx::array_t<3,std::complex<double>> outputArraySpiralDevice(fftx::global_ptr<std::complex<double>>(outputSpiralDevicePtr,0,1), test_comp::domain);

  /*
    Set up timers for cufft.
   */
  cudaEvent_t* startCufft = new cudaEvent_t[iterations];
  cudaEvent_t* stopCufft  = new cudaEvent_t[iterations];
  for (int itn = 0; itn < iterations; itn++ )
    {
      cudaEventCreate(&(startCufft[itn]));
      cudaEventCreate(&(stopCufft[itn]));
    }

  /*
    Get plan for cufft.
  */
  if (verbosity >= 1)
    {
      printf("get cufft plan\n");
    }
  cufftHandle plan;
  {
    auto result =
      cufftPlan3d(&plan, fftx_nx, fftx_ny, fftx_nz, CUFFT_Z2Z);
    if (result != CUFFT_SUCCESS)
      {
        exit(-1);
      }
  }

  /*
    Time iterations of complex-to-complex cufft calls using the plan.
   */
  if (verbosity >= 1)
    {
      printf("call cufftExecZ2Z %d times\n", iterations);
    }

  for (int itn = 0; itn < iterations; itn++ )
    {
      cudaEventRecord(startCufft[itn]);
      auto result = 
        cufftExecZ2Z(plan,
                     (cufftDoubleComplex *) inputDevicePtr,
                     (cufftDoubleComplex *) outputCufftDevicePtr,
                     CUFFT_FORWARD);
      cudaEventRecord(stopCufft[itn]);
      cudaEventSynchronize(stopCufft[itn]);
      if (result != CUFFT_SUCCESS)
        {
          printf("cufftExecZ2Z launch failed\n");
          exit(-1);
        }
    }
  cufftDestroy(plan);

  float* cufft_gpu = new float[iterations];
  for (int itn = 0; itn < iterations; itn++)
    {
      cudaEventElapsedTime(&(cufft_gpu[itn]), startCufft[itn], stopCufft[itn]);
    }
  delete[] startCufft;
  delete[] stopCufft;
 
  cudaDeviceSynchronize();

  /*
    Time iterations of complex-to-complex MDDFT with SPIRAL-generated code.
   */
  double* spiral_cpu = new double[iterations];
  float* spiral_gpu = new float[iterations];
  if (verbosity >= 1)
    {
      printf("call spiralmddft::init()\n");
    }
  spiralmddft::init();

  if (verbosity >= 1)
    {
      printf("call spiralmddft::transform() %d times\n", iterations);
    }

  for (int itn = 0; itn < iterations; itn++)
    {
      spiralmddft::transform(inputArrayDevice, outputArraySpiralDevice);
      spiral_gpu[itn] = spiralmddft::GPU_milliseconds;
      spiral_cpu[itn] = spiralmddft::CPU_milliseconds;
    }

  spiralmddft::destroy();

  /*
    Check that cufft and SPIRAL give the same results on last iteration.
  */
  cufftDoubleComplex* outputSpiralHostPtr = new cufftDoubleComplex[npts];
  cufftDoubleComplex* outputCufftHostPtr  = new cufftDoubleComplex[npts];
  cudaMemcpy(outputSpiralHostPtr, outputSpiralDevicePtr, // dest, source
             npts*sizeof(cufftDoubleComplex), // bytes
             cudaMemcpyDeviceToHost); // type
  cudaMemcpy(outputCufftHostPtr, outputCufftDevicePtr, // dest, source
             npts*sizeof(cufftDoubleComplex), // bytes
             cudaMemcpyDeviceToHost); // type

  cudaFree(bufferDevicePtr);

  const double tol = 1.e-7;
  bool match = true;
  double maxDiff = 0.;
  {
    for (size_t ind = 0; ind < npts; ind++)
      {
        auto outputSpiralPoint = outputSpiralHostPtr[ind];
        auto outputCufftPoint = outputCufftHostPtr[ind];
        double diffReal = outputSpiralPoint.x - outputCufftPoint.x;
        double diffImag = outputSpiralPoint.y - outputCufftPoint.y;
        updateMaxAbs(maxDiff, diffReal);
        updateMaxAbs(maxDiff, diffImag);
        bool matchPoint =
          (std::abs(diffReal) < tol) && (std::abs(diffImag) < tol);
        if (!matchPoint)
          {
            match = false;
            if (verbosity >= 3)
              {
                point_t<3> pt = pointFromPositionBox(ind, test_comp::domain);
                printf("error at (%d,%d,%d): SPIRAL %f+i*%f, cufft %f+i*%f\n",
                       pt[0], pt[1], pt[2],
                       outputSpiralPoint.x, outputSpiralPoint.y,
                       outputCufftPoint.x, outputCufftPoint.y);
              }
          }
      }
  }
  delete[] outputSpiralHostPtr;
  delete[] outputCufftHostPtr;
  if (match)
    {
      printf("YES, results match.  Max difference %11.5e\n", maxDiff);
    }
  else
    {
      printf("NO, results do not match.\n");
    }

  /*
    Get minimum, maximum, and average timings of iterations.
    First, with the first iteration excluded.
   */
  auto gpuMin = minSubarray(spiral_gpu, 1, iterations-1);
  auto gpuMax = maxSubarray(spiral_gpu, 1, iterations-1);
  auto gpuAvg = avgSubarray(spiral_gpu, 1, iterations-1);

  auto cpuMin = minSubarray(spiral_cpu, 1, iterations-1);
  auto cpuMax = maxSubarray(spiral_cpu, 1, iterations-1);
  auto cpuAvg = avgSubarray(spiral_cpu, 1, iterations-1);

  auto cufftMin = minSubarray(cufft_gpu, 1, iterations-1);
  auto cufftMax = maxSubarray(cufft_gpu, 1, iterations-1);
  auto cufftAvg = avgSubarray(cufft_gpu, 1, iterations-1);

  printf("Size %d %d %d, over %d iterations\n",
         fftx_nx, fftx_ny, fftx_nz, iterations);
  if (verbosity >= 2)
    {
      printf("itn    Spiral CPU     Spiral GPU     cufft\n");
      for (int itn = 0; itn < iterations; itn++)
        {
          printf("%3d    %.7e  %.7e  %.7e\n",
                 itn, spiral_cpu[itn], spiral_gpu[itn], cufft_gpu[itn]);
        }
    }
  else
    {
      printf("       Spiral CPU     Spiral GPU     cufft\n");
    }

  printf("NOT INCLUDING FIRST RUN:\n");
  printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, cufftAvg);
  if (verbosity >= 1)
    {
      printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, cufftMin);
      printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, cufftMax);
    }

  gpuMin = minSubarray(spiral_gpu, 0, iterations-1);
  gpuMax = maxSubarray(spiral_gpu, 0, iterations-1);
  gpuAvg = avgSubarray(spiral_gpu, 0, iterations-1);

  cpuMin = minSubarray(spiral_cpu, 0, iterations-1);
  cpuMax = maxSubarray(spiral_cpu, 0, iterations-1);
  cpuAvg = avgSubarray(spiral_cpu, 0, iterations-1);

  cufftMin = minSubarray(cufft_gpu, 0, iterations-1);
  cufftMax = maxSubarray(cufft_gpu, 0, iterations-1);
  cufftAvg = avgSubarray(cufft_gpu, 0, iterations-1);
  
  delete[] spiral_cpu;
  delete[] spiral_gpu;
  delete[] cufft_gpu;
  
  printf("INCLUDING FIRST RUN:\n");
  printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, cufftAvg);
  if (verbosity >= 1)
    {
      printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, cufftMin);
      printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, cufftMax);
    }

  printf("%s: All done, exiting\n", argv[0]);

  return 0;
}
