#include <stdio.h>

#ifdef FFTX_HIP
#include <hip/hip_runtime.h>
#include <hipfft.h>
#include "rocfft.h"
#endif

#include "fftx_mddft_public.h"
#include "fftx_imddft_public.h"
#include "fftx_rconv_public.h"

#include "mddft.fftx.precompile.hpp"
#include "imddft.fftx.precompile.hpp"
#include "rconv.fftx.precompile.hpp"

#include "fftx3utilities.h"

#include "test_comp.h"

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
  std::complex<double> * outputHipfftDevicePtr;
  hipMalloc(&bufferDevicePtr, 3 * npts*sizeof(std::complex<double>));
  std::complex<double> * thisDevicePtr = bufferDevicePtr;
  inputDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  outputSpiralDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  outputHipfftDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  hipMemcpy(inputDevicePtr, inputHostPtr, // dest, source
             npts*sizeof(std::complex<double>), // bytes
             hipMemcpyHostToDevice); // type
  
  fftx::array_t<3,std::complex<double>> inputArrayDevice(fftx::global_ptr<std::complex<double>>(inputDevicePtr,0,1), test_comp::domain);
  fftx::array_t<3,std::complex<double>> outputArraySpiralDevice(fftx::global_ptr<std::complex<double>>(outputSpiralDevicePtr,0,1), test_comp::domain);

  /*
    Set up timers for hipfft.
   */
  hipEvent_t* startHipfft = new hipEvent_t[iterations];
  hipEvent_t* stopHipfft  = new hipEvent_t[iterations];
  for (int itn = 0; itn < iterations; itn++ )
    {
      hipEventCreate(&(startHipfft[itn]));
      hipEventCreate(&(stopHipfft[itn]));
    }

  /*
    Get plan for hipfft.
  */
  if (verbosity >= 1)
    {
      printf("get hipfft plan\n");
    }
  hipfftHandle plan;
  {
    auto result =
      hipfftPlan3d(&plan, fftx_nx, fftx_ny, fftx_nz, HIPFFT_Z2Z);
    if (result != HIPFFT_SUCCESS)
      {
        exit(-1);
      }
  }

  /*
    Time iterations of complex-to-complex hipfft calls using the plan.
   */
  if (verbosity >= 1)
    {
      printf("call hipfftExecZ2Z %d times\n", iterations);
    }

  for (int itn = 0; itn < iterations; itn++ )
    {
      hipEventRecord(startHipfft[itn]);
      auto result = 
        hipfftExecZ2Z(plan,
                     (hipfftDoubleComplex *) inputDevicePtr,
                     (hipfftDoubleComplex *) outputHipfftDevicePtr,
                     HIPFFT_FORWARD);
      hipEventRecord(stopHipfft[itn]);
      hipEventSynchronize(stopHipfft[itn]);
      if (result != HIPFFT_SUCCESS)
        {
          printf("hipfftExecZ2Z launch failed\n");
          exit(-1);
        }
    }
  hipfftDestroy(plan);

  float* hipfft_gpu = new float[iterations];
  for (int itn = 0; itn < iterations; itn++)
    {
      hipEventElapsedTime(&(hipfft_gpu[itn]), startHipfft[itn], stopHipfft[itn]);
    }
  delete[] startHipfft;
  delete[] stopHipfft;

  hipDeviceSynchronize();

  if (verbosity >= 1)
    {
      printf("call spiralmddft::transform() %d times\n", iterations);
    }

  fftx::point_t<3> extents = test_comp::domain.extents();
  fftx::mddft<3> tfm(extents); // does initialization

  /*
    Time iterations of complex-to-complex MDDFT with SPIRAL-generated code.
   */
  double* spiral_cpu = new double[iterations];
  float* spiral_gpu = new float[iterations];
  for (int itn = 0; itn < iterations; itn++)
    {
      tfm.transform(inputArrayDevice, outputArraySpiralDevice);
      spiral_gpu[itn] = tfm.GPU_milliseconds();
      spiral_cpu[itn] = tfm.CPU_milliseconds();
    }

  /*
    Check that hipfft and SPIRAL give the same results on last iteration.
  */
  hipfftDoubleComplex* outputSpiralHostPtr = new hipfftDoubleComplex[npts];
  hipfftDoubleComplex* outputHipfftHostPtr  = new hipfftDoubleComplex[npts];
  hipMemcpy(outputSpiralHostPtr, outputSpiralDevicePtr, // dest, source
             npts*sizeof(hipfftDoubleComplex), // bytes
             hipMemcpyDeviceToHost); // type
  hipMemcpy(outputHipfftHostPtr, outputHipfftDevicePtr, // dest, source
             npts*sizeof(hipfftDoubleComplex), // bytes
             hipMemcpyDeviceToHost); // type

  hipFree(bufferDevicePtr);

  const double tol = 1.e-7;
  bool match = true;
  double maxDiff = 0.;
  {
    for (size_t ind = 0; ind < npts; ind++)
      {
        auto outputSpiralPoint = outputSpiralHostPtr[ind];
        auto outputHipfftPoint = outputHipfftHostPtr[ind];
        double diffReal = outputSpiralPoint.x - outputHipfftPoint.x;
        double diffImag = outputSpiralPoint.y - outputHipfftPoint.y;
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
                printf("error at (%d,%d,%d): SPIRAL %f+i*%f, hipfft %f+i*%f\n",
                       pt[0], pt[1], pt[2],
                       outputSpiralPoint.x, outputSpiralPoint.y,
                       outputHipfftPoint.x, outputHipfftPoint.y);
              }
          }
      }
  }
  
  delete[] outputSpiralHostPtr;
  delete[] outputHipfftHostPtr;
  if (match)
    {
      printf("YES, results match.  Max difference %11.5e\n", maxDiff);
    }
  else
    {
      printf("NO, results do not match.  Max difference %11.5e\n", maxDiff);
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

  auto hipfftMin = minSubarray(hipfft_gpu, 1, iterations-1);
  auto hipfftMax = maxSubarray(hipfft_gpu, 1, iterations-1);
  auto hipfftAvg = avgSubarray(hipfft_gpu, 1, iterations-1);

  printf("Size %d %d %d, over %d iterations\n",
         fftx_nx, fftx_ny, fftx_nz, iterations);
  if (verbosity >= 2)
    {
      printf("itn    Spiral CPU     Spiral GPU     hipfft\n");
      for (int itn = 0; itn < iterations; itn++)
        {
          printf("%3d    %.7e  %.7e  %.7e\n",
                 itn, spiral_cpu[itn], spiral_gpu[itn], hipfft_gpu[itn]);
        }
    }
  else
    {
      printf("       Spiral CPU     Spiral GPU     hipfft\n");
    }

  printf("NOT INCLUDING FIRST RUN:\n");
  printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, hipfftAvg);
  if (verbosity >= 1)
    {
      printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, hipfftMin);
      printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, hipfftMax);
    }

  gpuMin = minSubarray(spiral_gpu, 0, iterations-1);
  gpuMax = maxSubarray(spiral_gpu, 0, iterations-1);
  gpuAvg = avgSubarray(spiral_gpu, 0, iterations-1);

  cpuMin = minSubarray(spiral_cpu, 0, iterations-1);
  cpuMax = maxSubarray(spiral_cpu, 0, iterations-1);
  cpuAvg = avgSubarray(spiral_cpu, 0, iterations-1);

  hipfftMin = minSubarray(hipfft_gpu, 0, iterations-1);
  hipfftMax = maxSubarray(hipfft_gpu, 0, iterations-1);
  hipfftAvg = avgSubarray(hipfft_gpu, 0, iterations-1);
  
  delete[] spiral_cpu;
  delete[] spiral_gpu;
  delete[] hipfft_gpu;
  
  printf("INCLUDING FIRST RUN:\n");
  printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, hipfftAvg);
  if (verbosity >= 1)
    {
      printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, hipfftMin);
      printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, hipfftMax);
    }

  printf("%s: All done, exiting\n", argv[0]);

  return 0;
}
