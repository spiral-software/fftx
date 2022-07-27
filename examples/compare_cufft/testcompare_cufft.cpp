#include <stdio.h>

#include "device_macros.h"

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
  std::complex<double> * outputLibraryDevicePtr;
  DEVICE_MALLOC(&bufferDevicePtr, 3 * npts*sizeof(std::complex<double>));
  std::complex<double> * thisDevicePtr = bufferDevicePtr;
  inputDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  outputSpiralDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  outputLibraryDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  DEVICE_MEM_COPY(inputDevicePtr, inputHostPtr, // dest, source
                  npts*sizeof(std::complex<double>), // bytes
                  MEM_COPY_HOST_TO_DEVICE); // type
  
  fftx::array_t<3,std::complex<double>> inputArrayDevice(fftx::global_ptr<std::complex<double>>(inputDevicePtr,0,1), test_comp::domain);
  fftx::array_t<3,std::complex<double>> outputArraySpiralDevice(fftx::global_ptr<std::complex<double>>(outputSpiralDevicePtr,0,1), test_comp::domain);

  /*
    Set up timers for library (cufft or hipfft).
   */
  DEVICE_EVENT_T* startLibrary = new DEVICE_EVENT_T[iterations];
  DEVICE_EVENT_T* stopLibrary  = new DEVICE_EVENT_T[iterations];
  for (int itn = 0; itn < iterations; itn++ )
    {
      DEVICE_EVENT_CREATE(&(startLibrary[itn]));
      DEVICE_EVENT_CREATE(&(stopLibrary[itn]));
    }

  /*
    Get plan for library (cufft or hipfft).
  */
  if (verbosity >= 1)
    {
      printf("get library plan\n");
    }
  DEVICE_FFT_HANDLE plan;
  {
    auto result =
      DEVICE_FFT_PLAN3D(&plan, fftx_nx, fftx_ny, fftx_nz, DEVICE_FFT_Z2Z);
    if (result != DEVICE_FFT_SUCCESS)
      {
        exit(-1);
      }
  }

  /*
    Time iterations of complex-to-complex library calls using the plan.
   */
  if (verbosity >= 1)
    {
      printf("call DEVICE_FFT_EXECZ2Z %d times\n", iterations);
    }

  for (int itn = 0; itn < iterations; itn++ )
    {
      DEVICE_EVENT_RECORD(startLibrary[itn]);
      auto result = 
        DEVICE_FFT_EXECZ2Z(plan,
                           (DEVICE_FFT_DOUBLECOMPLEX *) inputDevicePtr,
                           (DEVICE_FFT_DOUBLECOMPLEX *) outputLibraryDevicePtr,
                           DEVICE_FFT_FORWARD);
      DEVICE_EVENT_RECORD(stopLibrary[itn]);
      DEVICE_EVENT_SYNCHRONIZE(stopLibrary[itn]);
      if (result != DEVICE_FFT_SUCCESS)
        {
          printf("DEVICE_FFT_EXECZ2Z launch failed\n");
          exit(-1);
        }
    }
  DEVICE_FFT_DESTROY(plan);

  float* library_gpu = new float[iterations];
  for (int itn = 0; itn < iterations; itn++)
    {
      DEVICE_EVENT_ELAPSED_TIME(&(library_gpu[itn]),
                                startLibrary[itn],
                                stopLibrary[itn]);
    }
  delete[] startLibrary;
  delete[] stopLibrary;
 
  DEVICE_SYNCHRONIZE();

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
    Check that library and SPIRAL give the same results on last iteration.
  */
  DEVICE_FFT_DOUBLECOMPLEX* outputSpiralHostPtr = new DEVICE_FFT_DOUBLECOMPLEX[npts];
  DEVICE_FFT_DOUBLECOMPLEX* outputLibraryHostPtr  = new DEVICE_FFT_DOUBLECOMPLEX[npts];
  DEVICE_MEM_COPY(outputSpiralHostPtr, outputSpiralDevicePtr, // dest, source
                  npts*sizeof(DEVICE_FFT_DOUBLECOMPLEX), // bytes
                  MEM_COPY_DEVICE_TO_HOST); // type
  DEVICE_MEM_COPY(outputLibraryHostPtr, outputLibraryDevicePtr, // dest, source
                  npts*sizeof(DEVICE_FFT_DOUBLECOMPLEX), // bytes
                  MEM_COPY_DEVICE_TO_HOST); // type

  DEVICE_FREE(bufferDevicePtr);

  const double tol = 1.e-7;
  bool match = true;
  double maxDiff = 0.;
  {
    for (size_t ind = 0; ind < npts; ind++)
      {
        auto outputSpiralPoint = outputSpiralHostPtr[ind];
        auto outputLibraryPoint = outputLibraryHostPtr[ind];
        double diffReal = outputSpiralPoint.x - outputLibraryPoint.x;
        double diffImag = outputSpiralPoint.y - outputLibraryPoint.y;
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
                printf("error at (%d,%d,%d): SPIRAL %f+i*%f, library %f+i*%f\n",
                       pt[0], pt[1], pt[2],
                       outputSpiralPoint.x, outputSpiralPoint.y,
                       outputLibraryPoint.x, outputLibraryPoint.y);
              }
          }
      }
  }
  delete[] outputSpiralHostPtr;
  delete[] outputLibraryHostPtr;
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

  auto libraryMin = minSubarray(library_gpu, 1, iterations-1);
  auto libraryMax = maxSubarray(library_gpu, 1, iterations-1);
  auto libraryAvg = avgSubarray(library_gpu, 1, iterations-1);

  printf("Size %d %d %d, over %d iterations\n",
         fftx_nx, fftx_ny, fftx_nz, iterations);
  if (verbosity >= 2)
    {
      printf("itn    Spiral CPU     Spiral GPU    library\n");
      for (int itn = 0; itn < iterations; itn++)
        {
          printf("%3d    %.7e  %.7e  %.7e\n",
                 itn, spiral_cpu[itn], spiral_gpu[itn], library_gpu[itn]);
        }
    }
  else
    {
      printf("       Spiral CPU     Spiral GPU    library\n");
    }

  printf("NOT INCLUDING FIRST RUN:\n");
  printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, libraryAvg);
  if (verbosity >= 1)
    {
      printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, libraryMin);
      printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, libraryMax);
    }

  gpuMin = minSubarray(spiral_gpu, 0, iterations-1);
  gpuMax = maxSubarray(spiral_gpu, 0, iterations-1);
  gpuAvg = avgSubarray(spiral_gpu, 0, iterations-1);

  cpuMin = minSubarray(spiral_cpu, 0, iterations-1);
  cpuMax = maxSubarray(spiral_cpu, 0, iterations-1);
  cpuAvg = avgSubarray(spiral_cpu, 0, iterations-1);

  libraryMin = minSubarray(library_gpu, 0, iterations-1);
  libraryMax = maxSubarray(library_gpu, 0, iterations-1);
  libraryAvg = avgSubarray(library_gpu, 0, iterations-1);
  
  delete[] spiral_cpu;
  delete[] spiral_gpu;
  delete[] library_gpu;
  
  printf("INCLUDING FIRST RUN:\n");
  printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, libraryAvg);
  if (verbosity >= 1)
    {
      printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, libraryMin);
      printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, libraryMax);
    }

  printf("%s: All done, exiting\n", argv[0]);

  return 0;
}
