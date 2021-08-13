#include <stdio.h>

#include "device_macros.h"

#include "fftx_mddft_public.h"
#include "fftx_mddft_decls.h"
// #include "fftx_imddft_public.h"
// #include "fftx_rconv_public.h"

#include "mddft.fftx.precompile.hpp"
// #include "imddft.fftx.precompile.hpp"
// #include "rconv.fftx.precompile.hpp"

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


void compareSize(fftx::point_t<3> a_size,
                 int a_iterations,
                 int a_verbosity)
{
  /*
    Allocate space for arrays, and set input array.
  */
  box_t<3> fulldomain(point_t<3>
                      ({{test_comp::offx+1,
                         test_comp::offy+1,
                         test_comp::offz+1}}),
                      point_t<3>
                      ({{test_comp::offx+a_size[0],
                         test_comp::offy+a_size[1],
                         test_comp::offz+a_size[2]}}));

  box_t<3> halfdomain(point_t<3>
                      ({{test_comp::offx+1,
                         test_comp::offy+1,
                         test_comp::offz+1}}),
                      point_t<3>
                      ({{test_comp::offx+a_size[0]/2+1,
                         test_comp::offy+a_size[1],
                         test_comp::offz+a_size[2]}}));
  
  // fftx::array_t<3,std::complex<double>> inputArrayHost(test_comp::domain);
  fftx::array_t<3,std::complex<double>> inputArrayHost(fulldomain);
  size_t npts = fulldomain.size();
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
  std::complex<double> * outputDeviceFFTDevicePtr;
  DEVICE_MALLOC(&bufferDevicePtr, 3 * npts*sizeof(std::complex<double>));
  std::complex<double> * thisDevicePtr = bufferDevicePtr;
  inputDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  outputSpiralDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  outputDeviceFFTDevicePtr = thisDevicePtr;
  thisDevicePtr += npts;
  //  DEVICE_MEM_COPY(inputDevicePtr, inputHostPtr, // dest, source
  //                  npts*sizeof(std::complex<double>), // bytes
  //                  MEM_COPY_HOST_TO_DEVICE); // type
  
  fftx::array_t<3,std::complex<double>>
    inputArrayDevice(fftx::global_ptr<std::complex<double>>
                     (inputDevicePtr,0,1), fulldomain);

  fftx::array_t<3,std::complex<double>>
    outputArraySpiralDevice(fftx::global_ptr<std::complex<double>>
                            (outputSpiralDevicePtr,0,1), fulldomain);

  /*
    Set up timers for deviceFFT.
   */
  DEVICE_EVENT_T* startDeviceFFT = new DEVICE_EVENT_T[a_iterations];
  DEVICE_EVENT_T* stopDeviceFFT  = new DEVICE_EVENT_T[a_iterations];
  for (int itn = 0; itn < a_iterations; itn++ )
    {
      DEVICE_EVENT_CREATE(&(startDeviceFFT[itn]));
      DEVICE_EVENT_CREATE(&(stopDeviceFFT[itn]));
    }

  /*
    Get plan for deviceFFT.
  */
  if (a_verbosity >= 1)
    {
      printf("get deviceFFT plan\n");
    }
  DEVICE_FFT_HANDLE plan;
  {
    auto result =
      DEVICE_FFT_PLAN3D(&plan, a_size[0], a_size[1], a_size[2], DEVICE_FFT_Z2Z);
    if (result != DEVICE_FFT_SUCCESS)
      {
        exit(-1);
      }
  }

  /*
    Time iterations of complex-to-complex deviceFFT calls using the plan.
   */
  if (a_verbosity >= 1)
    {
      printf("call deviceFFTExecZ2Z %d times\n", a_iterations);
    }

  for (int itn = 0; itn < a_iterations; itn++ )
    {
      DEVICE_MEM_COPY(inputDevicePtr, inputHostPtr, // dest, source
                      npts*sizeof(std::complex<double>), // bytes
                      MEM_COPY_HOST_TO_DEVICE); // type
      DEVICE_EVENT_RECORD(startDeviceFFT[itn]);
      auto result = 
        DEVICE_FFT_EXECZ2Z(plan,
                           (DEVICE_FFT_DOUBLECOMPLEX *) inputDevicePtr,
                           (DEVICE_FFT_DOUBLECOMPLEX *) outputDeviceFFTDevicePtr,
                           DEVICE_FFT_FORWARD);
      DEVICE_EVENT_RECORD(stopDeviceFFT[itn]);
      DEVICE_EVENT_SYNCHRONIZE(stopDeviceFFT[itn]);
      if (result != DEVICE_FFT_SUCCESS)
        {
          printf("deviceFFTExecZ2Z launch failed\n");
          exit(-1);
        }
    }
  DEVICE_FFT_DESTROY(plan);

  float* deviceFFT_gpu = new float[a_iterations];
  for (int itn = 0; itn < a_iterations; itn++)
    {
      DEVICE_EVENT_ELAPSED_TIME(&(deviceFFT_gpu[itn]), startDeviceFFT[itn], stopDeviceFFT[itn]);
    }
  delete[] startDeviceFFT;
  delete[] stopDeviceFFT;

  DEVICE_SYNCHRONIZE();

  if (a_verbosity >= 1)
    {
      printf("call spiralmddft::transform() %d times\n", a_iterations);
    }

  // fftx::point_t<3> extents = test_comp::domain.extents();
  // fftx::mddft<3> tfm(a_extents); // does initialization
  fftx::mddft<3> tfm(a_size); // does initialization

  /*
    Time iterations of complex-to-complex MDDFT with SPIRAL-generated code.
   */
  double* spiral_cpu = new double[a_iterations];
  float* spiral_gpu = new float[a_iterations];
  DEVICE_MEM_COPY(inputDevicePtr, inputHostPtr, // dest, source
                  npts*sizeof(std::complex<double>), // bytes
                  MEM_COPY_HOST_TO_DEVICE); // type
  for (int itn = 0; itn < a_iterations; itn++)
    {
      tfm.transform(inputArrayDevice, outputArraySpiralDevice);
      spiral_gpu[itn] = tfm.GPU_milliseconds();
      spiral_cpu[itn] = tfm.CPU_milliseconds();
    }

  /*
    Check that deviceFFT and SPIRAL give the same results on last iteration.
  */
  DEVICE_FFT_DOUBLECOMPLEX* outputSpiralHostPtr = new DEVICE_FFT_DOUBLECOMPLEX[npts];
  DEVICE_FFT_DOUBLECOMPLEX* outputDeviceFFTHostPtr  = new DEVICE_FFT_DOUBLECOMPLEX[npts];
  DEVICE_MEM_COPY(outputSpiralHostPtr, outputSpiralDevicePtr, // dest, source
                  npts*sizeof(DEVICE_FFT_DOUBLECOMPLEX), // bytes
                  MEM_COPY_DEVICE_TO_HOST); // type
  DEVICE_MEM_COPY(outputDeviceFFTHostPtr, outputDeviceFFTDevicePtr, // dest, source
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
        auto outputDeviceFFTPoint = outputDeviceFFTHostPtr[ind];
        double diffReal = outputSpiralPoint.x - outputDeviceFFTPoint.x;
        double diffImag = outputSpiralPoint.y - outputDeviceFFTPoint.y;
        updateMaxAbs(maxDiff, diffReal);
        updateMaxAbs(maxDiff, diffImag);
        bool matchPoint =
          (std::abs(diffReal) < tol) && (std::abs(diffImag) < tol);
        if (!matchPoint)
          {
            match = false;
            if (a_verbosity >= 3)
              {
                point_t<3> pt = pointFromPositionBox(ind, test_comp::domain);
                printf("error at (%d,%d,%d): SPIRAL %f+i*%f, deviceFFT %f+i*%f\n",
                       pt[0], pt[1], pt[2],
                       outputSpiralPoint.x, outputSpiralPoint.y,
                       outputDeviceFFTPoint.x, outputDeviceFFTPoint.y);
              }
          }
      }
  }
  
  delete[] outputSpiralHostPtr;
  delete[] outputDeviceFFTHostPtr;
  if (match)
    {
      printf("YES, results match for [%d,%d,%d].  Max difference %11.5e\n",
             a_size[0], a_size[1], a_size[2], maxDiff);
    }
  else
    {
      printf("NO, results do not match for [%d,%d,%d].  Max difference %11.5e\n",
             a_size[0], a_size[1], a_size[2], maxDiff);
    }

  /*
    Get minimum, maximum, and average timings of iterations.
    First, with the first iteration excluded.
   */
  auto gpuMin = minSubarray(spiral_gpu, 1, a_iterations-1);
  auto gpuMax = maxSubarray(spiral_gpu, 1, a_iterations-1);
  auto gpuAvg = avgSubarray(spiral_gpu, 1, a_iterations-1);

  auto cpuMin = minSubarray(spiral_cpu, 1, a_iterations-1);
  auto cpuMax = maxSubarray(spiral_cpu, 1, a_iterations-1);
  auto cpuAvg = avgSubarray(spiral_cpu, 1, a_iterations-1);

  auto deviceFFTMin = minSubarray(deviceFFT_gpu, 1, a_iterations-1);
  auto deviceFFTMax = maxSubarray(deviceFFT_gpu, 1, a_iterations-1);
  auto deviceFFTAvg = avgSubarray(deviceFFT_gpu, 1, a_iterations-1);

  printf("Size %d %d %d, over %d iterations\n",
         a_size[0], a_size[1], a_size[2], a_iterations);
  if (a_verbosity >= 2)
    {
      printf("itn    Spiral CPU     Spiral GPU     deviceFFT\n");
      for (int itn = 0; itn < a_iterations; itn++)
        {
          printf("%3d    %.7e  %.7e  %.7e\n",
                 itn, spiral_cpu[itn], spiral_gpu[itn], deviceFFT_gpu[itn]);
        }
    }
  else
    {
      printf("       Spiral CPU     Spiral GPU     deviceFFT\n");
    }

  printf("NOT INCLUDING FIRST RUN:\n");
  printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, deviceFFTAvg);
  if (a_verbosity >= 1)
    {
      printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, deviceFFTMin);
      printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, deviceFFTMax);
    }

  gpuMin = minSubarray(spiral_gpu, 0, a_iterations-1);
  gpuMax = maxSubarray(spiral_gpu, 0, a_iterations-1);
  gpuAvg = avgSubarray(spiral_gpu, 0, a_iterations-1);

  cpuMin = minSubarray(spiral_cpu, 0, a_iterations-1);
  cpuMax = maxSubarray(spiral_cpu, 0, a_iterations-1);
  cpuAvg = avgSubarray(spiral_cpu, 0, a_iterations-1);

  deviceFFTMin = minSubarray(deviceFFT_gpu, 0, a_iterations-1);
  deviceFFTMax = maxSubarray(deviceFFT_gpu, 0, a_iterations-1);
  deviceFFTAvg = avgSubarray(deviceFFT_gpu, 0, a_iterations-1);
  
  delete[] spiral_cpu;
  delete[] spiral_gpu;
  delete[] deviceFFT_gpu;
  
  printf("INCLUDING FIRST RUN:\n");
  printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, deviceFFTAvg);
  if (a_verbosity >= 1)
    {
      printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, deviceFFTMin);
      printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, deviceFFTMax);
    }
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

  int numentries = sizeof ( AllSizes3 ) / sizeof ( fftx::point_t<3> ) - 1; // last entry is { 0, 0, 0 }
  for ( int ind = 0; ind < numentries; ind++ )
    {
      compareSize(AllSizes3[ind], iterations, verbosity);
    }

  // compareSize(fftx::point_t<3>({{ 320, 320, 320 }}), iterations, verbosity);
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
