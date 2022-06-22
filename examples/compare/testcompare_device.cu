#include <stdio.h>

#include "device_macros.h"

#if defined(__CUDACC__) || defined(FFTX_HIP)
#include "fftx_mddft_gpu_public.h"
#include "fftx_imddft_gpu_public.h"
#include "fftx_mdprdft_gpu_public.h"
#include "fftx_imdprdft_gpu_public.h"
// #include "fftx_rconv_gpu_public.h"
#else
#include "fftx_mddft_cpu_public.h"
#include "fftx_imddft_cpu_public.h"
#include "fftx_mdprdft_cpu_public.h"
#include "fftx_imdprdft_cpu_public.h"
// #include "fftx_rconv_cpu_public.h"
#endif

#include "mddft.fftx.precompile.hpp"
#include "imddft.fftx.precompile.hpp"
#include "mdprdft.fftx.precompile.hpp"
#include "imdprdft.fftx.precompile.hpp"
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

void setRand(double& a_val)
{
  a_val = 1. - ((double) rand()) / (double) (RAND_MAX/2);
}

void setRand(std::complex<double>& a_val)
{
  double x, y;
  setRand(x);
  setRand(y);
  a_val = std::complex<double>(x, y);
}

double diffAbs(double a_x,
               double a_y)
{
  double diffNorm = a_x - a_y;
  if (diffNorm < 0.) diffNorm = -diffNorm;
  return diffNorm;
}

double diffAbs(std::complex<double>& a_x,
               std::complex<double>& a_y)
{
  double diffNorm = std::abs(a_x - a_y);
  return diffNorm;
}

DEVICE_FFT_RESULT deviceExecD2Z(DEVICE_FFT_HANDLE a_plan,
                                double* a_in,
                                std::complex<double>* a_out)
{
  return DEVICE_FFT_EXECD2Z(a_plan,
                            (DEVICE_FFT_DOUBLEREAL*) a_in,
                            (DEVICE_FFT_DOUBLECOMPLEX*) a_out);
}


template<typename T_IN, typename T_OUT>
struct deviceTransform
{
  deviceTransform(DEVICE_FFT_TYPE a_tp,
                  int a_dir = 0)
  {
    m_tp = a_tp;
    m_dir = a_dir;
  }
                  
  DEVICE_FFT_TYPE m_tp;

  int m_dir;

  DEVICE_FFT_RESULT plan3d(DEVICE_FFT_HANDLE& a_plan,
                           fftx::point_t<3> a_tfmSize)
  {
    return DEVICE_FFT_PLAN3D(&a_plan,
                             a_tfmSize[0], a_tfmSize[1], a_tfmSize[2],
                             m_tp);
  }

  DEVICE_FFT_RESULT exec(DEVICE_FFT_HANDLE a_plan,
                         T_IN* a_in,
                         T_OUT* a_out)
  {
    if (m_tp == DEVICE_FFT_Z2Z)
      {
        return DEVICE_FFT_EXECZ2Z(a_plan,
                                  (DEVICE_FFT_DOUBLECOMPLEX*) a_in,
                                  (DEVICE_FFT_DOUBLECOMPLEX*) a_out,
                                  m_dir);
      }
    else if (m_tp == DEVICE_FFT_D2Z)
      {
        return DEVICE_FFT_EXECD2Z(a_plan,
                                  (DEVICE_FFT_DOUBLEREAL*) a_in,
                                  (DEVICE_FFT_DOUBLECOMPLEX*) a_out);
      }
    else if (m_tp == DEVICE_FFT_Z2D)
      {
        return DEVICE_FFT_EXECZ2D(a_plan,
                                  (DEVICE_FFT_DOUBLECOMPLEX*) a_in,
                                  (DEVICE_FFT_DOUBLEREAL*) a_out);
      }
    else
      {
        return (DEVICE_FFT_RESULT) -1;
      }
  }
};
  

deviceTransform<std::complex<double>, std::complex<double> >
mddftDevice(DEVICE_FFT_Z2Z, DEVICE_FFT_FORWARD);

deviceTransform<std::complex<double>, std::complex<double> >
imddftDevice(DEVICE_FFT_Z2Z, DEVICE_FFT_INVERSE);

deviceTransform<double, std::complex<double> >
mdprdftDevice(DEVICE_FFT_D2Z);

deviceTransform<std::complex<double>, double>
imdprdftDevice(DEVICE_FFT_Z2D);

template<typename T_IN, typename T_OUT, class Transformer>
void compareSize(Transformer& a_tfm,
                 // fftx::mdprdft<3>& a_tfm,
                 // fftx::transformer<3, T_IN, T_OUT> a_tfm, // fftx::point_t<3> a_size,
                 deviceTransform<T_IN, T_OUT>& a_tfmDevice,
                 int a_iterations,
                 int a_verbosity)
{
  if (!a_tfm.isDefined())
    {
      return;
    }

  // fftx::point_t<3> extents = test_comp::domain.extents();
  // fftx::mddft<3> tfm(a_extents); // does initialization

  // fftx::mdprdft<3> tfm(a_size); // does initialization
  fftx::point_t<3> tfmSize = a_tfm.size();
  // deviceTransform<double, std::complex<double> > tfmDevice(DEVICE_FFT_D2Z);
  
  /*
    Allocate space for arrays, and set input array.
  */
  const fftx::point_t<3> inputSize = a_tfm.inputSize();
  const fftx::point_t<3> outputSize = a_tfm.outputSize();

  // This doesn't work. :/
  // const fftx::point_t<3> unit = fftx::point_t<3>::Unit();
  //   fftx::box_t<3> inputDomain(unit, inputSize);
  //   fftx::box_t<3> outputDomain(unit, outputSize);

  fftx::box_t<3> inputDomain(fftx::point_t<3>({{1, 1, 1}}),
                             fftx::point_t<3>({{inputSize[0],
                                                inputSize[1],
                                                inputSize[2]}}));
  fftx::box_t<3> outputDomain(fftx::point_t<3>({{1, 1, 1}}),
                              fftx::point_t<3>({{outputSize[0],
                                                 outputSize[1],
                                                 outputSize[2]}}));
  
  // fftx::array_t<3,std::complex<double>> inputArrayHost(test_comp::domain);
  fftx::array_t<3, T_IN> inputArrayHost(inputDomain);
  size_t nptsInput = inputDomain.size();
  size_t nptsOutput = outputDomain.size();
  size_t bytesInput = nptsInput * sizeof(T_IN);
  size_t bytesOutput = nptsOutput * sizeof(T_OUT);
  forall([](T_IN(&v), const fftx::point_t<3>& p)
         {
           setRand(v);
         }, inputArrayHost);
  // This symmetrizes only for complex input and real output,
  // in order to get a complex array that transforms to a real array.
  // symmetrize<T_IN, T_OUT>(inputArrayHost, outputDomain, a_verbosity);
  fftx::array_t<3, T_OUT> outputArrayHost(outputDomain);
  symmetrizeHermitian(inputArrayHost, outputArrayHost);
  T_IN* inputHostPtr = inputArrayHost.m_data.local();
  // additional code for GPU programs
  T_IN* inputDevicePtr;
  T_OUT* outputSpiralDevicePtr;
  T_OUT* outputDeviceFFTDevicePtr;
  DEVICE_MALLOC(&inputDevicePtr, bytesInput);
  DEVICE_MALLOC(&outputSpiralDevicePtr, bytesOutput);
  DEVICE_MALLOC(&outputDeviceFFTDevicePtr, bytesOutput);
  // Do this at the beginning of each iteration instead of here.
  //  DEVICE_MEM_COPY(inputDevicePtr, inputHostPtr, // dest, source
  //                  npts*sizeof(double), // bytes
  //                  MEM_COPY_HOST_TO_DEVICE); // type
  
  fftx::array_t<3, T_IN>
    inputArrayDevice(fftx::global_ptr<T_IN>
                     (inputDevicePtr,0,1), inputDomain);

  fftx::array_t<3, T_OUT>
    outputArraySpiralDevice(fftx::global_ptr<T_OUT>
                            (outputSpiralDevicePtr,0,1), outputDomain);

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
  DEVICE_FFT_CHECK(a_tfmDevice.plan3d(plan, tfmSize),
                   "device FFT define plan");

  /*
    Time iterations of real-to-complex deviceFFT calls using the plan.
   */
  if (a_verbosity >= 2)
    {
      printf("call deviceExec %d times\n", a_iterations);
    }

  for (int itn = 0; itn < a_iterations; itn++ )
    {
      DEVICE_MEM_COPY(inputDevicePtr, // dest
                      inputHostPtr, // source
                      bytesInput, // bytes
                      MEM_COPY_HOST_TO_DEVICE); // type
      DEVICE_EVENT_RECORD(startDeviceFFT[itn]);
      //      auto result = deviceExecD2Z(plan,
      //                                  inputDevicePtr,
      //                                  outputDeviceFFTDevicePtr);
      DEVICE_FFT_CHECK(a_tfmDevice.exec(plan,
                                        inputDevicePtr,
                                        outputDeviceFFTDevicePtr),
                       "device FFT exec launch");
      //      auto result = 
      //        DEVICE_FFT_EXECD2Z(plan,
      //                           (T_IN*) inputDevicePtr,
      //                           (DEVICE_FFT_DOUBLECOMPLEX*) outputDeviceFFTDevicePtr); // FIXME: (T_OUT*)
      DEVICE_EVENT_RECORD(stopDeviceFFT[itn]);
      DEVICE_EVENT_SYNCHRONIZE(stopDeviceFFT[itn]);
    }
  DEVICE_FFT_DESTROY(plan);

  float* deviceFFT_gpu = new float[a_iterations];
  for (int itn = 0; itn < a_iterations; itn++)
    {
      DEVICE_EVENT_ELAPSED_TIME(&(deviceFFT_gpu[itn]),
                                startDeviceFFT[itn],
                                stopDeviceFFT[itn]);
    }
  delete[] startDeviceFFT;
  delete[] stopDeviceFFT;

  DEVICE_SYNCHRONIZE();

  if (a_verbosity >= 2)
    {
      printf("call Spiral transform %d times\n", a_iterations);
    }

  /*
    Time iterations of transform with SPIRAL-generated code.
   */
  double* spiral_cpu = new double[a_iterations];
  float* spiral_gpu = new float[a_iterations];
  DEVICE_MEM_COPY(inputDevicePtr, // dest
                  inputHostPtr, // source
                  bytesInput, // bytes
                  MEM_COPY_HOST_TO_DEVICE); // type
  for (int itn = 0; itn < a_iterations; itn++)
    {
      a_tfm.transform(inputArrayDevice, outputArraySpiralDevice);
      spiral_gpu[itn] = a_tfm.GPU_milliseconds();
      spiral_cpu[itn] = a_tfm.CPU_milliseconds();
    }

  /*
    Check that deviceFFT and SPIRAL give the same results on last iteration.
  */
  T_OUT* outputSpiralHostPtr = new T_OUT[nptsOutput];
  T_OUT* outputDeviceFFTHostPtr = new T_OUT[nptsOutput];
  DEVICE_MEM_COPY(outputSpiralHostPtr, // dest
                  outputSpiralDevicePtr, // source
                  bytesOutput, // bytes
                  MEM_COPY_DEVICE_TO_HOST); // type
  DEVICE_MEM_COPY(outputDeviceFFTHostPtr, // dest
                  outputDeviceFFTDevicePtr, // source
                  bytesOutput, // bytes
                  MEM_COPY_DEVICE_TO_HOST); // type

  DEVICE_FREE(inputDevicePtr);
  DEVICE_FREE(outputSpiralDevicePtr);
  DEVICE_FREE(outputDeviceFFTDevicePtr);

  const double tol = 1.e-7;
  bool match = true;
  double maxDiff = 0.;
  {
    for (size_t ind = 0; ind < nptsOutput; ind++)
      {
        T_OUT outputSpiralPoint = outputSpiralHostPtr[ind];
        T_OUT outputDeviceFFTPoint = outputDeviceFFTHostPtr[ind];
        // auto diffPoint = outputSpiralPoint - outputDeviceFFTPoint;
        // double diffReal = outputSpiralPoint.x - outputDeviceFFTPoint.x;
        // double diffImag = outputSpiralPoint.y - outputDeviceFFTPoint.y;
        double diffAbsPoint = diffAbs(outputSpiralPoint, outputDeviceFFTPoint);
        updateMaxAbs(maxDiff, diffAbsPoint);
        bool matchPoint = (diffAbsPoint < tol);
        if (!matchPoint)
          {
            match = false;
            if (a_verbosity >= 3)
              {
                point_t<3> pt = pointFromPositionBox(ind, outputDomain);
                std::cout << "error at " << pt
                          << ": SPIRAL " << outputSpiralPoint
                          << ", deviceFFT " << outputDeviceFFTPoint
                          << std::endl;
                // printf("error at (%d,%d,%d): SPIRAL %f+i*%f, deviceFFT %f+i*%f\n",
                // pt[0], pt[1], pt[2],
                // outputSpiralPoint.x, outputSpiralPoint.y,
                // outputDeviceFFTPoint.x, outputDeviceFFTPoint.y);
              }
          }
      }
  }
  
  delete[] outputSpiralHostPtr;
  delete[] outputDeviceFFTHostPtr;
  if (match)
    {
      printf("YES, results match for %s. Max diff %11.5e\n",
             a_tfm.name().c_str(), maxDiff);
    }
  else
    {
      printf("NO, results do not match for %s. Max diff %11.5e\n",
             a_tfm.name().c_str(), maxDiff);
    }

  /*
    Get minimum, maximum, and average timings of iterations.
    First, with the first iteration included.
   */
  auto gpuMin = minSubarray(spiral_gpu, 0, a_iterations-1);
  auto gpuMax = maxSubarray(spiral_gpu, 0, a_iterations-1);
  auto gpuAvg = avgSubarray(spiral_gpu, 0, a_iterations-1);

  auto cpuMin = minSubarray(spiral_cpu, 0, a_iterations-1);
  auto cpuMax = maxSubarray(spiral_cpu, 0, a_iterations-1);
  auto cpuAvg = avgSubarray(spiral_cpu, 0, a_iterations-1);

  auto deviceFFTMin = minSubarray(deviceFFT_gpu, 0, a_iterations-1);
  auto deviceFFTMax = maxSubarray(deviceFFT_gpu, 0, a_iterations-1);
  auto deviceFFTAvg = avgSubarray(deviceFFT_gpu, 0, a_iterations-1);

  //  printf("Size %d %d %d, over %d iterations\n",
  //         tfmSize[0], tfmSize[1], tfmSize[2], a_iterations);
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

  printf("INCLUDING FIRST RUN:\n");
  printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, deviceFFTAvg);
  if (a_verbosity >= 1)
    {
      printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, deviceFFTMin);
      printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, deviceFFTMax);
    }

  if (a_iterations > 1)
    {
      gpuMin = minSubarray(spiral_gpu, 1, a_iterations-1);
      gpuMax = maxSubarray(spiral_gpu, 1, a_iterations-1);
      gpuAvg = avgSubarray(spiral_gpu, 1, a_iterations-1);

      cpuMin = minSubarray(spiral_cpu, 1, a_iterations-1);
      cpuMax = maxSubarray(spiral_cpu, 1, a_iterations-1);
      cpuAvg = avgSubarray(spiral_cpu, 1, a_iterations-1);

      deviceFFTMin = minSubarray(deviceFFT_gpu, 1, a_iterations-1);
      deviceFFTMax = maxSubarray(deviceFFT_gpu, 1, a_iterations-1);
      deviceFFTAvg = avgSubarray(deviceFFT_gpu, 1, a_iterations-1);
  
      printf("NOT INCLUDING FIRST RUN:\n");
      printf("Avg    %.7e  %.7e  %.7e\n", cpuAvg, gpuAvg, deviceFFTAvg);
      if (a_verbosity >= 1)
        {
          printf("Min    %.7e  %.7e  %.7e\n", cpuMin, gpuMin, deviceFFTMin);
          printf("Max    %.7e  %.7e  %.7e\n", cpuMax, gpuMax, deviceFFTMax);
        }
    }

  delete[] spiral_cpu;
  delete[] spiral_gpu;
  delete[] deviceFFT_gpu;
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

  // last entry is { 0, 0, 0 }
  fftx::point_t<3> *ents = fftx_mddft_QuerySizes ();

  for ( int ind = 0; ents[ind][0] != 0; ind++ )
    {
      fftx::point_t<3> sz = ents[ind];

#ifdef FFTX_HIP
      // Avoid size on which hipFFT fails in rocm 4.5.
      if (sz == fftx::point_t<3>({{128, 128, 680}}) ) continue;
#endif
      {
        fftx::mddft<3> tfm(sz);
        compareSize(tfm, mddftDevice, iterations, verbosity);
      }

      {
        fftx::imddft<3> tfm(sz);
        compareSize(tfm, imddftDevice, iterations, verbosity);
      }

      {
        fftx::mdprdft<3> tfm(sz);
        compareSize(tfm, mdprdftDevice, iterations, verbosity);
      }

      {
        fftx::imdprdft<3> tfm(sz);
        compareSize(tfm, imdprdftDevice, iterations, verbosity);
      }
    }
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
