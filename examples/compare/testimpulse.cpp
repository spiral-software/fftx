#include <stdio.h>

#include "device_macros.h"

#include "fftx_mddft_public.h"
#include "fftx_imddft_public.h"
#include "fftx_mdprdft_public.h"
#include "fftx_imdprdft_public.h"
// #include "fftx_rconv_public.h"

#include "mddft.fftx.precompile.hpp"
#include "imddft.fftx.precompile.hpp"
#include "mdprdft.fftx.precompile.hpp"
#include "imdprdft.fftx.precompile.hpp"
// #include "rconv.fftx.precompile.hpp"

#include "fftx3utilities.h"

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

void set0(double& a_val)
{
  a_val = 0.;
}

void set1(double& a_val)
{
  a_val = 1.;
}

void set0(std::complex<double>& a_val)
{
  a_val = std::complex<double>(0., 0.);
}

void set1(std::complex<double>& a_val)
{
  a_val = std::complex<double>(1., 0.);
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

fftx::box_t<3> domain1(fftx::point_t<3> a_pt)
{
  return fftx::box_t<3>(fftx::point_t<3>({{1, 1, 1}}),
                        fftx::point_t<3>({{a_pt[0], a_pt[1], a_pt[2]}}));
}
                                

// Example of use: DEVICE_FFT_CHECK(exec(...), "exec");
inline void DEVICE_FFT_CHECK(DEVICE_FFT_RESULT a_rc,
                             const std::string& a_name)
{
  if (a_rc != DEVICE_FFT_SUCCESS)
    {
      std::cerr << a_name << " failed with code " << a_rc
#if defined(__CUDACC__)
                << " meaning " << _cudaGetErrorEnum(a_rc)
#endif
                << std::endl;
      exit(-1);
    }
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

  fftx::point_t<3> inputSize(fftx::point_t<3> a_tfmSize)
  {
    fftx::point_t<3> ret = a_tfmSize;    
    if (m_tp == DEVICE_FFT_Z2D)
      {
        ret[0] = a_tfmSize[0]/2 + 1;
      }
    return ret;
  }

  fftx::point_t<3> outputSize(fftx::point_t<3> a_tfmSize)
  {
    fftx::point_t<3> ret = a_tfmSize;    
    if (m_tp == DEVICE_FFT_D2Z)
      {
        ret[0] = a_tfmSize[0]/2 + 1;
      }
    return ret;
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

void setArray3dElementConj(fftx::array_t<3, std::complex<double> >& a_arr,
                           int a_dst0, int a_dst1, int a_dst2,
                           int a_src0, int a_src1, int a_src2)
{
  fftx::box_t<3> arrDomain = a_arr.m_domain;
  std::complex<double>* arrPtr = a_arr.m_data.local();
  fftx::point_t<3> pointDst({{a_dst0, a_dst1, a_dst2}});
  fftx::point_t<3> pointSrc({{a_src0, a_src1, a_src2}});
  auto indDst = positionInBox(pointDst, arrDomain);
  auto indSrc = positionInBox(pointSrc, arrDomain);
  arrPtr[indDst] = std::conj(arrPtr[indSrc]);
}

template<typename T_IN, typename T_OUT>
void symmetrize(fftx::array_t<3, T_IN>& a_arr,
                fftx::box_t<3> a_fullDomain)
{ };

// Modify complex array so that it actually does transform to a real array.
template<>
void symmetrize<std::complex<double>, double>(fftx::array_t<3, std::complex<double> >& a_arr,
                                              fftx::box_t<3> a_fullDomain)
{
  fftx::box_t<3> arrDomain = a_arr.m_domain;
  std::complex<double>* arrPtr = a_arr.m_data.local();
  
  fftx::point_t<3> lo = a_fullDomain.lo;
  fftx::point_t<3> extent = a_fullDomain.extents();

  int k0 = extent[0]/2;
  int k1 = extent[1]/2; 
  int k2 = extent[2]/2;
  
  fftx::point_t<3> maxpt = lo;
  for (int d = 0; d < 3; d++)
    {
      if (extent[d] % 2 == 0)
        { // length is even in dimension d
          maxpt[d] += extent[d]/2;
        }
    }

  // POINT must be real.
  for (int pt0 = lo[0]; pt0 <= maxpt[0]; pt0 += k0)
    for (int pt1 = lo[1]; pt1 <= maxpt[1]; pt1 += k1)
      for (int pt2 = lo[2]; pt2 <= maxpt[2]; pt2 += k2)
        {
          fftx::point_t<3> point({{pt0, pt1, pt2}});
          auto ind = positionInBox(point, arrDomain);
          arrPtr[ind].imag(0.);
        }

  // ROWS in first dimension are truncated from the array.
  
  // ROWS in second dimension must have symmetry.
  for (int pt0 = lo[0]; pt0 <= maxpt[0]; pt0 += k0)
    {
      for (int off1 = k1+1; off1 < extent[1]; off1++)
        {
          setArray3dElementConj(a_arr,
                                pt0, lo[1] + off1, lo[2],
                                pt0, lo[1] + extent[1] - off1, lo[2]);
        }
    }

  // ROWS in third dimension must have symmetry.
  for (int pt0 = lo[0]; pt0 <= maxpt[0]; pt0 += k0)
    for (int pt1 = lo[1]; pt1 <= maxpt[1]; pt1 += k1)
      {
        for (int off2 = k2+1; off2 < extent[2]; off2++)
          {
            setArray3dElementConj(a_arr,
                                  pt0, pt1, lo[2] + off2,
                                  pt0, pt1, lo[2] + extent[2] - off2);
          }
      }

  // PLANE through the point must have symmetry.
  for (int pt0 = lo[0]; pt0 <= maxpt[0]; pt0 += k0)
    {
      for (int off1 = k1+1; off1 < extent[1]; off1++)
        for (int off2 = 1; off2 < extent[2]; off2++)
          {
            setArray3dElementConj(a_arr,
                                  pt0, lo[1] + off1, lo[2] + off2,
                                  pt0, lo[1] + extent[1] - off1, lo[2] + extent[2] - off2);
          }
    }
}

void writeErrorValue(fftx::point_t<3>& a_pt,
                     double a_val,
                     double a_correct)
{
  if (!(abs(a_val - a_correct) < 1.e-9))
    {
      printf(" %4d %4d %4d %17.9e\n",
             a_pt[0], a_pt[1], a_pt[2], a_val);
    }
}
                     
void writeErrorValue(fftx::point_t<3>& a_pt,
                     std::complex<double> a_val,
                     std::complex<double> a_correct)
{
  if (!(abs(a_val - a_correct) < 1.e-9))
    {
      printf(" %4d %4d %4d %17.9e %17.9e\n",
             a_pt[0], a_pt[1], a_pt[2], real(a_val), imag(a_val));
    }
}
                     
template<typename T_IN, typename T_OUT>
void runDeviceFFT(deviceTransform<T_IN, T_OUT>& a_tfmDevice,
                  fftx::point_t<3> a_tfmSize,
                  int a_verbosity)
{
  fftx::point_t<3> inputSize = a_tfmDevice.inputSize(a_tfmSize);
  fftx::point_t<3> outputSize = a_tfmDevice.outputSize(a_tfmSize);
  if (a_verbosity >= 2)
    {
      std::cout << "Input " << inputSize
                << " output " << outputSize << std::endl;
    }
  fftx::box_t<3> inputDomain = domain1(inputSize);
  fftx::box_t<3> outputDomain = domain1(outputSize);

  /*
    Define and set input.
  */
  fftx::array_t<3, T_IN> inputArrayHost(inputDomain);
  auto nptsInput = inputDomain.size();
  auto bytesInput = nptsInput * sizeof(T_IN);
  T_IN* inputDevicePtr = NULL;
  DEVICE_MALLOC(&inputDevicePtr, bytesInput);
  forall([](T_IN(&v), const fftx::point_t<3>& p)
         {
           if (p == fftx::point_t<3>::Unit())
             {
               set1(v);
             }
           else
             {
               set0(v);
             }
         }, inputArrayHost);
  symmetrize<T_IN, T_OUT>(inputArrayHost, outputDomain);
  T_IN* inputHostPtr = inputArrayHost.m_data.local();
  DEVICE_MEM_COPY(inputDevicePtr, // dest
                  inputHostPtr, // source
                  bytesInput, // bytes
                  MEM_COPY_HOST_TO_DEVICE); // type
  
  /*
    Define output.
  */

  T_OUT* outputDevicePtr = NULL;
  auto nptsOutput = outputDomain.size();
  auto bytesOutput = nptsOutput * sizeof(T_OUT);
  DEVICE_MALLOC(&outputDevicePtr, bytesOutput);

  /*
    Get plan for deviceFFT.
  */
  if (a_verbosity >= 2)
    {
      std::cout << "get deviceFFT plan on " << a_tfmSize << std::endl;
      printf("get deviceFFT plan\n");
    }
  DEVICE_FFT_HANDLE plan;
  DEVICE_FFT_CHECK(DEVICE_FFT_PLAN3D(&plan,
                                     a_tfmSize[0], a_tfmSize[1], a_tfmSize[2],
                                     a_tfmDevice.m_tp), // DEVICE_FFT_D2Z
                   "DEVICE_FFT_PLAN3D");

  /*
    Run deviceFFT using the plan.
  */
  if (a_verbosity >= 2)
    {
      std::cout << "call deviceExec on " << a_tfmSize << std::endl;
    }
  DEVICE_FFT_CHECK(a_tfmDevice.exec(plan, inputDevicePtr, outputDevicePtr),
                   "deviceFFTExec launch");

  DEVICE_FFT_DESTROY(plan);

  DEVICE_SYNCHRONIZE();

  /*
    Copy output from device to host.
  */
  T_OUT* outputHostPtr = new T_OUT[nptsOutput];
  DEVICE_MEM_COPY(outputHostPtr, // dest
                  outputDevicePtr, // source
                  bytesOutput, // bytes
                  MEM_COPY_DEVICE_TO_HOST); // type

  DEVICE_FREE(inputDevicePtr);
  DEVICE_FREE(outputDevicePtr);

  double maxAbsDiff = 0.;
  for (size_t ind = 0; ind < nptsOutput; ind++)
    {
      double absDiff = abs(outputHostPtr[ind] - 1.);
      if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
    }
  std::cout << "device FFT max error " << maxAbsDiff << std::endl;

  if (a_verbosity >= 3)
    {
      std::cout << "Writing out anything not 1 in output" << std::endl;
      for (size_t ind = 0; ind < nptsOutput; ind++)
        {
          T_OUT outputPoint = outputHostPtr[ind];
          fftx::point_t<3> pt = pointFromPositionBox(ind, outputDomain);
          writeErrorValue(pt, outputPoint, 1.);
        }
    }
  delete[] outputHostPtr;
}


template<typename T_IN, typename T_OUT, class Transformer>
void runSpiral(Transformer& a_tfm,
               int a_verbosity)
{
  if (!a_tfm.isDefined())
    {
      return;
    }

  fftx::point_t<3> inputSize = a_tfm.inputSize();
  fftx::point_t<3> outputSize = a_tfm.outputSize();
  if (a_verbosity >= 2)
    {
      std::cout << "Input " << inputSize
                << " output " << outputSize << std::endl;
    }
  fftx::box_t<3> inputDomain = domain1(inputSize);
  fftx::box_t<3> outputDomain = domain1(outputSize);
  
  /*
    Define and set input.
  */
  fftx::array_t<3, T_IN> inputArrayHost(inputDomain);
  auto nptsInput = inputDomain.size();
  auto bytesInput = nptsInput * sizeof(T_IN);
  T_IN* inputDevicePtr = NULL;
  DEVICE_MALLOC(&inputDevicePtr, bytesInput);
  forall([](T_IN(&v), const fftx::point_t<3>& p)
         {
           if (p == fftx::point_t<3>::Unit())
             {
               set1(v);
             }
           else
             {
               set0(v);
             }
         }, inputArrayHost);
  symmetrize<T_IN, T_OUT>(inputArrayHost, outputDomain);
  T_IN* inputHostPtr = inputArrayHost.m_data.local();
  DEVICE_MEM_COPY(inputDevicePtr, // dest
                  inputHostPtr, // source
                  bytesInput, // bytes
                  MEM_COPY_HOST_TO_DEVICE); // type

  /*
    Define output.
  */
  
  T_OUT* outputDevicePtr = NULL;
  auto nptsOutput = outputDomain.size();
  auto bytesOutput = nptsOutput * sizeof(T_OUT);
  DEVICE_MALLOC(&outputDevicePtr, bytesOutput);

  /*
    Run transform with SPIRAL-generated code.
   */
  fftx::array_t<3, T_IN> inputArrayDevice
    (fftx::global_ptr<T_IN>(inputDevicePtr,0,1), inputDomain);
  fftx::array_t<3, T_OUT> outputArrayDevice
    (fftx::global_ptr<T_OUT>(outputDevicePtr,0,1), outputDomain);

  if (a_verbosity >= 2)
    {
      std::cout << "call transform on " << a_tfm.size() << std::endl;
    }
  a_tfm.transform(inputArrayDevice, outputArrayDevice);

  /*
    Copy output from device to host.
  */
  T_OUT* outputHostPtr = new T_OUT[nptsOutput];
  DEVICE_MEM_COPY(outputHostPtr, // dest
                  outputDevicePtr, // source
                  bytesOutput, // bytes
                  MEM_COPY_DEVICE_TO_HOST); // type

  DEVICE_FREE(inputDevicePtr);
  DEVICE_FREE(outputDevicePtr);

  double maxAbsDiff = 0.;
  for (size_t ind = 0; ind < nptsOutput; ind++)
    {
      double absDiff = abs(outputHostPtr[ind] - 1.);
      if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
    }
  std::cout << "Spiral FFT max error " << maxAbsDiff << std::endl;

  if (a_verbosity >= 3)
    {
      std::cout << "Writing out anything not 1 in output" << std::endl;
      for (size_t ind = 0; ind < nptsOutput; ind++)
        {
          T_OUT outputPoint = outputHostPtr[ind];
          fftx::point_t<3> pt = pointFromPositionBox(ind, outputDomain);
          writeErrorValue(pt, outputPoint, 1.);
        }
    }
  delete[] outputHostPtr;
}


int main(int argc, char* argv[])
{
  printf("Usage:  %s [nx] [ny] [nz] [runwhich=0] [verbosity=0]\n", argv[0]);
  printf("runwhich 0 for device FFT, 1 for Spiral FFT, 2 for both\n");
  printf("verbosity 0 for least, 2 for subparts, 3 for error at every point\n");
  if (argc < 3)
    {
      printf("Missing dimensions\n");
      exit(0);
    }
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int nz = atoi(argv[3]);
  fftx::point_t<3> sz({{nx, ny, nz}});

  int runwhich = 0;
  int verbosity = 0;
  if (argc > 4)
    {
      runwhich = atoi(argv[4]);
      if (argc > 5)
        {
          verbosity = atoi(argv[5]);
        }
    }
  bool doDeviceFFT = (runwhich % 2 == 0); // 0 or 2
  bool doSpiral = (runwhich >= 1); // 1 or 2
  std::cout << "Running " << sz
            << " which " << runwhich
            << " device FFT " << (doDeviceFFT ? "YES" : "NO")
            << " Spiral " << (doSpiral ? "YES" : "NO")
            << " verbosity " << verbosity << std::endl;

  if (doDeviceFFT)
    {
      {
        std::cout << "=== mddft<" << sz << "> on device FFT ===" << std::endl;
        runDeviceFFT(mddftDevice, sz, verbosity);
      }
      {
        std::cout << "=== imddft<" << sz << "> on device FFT ===" << std::endl;
        runDeviceFFT(imddftDevice, sz, verbosity);
      }
      {
        std::cout << "=== mdprdft<" << sz << "> on device FFT ===" << std::endl;
        runDeviceFFT(mdprdftDevice, sz, verbosity);
      }
      {
        std::cout << "=== imdprdft<" << sz << "> on device FFT ===" << std::endl;
        runDeviceFFT(imdprdftDevice, sz, verbosity);
      }
    }
  if (doSpiral)
    {
      {
        std::cout << "=== mddft<" << sz << "> on Spiral ===" << std::endl;
        fftx::mddft<3> tfm(sz);
        runSpiral<std::complex<double>, std::complex<double>>
          (tfm, verbosity);
      }
      {
        std::cout << "=== imddft<" << sz << "> on Spiral ===" << std::endl;
        fftx::imddft<3> tfm(sz);
        runSpiral<std::complex<double>, std::complex<double>>
          (tfm, verbosity);
      }
      {
        std::cout << "=== mdprdft<" << sz << "> on Spiral ===" << std::endl;
        fftx::mdprdft<3> tfm(sz);
        runSpiral<double, std::complex<double>>
          (tfm, verbosity);
      }
      {
        std::cout << "=== imdprdft<" << sz << "> on Spiral ===" << std::endl;
        fftx::imdprdft<3> tfm(sz);
        runSpiral<std::complex<double>, double>
          (tfm, verbosity);
      }
    }
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
