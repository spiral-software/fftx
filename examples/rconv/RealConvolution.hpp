#ifndef REAL_CONVOLUTION_HEADER
#define REAL_CONVOLUTION_HEADER

#include <cmath> // Without this, abs returns zero!
#include <random>

#include "fftx3.hpp"
#include "fftx3utilities.h"
#include "interface.hpp"
#include "rconvObj.hpp"

// #if defined(FFTX_CUDA) || defined(FFTX_HIP)
// #include "fftx_rconv_gpu_public.h"
// #else
// #include "fftx_rconv_cpu_public.h"
// #endif

#include "device_macros.h"
// #include "rconv.fftx.precompile.hpp"

std::mt19937 generator;
// unifRealDist is uniform over the reals in (-1/2, 1/2).
std::uniform_real_distribution<double> unifRealDist;

// Return random real number.
double unifReal()
{
  return unifRealDist(generator);
}

// Fill a_arr with real numbers distributed uniformly in (-1/2, 1/2).
template<int DIM>
void unifRealArray(fftx::array_t<DIM, double>& a_arr)
{
  forall([](double(&v),
            const fftx::point_t<DIM>& p)
         {
           v = unifReal();
         }, a_arr);
}

template<int DIM>
class RealConvolution
{
public:
  RealConvolution()
  {
    m_tp = EMPTY;
  }

  // constructor with FFTX handle
  RealConvolution(
                  // (fftx::array_t<DIM, double>&,
                  //  fftx::array_t<DIM, double>&,
                  //  fftx::array_t<DIM, double>&),
                  RCONVProblem rp,
                  std::vector<int>& sizes,
                  fftx::box_t<DIM> a_domain,
                  fftx::box_t<DIM> a_fdomain)
  {
    // m_functionPtr = prob;
    m_rp = rp;
    m_sizes = sizes;
    m_domain = a_domain;
    m_fdomain = a_fdomain;
    m_tp = FFTX_HANDLE;
  }
  
  // // constructor with FFTX library transformer
  // RealConvolution(fftx::rconv<DIM>* a_transformerPtr)
  // {
  //   m_transformerPtr = a_transformerPtr;
  //   assert(m_transformerPtr->inputSize() == m_transformerPtr->outputSize());
  //   m_domain = domainFromSize<DIM>(m_transformerPtr->inputSize());
  //   m_fdomain = domainFromSize<DIM>(m_transformerPtr->sizeHalf());
  //   m_tp = FFTX_LIB;
  // }
  
  virtual bool isDefined()
  { return (m_tp != EMPTY); }

  fftx::box_t<DIM>& domain()
  { return m_domain; }
      
  fftx::box_t<DIM>& fdomain()
  { return m_fdomain; }

  virtual void exec(fftx::array_t<DIM, double>& a_input,
                    fftx::array_t<DIM, double>& a_output,
                    fftx::array_t<DIM, double>& a_symbol)
  {
    if (m_tp == EMPTY)
      {
        std::cout << "calling exec on empty RealConvolution" << std::endl;
      }
    else if (m_tp == FFTX_HANDLE || m_tp == FFTX_LIB)
      {
#if defined(FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
        // on GPU
        auto input_size = m_domain.size();
        auto output_size = m_domain.size();
        auto symbol_size = m_fdomain.size();

        auto input_bytes = input_size * sizeof(double);
        auto output_bytes = output_size * sizeof(double);
        auto symbol_bytes = symbol_size * sizeof(double);

#if defined FFTX_HIP 
        hipDeviceptr_t inputPtr;
        hipDeviceptr_t outputPtr;
        hipDeviceptr_t symbolPtr;
#elif defined FFTX_CUDA
        CUdeviceptr inputPtr;
        CUdeviceptr outputPtr;
        CUdeviceptr symbolPtr;
#endif

#if defined(FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MALLOC((void **)&inputPtr, input_bytes);
        DEVICE_MALLOC((void **)&outputPtr, output_bytes);
        DEVICE_MALLOC((void **)&symbolPtr, symbol_bytes);

        DEVICE_MEM_COPY((void*)inputPtr, a_input.m_data.local(), input_bytes,
                        MEM_COPY_HOST_TO_DEVICE);
        DEVICE_MEM_COPY((void*)symbolPtr, a_symbol.m_data.local(), symbol_bytes,
                        MEM_COPY_HOST_TO_DEVICE);
#else
		sycl::buffer<double> buf_inputPtr(a_input.m_data.local(), input_size);
		sycl::buffer<double> buf_outputPtr(a_output.m_data.local(), output_size);
		sycl::buffer<double> buf_symbolPtr(a_symbol.m_data.local(), symbol_size);
#endif

#if defined FFTX_HIP
        fftx::array_t<DIM, hipDeviceptr_t> inputDevice(fftx::global_ptr<hipDeviceptr_t>
                                               (&inputPtr, 0, 1), m_domain);
        fftx::array_t<DIM, hipDeviceptr_t> outputDevice(fftx::global_ptr<hipDeviceptr_t>
                                                (&outputPtr, 0, 1), m_domain);
        fftx::array_t<DIM, hipDeviceptr_t> symbolDevice(fftx::global_ptr<hipDeviceptr_t>
                                                (&symbolPtr, 0, 1), m_fdomain);

        std::vector<void*> args{outputPtr, inputPtr, symbolPtr};
#elif defined FFTX_SYCL
		/*fftx::array_t<DIM, sycl::buffer<double>> inputDevice(fftx::global_ptr<sycl::buffer<double>>
                                               (&buf_inputPtr, 0, 1), m_domain);
        fftx::array_t<DIM, sycl::buffer<double>> outputDevice(fftx::global_ptr<sycl::buffer<double>>
                                                (&buf_outputPtr, 0, 1), m_domain);
        fftx::array_t<DIM, sycl::buffer<double>> symbolDevice(fftx::global_ptr<sycl::buffer<double>>
                                                (&buf_symbolPtr, 0, 1), m_fdomain);
*/
        std::vector<void*> args{(void*)&(buf_outputPtr), (void*)&(buf_inputPtr), (void*)&(buf_symbolPtr)};
#else
        fftx::array_t<DIM, CUdeviceptr> inputDevice(fftx::global_ptr<CUdeviceptr>
                                               (&inputPtr, 0, 1), m_domain);
        fftx::array_t<DIM, CUdeviceptr> outputDevice(fftx::global_ptr<CUdeviceptr>
                                                (&outputPtr, 0, 1), m_domain);
        fftx::array_t<DIM, CUdeviceptr> symbolDevice(fftx::global_ptr<CUdeviceptr>
                                                (&symbolPtr, 0, 1), m_fdomain);

        std::vector<void*> args{&outputPtr, &inputPtr, &symbolPtr};
#endif      
        m_rp.setArgs(args);
        m_rp.setSizes(m_sizes);
        m_rp.transform();

        // if (m_tp == FFTX_HANDLE)
        //   {
        //     (*m_functionPtr)(inputDevice, outputDevice, symbolDevice);
        //   }
        // else if (m_tp == FFTX_LIB)
        //   {
        //     m_transformerPtr->transform(inputDevice, outputDevice, symbolDevice);
        //   }
#if defined(FFTX_HIP) || defined(FFTX_CUDA)
		DEVICE_MEM_COPY(a_output.m_data.local(), (void*)outputPtr, output_bytes,
                        MEM_COPY_DEVICE_TO_HOST);

        DEVICE_FREE((void*)inputPtr);
        DEVICE_FREE((void*)outputPtr);
#endif
#else
        std::vector<void*> args{(void*)a_output.m_data.local(),
                                (void*)a_input.m_data.local(),
                                (void*)a_symbol.m_data.local()};
        m_rp.setArgs(args);
        m_rp.setSizes(m_sizes);
        m_rp.transform();
#endif
      }
  }

protected:

  enum TransformType { EMPTY = 0, FFTX_HANDLE = 1, FFTX_LIB = 2};

  TransformType m_tp;
  RCONVProblem m_rp;
  std::vector<int> m_sizes;
  fftx::box_t<DIM> m_domain;
  fftx::box_t<DIM> m_fdomain;

  // case FFTX_HANDLE
  fftx::handle_t (*m_functionPtr) (fftx::array_t<DIM, double>&,
                                   fftx::array_t<DIM, double>&,
                                   fftx::array_t<DIM, double>&);
  
  // case FFTX_LIB
  // fftx::rconv<DIM>* m_transformerPtr;
};

template<int DIM>
class TestRealConvolution
{
public:

  TestRealConvolution(RealConvolution<DIM> a_tfm,
                      int a_rounds,
                      int a_verbosity)
  {
    if (!a_tfm.isDefined())
      {
        std::cout << "transformation not defined" << std::endl;
        return;
      }

    m_tfm = a_tfm;
    m_rounds = a_rounds;
    m_verbosity = a_verbosity;

    m_domain = m_tfm.domain();
    m_fdomain = m_tfm.fdomain();

    double err = 0.;
    updateMax(err, testConstantSymbol());
    updateMax(err, testDelta());
    updateMax(err, testPoisson());
    printf("%dD tests in %d rounds max error %11.5e\n",
           DIM, m_rounds, err);
  }

protected:

  enum VerbosityLevel { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};

  RealConvolution<DIM> m_tfm;
  
  int m_rounds;
  
  int m_verbosity;

  fftx::box_t<DIM> m_domain;
  fftx::box_t<DIM> m_fdomain;

  // Fill a_arr with real numbers distributed uniformly in (-1/2, 1/2).
  void unifRealArray(fftx::array_t<DIM, double>& a_arr)
  {
    forall([](double(&v),
              const fftx::point_t<DIM>& p)
           {
             v = unifReal();
           }, a_arr);
  }

  double testConstantSymbol()
  {
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        printf("calling testConstantSymbol<%d>\n", DIM);
      }
    fftx::array_t<DIM, double> input(m_domain);
    fftx::array_t<DIM, double> output(m_domain);
    fftx::array_t<DIM, double> symbol(m_fdomain);
    
    double scaling = 1. / (m_domain.size()*1.);
    setConstant(symbol, scaling);
    double errConstantSymbol = 0.;
    for (int itn = 1; itn <= m_rounds; itn++)
      {
        unifRealArray(input);
        m_tfm.exec(input, output, symbol);
        double err = absMaxDiffArray(input, output);
        updateMax(errConstantSymbol, err);
        if (m_verbosity >= SHOW_ROUNDS)
          {
            printf("%dD random input with constant symbol max error %11.5e\n",
                   DIM, err);
          }
      }
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        printf("%dD random input with constant symbol in %d rounds: max error %11.5e\n",
               DIM, m_rounds, errConstantSymbol);
      }
    return errConstantSymbol;
  }

  double testDelta()
  {
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        printf("calling testDelta<%d>\n", DIM);
      }
    fftx::array_t<DIM, double> input(m_domain);
    fftx::array_t<DIM, double> output(m_domain);
    fftx::array_t<DIM, double> symbol(m_fdomain);

    setConstant(input, 2.);

    fftx::point_t<DIM> cornerLo = m_domain.lo;
    double scaling = 1. / (m_domain.size()*1.);
    /*
    forall([cornerLo, scaling](double(&v), const fftx::point_t<DIM>& p)
           {
             if (p == cornerLo)
               {
                 v = scaling; // WAS 1;
               }
             else
               {
                 v = 0.;
               }
           }, symbol);
    */
    // Substitute for forall.
    setConstant(symbol, 0.);
    auto symbolPtr = symbol.m_data.local();
    auto indCornerLo = positionInBox(cornerLo, m_fdomain);
    symbolPtr[indCornerLo] = scaling;

    m_tfm.exec(input, output, symbol);
    double errDelta = absMaxDiffArray(input, output);
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        printf("%dD delta function test: max error %11.5e\n", DIM, errDelta);
      }
    return errDelta;
  }

  double testPoisson()
  {
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        printf("calling testPoisson<%d>\n", DIM);
      }
    fftx::array_t<DIM, double> input(m_domain);
    fftx::array_t<DIM, double> output(m_domain);
    fftx::array_t<DIM, double> symbol(m_fdomain);

    fftx::point_t<DIM> lo = m_domain.lo;
    fftx::point_t<DIM> hi = m_domain.hi;
    double center[DIM];
    fftx::point_t<DIM> extents = m_domain.extents();
    int extentMin = extents[0];
    for (int d = 0; d < DIM; d++)
      {
        center[d] = (lo[d] + hi[d]) * 0.5;
        if (extents[d] < extentMin)
          {
            extentMin = extents[d];
          }
      }
    // Set radius to extentMin/sqrt(2)/2.
    double radius2 = (extentMin * extentMin) * (1./8.);
    /*
    forall([center, radius2](double(&v), const fftx::point_t<DIM>& p)
           {
             double dist2 = 0.;
             for (int d = 0; d < DIM; d++)
               {
                 double displacement2 = p[d] - center[d];
                 displacement2 *= displacement2;
                 dist2 += displacement2;
               }
             if (dist2 < radius2)
               {
                 // v = 1.;
                 // For periodicity, need sum of rhs over all points to be zero.
                 v = p[0] - center[0];
               }
             else
               {
                 v = 0.;
               }
           }, input);
    */
    // Substitute for forall.
    auto inputPtr = input.m_data.local();
    auto input_size = m_domain.size();
    for (size_t ind = 0; ind < input_size; ind++)
      {
        fftx::point_t<DIM> p = pointFromPositionBox(ind, m_domain);
        double dist2 = 0.;
        for (int d = 0; d < DIM; d++)
          {
            double displacement2 = p[d] - center[d];
            displacement2 *= displacement2;
            dist2 += displacement2;
          }
        if (dist2 < radius2)
          {
            // v = 1.;
            // For periodicity, need sum of rhs over all points to be zero.
            inputPtr[ind] = p[0] - center[0];
          }
        else
          {
            inputPtr[ind] = 0.;
          }
      }

    fftx::point_t<DIM> cornerLo = m_domain.lo;
    /*
    size_t normalize = m_domain.size();
    forall([cornerLo, extents, normalize](double(&v), const fftx::point_t<DIM>& p)
           {
             if (p == cornerLo)
               {
                 v = 0.;
               }
             else
               {
                 double sin2sum = 0.;
                 for (int d = 0; d < DIM; d++)
                   {
                     double sin1 = sin((p[d]-cornerLo[d])*M_PI/(extents[d]*1.));
                     sin2sum += sin1 * sin1;
                   }
                 v = -1. / ((4 * normalize) * sin2sum);
               }
           }, symbol);
    */
    // Substitute for forall.
    auto symbolPtr = symbol.m_data.local();
    auto symbol_size = m_fdomain.size();
    for (size_t ind = 0; ind < symbol_size; ind++)
      {
        fftx::point_t<DIM> p = pointFromPositionBox(ind, m_fdomain);
        if (p == cornerLo)
          {
            symbolPtr[ind] = 0.;
          }
        else
          {
            double sin2sum = 0.;
            for (int d = 0; d < DIM; d++)
              {
                double sin1 = sin((p[d]-cornerLo[d])*M_PI/(extents[d]*1.));
                sin2sum += sin1 * sin1;
              }
            symbolPtr[ind] = -1. / ((4*input_size) * sin2sum);
          }
      }

    m_tfm.exec(input, output, symbol);

    fftx::array_t<DIM, double> lap2output(m_domain);
    laplacian2periodic(lap2output, output);
  
    double errPoisson = absMaxDiffArray(lap2output, input);
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        printf("%dD Poisson test: max error %11.5e\n",
               DIM, errPoisson);
      }
    return errPoisson;
  }
};
#endif
