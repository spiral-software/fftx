#ifndef FFTX_REAL_CONVOLUTION_HEADER
#define FFTX_REAL_CONVOLUTION_HEADER

#include <cmath> // Without this, abs returns zero!
#include <random>

#include "fftx.hpp"
#include "fftxutilities.hpp"
#include "fftxinterface.hpp"
#include "fftxrconvObj.hpp"
// for testRandomSymbol
#include "fftxmdprdftObj.hpp"
#include "fftximdprdftObj.hpp"

// #if defined(FFTX_CUDA) || defined(FFTX_HIP)
// #include "fftx_rconv_gpu_public.h"
// #else
// #include "fftx_rconv_cpu_public.h"
// #endif

#include "fftxdevice_macros.h"
// #include "fftxrconv.precompile.hpp"

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
        fftx::ErrStream() << "calling exec on empty RealConvolution" << std::endl;
      }
    else if (m_tp == FFTX_HANDLE || m_tp == FFTX_LIB)
      {
        double* inputHostPtr = a_input.m_data.local();
        double* outputHostPtr = a_output.m_data.local();
	double* symbolHostPtr = a_symbol.m_data.local();
#if defined(FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
        // on GPU

#if defined(FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_PTR inputDevicePtr = fftxDeviceMallocForHostArray(a_input);
        FFTX_DEVICE_PTR outputDevicePtr = fftxDeviceMallocForHostArray(a_output);
        FFTX_DEVICE_PTR symbolDevicePtr = fftxDeviceMallocForHostArray(a_symbol);

        fftxCopyHostArrayToDevice(inputDevicePtr, a_input);
        fftxCopyHostArrayToDevice(symbolDevicePtr, a_symbol);
	
        //        fftx::array_t<DIM, FFTX_DEVICE_PTR> inputDevice(fftx::global_ptr<FFTX_DEVICE_PTR>
        //						   (&inputDevicePtr, 0, 1), m_domain);
        //        fftx::array_t<DIM, FFTX_DEVICE_PTR> outputDevice(fftx::global_ptr<FFTX_DEVICE_PTR>
        //						    (&outputDevicePtr, 0, 1), m_domain);
        //        fftx::array_t<DIM, FFTX_DEVICE_PTR> symbolDevice(fftx::global_ptr<FFTX_DEVICE_PTR>
        //						    (&symbolDevicePtr, 0, 1), m_fdomain);

#if defined(FFTX_CUDA)
        std::vector<void*> args{&outputDevicePtr, &inputDevicePtr, &symbolDevicePtr};
#elif defined(FFTX_HIP)
        std::vector<void*> args{outputDevicePtr, inputDevicePtr, symbolDevicePtr};
#endif

#elif defined(FFTX_SYCL)
        auto input_pts = m_domain.size();
        auto output_pts = m_domain.size();
        auto symbol_pts = m_fdomain.size();

	sycl::buffer<double> inputBuffer(inputHostPtr, input_pts);
	sycl::buffer<double> outputBuffer(outputHostPtr, output_pts);
	sycl::buffer<double> symbolBuffer(symbolHostPtr, symbol_pts);
        std::vector<void*> args{(void*)&(outputBuffer), (void*)&(inputBuffer), (void*)&(symbolBuffer)};
#endif

#else // neither CUDA nor HIP nor SYCL
        std::vector<void*> args{(void*)outputHostPtr, (void*)inputHostPtr, (void*)symbolHostPtr };
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
        fftxCopyDeviceToHostArray(a_output, outputDevicePtr);

        fftxDeviceFree(inputDevicePtr);
        fftxDeviceFree(outputDevicePtr);
        fftxDeviceFree(symbolDevicePtr);
#endif
      }
  }

  virtual void exec2(fftx::array_t<DIM, double>& a_input,
                     fftx::array_t<DIM, double>& a_output,
                     fftx::array_t<DIM, double>& a_symbol)
  {
    if (m_tp == EMPTY)
      {
        fftx::ErrStream() << "calling exec2 on empty RealConvolution" << std::endl;
      }
    else if (m_tp == FFTX_HANDLE || m_tp == FFTX_LIB)
      {
        double* inputHostPtr = a_input.m_data.local();
        double* outputHostPtr = a_output.m_data.local();
	double* symbolHostPtr = a_symbol.m_data.local();

        // intermediate arrays
        fftx::array_t<DIM, std::complex<double> > outComplex(m_fdomain);
        fftx::array_t<DIM, std::complex<double> > inComplex(m_fdomain);
        std::complex<double>* outComplexHostPtr = outComplex.m_data.local();
        std::complex<double>* inComplexHostPtr = inComplex.m_data.local();
#if defined(FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
        // on GPU

#if defined(FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_PTR inputDevicePtr = fftxDeviceMallocForHostArray(a_input);
        FFTX_DEVICE_PTR outputDevicePtr = fftxDeviceMallocForHostArray(a_output);
        FFTX_DEVICE_PTR symbolDevicePtr = fftxDeviceMallocForHostArray(a_symbol);

        fftxCopyHostArrayToDevice(inputDevicePtr, a_input);
        fftxCopyHostArrayToDevice(symbolDevicePtr, a_symbol);
	
        //        fftx::array_t<DIM, FFTX_DEVICE_PTR> inputDevice(fftx::global_ptr<FFTX_DEVICE_PTR>
        //						   (&inputDevicePtr, 0, 1), m_domain);
        //        fftx::array_t<DIM, FFTX_DEVICE_PTR> outputDevice(fftx::global_ptr<FFTX_DEVICE_PTR>
        //						    (&outputDevicePtr, 0, 1), m_domain);
        //        fftx::array_t<DIM, FFTX_DEVICE_PTR> symbolDevice(fftx::global_ptr<FFTX_DEVICE_PTR>
        //						    (&symbolDevicePtr, 0, 1), m_fdomain);

#if defined(FFTX_CUDA)
        std::vector<void*> args{&outputDevicePtr, &inputDevicePtr, &symbolDevicePtr};
#elif defined(FFTX_HIP)
        std::vector<void*> args{outputDevicePtr, inputDevicePtr, symbolDevicePtr};
#endif

#elif defined(FFTX_SYCL)
        auto input_pts = m_domain.size();
        auto output_pts = m_domain.size();
        auto symbol_pts = m_fdomain.size();

	sycl::buffer<double> inputBuffer(inputHostPtr, input_pts);
	sycl::buffer<double> outputBuffer(outputHostPtr, output_pts);
	sycl::buffer<double> symbolBuffer(symbolHostPtr, symbol_pts);
        std::vector<void*> args{(void*)&(outputBuffer), (void*)&(inputBuffer), (void*)&(symbolBuffer)};
#endif

#else // neither CUDA nor HIP nor SYCL
        // std::vector<void*> args{(void*)outputHostPtr, (void*)inputHostPtr, (void*)symbolHostPtr };
        std::vector<void*> argsR2C{(void*)outComplexHostPtr, (void*)inputHostPtr, (void*) NULL };
        std::vector<void*> argsC2R{(void*)outputHostPtr, (void*)inComplexHostPtr, (void*) NULL };
#endif
        MDPRDFTProblem tfmR2C(argsR2C, m_sizes, "mdprdft");
        IMDPRDFTProblem tfmC2R(argsC2R, m_sizes, "imdprdft");

        // output outComplexHostPtr, input inputHostPtr
        tfmR2C.transform();

        // Set inComplex = outComplex * symbol.
        auto fpts = m_fdomain.size();
        // std::cout << "m_fdomain points: " << fpts << std::endl;
        for (size_t ind = 0; ind < fpts; ind++)
          {
            inComplexHostPtr[ind] =
              outComplexHostPtr[ind] * symbolHostPtr[ind];
          }

        // output outHostPtr, input inComplexHostPtr
        tfmC2R.transform();

#if defined(FFTX_HIP) || defined(FFTX_CUDA)
        fftxCopyDeviceToHostArray(a_output, outputDevicePtr);

        fftxDeviceFree(inputDevicePtr);
        fftxDeviceFree(outputDevicePtr);
        fftxDeviceFree(symbolDevicePtr);
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
        fftx::ErrStream() << "transformation not defined" << std::endl;
        return;
      }
    fftx::OutStream() << std::scientific << std::setprecision(5);

    m_tfm = a_tfm;
    m_rounds = a_rounds;
    m_verbosity = a_verbosity;

    m_domain = m_tfm.domain();
    m_fdomain = m_tfm.fdomain();

    double err = 0.;
    updateMax(err, testConstantSymbol());
    updateMax(err, testDelta());
    updateMax(err, testPoisson());
    updateMax(err, testRandomSymbol());
    fftx::OutStream() << DIM << "D tests in "
                      << m_rounds << " rounds max error " << err
                      << std::endl;
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
        fftx::OutStream() << "calling testConstantSymbol<"
                          << DIM << ">" << std::endl;
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
            fftx::OutStream() << DIM
                              << "D random input with constant symbol max error "
                              << err << std::endl;
          }
      }
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        fftx::OutStream() << DIM
                          << "D random input with constant symbol in "
                          << m_rounds << " rounds: max error "
                          << errConstantSymbol << std::endl;
      }
    return errConstantSymbol;
  }

  double testDelta()
  {
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        fftx::OutStream() << "calling testDelta<"
                          << DIM << ">" << std::endl;
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
        fftx::OutStream() << DIM
                          << "D delta function test: max error "
                          << errDelta << std::endl;
      }
    return errDelta;
  }

  double testPoisson()
  {
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        fftx::OutStream() << "calling testPoisson<"
                          << DIM << ">" << std::endl;
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
    auto input_pts = m_domain.size();
    for (size_t ind = 0; ind < input_pts; ind++)
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
    auto symbol_pts = m_fdomain.size();
    for (size_t ind = 0; ind < symbol_pts; ind++)
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
            symbolPtr[ind] = -1. / ((4*input_pts) * sin2sum);
          }
      }

    m_tfm.exec(input, output, symbol);

    fftx::array_t<DIM, double> lap2output(m_domain);
    laplacian2periodic(lap2output, output);
  
    double errPoisson = absMaxDiffArray(lap2output, input);
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        fftx::OutStream() << DIM
                          << "D Poisson test: max error "
                          << errPoisson << std::endl;
      }
    return errPoisson;
  }

  double testRandomSymbol()
  {
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        fftx::OutStream() << "calling testRandomSymbol<"
                          << DIM << ">" << std::endl;
      }
    fftx::array_t<DIM, double> input(m_domain);
    fftx::array_t<DIM, double> output(m_domain);
    fftx::array_t<DIM, double> symbol(m_fdomain);

    fftx::array_t<DIM, double> output2(m_domain);
    
    double errRandomSymbol = 0.;
    for (int itn = 1; itn <= m_rounds; itn++)
      { // FIXME: after first iteration, m_tfm doesn't work.
        unifRealArray(input);
        unifRealArray(symbol); // FIXME: set Hermitian symmetry?

        m_tfm.exec(input, output, symbol);

        m_tfm.exec2(input, output2, symbol);

        double err = absMaxDiffArray(output, output2);
        updateMax(errRandomSymbol, err);
        if (m_verbosity >= SHOW_ROUNDS)
          {
            fftx::OutStream() << DIM
                              << "D random input with random symbol max error "
                              << err << std::endl;
          }
      }
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        fftx::OutStream() << DIM
                          << "D random input with random symbol in "
                          << m_rounds << " rounds: max error "
                          << errRandomSymbol << std::endl;
      }
    return errRandomSymbol;
  }

};
#endif
