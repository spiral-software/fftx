#include <cmath> // Without this, abs returns zero!
#include <random>

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

#include "device_macros.h"

enum VerbosityLevel { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};
  
std::mt19937 generator;
// unifRealDist is uniform over the reals in (-1/2, 1/2).
std::uniform_real_distribution<double> unifRealDist;
// unifInt[d] is uniform over the integers in domain.lo[d] : domain.hi[d]
std::uniform_int_distribution<int> unifInt[3];

// Return random point in domain.
template<int DIM>
fftx::point_t<DIM> unifPoint()
{
  fftx::point_t<DIM> ret;
  for (int d = 0; d < DIM; d++)
    {
      ret[d] = unifInt[d](generator);
    }
  return ret;
}

// Return random real number.
double unifReal()
{
  return unifRealDist(generator);
}

// Return random complex number.
std::complex<double> unifComplex()
{
  return std::complex<double>(unifReal(), unifReal());
}

inline void getUnifScalar(double& a_scalar)
{
  a_scalar = unifReal();
}

inline void getUnifScalar(std::complex<double>& a_scalar)
{
  a_scalar = unifComplex();
}

template<typename T>
inline T unifScalar()
{
  T ret;
  getUnifScalar(ret);
  return ret;
}

template<typename T_IN, typename T_OUT>
void getUnifScalarPair(T_IN& a_scalarIn,
                       T_OUT& a_scalarOut);

void getUnifScalarPair(std::complex<double>& a_scalarIn,
                       std::complex<double>& a_scalarOut)
{
  a_scalarIn = unifComplex();
  a_scalarOut = a_scalarIn;
}

void getUnifScalarPair(double& a_scalarIn,
                       std::complex<double>& a_scalarOut)
{
  a_scalarIn = unifReal();
  a_scalarOut = std::complex<double>(a_scalarIn, 0.);
}

void getUnifScalarPair(std::complex<double>& a_scalarIn,
                       double& a_scalarOut)
                              
{
  a_scalarOut = unifReal();
  a_scalarIn = std::complex<double>(a_scalarOut, 0.);
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

// Fill a_arr with complex numbers with real and imaginary components distributed uniformly in (-1/2, 1/2).
template<int DIM>
void unifComplexArray(fftx::array_t<DIM, std::complex<double>>& a_arr)
{
  forall([](std::complex<double>(&v),
            const fftx::point_t<DIM>& p)
         {
           v = unifComplex();
         }, a_arr);
}

template<int DIM, typename T>
void unifArray(fftx::array_t<DIM, T>& a_arr);

template<int DIM>
void unifArray(fftx::array_t<DIM, double>& a_arr)
{
  unifRealArray(a_arr);
}

template<int DIM>
void unifArray(fftx::array_t<DIM, std::complex<double>>& a_arr)
{
  unifComplexArray(a_arr);
}

fftx::box_t<3> domainFromSize(const fftx::point_t<3>& a_size)
{
  fftx::box_t<3> bx(fftx::point_t<3>({{1, 1, 1}}),
                    fftx::point_t<3>({{a_size[0], a_size[1], a_size[2]}}));
  return bx;
}

template<int DIM>
size_t pointProduct(const fftx::point_t<DIM>& a_pt)
{
  size_t prod = 1;
  for (int d = 0; d < DIM; d++)
    {
      prod *= a_pt[d];
    }
  return prod;
}

// Set a_arr to a_scaling at point a_fixed, and 0 elsewhere.
template<int DIM, typename T>
void setUnitImpulse(fftx::array_t<DIM, T>& a_arr,
                    const fftx::point_t<DIM>& a_fixed,
                    T a_scaling = scalarVal<T>(1.) )
{
  forall([a_fixed, a_scaling](T(&v),
                              const fftx::point_t<DIM>& p)
         {
           if (p == a_fixed)
             {
               v = a_scaling;
             }
           else
             {
               v = scalarVal<T>(0.);
             }
         }, a_arr);
}

// Set a_arr to product of waves from impulse at a_fixed.
template<int DIM>
void setProductWaves(fftx::array_t<DIM, std::complex<double>>& a_arr,
                     const fftx::point_t<DIM>& a_extent,
                     const fftx::point_t<DIM>& a_fixed,
                     int a_sign)
{
  fftx::point_t<DIM> lo = a_arr.m_domain.lo;
  std::complex<double> omega[DIM];
  for (int d = 0; d < DIM; d++)
    {
      double th = (a_sign*2*(a_fixed[d] - lo[d])) * M_PI / (a_extent[d] * 1.);
      omega[d] = std::complex<double>(cos(th), sin(th));
    }
  forall([omega, lo](std::complex<double>(&v),
                     const fftx::point_t<DIM>& p)
         {
           v = std::complex<double>(1., 0.);
           for (int d = 0; d < DIM; d++)
             {
               v *= pow(omega[d], p[d] - lo[d]);
             }
         }, a_arr);
}

template<int DIM>
void setRotator(fftx::array_t<DIM, std::complex<double>>& a_arr,
                const fftx::box_t<DIM>& a_dom,
                int a_dim,
                int a_shift)
{
  fftx::point_t<DIM> lo = a_dom.lo;
  fftx::point_t<DIM> hi = a_dom.hi;
  fftx::point_t<DIM> fixed = lo;
  if (a_shift > 0)
    {
      fixed[a_dim] = lo[a_dim] + a_shift;
    }
  else if (a_shift < 0)
    {
      fixed[a_dim] = hi[a_dim] - (a_shift+1);
    }
  // std::cout << "setRotator in " << a_dim << " shift " << a_shift
  // << " waves " << fixed << " of " << a_arr.m_domain << std::endl;
  setProductWaves(a_arr, a_dom.extents(), fixed, -1);
}

template<int DIM, typename T_IN, typename T_OUT>
void TransformCall(fftx::transformer<DIM, T_IN, T_OUT>& a_tfm,
                   fftx::array_t<DIM, T_IN>& a_input, // make this const?
                   fftx::array_t<DIM, T_OUT>& a_output)
{
  auto inputDomain = a_input.m_domain;
  auto outputDomain = a_output.m_domain;

  auto input_size = inputDomain.size();
  auto output_size = outputDomain.size();

  auto input_bytes = input_size * sizeof(T_IN);
  auto output_bytes = output_size * sizeof(T_OUT);

  T_IN* inputHostPtr = a_input.m_data.local();
  T_OUT* outputHostPtr = a_output.m_data.local();

  T_IN* inputDevicePtr;
  T_OUT* outputDevicePtr;

  DEVICE_MALLOC(&inputDevicePtr, input_bytes);
  DEVICE_MALLOC(&outputDevicePtr, output_bytes);

  fftx::array_t<DIM, T_IN> inputDevice(fftx::global_ptr<T_IN>
                                       (inputDevicePtr, 0, 1), inputDomain);
  fftx::array_t<DIM, T_OUT> outputDevice(fftx::global_ptr<T_OUT>
                                         (outputDevicePtr, 0, 1), outputDomain);

  DEVICE_MEM_COPY(inputDevicePtr, inputHostPtr, input_bytes,
                  MEM_COPY_HOST_TO_DEVICE);

  a_tfm.transform2(inputDevice, outputDevice);

  DEVICE_MEM_COPY(outputHostPtr, outputDevicePtr, output_bytes,
                  MEM_COPY_DEVICE_TO_HOST);

  DEVICE_FREE(inputDevicePtr);
  DEVICE_FREE(outputDevicePtr);
}

template<int DIM, typename T_IN, typename T_OUT>
double test1Transform(fftx::transformer<DIM, T_IN, T_OUT>& a_tfm,
                      int a_rounds,
                      int a_verbosity)
{
  fftx::box_t<DIM> inputDomain = domainFromSize(a_tfm.inputSize());
  fftx::box_t<DIM> outputDomain = domainFromSize(a_tfm.outputSize());
  
  fftx::array_t<DIM, T_IN> inA(inputDomain);
  fftx::array_t<DIM, T_IN> inB(inputDomain);
  fftx::array_t<DIM, T_IN> LCin(inputDomain);

  fftx::array_t<DIM, T_OUT> outA(outputDomain);
  fftx::array_t<DIM, T_OUT> outB(outputDomain);
  fftx::array_t<DIM, T_OUT> LCout(outputDomain);
  fftx::array_t<DIM, T_OUT> outLCin(outputDomain);

  double errtest1 = 0.;
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      T_IN alphaIn, betaIn;
      T_OUT alphaOut, betaOut;
      getUnifScalarPair(alphaIn, alphaOut);
      getUnifScalarPair(betaIn, betaOut);
      unifArray(inA);
      unifArray(inB);
      sumArrays(LCin, inA, inB, alphaIn, betaIn);

      TransformCall(a_tfm, inA, outA);
      TransformCall(a_tfm, inB, outB);
      sumArrays(LCout, outA, outB, alphaOut, betaOut);
      TransformCall(a_tfm, LCin, outLCin);
      double err = absMaxDiffArray(outLCin, LCout);
      updateMax(errtest1, err);
      if (a_verbosity >= SHOW_ROUNDS)
        {
          printf("%dD linearity test round %d max error %11.5e\n", DIM, itn, err);
        }
    }
  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD Test 1 (linearity) in %d rounds: max error %11.5e\n", DIM, a_rounds, errtest1);
    }
  return errtest1;
}


template<int DIM, typename T_IN, typename T_OUT>
double test2impulse1(fftx::transformer<DIM, T_IN, T_OUT>& a_tfm,
                     int a_verbosity)
{ // Unit impulse at low corner.
  fftx::box_t<DIM> inputDomain = domainFromSize(a_tfm.inputSize());
  fftx::box_t<DIM> outputDomain = domainFromSize(a_tfm.outputSize());

  fftx::array_t<DIM, T_IN> inImpulse(inputDomain);
  fftx::array_t<DIM, T_OUT> outImpulse(outputDomain);
  fftx::array_t<DIM, T_OUT> all1out(outputDomain);
  setUnitImpulse(inImpulse, inputDomain.lo);
  setConstant(all1out, scalarVal<T_OUT>(1.));
  TransformCall(a_tfm, inImpulse, outImpulse);
  double errtest2impulse1 = absMaxDiffArray(outImpulse, all1out);
  if (a_verbosity >= SHOW_SUBTESTS)
    {
       printf("%dD unit impulse low corner test: max error %11.5e\n",
              DIM, errtest2impulse1);
    }
  return errtest2impulse1;
}

template<int DIM, typename T_IN, typename T_OUT>
double test2impulsePlus(fftx::transformer<DIM, T_IN, T_OUT>& a_tfm,
                        int a_rounds,
                        int a_verbosity)
{ // Unit impulse at low corner.
  fftx::box_t<DIM> inputDomain = domainFromSize(a_tfm.inputSize());
  fftx::box_t<DIM> outputDomain = domainFromSize(a_tfm.outputSize());

  fftx::array_t<DIM, T_IN> inImpulse(inputDomain);
  fftx::array_t<DIM, T_OUT> outImpulse(outputDomain);
  fftx::array_t<DIM, T_OUT> all1out(outputDomain);
  setUnitImpulse(inImpulse, inputDomain.lo);
  setConstant(all1out, scalarVal<T_OUT>(1.));
  TransformCall(a_tfm, inImpulse, outImpulse);

  fftx::array_t<DIM, T_IN> inRand(inputDomain);
  fftx::array_t<DIM, T_IN> inImpulseMinusRand(inputDomain);

  fftx::array_t<DIM, T_OUT> outRand(outputDomain);
  fftx::array_t<DIM, T_OUT> outImpulseMinusRand(outputDomain);
  fftx::array_t<DIM, T_OUT> mysum(outputDomain);
  
  // Check that for random arrays inRand,
  // fft(inRand) + fft(inImpulse - inRand) = fft(inImpulse) = all1out.
  double errtest2impulsePlus = 0.;
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      unifArray(inRand);
      TransformCall(a_tfm, inRand, outRand);
      diffArrays(inImpulseMinusRand, inImpulse, inRand);
      TransformCall(a_tfm, inImpulseMinusRand, outImpulseMinusRand);
      sumArrays(mysum, outRand, outImpulseMinusRand);
      double err = absMaxDiffArray(mysum, all1out);
      updateMax(errtest2impulsePlus, err);
      if (a_verbosity >= SHOW_ROUNDS)
          {
            printf("%dD random + unit impulse low corner test round %d max error %11.5e\n", DIM, itn, err);
          }
    }

  if (a_verbosity >= SHOW_SUBTESTS)
    {
      printf("%dD unit impulse low corner test in %d rounds: max error %11.5e\n",
             DIM, a_rounds, errtest2impulsePlus);
    }
  return errtest2impulsePlus;
}

template<int DIM, typename T_IN, typename T_OUT>
double test2constant(fftx::transformer<DIM, T_IN, T_OUT>& a_tfm,
                     int a_verbosity)
{ // Check that constant maps back to unit impulse at low corner.
  fftx::box_t<DIM> inputDomain = domainFromSize(a_tfm.inputSize());
  fftx::box_t<DIM> outputDomain = domainFromSize(a_tfm.outputSize());
  fftx::array_t<DIM, T_IN> all1in(inputDomain);
  setConstant(all1in, scalarVal<T_IN>(1.));

  fftx::array_t<DIM, T_OUT> magImpulse(outputDomain);
  size_t npts = pointProduct(a_tfm.size());
  T_OUT mag = scalarVal<T_OUT>(npts * 1.);
  setUnitImpulse(magImpulse, outputDomain.lo, mag);

  fftx::array_t<DIM, T_OUT> outImpulse(outputDomain);
  TransformCall(a_tfm, all1in, outImpulse);

  double errtest2constant = absMaxDiffArray(outImpulse, magImpulse);
  if (a_verbosity >= SHOW_SUBTESTS)
    {
       printf("%dD constant test: max error %11.5e\n", DIM, errtest2constant);
    }
  return errtest2constant;
}

template<int DIM, typename T_IN, typename T_OUT>
double test2constantPlus(fftx::transformer<DIM, T_IN, T_OUT>& a_tfm,
                         int a_rounds,
                         int a_verbosity)
{
  fftx::box_t<DIM> inputDomain = domainFromSize(a_tfm.inputSize());
  fftx::box_t<DIM> outputDomain = domainFromSize(a_tfm.outputSize());

  fftx::array_t<DIM, T_IN> all1in(inputDomain);
  setConstant(all1in, scalarVal<T_IN>(1.));

  fftx::array_t<DIM, T_OUT> magImpulse(outputDomain);
  size_t npts = pointProduct(a_tfm.size());
  T_OUT mag = scalarVal<T_OUT>(npts * 1.);
  setUnitImpulse(magImpulse, outputDomain.lo, mag);

  fftx::array_t<DIM, T_IN> inRand(inputDomain);
  fftx::array_t<DIM, T_IN> inConstantMinusRand(inputDomain);

  fftx::array_t<DIM, T_OUT> outRand(outputDomain);
  fftx::array_t<DIM, T_OUT> outConstantMinusRand(outputDomain);
  fftx::array_t<DIM, T_OUT> outSum(outputDomain);

  // Check that for random arrays inRand,
  // fft(inRand) + fft(all1 - inRand) = fft(all1) = magImpulse.
  double errtest2constantPlus = 0.;
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      unifArray(inRand);
      TransformCall(a_tfm, inRand, outRand);

      diffArrays(inConstantMinusRand, all1in, inRand);
      TransformCall(a_tfm, inConstantMinusRand, outConstantMinusRand);

      sumArrays(outSum, outRand, outConstantMinusRand);
      
      double err = absMaxDiffArray(outSum, magImpulse);
      updateMax(errtest2constantPlus, err);
      if (a_verbosity >= SHOW_ROUNDS)
          {
            printf("%dD random + constant test round %d max error %11.5e\n",
                   DIM, itn, err);
          }
    }

  if (a_verbosity >= SHOW_SUBTESTS)
    {
      printf("%dD random + constant test in %d rounds: max error %11.5e\n",
             DIM, a_rounds, errtest2constantPlus);
  
    }
  return errtest2constantPlus;
}

template<int DIM, typename T_IN>
double test2impulseRandom(fftx::transformer<DIM, T_IN, double>& a_tfm,
                          int a_sign,
                          int a_rounds,
                          int a_verbosity)
{
  // Do nothing if output is real.  Run this test only if output is complex.
  return 0.;
}

template<int DIM, typename T_IN>
double test2impulseRandom(fftx::transformer<DIM, T_IN, std::complex<double>>& a_tfm,
                          int a_sign,
                          int a_rounds,
                          int a_verbosity)
{
  // Check unit impulse at random position.
  fftx::box_t<DIM> inputDomain = domainFromSize(a_tfm.inputSize());
  fftx::box_t<DIM> outputDomain = domainFromSize(a_tfm.outputSize());

  fftx::array_t<DIM, T_IN> inImpulse(inputDomain);
  fftx::array_t<DIM, std::complex<double>> outImpulse(outputDomain);
  fftx::array_t<DIM, std::complex<double>> outCheck(outputDomain);
  double errtest2impulseRandom = 0.;
  // fftx::point_t<DIM> fullExtents = a_tfm.size();
  fftx::point_t<DIM> fullExtents = inputDomain.extents();
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      fftx::point_t<DIM> rpoint = unifPoint<DIM>();
      setUnitImpulse(inImpulse, rpoint);
      TransformCall(a_tfm, inImpulse, outImpulse);
      // Recall inputDomain is whole domain,
      // but outputDomain may be truncated;
      // waves defined on outputDomain,
      // but based on the full inputDomain extents.
      setProductWaves(outCheck, fullExtents, rpoint, a_sign);
      double err = absMaxDiffArray(outImpulse, outCheck);
      updateMax(errtest2impulseRandom, err);
      if (a_verbosity >= SHOW_ROUNDS)
        {
          printf("%dD random impulse test round %d max error %11.5e\n", DIM, itn, err);
        }
    }
  return errtest2impulseRandom;
}


template<int DIM, typename T_IN, typename T_OUT>
double test2Transform(fftx::transformer<DIM, T_IN, T_OUT>& a_tfm,
                      int a_sign,
                      int a_rounds,
                      int a_verbosity)
{
  double errtest2 = 0.;

  updateMax(errtest2,
            test2impulse1(a_tfm, a_verbosity));

  updateMax(errtest2,
            test2impulsePlus(a_tfm, a_rounds, a_verbosity));

  updateMax(errtest2,
            test2constant(a_tfm, a_verbosity));

  updateMax(errtest2,
            test2constantPlus(a_tfm, a_rounds, a_verbosity));

  updateMax(errtest2,
            test2impulseRandom(a_tfm, a_sign, a_rounds, a_verbosity));

  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD Test 2 (impulses) in %d rounds: max error %11.5e\n", DIM, a_rounds, errtest2);
    }
  return errtest2;
}


template<int DIM, typename T_IN>
double test3time(fftx::transformer<DIM, T_IN, double>& a_tfm,
                 int a_sign,
                 int a_rounds,
                 int a_verbosity)
{
  // Do nothing if output is real.  Run this test only if output is complex.
  return 0.;
}

template<int DIM, typename T_IN>
double test3time(fftx::transformer<DIM, T_IN, std::complex<double>>& a_tfm,
                 int a_sign,
                 int a_rounds,
                 int a_verbosity)
{
  fftx::box_t<DIM> inputDomain = domainFromSize(a_tfm.inputSize());
  fftx::box_t<DIM> outputDomain = domainFromSize(a_tfm.outputSize());

  fftx::array_t<DIM, T_IN> inRand(inputDomain);
  fftx::array_t<DIM, T_IN> inRandRot(inputDomain);
  fftx::array_t<DIM, std::complex<double>> outRand(outputDomain);
  fftx::array_t<DIM, std::complex<double>> outRandRot(outputDomain);
  fftx::array_t<DIM, std::complex<double>> rotator(outputDomain);
  fftx::array_t<DIM, std::complex<double>> outRandRotMult(outputDomain);
  double errtest3timeDim[DIM];
  double errtest3time = 0.;
  for (int d = 0; d < DIM; d++)
    {
      errtest3timeDim[d] = 0.;
      setRotator(rotator, inputDomain, d, -a_sign); // +1 for MDDFT, -1 for IMDDFT, -1 for PRDFT
      for (int itn = 1; itn <= a_rounds; itn++)
        {
          unifArray(inRand);
          
          // time-shift test in dimension d
          rotate(inRandRot, inRand, d, 1); // +1 for MDDFT, +1 for IMDDFT, +1 for PRDFT
          TransformCall(a_tfm, inRand, outRand);
          TransformCall(a_tfm, inRandRot, outRandRot);
          productArrays(outRandRotMult, outRandRot, rotator);
          double err = absMaxDiffArray(outRandRotMult, outRand);
          updateMax(errtest3timeDim[d], err);
          updateMax(errtest3time, errtest3timeDim[d]);
          if (a_verbosity >= SHOW_ROUNDS)
            {
              printf("%dD dim %d time-shift test %d max error %11.5e\n",
                     DIM, d, itn, err);
            }
        }
      if (a_verbosity >= SHOW_SUBTESTS)
        {
          printf("%dD dim %d time-shift test in %d rounds: max error %11.5e\n",
                 DIM, d, a_rounds, errtest3timeDim[d]);
        }
    }
  return errtest3time;
}

template<int DIM, typename T_OUT>
double test3frequency(fftx::transformer<DIM, double, T_OUT>& a_tfm,
                      int a_sign,
                      int a_rounds,
                      int a_verbosity)
{
  // Do nothing if input is real. Run this test only if input is complex.
  return 0.;
}

template<int DIM, typename T_OUT>
double test3frequency(fftx::transformer<DIM, std::complex<double>, T_OUT>& a_tfm,
                      int a_sign,
                      int a_rounds,
                      int a_verbosity)
{
  fftx::box_t<DIM> inputDomain = domainFromSize(a_tfm.inputSize());
  fftx::box_t<DIM> outputDomain = domainFromSize(a_tfm.outputSize());

  fftx::array_t<DIM, std::complex<double>> inRand(inputDomain);
  fftx::array_t<DIM, std::complex<double>> inRandMult(inputDomain);
  fftx::array_t<DIM, T_OUT> outRand(outputDomain);
  fftx::array_t<DIM, T_OUT> outRandMult(outputDomain);
  fftx::array_t<DIM, std::complex<double>> rotatorUp(inputDomain);
  fftx::array_t<DIM, T_OUT> outRandMultRot(outputDomain);
  double errtest3frequencyDim[DIM];
  double errtest3frequency = 0.;
  for (int d = 0; d < DIM; d++)
    {
      // frequency-shift test in dimension d
      errtest3frequencyDim[d] = 0.;
      // Recall outputDomain is whole domain,
      // but inputDomain may be truncated;
      // rotatorUp is defined on inputDomain,
      // but based on full outputDomain.
      setRotator(rotatorUp, outputDomain, d, 1);
      for (int itn = 1; itn <= a_rounds; itn++)
        {
          unifComplexArray(inRand);

          productArrays(inRandMult, inRand, rotatorUp);
          TransformCall(a_tfm, inRand, outRand);
          TransformCall(a_tfm, inRandMult, outRandMult);
          rotate(outRandMultRot, outRandMult, d, a_sign);
          double err = absMaxDiffArray(outRandMultRot, outRand);
          updateMax(errtest3frequencyDim[d], err);
          updateMax(errtest3frequency, errtest3frequencyDim[d]);
          if (a_verbosity >= SHOW_ROUNDS)
            {
              printf("%dD dim %d frequency-shift test %d max error %11.5e\n", DIM, d, itn, err);
            }
        }
      if (a_verbosity >= SHOW_SUBTESTS)
        {
          printf("%dD dim %d frequency-shift test in %d rounds: max error %11.5e\n", DIM, d, a_rounds, errtest3frequencyDim[d]);
        }
    }
  return errtest3frequency;
}

template<int DIM, typename T_IN, typename T_OUT>
double test3Transform(fftx::transformer<DIM, T_IN, T_OUT>& a_tfm,
                      int a_sign,
                      int a_rounds,
                      int a_verbosity)
{
  double errtest3 = 0.;

  updateMax(errtest3,
            test3time(a_tfm, a_sign, a_rounds, a_verbosity));

  updateMax(errtest3,
            test3frequency(a_tfm, a_sign, a_rounds, a_verbosity));

  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD Test 3 (shifts) in %d rounds: max error %11.5e\n",
             DIM, a_rounds, errtest3);
    }
  return errtest3;
}


template<int DIM, typename T_IN, typename T_OUT>
void verifyTransform(fftx::transformer<DIM, T_IN, T_OUT>& a_tfm,
                     int a_sign,
                     int a_rounds,
                     int a_verbosity)
{
  if (!a_tfm.isDefined())
    {
      return;
    }

  double err = 0.;

  updateMax(err,
            test1Transform(a_tfm, a_rounds, a_verbosity));

  updateMax(err,
            test2Transform(a_tfm, a_sign, a_rounds, a_verbosity));

  updateMax(err,
            test3Transform(a_tfm, a_sign, a_rounds, a_verbosity));

  printf("%dD test on %s in %d rounds max error %11.5e\n",
         DIM, a_tfm.name().c_str(), a_rounds, err);
}


int main(int argc, char* argv[])
{
  // { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};
  printf("Usage:  %s [verbosity=0] [rounds=20]\n", argv[0]);
  printf("verbosity 0 for summary, 1 for categories, 2 for subtests, 3 for rounds\n");
  int verbosity = 0;
  int rounds = 20;
  if (argc > 1)
    {
      verbosity = atoi(argv[1]);
      if (argc > 2)
        {
          rounds = atoi(argv[2]);
        }
    }
  printf("Running with verbosity %d, random %d rounds\n", verbosity, rounds);

  /*
    Set up random number generator.
  */
  std::random_device rd;
  generator = std::mt19937(rd());
  unifRealDist = std::uniform_real_distribution<double>(-0.5, 0.5);

    // last entry is { 0, 0, 0 }
  fftx::point_t<3> *ents = fftx_mddft_QuerySizes ();
  
  for ( int ind = 0; ents[ind][0] != 0; ind++ )
    {
      fftx::point_t<3> sz = ents[ind];

      for (int d = 0; d < 3; d++)
        {
          unifInt[d] = std::uniform_int_distribution<int>(1, sz[d]);
        }

      {
        fftx::mddft<3> tfm(sz);
        verifyTransform(tfm, -1, rounds, verbosity);
       }

      {
        fftx::imddft<3> tfm(sz);
        verifyTransform(tfm, 1, rounds, verbosity);
       }

      {
        fftx::mdprdft<3> tfm(sz);
        verifyTransform(tfm, -1, rounds, verbosity);
      }

      {
        fftx::imdprdft<3> tfm(sz);
        verifyTransform(tfm, 1, rounds, verbosity);
      }
    }

  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
