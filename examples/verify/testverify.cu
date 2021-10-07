#include <cmath> // Without this, abs returns zero!
#include <random>
/*
#include "mddft1.fftx.codegen.hpp"
#include "imddft1.fftx.codegen.hpp"
#include "mddft2.fftx.codegen.hpp"
#include "imddft2.fftx.codegen.hpp"
*/
#include "mddft3.fftx.codegen.hpp"
#include "imddft3.fftx.codegen.hpp"
/*
#include "prdft1.fftx.codegen.hpp"
#include "iprdft1.fftx.codegen.hpp"
#include "prdft2.fftx.codegen.hpp"
#include "iprdft2.fftx.codegen.hpp"
*/
#include "prdft3.fftx.codegen.hpp"
#include "iprdft3.fftx.codegen.hpp"
#include "fftx3utilities.h"
#include "verify.h"

enum VerbosityLevel { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};
  
// using namespace fftx;

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
void DFTfunctionDevice(fftx::handle_t (a_dftFunction)
                       (fftx::array_t<DIM, T_IN>&,
                        fftx::array_t<DIM, T_OUT>&),
                       fftx::array_t<DIM, T_IN>& a_input, // make this const?
                       fftx::array_t<DIM, T_OUT>& a_output)

{
  auto inputDomain = a_input.m_domain;
  auto outputDomain = a_output.m_domain;

  auto input_size = inputDomain.size();
  auto output_size = outputDomain.size();

  auto input_bytes = input_size * sizeof(T_IN);
  auto output_bytes = output_size * sizeof(T_OUT);

  char* bufferPtr;
  cudaMalloc(&bufferPtr, input_bytes + output_bytes);
  T_IN* inputPtr = (T_IN*) bufferPtr;
  bufferPtr += input_bytes;
  T_OUT* outputPtr = (T_OUT*) bufferPtr;

  cudaMemcpy(inputPtr, a_input.m_data.local(), input_bytes,
             cudaMemcpyHostToDevice);

  fftx::array_t<DIM, T_IN> inputDevice(fftx::global_ptr<T_IN>
                                       (inputPtr, 0, 1), inputDomain);
  fftx::array_t<DIM, T_OUT> outputDevice(fftx::global_ptr<T_OUT>
                                         (outputPtr, 0, 1), outputDomain);
  
  a_dftFunction(inputDevice, outputDevice);

  cudaMemcpy(a_output.m_data.local(), outputPtr, output_bytes,
             cudaMemcpyDeviceToHost);
}

template<int DIM, typename T_IN, typename T_OUT>
double test1DFTfunction(fftx::handle_t (a_dftFunction)
                       (fftx::array_t<DIM, T_IN>&,
                        fftx::array_t<DIM, T_OUT>&),
                        fftx::box_t<DIM> a_inDomain,
                        fftx::box_t<DIM> a_outDomain,
                        int a_rounds,
                        int a_verbosity)
{
  fftx::array_t<DIM, T_IN> inA(a_inDomain);
  fftx::array_t<DIM, T_IN> inB(a_inDomain);
  fftx::array_t<DIM, T_IN> LCin(a_inDomain);

  fftx::array_t<DIM, T_OUT> outA(a_outDomain);
  fftx::array_t<DIM, T_OUT> outB(a_outDomain);
  fftx::array_t<DIM, T_OUT> LCout(a_outDomain);
  fftx::array_t<DIM, T_OUT> outLCin(a_outDomain);

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
      
      DFTfunctionDevice(a_dftFunction, inA, outA);
      DFTfunctionDevice(a_dftFunction, inB, outB);
      sumArrays(LCout, outA, outB, alphaOut, betaOut);
      DFTfunctionDevice(a_dftFunction, LCin, outLCin);
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
double test2impulse1(fftx::handle_t (a_dftFunction)
                     (fftx::array_t<DIM, T_IN>&,
                      fftx::array_t<DIM, T_OUT>&),
                     fftx::box_t<DIM> a_inDomain,
                     fftx::box_t<DIM> a_outDomain,
                     int a_verbosity)
{ // Unit impulse at low corner.
  fftx::array_t<DIM, T_IN> inImpulse(a_inDomain);
  fftx::array_t<DIM, T_OUT> outImpulse(a_outDomain);
  fftx::array_t<DIM, T_OUT> all1out(a_outDomain);
  setUnitImpulse(inImpulse, a_inDomain.lo);
  setConstant(all1out, scalarVal<T_OUT>(1.));
  DFTfunctionDevice(a_dftFunction, inImpulse, outImpulse);
  double errtest2impulse1 = absMaxDiffArray(outImpulse, all1out);
  if (a_verbosity >= SHOW_SUBTESTS)
    {
       printf("%dD unit impulse low corner test: max error %11.5e\n", DIM, errtest2impulse1);
    }
  return errtest2impulse1;
}

template<int DIM, typename T_IN, typename T_OUT>
double test2impulsePlus(fftx::handle_t (a_dftFunction)
                        (fftx::array_t<DIM, T_IN>&,
                         fftx::array_t<DIM, T_OUT>&),
                        fftx::box_t<DIM> a_inDomain,
                        fftx::box_t<DIM> a_outDomain,
                        int a_rounds,
                        int a_verbosity)
{ // Unit impulse at low corner.
  fftx::array_t<DIM, T_IN> inImpulse(a_inDomain);
  fftx::array_t<DIM, T_OUT> outImpulse(a_outDomain);
  fftx::array_t<DIM, T_OUT> all1out(a_outDomain);
  setUnitImpulse(inImpulse, a_inDomain.lo);
  setConstant(all1out, scalarVal<T_OUT>(1.));
  DFTfunctionDevice(a_dftFunction, inImpulse, outImpulse);

  fftx::array_t<DIM, T_IN> inRand(a_inDomain);
  fftx::array_t<DIM, T_IN> inImpulseMinusRand(a_inDomain);

  fftx::array_t<DIM, T_OUT> outRand(a_outDomain);
  fftx::array_t<DIM, T_OUT> outImpulseMinusRand(a_outDomain);
  fftx::array_t<DIM, T_OUT> mysum(a_outDomain);
  
  // Check that for random arrays inRand,
  // fft(inRand) + fft(inImpulse - inRand) = fft(inImpulse) = all1out.
  double errtest2impulsePlus = 0.;
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      unifArray(inRand);
      DFTfunctionDevice(a_dftFunction, inRand, outRand);
      diffArrays(inImpulseMinusRand, inImpulse, inRand);
      DFTfunctionDevice(a_dftFunction, inImpulseMinusRand, outImpulseMinusRand);
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
      printf("%dD unit impulse low corner test in %d rounds: max error %11.5e\n", DIM, a_rounds, errtest2impulsePlus);
    }
  return errtest2impulsePlus;
}

template<int DIM, typename T_IN, typename T_OUT>
double test2constant(fftx::handle_t (a_dftFunction)
                     (fftx::array_t<DIM, T_IN>&,
                      fftx::array_t<DIM, T_OUT>&),
                     const fftx::box_t<DIM>& a_inDomain,
                     const fftx::box_t<DIM>& a_outDomain,
                     const fftx::point_t<DIM>& a_fullExtents,
                     int a_verbosity)
{ // Check that constant maps back to unit impulse at low corner.
  fftx::array_t<DIM, T_IN> all1in(a_inDomain);
  setConstant(all1in, scalarVal<T_IN>(1.));

  fftx::array_t<DIM, T_OUT> magImpulse(a_outDomain);
  size_t npts = 1;
  for (int d = 0; d < DIM; d++)
    {
      npts *= a_fullExtents[d];
    }
  T_OUT mag = scalarVal<T_OUT>(npts * 1.);
  setUnitImpulse(magImpulse, a_outDomain.lo, mag);

  fftx::array_t<DIM, T_OUT> outImpulse(a_outDomain);
  DFTfunctionDevice(a_dftFunction, all1in, outImpulse);

  double errtest2constant = absMaxDiffArray(outImpulse, magImpulse);
  if (a_verbosity >= SHOW_SUBTESTS)
    {
       printf("%dD constant test: max error %11.5e\n", DIM, errtest2constant);
    }
  return errtest2constant;
}

template<int DIM, typename T_IN, typename T_OUT>
double test2constantPlus(fftx::handle_t (a_dftFunction)
                         (fftx::array_t<DIM, T_IN>&,
                          fftx::array_t<DIM, T_OUT>&),
                         const fftx::box_t<DIM>& a_inDomain,
                         const fftx::box_t<DIM>& a_outDomain,
                         const fftx::point_t<DIM>& a_fullExtents,
                         int a_rounds,
                         int a_verbosity)
{
  fftx::array_t<DIM, T_IN> all1in(a_inDomain);
  setConstant(all1in, scalarVal<T_IN>(1.));

  fftx::array_t<DIM, T_OUT> magImpulse(a_outDomain);
  size_t npts = 1;
  for (int d = 0; d < DIM; d++)
    {
      npts *= a_fullExtents[d];
    }
  T_OUT mag = scalarVal<T_OUT>(npts * 1.);
  setUnitImpulse(magImpulse, a_outDomain.lo, mag);

  fftx::array_t<DIM, T_IN> inRand(a_inDomain);
  fftx::array_t<DIM, T_IN> inConstantMinusRand(a_inDomain);

  fftx::array_t<DIM, T_OUT> outRand(a_outDomain);
  fftx::array_t<DIM, T_OUT> outConstantMinusRand(a_outDomain);
  fftx::array_t<DIM, T_OUT> outSum(a_outDomain);

  // Check that for random arrays inRand,
  // fft(inRand) + fft(all1 - inRand) = fft(all1) = magImpulse.
  double errtest2constantPlus = 0.;
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      unifArray(inRand);
      DFTfunctionDevice(a_dftFunction, inRand, outRand);

      diffArrays(inConstantMinusRand, all1in, inRand);
      DFTfunctionDevice(a_dftFunction, inConstantMinusRand, outConstantMinusRand);

      sumArrays(outSum, outRand, outConstantMinusRand);
      
      double err = absMaxDiffArray(outSum, magImpulse);
      updateMax(errtest2constantPlus, err);
      if (a_verbosity >= SHOW_ROUNDS)
          {
            printf("%dD random + constant test round %d max error %11.5e\n", DIM, itn, err);
          }
    }

  if (a_verbosity >= SHOW_SUBTESTS)
    {
      printf("%dD random + constant test in %d rounds: max error %11.5e\n", DIM, a_rounds, errtest2constantPlus);
  
    }
  return errtest2constantPlus;
}

template<int DIM, typename T_IN, typename T_OUT>
double test2impulseRandom(fftx::handle_t (a_dftFunction)
                          (fftx::array_t<DIM, T_IN>&,
                           fftx::array_t<DIM, T_OUT>&),
                          const fftx::box_t<DIM>& a_inDomain,
                          const fftx::box_t<DIM>& a_outDomain,
                          int a_sign,
                          int a_rounds,
                          int a_verbosity)
{
  return 0.;
}

template<int DIM, typename T_IN>
double test2impulseRandom(fftx::handle_t (a_dftFunction)
                          (fftx::array_t<DIM, T_IN>&,
                           fftx::array_t<DIM, std::complex<double>>&),
                          const fftx::box_t<DIM>& a_inDomain,
                          const fftx::box_t<DIM>& a_outDomain,
                          int a_sign,
                          int a_rounds,
                          int a_verbosity)
{
  // Check unit impulse at random position.
  fftx::array_t<DIM, T_IN> inImpulse(a_inDomain);
  fftx::array_t<DIM, std::complex<double>> outImpulse(a_outDomain);
  fftx::array_t<DIM, std::complex<double>> outCheck(a_outDomain);
  double errtest2impulseRandom = 0.;
  fftx::point_t<DIM> fullExtents = a_inDomain.extents();
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      fftx::point_t<DIM> rpoint = unifPoint<DIM>();
      setUnitImpulse(inImpulse, rpoint);
      DFTfunctionDevice(a_dftFunction, inImpulse, outImpulse);
      // Recall a_inDomain is whole domain, but a_outDomain may be truncated;
      // waves defined on a_outDomain, but based on the full a_inDomain extents.
      setProductWaves(outCheck, fullExtents, rpoint,  a_sign);
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
double test2DFTfunction(fftx::handle_t (a_dftFunction)
                       (fftx::array_t<DIM, T_IN>&,
                        fftx::array_t<DIM, T_OUT>&),
                        const fftx::box_t<DIM>& a_inDomain,
                        const fftx::box_t<DIM>& a_outDomain,
                        const fftx::point_t<DIM>& a_fullExtents,
                        int a_sign,
                        int a_rounds,
                        int a_verbosity)
{
  double errtest2 = 0.;

  updateMax(errtest2,
            test2impulse1(a_dftFunction, a_inDomain, a_outDomain,
                          a_verbosity));
  updateMax(errtest2,
            test2impulsePlus(a_dftFunction, a_inDomain, a_outDomain,
                             a_rounds, a_verbosity));
  
  updateMax(errtest2,
            test2constant(a_dftFunction, a_inDomain, a_outDomain,
                          a_fullExtents, a_verbosity));
  
  updateMax(errtest2,
            test2constantPlus(a_dftFunction, a_inDomain, a_outDomain,
                              a_fullExtents, a_rounds, a_verbosity));
  
  updateMax(errtest2,
            test2impulseRandom(a_dftFunction, a_inDomain, a_outDomain,
                               a_sign, a_rounds, a_verbosity));
  
  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD Test 2 (impulses) in %d rounds: max error %11.5e\n", DIM, a_rounds, errtest2);
    }
  return errtest2;
}


template<int DIM, typename T_IN, typename T_OUT>
double test3time(fftx::handle_t (a_dftFunction)
                 (fftx::array_t<DIM, T_IN>&,
                  fftx::array_t<DIM, T_OUT>&),
                 const fftx::box_t<DIM>& a_inDomain,
                 const fftx::box_t<DIM>& a_outDomain,
                 int a_sign,
                 int a_rounds,
                 int a_verbosity)
{
  return 0.;
}

template<int DIM, typename T_IN>
double test3time(fftx::handle_t (a_dftFunction)
                 (fftx::array_t<DIM, T_IN>&,
                  fftx::array_t<DIM, std::complex<double>>&),
                 const fftx::box_t<DIM>& a_inDomain,
                 const fftx::box_t<DIM>& a_outDomain,
                 int a_sign,
                 int a_rounds,
                 int a_verbosity)
{
  fftx::array_t<DIM, T_IN> inRand(a_inDomain);
  fftx::array_t<DIM, T_IN> inRandRot(a_inDomain);
  fftx::array_t<DIM, std::complex<double>> outRand(a_outDomain);
  fftx::array_t<DIM, std::complex<double>> outRandRot(a_outDomain);
  fftx::array_t<DIM, std::complex<double>> rotator(a_outDomain);
  fftx::array_t<DIM, std::complex<double>> outRandRotMult(a_outDomain);
  double errtest3timeDim[DIM];
  double errtest3time = 0.;
  for (int d = 0; d < DIM; d++)
    {
      errtest3timeDim[d] = 0.;
      setRotator(rotator, a_inDomain, d, -a_sign); // +1 for MDDFT, -1 for IMDDFT, -1 for PRDFT
      for (int itn = 1; itn <= a_rounds; itn++)
        {
          unifArray(inRand);
          
          // time-shift test in dimension d
          rotate(inRandRot, inRand, d, 1); // +1 for MDDFT, +1 for IMDDFT, +1 for PRDFT
          DFTfunctionDevice(a_dftFunction, inRand, outRand);
          DFTfunctionDevice(a_dftFunction, inRandRot, outRandRot);
          productArrays(outRandRotMult, outRandRot, rotator);
          double err = absMaxDiffArray(outRandRotMult, outRand);
          updateMax(errtest3timeDim[d], err);
          updateMax(errtest3time, errtest3timeDim[d]);
          if (a_verbosity >= SHOW_ROUNDS)
            {
              printf("%dD dim %d time-shift test %d max error %11.5e\n", DIM, d, itn, err);
            }
        }
      if (a_verbosity >= SHOW_SUBTESTS)
        {
          printf("%dD dim %d time-shift test in %d rounds: max error %11.5e\n", DIM, d, a_rounds, errtest3timeDim[d]);
        }
    }
  return errtest3time;
}

template<int DIM, typename T_IN, typename T_OUT>
double test3frequency(fftx::handle_t (a_dftFunction)
                      (fftx::array_t<DIM, T_IN>&,
                       fftx::array_t<DIM, T_OUT>&),
                      const fftx::box_t<DIM>& a_inDomain,
                      const fftx::box_t<DIM>& a_outDomain,
                      int a_sign,
                      int a_rounds,
                      int a_verbosity)
{
  return 0.;
}

template<int DIM, typename T_OUT>
double test3frequency(fftx::handle_t (a_dftFunction)
                      (fftx::array_t<DIM, std::complex<double>>&,
                       fftx::array_t<DIM, T_OUT>&),
                      const fftx::box_t<DIM>& a_inDomain,
                      const fftx::box_t<DIM>& a_outDomain,
                      int a_sign,
                      int a_rounds,
                      int a_verbosity)
{
  fftx::array_t<DIM, std::complex<double>> inRand(a_inDomain);
  fftx::array_t<DIM, std::complex<double>> inRandMult(a_inDomain);
  fftx::array_t<DIM, T_OUT> outRand(a_outDomain);
  fftx::array_t<DIM, T_OUT> outRandMult(a_outDomain);
  fftx::array_t<DIM, std::complex<double>> rotatorUp(a_inDomain);
  fftx::array_t<DIM, T_OUT> outRandMultRot(a_outDomain);
  double errtest3frequencyDim[DIM];
  double errtest3frequency = 0.;
  for (int d = 0; d < DIM; d++)
    {
      // frequency-shift test in dimension d
      errtest3frequencyDim[d] = 0.;
      // Recall a_outDomain is whole domain, but a_inDomain may be truncated;
      // rotatorUp is defined on a_inDomain, but based on the full a_outDomain.
      setRotator(rotatorUp, a_outDomain, d, 1);
      for (int itn = 1; itn <= a_rounds; itn++)
        {
          unifComplexArray(inRand);

          productArrays(inRandMult, inRand, rotatorUp);
          DFTfunctionDevice(a_dftFunction, inRand, outRand);
          DFTfunctionDevice(a_dftFunction, inRandMult, outRandMult);
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
double test3DFTfunction(fftx::handle_t (a_dftFunction)
                        (fftx::array_t<DIM, T_IN>&,
                         fftx::array_t<DIM, T_OUT>&),
                        const fftx::box_t<DIM>& a_inDomain,
                        const fftx::box_t<DIM>& a_outDomain,
                        int a_sign,
                        int a_rounds,
                        int a_verbosity)
{
  double errtest3 = 0.;

  updateMax(errtest3,
            test3time(a_dftFunction, a_inDomain, a_outDomain,
                      a_sign, a_rounds, a_verbosity));
  
  updateMax(errtest3,
            test3frequency(a_dftFunction, a_inDomain, a_outDomain,
                           a_sign, a_rounds, a_verbosity));
  
  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD Test 3 (shifts) in %d rounds: max error %11.5e\n", DIM, a_rounds, errtest3);
    }
  return errtest3;
}


template<int DIM, typename T_IN, typename T_OUT>
void verifyDFTfunction(fftx::handle_t (a_dftFunction)
                       (fftx::array_t<DIM, T_IN>&,
                        fftx::array_t<DIM, T_OUT>&),
                       const fftx::box_t<DIM>& a_inDomain,
                       const fftx::box_t<DIM>& a_outDomain,
                       const fftx::point_t<DIM>& a_fullExtents,
                       int a_sign,
                       int a_rounds,
                       int a_verbosity)
{
  double err = 0.;

  updateMax(err,
            test1DFTfunction(a_dftFunction, a_inDomain, a_outDomain,
                             a_rounds, a_verbosity));

  updateMax(err,
            test2DFTfunction(a_dftFunction, a_inDomain, a_outDomain,
                             a_fullExtents, a_sign, a_rounds, a_verbosity));

  updateMax(err,
            test3DFTfunction(a_dftFunction, a_inDomain, a_outDomain,
                             a_sign, a_rounds, a_verbosity));

  printf("%dD test in %d rounds max error %11.5e\n", DIM, a_rounds, err);
}


template<int DIM>
void verifyDimension(fftx::box_t<DIM> a_domain,
                     fftx::box_t<DIM> a_fdomain,
                     int a_rounds,
                     fftx::handle_t (a_mddft)
                     (fftx::array_t<DIM, std::complex<double>>&,
                      fftx::array_t<DIM, std::complex<double>>&),
                     fftx::handle_t (a_imddft)
                     (fftx::array_t<DIM, std::complex<double>>&,
                      fftx::array_t<DIM, std::complex<double>>&),
                     fftx::handle_t (a_prdft)
                     (fftx::array_t<DIM, double>&,
                      fftx::array_t<DIM, std::complex<double>>&),
                     fftx::handle_t (a_iprdft)
                     (fftx::array_t<DIM, std::complex<double>>&,
                      fftx::array_t<DIM, double>&),
                     int a_verbosity)
{
  // std::cout << "*******************************************" << std::endl;
  // std::cout << DIM << "D domain = " << verify::domain1 << std::endl;
  // std::cout << "*******************************************" << std::endl;

  fftx::point_t<DIM> fullextents = a_domain.extents();

  std::cout << "***** test " << DIM << "D MDDFT on complex "
            << a_domain << std::endl;
  verifyDFTfunction(a_mddft, a_domain, a_domain, fullextents, -1, a_rounds, a_verbosity);

  std::cout << "***** test " << DIM << "D IMDDFT on complex "
            << a_domain << std::endl;
  verifyDFTfunction(a_imddft, a_domain, a_domain, fullextents, 1, a_rounds, a_verbosity);
  
  std::cout << "***** test " << DIM << "D PRDFT from real "
            << a_domain << " to complex " << a_fdomain << std::endl;
  verifyDFTfunction(a_prdft, a_domain, a_fdomain, fullextents, -1, a_rounds, a_verbosity);

  std::cout << "***** test " << DIM << "D IPRDFT from complex "
            << a_fdomain << " to real " << a_domain << std::endl;
  verifyDFTfunction(a_iprdft, a_fdomain, a_domain, fullextents, 1, a_rounds, a_verbosity);
}
                    

int main(int argc, char* argv[])
{
  // { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};
  printf("Usage:  %s [verbosity=0] [rounds=20]\n", argv[0]);
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
  for (int d = 0; d < 3; d++)
    {
      unifInt[d] = std::uniform_int_distribution<int>(verify::domain3.lo[d],
                                                      verify::domain3.hi[d]);
    }

  /*
  // printf("Call mddft1::init()\n");
  mddft1::init();
  // printf("Call imddft1::init()\n");
  imddft1::init();
  // printf("Call prdft1::init()\n");
  prdft1::init();
  // printf("Call iprdft1::init()\n");
  iprdft1::init();

  verifyDimension(verify::domain1, verify::fdomain1, rounds,
                  mddft1::transform, imddft1::transform,
                  prdft1::transform, iprdft1::transform,
                  verbosity);
  
  mddft1::destroy();
  imddft1::destroy();
  prdft1::destroy();
  iprdft1::destroy();
  */

  /*
  // printf("Call mddft2::init()\n");
  mddft2::init();
  // printf("Call imddft2::init()\n");
  imddft2::init();
  // printf("Call prdft2::init()\n");
  prdft2::init();
  // printf("Call iprdft2::init()\n");
  iprdft2::init();

  verifyDimension(verify::domain2, verify::fdomain2, rounds,
                  mddft2::transform, imddft2::transform,
                  prdft2::transform, iprdft2::transform,
                  verbosity);
  
  mddft2::destroy();
  imddft2::destroy();
  prdft2::destroy();
  iprdft2::destroy();
  */

  // printf("Call mddft3::init()\n");
  mddft3::init();
  // printf("Call imddft3::init()\n");
  imddft3::init();
  // printf("Call prdft3::init()\n");
  prdft3::init();
  // printf("Call iprdft3::init()\n");
  iprdft3::init();

  verifyDimension(verify::domain3, verify::fdomain3, rounds,
                  mddft3::transform, imddft3::transform,
                  prdft3::transform, iprdft3::transform,
                  verbosity);
  
  mddft3::destroy();
  imddft3::destroy();
  prdft3::destroy();
  iprdft3::destroy();

  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
