#include <stdio.h>
#include <cmath> // Without this, abs is the wrong function!
#include <random>

#include "device_macros.h"

// #include "fftx_mddft_public.h"
// #include "fftx_imddft_public.h"
#include "fftx_rconv_public.h"

// #include "mddft.fftx.precompile.hpp"
// #include "imddft.fftx.precompile.hpp"
#include "rconv.fftx.precompile.hpp"
#include "transformer.fftx.precompile.hpp"

#include "fftx3utilities.h"
#include "rconv_dims.h"

enum VerbosityLevel { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};
  
// using namespace fftx;

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

template<int DIM, class Transformer>
void convolutionDevice(Transformer& a_transformer,
                       fftx::array_t<DIM, double>& a_input,
                       fftx::array_t<DIM, double>& a_output,
                       fftx::array_t<DIM, double>& a_symbol)
{
  auto inputDomain = a_input.m_domain;
  auto outputDomain = a_output.m_domain;
  auto symbolDomain = a_symbol.m_domain;
  
  auto input_size = inputDomain.size();
  auto output_size = outputDomain.size();
  auto symbol_size = symbolDomain.size();
  
  auto input_bytes = input_size * sizeof(double);
  auto output_bytes = output_size * sizeof(double);
  auto symbol_bytes = symbol_size * sizeof(double);

  double* inputPtr;
  double* outputPtr;
  double* symbolPtr;
  DEVICE_MALLOC(&inputPtr, input_bytes);
  DEVICE_MALLOC(&outputPtr, output_bytes);
  DEVICE_MALLOC(&symbolPtr, symbol_bytes);
  
  DEVICE_MEM_COPY(inputPtr, a_input.m_data.local(), input_bytes,
                  MEM_COPY_HOST_TO_DEVICE);
  DEVICE_MEM_COPY(symbolPtr, a_symbol.m_data.local(), symbol_bytes,
                  MEM_COPY_HOST_TO_DEVICE);
  
  fftx::array_t<DIM, double> inputDevice(fftx::global_ptr<double>
                                         (inputPtr, 0, 1), inputDomain);
  fftx::array_t<DIM, double> outputDevice(fftx::global_ptr<double>
                                          (outputPtr, 0, 1), outputDomain);
  fftx::array_t<DIM, double> symbolDevice(fftx::global_ptr<double>
                                          (symbolPtr, 0, 1), symbolDomain);

  a_transformer.transform(inputDevice, outputDevice, symbolDevice);

  DEVICE_MEM_COPY(a_output.m_data.local(), outputPtr, output_bytes,
                  MEM_COPY_DEVICE_TO_HOST);
  DEVICE_FREE(inputPtr);
  DEVICE_FREE(outputPtr);
  DEVICE_FREE(symbolPtr);
}


template<int DIM, class Transformer>
double testConstantSymbol(Transformer& a_transformer,
                          fftx::box_t<DIM> a_domain,
                          fftx::box_t<DIM> a_fdomain,
                          int a_rounds,
                          int a_verbosity)
{
  printf("calling testConstantSymbol<%d>\n", DIM);
  fftx::array_t<DIM, double> input(a_domain);
  fftx::array_t<DIM, double> output(a_domain);
  fftx::array_t<DIM, double> symbol(a_fdomain);

  double scaling = 1. / (a_domain.size()*1.);
  setConstant(symbol, scaling);
  double errConstantSymbol = 0.;
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      unifRealArray(input);
      convolutionDevice(a_transformer, input, output, symbol);
      double err = absMaxDiffArray(input, output);
      updateMax(errConstantSymbol, err);
      if (a_verbosity >= SHOW_ROUNDS)
        {
          printf("%dD random input with constant symbol max error %11.5e\n",
                 DIM, err);
        }
    }
  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD random input with constant symbol in %d rounds: max error %11.5e\n", DIM, a_rounds, errConstantSymbol);
    }
  return errConstantSymbol;
}

template<int DIM, class Transformer>
double testDelta(Transformer& a_transformer,
                 fftx::box_t<DIM> a_domain,
                 fftx::box_t<DIM> a_fdomain,
                 int a_verbosity)
{
  printf("calling testDelta<%d>\n", DIM);
  fftx::array_t<DIM, double> input(a_domain);
  fftx::array_t<DIM, double> output(a_domain);
  fftx::array_t<DIM, double> symbol(a_fdomain);

  setConstant(input, 2.);

  fftx::point_t<DIM> cornerLo = a_domain.lo;
  double scaling = 1. / (a_domain.size()*1.);
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
  auto indCornerLo = positionInBox(cornerLo, a_fdomain);
  symbolPtr[indCornerLo] = scaling;

  convolutionDevice(a_transformer, input, output, symbol);
  double errDelta = absMaxDiffArray(input, output);
  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD delta function test: max error %11.5e\n", DIM, errDelta);
    }
  return errDelta;
}

template<int DIM, class Transformer>
double testPoisson(Transformer& a_transformer,
                   fftx::box_t<DIM> a_domain,
                   fftx::box_t<DIM> a_fdomain,
                   int a_verbosity)
{
  printf("calling testPoisson<%d>\n", DIM);
  fftx::array_t<DIM, double> input(a_domain);
  fftx::array_t<DIM, double> output(a_domain);
  fftx::array_t<DIM, double> symbol(a_fdomain);

  fftx::point_t<DIM> lo = a_domain.lo;
  fftx::point_t<DIM> hi = a_domain.hi;
  double center[DIM];
  fftx::point_t<DIM> extents = a_domain.extents();
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
  auto input_size = a_domain.size();
  for (size_t ind = 0; ind < input_size; ind++)
    {
      fftx::point_t<DIM> p = pointFromPositionBox(ind, a_domain);
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

  fftx::point_t<DIM> cornerLo = a_domain.lo;
  /*
  size_t normalize = a_domain.size();
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
  auto symbol_size = a_fdomain.size();
  for (size_t ind = 0; ind < symbol_size; ind++)
    {
      fftx::point_t<DIM> p = pointFromPositionBox(ind, a_fdomain);
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
  
  convolutionDevice(a_transformer, input, output, symbol);

  fftx::array_t<DIM,double> lap2output(a_domain);
  laplacian2periodic(lap2output, output);
  
  double errPoisson = absMaxDiffArray(lap2output, input);
  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD Poisson test: max error %11.5e\n", DIM, errPoisson);
    }
  return errPoisson;
}

template<int DIM, class Transformer>
void rconvDimension(Transformer& a_transformer,
                    fftx::box_t<DIM> a_domain,
                    fftx::box_t<DIM> a_fdomain,
                    int a_rounds,
                    int a_verbosity)
{
  if (!a_transformer.isDefined())
    {
      return;
    }

  std::cout << "***** test " << DIM << "D real convolution on "
            << a_domain << std::endl;

  double err = 0.;

  updateMax(err,
            testConstantSymbol(a_transformer, a_domain, a_fdomain,
                               a_rounds, a_verbosity));

  updateMax(err,
            testDelta(a_transformer, a_domain, a_fdomain,
                      a_verbosity));

  updateMax(err,
            testPoisson(a_transformer, a_domain, a_fdomain,
                        a_verbosity));

  fftx::point_t<DIM> sz = a_domain.extents();
  printf("%dD tests of rconv<3>[%d,%d,%d] in %d rounds max error %11.5e\n",
         DIM, sz[0], sz[1], sz[2], a_rounds, err);
}
                    

template<int DIM>
void rconvSize(fftx::point_t<DIM> a_size,
               int a_rounds,
               int a_verbosity)
{
  fftx::box_t<3> fulldomain(fftx::point_t<3>
                            ({{rconv_dims::offx+1,
                                  rconv_dims::offy+1,
                                  rconv_dims::offz+1}}),
                            fftx::point_t<3>
                            ({{rconv_dims::offx+a_size[0],
                                  rconv_dims::offy+a_size[1],
                                  rconv_dims::offz+a_size[2]}}));
  
  fftx::box_t<3> halfdomain(fftx::point_t<3>
                            ({{rconv_dims::offx+1,
                                  rconv_dims::offy+1,
                                  rconv_dims::offz+1}}),
                            fftx::point_t<3>
#if FFTX_COMPLEX_TRUNC_LAST
                            ({{rconv_dims::offx+a_size[0],
                                  rconv_dims::offy+a_size[1],
                                  rconv_dims::offz+a_size[2]/2+1}})
#else
                            ({{rconv_dims::offx+a_size[0]/2+1,
                                  rconv_dims::offy+a_size[1],
                                  rconv_dims::offz+a_size[2]}})
#endif
                            );
  fftx::rconv<3> tfm(a_size); // does initialization
  rconvDimension(tfm, fulldomain, halfdomain, a_rounds, a_verbosity);
}
  
int main(int argc, char* argv[])
{
  // { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};
  printf("Usage:  %s [verbosity=0] [rounds=20]\n", argv[0]);
  printf("verbosity 0 for summary, 1 for categories, 2 for subtests, 3 for all iterations\n");
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

  /*
    2-dimensional tests.
  */
  //    rconv2::init();
  //    rconvDimension(rconv2::transform, rconv_dims::domain2, rconv_dims::fdomain2,
  //                   rounds, verbosity);
  //    rconv2::destroy();
  
  /*
    3-dimensional tests.
  */

  // rconvSize(fftx::point_t<3>({{  48,  48,  48 }}), rounds, verbosity);

  fftx::point_t<3> *ents = fftx_rconv_QuerySizes ();

  for ( int ind = 0; ents[ind][0] != 0; ind++ )
    {
      rconvSize(ents[ind], rounds, verbosity);
    }
  
  // rconvSize(fftx::point_t<3>({{  48,  48,  48 }}), rounds, verbosity);
  
  // fftx::point_t<3> extents = rconv_dims::domain3.extents();
  // fftx::rconv<3> tfm(extents); // does initialization
  // rconvDimension(tfm, rconv_dims::domain3, rconv_dims::fdomain3,
  //                rounds, verbosity);

  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
