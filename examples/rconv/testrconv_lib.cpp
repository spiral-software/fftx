#include <stdio.h>
#include <cmath> // Without this, abs is the wrong function!
#include <random>

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
double testConstantSymbol(Transformer& a_transformer,
                          fftx::box_t<DIM> a_domain,
                          fftx::box_t<DIM> a_fdomain,
                          int a_rounds,
                          int a_verbosity)
{
  printf("calling testConstantSymbol<%d>\n", DIM);
  array_t<DIM, double> input(a_domain);
  array_t<DIM, double> output(a_domain);
  array_t<DIM, double> symbol(a_fdomain);

  double scaling = 1. / (a_domain.size()*1.);
  setConstant(symbol, scaling);
  double errConstantSymbol = 0.;
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      unifRealArray(input);
      a_transformer.transform(input, output, symbol);
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
  array_t<DIM, double> input(a_domain);
  array_t<DIM, double> output(a_domain);
  array_t<DIM, double> symbol(a_fdomain);

  setConstant(input, 2.);

  point_t<DIM> cornerLo = a_domain.lo;
  double scaling = 1. / (a_domain.size()*1.);
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

  a_transformer.transform(input, output, symbol);
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
  array_t<DIM, double> input(a_domain);
  array_t<DIM, double> output(a_domain);
  array_t<DIM, double> symbol(a_fdomain);

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

  point_t<DIM> cornerLo = a_domain.lo;
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

  a_transformer.transform(input, output, symbol);

  array_t<DIM,double> lap2output(a_domain);
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
  box_t<3> fulldomain(point_t<3>
                      ({{rconv_dims::offx+1,
                         rconv_dims::offy+1,
                         rconv_dims::offz+1}}),
                      point_t<3>
                      ({{rconv_dims::offx+a_size[0],
                         rconv_dims::offy+a_size[1],
                         rconv_dims::offz+a_size[2]}}));
  
  box_t<3> halfdomain(point_t<3>
                      ({{rconv_dims::offx+1,
                         rconv_dims::offy+1,
                         rconv_dims::offz+1}}),
                      point_t<3>
                      ({{rconv_dims::offx+a_size[0]/2+1,
                         rconv_dims::offy+a_size[1],
                         rconv_dims::offz+a_size[2]}}));

  fftx::rconv<3> tfm(a_size); // does initialization
  rconvDimension(tfm, fulldomain, halfdomain, a_rounds, a_verbosity);

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
