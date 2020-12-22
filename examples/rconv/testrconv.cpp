#define _USE_MATH_DEFINES
#include <cmath> // Without this, abs is the wrong function!
#include <random>
#include "rconv2.fftx.codegen.hpp"
#include "rconv3.fftx.codegen.hpp"
#include "fftx3utilities.h"
#include "rconv.h"

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

template<int DIM>
double testConstantSymbol(fftx::handle_t (a_transform)
                          (fftx::array_t<DIM, double>&,
                           fftx::array_t<DIM, double>&,
                           fftx::array_t<DIM, double>&),
                          fftx::box_t<DIM> a_domain,
                          fftx::box_t<DIM> a_fdomain,
                          int a_rounds,
                          int a_verbosity)
{
  array_t<DIM, double> input(a_domain);
  array_t<DIM, double> output(a_domain);
  array_t<DIM, double> symbol(a_fdomain);

  double scaling = 1. / (a_domain.size()*1.);
  setConstant(symbol, scaling);
  double errConstantSymbol = 0.;
  for (int itn = 1; itn <= a_rounds; itn++)
    {
      unifRealArray(input);
      a_transform(input, output, symbol);
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

template<int DIM>
double testDelta(fftx::handle_t (a_transform)
                 (fftx::array_t<DIM, double>&,
                  fftx::array_t<DIM, double>&,
                  fftx::array_t<DIM, double>&),
                 fftx::box_t<DIM> a_domain,
                 fftx::box_t<DIM> a_fdomain,
                 int a_verbosity)
{
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
  
  a_transform(input, output, symbol);
  double errDelta = absMaxDiffArray(input, output);
  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD delta function test: max error %11.5e\n", DIM, errDelta);
    }
  return errDelta;
}

template<int DIM>
double testPoisson(fftx::handle_t (a_transform)
                   (fftx::array_t<DIM, double>&,
                    fftx::array_t<DIM, double>&,
                    fftx::array_t<DIM, double>&),
                   fftx::box_t<DIM> a_domain,
                   fftx::box_t<DIM> a_fdomain,
                   int a_verbosity)
{
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
  
  a_transform(input, output, symbol);

  array_t<DIM,double> lap2output(a_domain);
  laplacian2periodic(lap2output, output);
  
  double errPoisson = absMaxDiffArray(lap2output, input);
  if (a_verbosity >= SHOW_CATEGORIES)
    {
      printf("%dD Poisson test: max error %11.5e\n", DIM, errPoisson);
    }
  return errPoisson;
}

template<int DIM>
void rconvDimension(fftx::handle_t (a_transform)
                    (fftx::array_t<DIM, double>&,
                     fftx::array_t<DIM, double>&,
                     fftx::array_t<DIM, double>&),
                    fftx::box_t<DIM> a_domain,
                    fftx::box_t<DIM> a_fdomain,
                    int a_rounds,
                    int a_verbosity)
{
  std::cout << "***** test " << DIM << "D real convolution on "
            << a_domain << std::endl;

  double err = 0.;
  
  updateMax(err,
            testConstantSymbol(a_transform, a_domain, a_fdomain,
                               a_rounds, a_verbosity));

  updateMax(err,
            testDelta(a_transform, a_domain, a_fdomain,
                      a_verbosity));

  updateMax(err,
            testPoisson(a_transform, a_domain, a_fdomain,
                        a_verbosity));

  printf("%dD tests in %d rounds max error %11.5e\n", DIM, a_rounds, err);
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
  rconv2::init();
  rconvDimension(rconv2::transform, rconv::domain2, rconv::fdomain2,
                 rounds, verbosity);
  rconv2::destroy();
  
  /*
    3-dimensional tests.
  */
  rconv3::init();
  rconvDimension(rconv3::transform, rconv::domain3, rconv::fdomain3,
                 rounds, verbosity);
  rconv3::destroy();
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
