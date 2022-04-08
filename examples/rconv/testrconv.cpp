#include <cmath> // Without this, abs is the wrong function!
#include <random>
// #include "rconv2.fftx.codegen.hpp"
#include "rconv3.fftx.codegen.hpp"
#include "fftx3utilities.h"
#include "device_macros.h"
#include "rconv.h"

#include "RealConvolution.hpp"
  
// using namespace fftx;

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

  RealConvolution<DIM> fun(a_transform, a_domain, a_fdomain);
  TestRealConvolution<DIM>(fun, a_rounds, a_verbosity);
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
  //    rconvDimension(rconv2::transform, rconv_once::domain2, rconv_once::fdomain2,
  //                   rounds, verbosity);
  //    rconv2::destroy();
  
  /*
    3-dimensional tests.
  */
  rconv3::init();
  rconvDimension(rconv3::transform, rconv_once::domain3, rconv_once::fdomain3,
                 rounds, verbosity);
  rconv3::destroy();
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
