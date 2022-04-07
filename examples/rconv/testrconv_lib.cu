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

#include "RealConvolution.hpp"

template<int DIM>
void rconvDimension(fftx::rconv<DIM>& a_transformer,
                    int a_rounds,
                    int a_verbosity)
{
  if (!a_transformer.isDefined())
    {
      return;
    }

  std::cout << "***** test " << DIM << "D real convolution on "
            << a_transformer.inputSize() << std::endl;

  RealConvolution<DIM> fun(&a_transformer);
  TestRealConvolution<DIM>(fun, a_rounds, a_verbosity);
}

void rconvSize(fftx::point_t<3> a_size,
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
  rconvDimension(tfm, a_rounds, a_verbosity);
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
