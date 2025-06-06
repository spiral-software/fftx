//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.
//

#include <cmath> // Without this, abs returns zero!
#include <random>

#if defined(FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
#include "fftx_rconv_gpu_public.h"
#else
#include "fftx_rconv_cpu_public.h"
#endif

// #include "fftxrconv.precompile.hpp"

#include "fftxRealConvolution.hpp"

template<int DIM>
int rconvDimension(std::vector<int> sizes,
                   fftx::box_t<DIM> a_domain,
                   fftx::box_t<DIM> a_fdomain,
                   int a_rounds,
                   int a_verbosity)
{
  fftx::OutStream() << "***** test " << DIM << "D real convolution on "
                    << a_domain << std::endl;
  RealConvolution<DIM> fun(sizes, a_domain, a_fdomain);
  return fun.testAll(a_rounds, a_verbosity);
}


int rconvSize(fftx::point_t<3> a_size,
              int a_rounds,
              int a_verbosity)
{
  int offx = 3, offy = 5, offz = 11;
  fftx::point_t<3> offsets({{offx, offy, offz}});

  fftx::point_t<3> truncSize = truncatedComplexDimensions(a_size);

  fftx::box_t<3> fulldomain = domainFromSize(a_size, offsets);
  fftx::box_t<3> halfdomain = domainFromSize(truncSize, offsets);
  
  // fftx::rconv<3> tfm(a_size); // does initialization
  // rconvDimension(tfm, a_rounds, a_verbosity);
  std::vector<int> sizes{a_size[0], a_size[1], a_size[2]};
  return rconvDimension(sizes, fulldomain, halfdomain, a_rounds, a_verbosity);
}
  
int main(int argc, char* argv[])
{
  // { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};
  char *prog = argv[0];
  int verbosity = 0;
  int rounds = 2;
  int baz = 0;
  while ( argc > 1 && argv[1][0] == '-' ) {
      switch ( argv[1][1] ) {
      case 'i':
          if(strlen(argv[1]) > 2) {
            baz = 2;
          } else {
            baz = 0;
            argv++, argc--;
          }
          rounds = atoi ( & argv[1][baz] );
          break;
      case 'v':
          if(strlen(argv[1]) > 2) {
            baz = 2;
          } else {
            baz = 0;
            argv++, argc--;
          }
          verbosity = atoi ( & argv[1][baz] );
          break;
      case 'h':
          fftx::OutStream() << "Usage: " << argv[0]
                            << ": [ -i rounds ] [-v verbosity: 0 for summary, 1 for categories, 2 for subtests, 3 for all iterations] [ -h (print help message) ]"
                            << std::endl;
          exit (0);
      default:
          fftx::OutStream() << prog << ": unknown argument: "
                            << argv[1] << " ... ignored" << std::endl;
      }
      argv++, argc--;
  }

  fftx::OutStream() << "Running with verbosity " << verbosity
                    << ", random " << rounds << " rounds" << std::endl;

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

  int status = 0;

  fftx::point_t<3> *ents = fftx_rconv_QuerySizes ();

  for ( int ind = 0; ents[ind][0] != 0; ind++ )
    {
      status += rconvSize(ents[ind], rounds, verbosity);
    }
  
  // rconvSize(fftx::point_t<3>({{  48,  48,  48 }}), rounds, verbosity);
  
  // fftx::point_t<3> extents = rconv_dims::domain3.extents();
  // fftx::rconv<3> tfm(extents); // does initialization
  // rconvDimension(tfm, rconv_dims::domain3, rconv_dims::fdomain3,
  //                rounds, verbosity);

  fftx::OutStream() << prog << ": All done, exiting with status "
                    << status << std::endl;
  std::flush(fftx::OutStream());

  return status;
}
