#include <cmath> // Without this, abs returns zero!
#include <random>

#if defined(FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
#include "fftx_mddft_gpu_public.h"
#include "fftx_imddft_gpu_public.h"
#include "fftx_mdprdft_gpu_public.h"
#include "fftx_imdprdft_gpu_public.h"
// #include "fftx_rconv_gpu_public.h"
#else
#include "fftx_mddft_cpu_public.h"
#include "fftx_imddft_cpu_public.h"
#include "fftx_mdprdft_cpu_public.h"
#include "fftx_imdprdft_cpu_public.h"
// #include "fftx_rconv_cpu_public.h"
#endif

#include "mddft.fftx.precompile.hpp"
#include "imddft.fftx.precompile.hpp"
#include "mdprdft.fftx.precompile.hpp"
#include "imdprdft.fftx.precompile.hpp"
// #include "rconv.fftx.precompile.hpp"

#include "fftx3utilities.h"

#include "device_macros.h"
#include "VerifyTransform.hpp"

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
          printf ( "Usage: %s: [ -i rounds ] [-v verbosity: 0 for summary, 1 for categories, 2 for subtests, 3 for all iterations] [ -h (print help message) ]\n", argv[0] );
          exit (0);
      default:
          printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
      }
      argv++, argc--;
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

      {
        fftx::mddft<3> tfm(sz);
        if (tfm.isDefined())
          {
            TransformFunction<3, std::complex<double>, std::complex<double>>
              fun(&tfm, -1);
            VerifyTransform<3, std::complex<double>, std::complex<double>>
              (fun, rounds, verbosity);
          }
       }

      {
        fftx::imddft<3> tfm(sz);
        if (tfm.isDefined())
          {
            TransformFunction<3, std::complex<double>, std::complex<double>>
              fun(&tfm, 1);
            VerifyTransform<3, std::complex<double>, std::complex<double>>
              (fun, rounds, verbosity);
          }
       }

      {
        fftx::mdprdft<3> tfm(sz);
        if (tfm.isDefined())
          {
            TransformFunction<3, double, std::complex<double>>
              fun(&tfm, -1);
            VerifyTransform<3, double, std::complex<double>>
              (fun, rounds, verbosity);
          }
      }

      {
        fftx::imdprdft<3> tfm(sz);
        if (tfm.isDefined())
          {
            TransformFunction<3, std::complex<double>, double>
              fun(&tfm, 1);
            VerifyTransform<3, std::complex<double>, double>
              (fun, rounds, verbosity);
          }
      }
    }

  printf("%s: All done, exiting\n", prog);
  return 0;
}
