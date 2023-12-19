#include <cmath> // Without this, abs returns zero!
#include <random>
#include "fftx3.hpp"
#include "fftx3utilities.h"

#include "VerifyTransform.hpp"

void verify3d(fftx::point_t<3> a_fullExtents,
              int a_rounds,
              deviceTransform3dType<std::complex<double>, std::complex<double>>& a_mddft,
              deviceTransform3dType<std::complex<double>, std::complex<double>>& a_imddft,
              deviceTransform3dType<double, std::complex<double>>& a_prdft,
              deviceTransform3dType<std::complex<double>, double>& a_iprdft,
              int a_verbosity)
{
  {
    std::string name = "mddft";
    std::cout << "***** test 3D MDDFT complex-to-complex size "
              << a_fullExtents << std::endl;
    TransformFunction<3, std::complex<double>, std::complex<double>>
      fun(a_mddft, a_fullExtents, name, -1);
    VerifyTransform<3, std::complex<double>, std::complex<double>>
      (fun, a_rounds, a_verbosity);
  }

  {
    std::string name = "imddft";
    std::cout << "***** test 3D IMDDFT complex-to-complex size "
              << a_fullExtents << std::endl;
    TransformFunction<3, std::complex<double>, std::complex<double>>
      fun(a_imddft, a_fullExtents, name, 1);
    VerifyTransform<3, std::complex<double>, std::complex<double>>
      (fun, a_rounds, a_verbosity);
  }

  {
    std::string name = "mdprdft";
    std::cout << "***** test 3D PRDFT real-to-complex size "
              << a_fullExtents << std::endl;
    TransformFunction<3, double, std::complex<double>>
      fun(a_prdft, a_fullExtents, name, -1);
    VerifyTransform<3, double, std::complex<double>>
      (fun, a_rounds, a_verbosity);
  }

  {
    std::string name = "imdprdft";
    std::cout << "***** test 3D IPRDFT complex-to-real size "
              << a_fullExtents << std::endl;
    TransformFunction<3, std::complex<double>, double>
      fun(a_iprdft, a_fullExtents, name, 1);
    VerifyTransform<3, std::complex<double>, double>
      (fun, a_rounds, a_verbosity);
  }
}
                    

int main(int argc, char* argv[])
{
  // { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};
  int mm = 24, nn = 32, kk = 40; // default cube dimensions
  char *prog = argv[0];
  int baz = 0;
  int verbosity = 0;
  int rounds = 2;
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
      case 's':
          if(strlen(argv[1]) > 2) {
            baz = 2;
          } else {
            baz = 0;
            argv++, argc--;
          }
          mm = atoi ( & argv[1][baz] );
          while ( argv[1][baz] != 'x' ) baz++;
          baz++ ;
          nn = atoi ( & argv[1][baz] );
          while ( argv[1][baz] != 'x' ) baz++;
          baz++ ;
          kk = atoi ( & argv[1][baz] );
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
          printf ( "Usage: %s: [ -i rounds ] [-v verbosity: 0 for summary, 1 for categories, 2 for subtests, 3 for all iterations] [ -s MMxNNxKK ] [ -h (print help message) ]\n", argv[0] );
          exit (0);
      default:
          printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
      }
      argv++, argc--;
  }

  printf("Running size %dx%dx%d with verbosity %d, random %d rounds\n",
         mm, nn, kk, verbosity, rounds);

  /*
    Set up random number generator.
  */
  std::random_device rd;
  generator = std::mt19937(rd());
  unifRealDist = std::uniform_real_distribution<double>(-0.5, 0.5);

  fftx::point_t<3> fullExtents({{mm, nn, kk}});
  verify3d(fullExtents, rounds,
           mddft3dDevice, imddft3dDevice,
           mdprdft3dDevice, imdprdft3dDevice,
           verbosity);
  
  printf("%s: All done, exiting\n", prog);
  return 0;
}
