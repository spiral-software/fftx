#include <cmath> // Without this, abs returns zero!
#include <random>
#include "fftx3.hpp"
#include "fftx3utilities.h"
#include "device_macros.h"

#include "VerifyTransform.hpp"
#include "mddftObj.hpp"
#include "imddftObj.hpp"
#include "mdprdftObj.hpp"
#include "imdprdftObj.hpp"

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

  fftx::point_t<DIM> fullExtents = a_domain.extents();

  {
    std::string name = "mddft";
    std::cout << "***** test " << DIM << "D MDDFT on complex "
              << a_domain << std::endl;


    std::cout << "Now MDDFTProblem" << std::endl;
    // std::vector<int> sizes{mm,nn,kk};
    std::vector<int> sizes{fullExtents[0], fullExtents[1], fullExtents[2]};
    std::vector<void*> args; // fill this in later
    MDDFTProblem mdp(args, sizes, "mddft");
    TransformFunction<DIM, std::complex<double>, std::complex<double>>
      // funprob(&mdp, a_domain, a_domain, fullExtents, name, -1);
      funprob(&mdp, -1);
    VerifyTransform<DIM, std::complex<double>, std::complex<double>>
      (funprob, a_rounds, a_verbosity);

    std::cout << "########### with a_mddft now" << std::endl;
    
    TransformFunction<DIM, std::complex<double>, std::complex<double>>
      fun(a_mddft, a_domain, a_domain, fullExtents, name, -1);
    VerifyTransform<DIM, std::complex<double>, std::complex<double>>
      (fun, a_rounds, a_verbosity);
  }

  {
    std::string name = "imddft";
    std::cout << "***** test " << DIM << "D IMDDFT on complex "
              << a_domain << std::endl;
    TransformFunction<DIM, std::complex<double>, std::complex<double>>
      fun(a_imddft, a_domain, a_domain, fullExtents, name, 1);
    VerifyTransform<DIM, std::complex<double>, std::complex<double>>
      (fun, a_rounds, a_verbosity);
  }

  {
    std::string name = "mdprdft";
    std::cout << "***** test " << DIM << "D PRDFT from real "
              << a_domain << " to complex " << a_fdomain << std::endl;
    TransformFunction<DIM, double, std::complex<double>>
      fun(a_prdft, a_domain, a_fdomain, fullExtents, name, -1);
    VerifyTransform<DIM, double, std::complex<double>>
      (fun, a_rounds, a_verbosity);
  }

  {
    std::string name = "imdprdft";
    std::cout << "***** test " << DIM << "D IPRDFT from complex "
              << a_fdomain << " to real " << a_domain << std::endl;
    TransformFunction<DIM, std::complex<double>, double>
      fun(a_iprdft, a_fdomain, a_domain, fullExtents, name, 1);
    VerifyTransform<DIM, std::complex<double>, double>
      (fun, a_rounds, a_verbosity);
  }
}
                    

template<int DIM>
void verifyDimension(fftx::point_t<DIM>& a_fullExtents, // need for templating
                     MDDFTProblem& a_mddft,
                     IMDDFTProblem& a_imddft,
                     MDPRDFTProblem& a_mdprdft,
                     IMDPRDFTProblem& a_imdprdft,
                     int a_rounds,
                     int a_verbosity)
{
  {
    std::string name = "mddft";
    std::cout << "***** test " << DIM << "D MDDFT complex-to-complex size "
              << a_fullExtents << std::endl;
    TransformFunction<DIM, std::complex<double>, std::complex<double>>
      funprob(&a_mddft, -1);
    VerifyTransform<DIM, std::complex<double>, std::complex<double>>
      (funprob, a_rounds, a_verbosity);
  }

  {
    std::string name = "imddft";
    std::cout << "***** test " << DIM << "D IMDDFT complex-to-complex size "
              << a_fullExtents << std::endl;
    TransformFunction<DIM, std::complex<double>, std::complex<double>>
      funprob(&a_imddft, 1);
    VerifyTransform<DIM, std::complex<double>, std::complex<double>>
      (funprob, a_rounds, a_verbosity);
  }

  {
    std::string name = "mdprdft";
    std::cout << "***** test " << DIM << "D PRDFT real-to-complex size "
              << a_fullExtents << std::endl;
    TransformFunction<DIM, double, std::complex<double>>
      funprob(&a_mdprdft, -1);
    VerifyTransform<DIM, double, std::complex<double>>
      (funprob, a_rounds, a_verbosity);
  }

  {
    std::string name = "imdprdft";
    std::cout << "***** test " << DIM << "D IPRDFT complex-to-real size "
              << a_fullExtents << std::endl;
    TransformFunction<DIM, std::complex<double>, double>
      funprob(&a_imdprdft, 1);
    VerifyTransform<DIM, std::complex<double>, double>
      (funprob, a_rounds, a_verbosity);
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

  std::vector<int> sizes{mm, nn, kk};
  std::vector<void*> args; // fill this in later
  MDDFTProblem mddft(args, sizes, "mddft");
  IMDDFTProblem imddft(args, sizes, "imddft");
  MDPRDFTProblem mdprdft(args, sizes, "mdprdft");
  IMDPRDFTProblem imdprdft(args, sizes, "imdprdft");
  fftx::point_t<3> fullExtents({{mm, nn, kk}});
  verifyDimension(fullExtents, mddft, imddft, mdprdft, imdprdft,
                  rounds, verbosity);

  /*
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
  */

  printf("%s: All done, exiting\n", prog);
  return 0;
}
