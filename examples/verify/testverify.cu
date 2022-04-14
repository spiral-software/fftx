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
#include "device_macros.h"

#include "VerifyTransform.hpp"

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
    TransformFunction<DIM, std::complex<double>, std::complex<double>>
      fun(a_mddft, a_domain, a_domain, fullExtents, name, -1);
    VerifyTransform<DIM, std::complex<double>, std::complex<double>>
      (fun, a_rounds, a_verbosity);
    // verifyTransform(a_mddft, a_domain, a_domain, fullextents, -1, a_rounds, a_verbosity);
  }

  {
    std::string name = "imddft";
    std::cout << "***** test " << DIM << "D IMDDFT on complex "
              << a_domain << std::endl;
    TransformFunction<DIM, std::complex<double>, std::complex<double>>
      fun(a_imddft, a_domain, a_domain, fullExtents, name, 1);
    VerifyTransform<DIM, std::complex<double>, std::complex<double>>
      (fun, a_rounds, a_verbosity);
    // verifyTransform(a_imddft, a_domain, a_domain, fullextents, 1, a_rounds, a_verbosity);
  }

  {
    std::string name = "mdprdft";
    std::cout << "***** test " << DIM << "D PRDFT from real "
              << a_domain << " to complex " << a_fdomain << std::endl;
    TransformFunction<DIM, double, std::complex<double>>
      fun(a_prdft, a_domain, a_fdomain, fullExtents, name, -1);
    VerifyTransform<DIM, double, std::complex<double>>
      (fun, a_rounds, a_verbosity);
    // verifyTransform(a_prdft, a_domain, a_fdomain, fullextents, -1, a_rounds, a_verbosity);
  }

  {
    std::string name = "imdprdft";
    std::cout << "***** test " << DIM << "D IPRDFT from complex "
              << a_fdomain << " to real " << a_domain << std::endl;
    TransformFunction<DIM, std::complex<double>, double>
      fun(a_iprdft, a_fdomain, a_domain, fullExtents, name, 1);
    VerifyTransform<DIM, std::complex<double>, double>
      (fun, a_rounds, a_verbosity);
    // verifyTransform(a_iprdft, a_fdomain, a_domain, fullextents, 1, a_rounds, a_verbosity);
  }
}
                    

int main(int argc, char* argv[])
{
  // { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};
  printf("Usage:  %s [verbosity=0] [rounds=20]\n", argv[0]);
  printf("verbosity 0 for summary, 1 for categories, 2 for subtests, 3 for rounds\n");
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
