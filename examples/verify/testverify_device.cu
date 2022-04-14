#include <cmath> // Without this, abs returns zero!
#include <random>
#include "fftx3.hpp"
#include "fftx3utilities.h"
#include "verify_device.h"
// #include "device_macros.h"

#include "VerifyTransform.hpp"

void verify3d(fftx::box_t<3> a_domain,
              fftx::box_t<3> a_fdomain,
              int a_rounds,
              deviceTransform3dType<std::complex<double>, std::complex<double>>& a_mddft,
              deviceTransform3dType<std::complex<double>, std::complex<double>>& a_imddft,
              deviceTransform3dType<double, std::complex<double>>& a_prdft,
              deviceTransform3dType<std::complex<double>, double>& a_iprdft,
              int a_verbosity)
{
  fftx::point_t<3> fullExtents = a_domain.extents();

  {
    std::string name = "mddft";
    std::cout << "***** test 3D MDDFT on complex "
              << a_domain << std::endl;
    TransformFunction<3, std::complex<double>, std::complex<double>>
      fun(a_mddft, a_domain, a_domain, fullExtents, name, -1);
    VerifyTransform<3, std::complex<double>, std::complex<double>>
      (fun, a_rounds, a_verbosity);
    // verifyTransform(a_mddft, a_domain, a_domain, fullextents, -1, a_rounds, a_verbosity);
  }

  {
    std::string name = "imddft";
    std::cout << "***** test 3D IMDDFT on complex "
              << a_domain << std::endl;
    TransformFunction<3, std::complex<double>, std::complex<double>>
      fun(a_imddft, a_domain, a_domain, fullExtents, name, 1);
    VerifyTransform<3, std::complex<double>, std::complex<double>>
      (fun, a_rounds, a_verbosity);
    // verifyTransform(a_imddft, a_domain, a_domain, fullextents, 1, a_rounds, a_verbosity);
  }

  {
    std::string name = "mdprdft";
    std::cout << "***** test 3D PRDFT from real "
              << a_domain << " to complex " << a_fdomain << std::endl;
    TransformFunction<3, double, std::complex<double>>
      fun(a_prdft, a_domain, a_fdomain, fullExtents, name, -1);
    VerifyTransform<3, double, std::complex<double>>
      (fun, a_rounds, a_verbosity);
    // verifyTransform(a_prdft, a_domain, a_fdomain, fullextents, -1, a_rounds, a_verbosity);
  }

  {
    std::string name = "imdprdft";
    std::cout << "***** test 3D IPRDFT from complex "
              << a_fdomain << " to real " << a_domain << std::endl;
    TransformFunction<3, std::complex<double>, double>
      fun(a_iprdft, a_fdomain, a_domain, fullExtents, name, 1);
    VerifyTransform<3, std::complex<double>, double>
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

  verify3d(verify::domain3, verify::fdomain3, rounds,
           mddft3dDevice, imddft3dDevice,
           mdprdft3dDevice, imdprdft3dDevice,
           verbosity);
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
