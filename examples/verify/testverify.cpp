
#include <cmath> // Without this, abs returns zero!
#include <random>
#include "verify_mddft.fftx.codegen.hpp"
#include "verify_imddft.fftx.codegen.hpp"
#include "verify.h"

// using namespace fftx;

std::mt19937 generator;
std::uniform_real_distribution<double> unifReal;
std::uniform_int_distribution<int> unifInt[3];

fftx::point_t<3> unifPoint()
{
  return fftx::point_t<3>({{unifInt[0](generator),
                            unifInt[1](generator),
                            unifInt[2](generator)}});
          
}

std::complex<double> unifComplex()
{
  return std::complex<double>(unifReal(generator),
                              unifReal(generator));
}

std::complex<double> unifComplexValPoint(std::complex<double>& a_v,
                                         fftx::point_t<3> a_pt)
{
  return unifComplex();
}

void unifComplexArray(fftx::array_t<3, std::complex<double>> a_arr)
{
  auto arrPtr = a_arr.m_data.local();
  auto domain = a_arr.m_domain;
  size_t npts = domain.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      arrPtr[ind] = unifComplex();
    }
}

void sumScaledArrays(fftx::array_t<3, std::complex<double>> a_sum,
                     const fftx::array_t<3, std::complex<double>> a_arr1,
                     std::complex<double> a_scale1,
                     const fftx::array_t<3, std::complex<double>> a_arr2,
                     std::complex<double> a_scale2)
{
  auto domain = a_sum.m_domain;
  assert(domain == a_arr1.m_domain);
  assert(domain == a_arr2.m_domain);
  auto sumPtr = a_sum.m_data.local();
  auto arr1Ptr = a_arr1.m_data.local();
  auto arr2Ptr = a_arr2.m_data.local();
  size_t npts = domain.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      sumPtr[ind] =
        a_scale1 * arr1Ptr[ind] +
        a_scale2 * arr2Ptr[ind];
    }
}

void setConstant(fftx::array_t<3, std::complex<double>>& a_arr,
                 std::complex<double> a_val)
{
  forall([a_val](std::complex<double>(&v), const fftx::point_t<3>& p)
         {
           v = a_val;
         }, a_arr);
}

void setUnitImpulse(fftx::array_t<3, std::complex<double>>& a_arr,
                    const fftx::point_t<3>& a_fixed)
{
  forall([a_fixed](std::complex<double>(&v), const fftx::point_t<3>& p)
         {
           if (p == a_fixed)
             {
               v = std::complex<double>(1.0, 0.0);
             }
           else
             {
               v = std::complex<double>(0.0, 0.0);
             }
         }, a_arr);
}

void setProductWaves(fftx::array_t<3, std::complex<double>>& a_arr,
                     const fftx::point_t<3>& a_fixed)
{
  fftx::point_t<3> extent = a_arr.m_domain.extents();
  fftx::point_t<3> cornerLo = a_arr.m_domain.lo;
  std::complex<double> omega[3];
  for (int d = 0; d < 3; d++)
    {
      double th = (-2 * (a_fixed[d] - cornerLo[d])) * M_PI / (extent[d] * 1.);
      omega[d] = std::complex<double>(cos(th), sin(th));
      std::cout << d << " th = " << th << " omega = " << omega[d] << std::endl;
    }
  forall([omega, cornerLo](std::complex<double>(&v), const fftx::point_t<3>& p)
         {
           v = pow(omega[0], p[0] - cornerLo[0]) *
               pow(omega[1], p[1] - cornerLo[1]) *
               pow(omega[2], p[2] - cornerLo[2]);
         }, a_arr);
}

double absMaxDiffArray(fftx::array_t<3, std::complex<double>>& a_arr1,
                       fftx::array_t<3, std::complex<double>>& a_arr2)
{
  auto domain = a_arr1.m_domain;
  assert(domain == a_arr2.m_domain);
  auto arr1Ptr = a_arr1.m_data.local();
  auto arr2Ptr = a_arr2.m_data.local();
  size_t npts = domain.size();
  double absDiffMax = 0.;
  for (size_t ind = 0; ind < npts; ind++)
    {
      //      std::cout << ind << "   "
      //                << arr1Ptr[ind] << "   " << arr2Ptr[ind] << std::endl;
      double absDiffHere = abs(arr1Ptr[ind] - arr2Ptr[ind]);
      if (absDiffHere > absDiffMax) absDiffMax = absDiffHere;
    }
  return absDiffMax;
}

int main(int argc, char* argv[])
{
  printf("%s: Entered test program\n", argv[0]);
  std::cout << "domain = " << verify::domain << std::endl;
  int rounds = 20;
  if (argc > 1)
    {
      rounds = atoi(argv[1]);
    }
  printf("Number of rounds: %d\n", rounds);
	
  printf("Call verify_mddft::init()\n");
  verify_mddft::init();
  printf("Call verify_imddft::init()\n");
  verify_imddft::init();

  fftx::array_t<3, std::complex<double>> input(verify::domain);
  fftx::array_t<3, std::complex<double>> output(verify::domain);

  /*
    Set up random number generator.
  */
  std::random_device rd;
  generator = std::mt19937(rd());
  unifReal = std::uniform_real_distribution<double>(-0.5, 0.5);
  for (int d = 0; d < 3; d++)
    {
      unifInt[d] = std::uniform_int_distribution<int>(verify::domain.lo[d],
                                                      verify::domain.hi[d]);
    }
  
  /*
    Test 1: Check linearity.
   */
  {
    fftx::array_t<3, std::complex<double>> inA(verify::domain);
    fftx::array_t<3, std::complex<double>> inB(verify::domain);
    fftx::array_t<3, std::complex<double>> LCin(verify::domain);

    fftx::array_t<3, std::complex<double>> outA(verify::domain);
    fftx::array_t<3, std::complex<double>> outB(verify::domain);
    fftx::array_t<3, std::complex<double>> LCout(verify::domain);
    fftx::array_t<3, std::complex<double>> outLCin(verify::domain);
    for (int itn = 0; itn < rounds; itn++)
      {
        std::complex<double> alpha = unifComplex();
        std::complex<double> beta = unifComplex();
        // These do not seem to have any effect.  Sorry.
        // forall(unifComplexValPoint, inA);
        // forall(unifComplexValPoint, inB);
        unifComplexArray(inA);
        unifComplexArray(inB);
        sumScaledArrays(LCin, inA, alpha, inB, beta);
        
        { // Test MDDFT.
          verify_mddft::transform(inA, outA);
          verify_mddft::transform(inB, outB);
          sumScaledArrays(LCout, outA, alpha, outB, beta);
          verify_mddft::transform(LCin, outLCin);

          double dd = absMaxDiffArray(outLCin, LCout);
          printf("MDDFT linearity test %d max abs diff = %e\n", itn+1, dd);
        }

        { // Test IMDDFT.
          verify_imddft::transform(inA, outA);
          verify_imddft::transform(inB, outB);
          sumScaledArrays(LCout, outA, alpha, outB, beta);
          verify_imddft::transform(LCin, outLCin);

          double dd = absMaxDiffArray(outLCin, LCout);
          printf("IMDDFT linearity test %d max abs diff = %e\n", itn+1, dd);
        }
      }
    }

  /*
    Test 2: Check that unit impulse is transformed properly.
  */
  {
    fftx::array_t<3, std::complex<double>> inImpulse(verify::domain);
    fftx::array_t<3, std::complex<double>> outImpulse(verify::domain);
    fftx::array_t<3, std::complex<double>> outCheck(verify::domain);

    // Unit impulse at low corner.
    setUnitImpulse(inImpulse, verify::domain.lo);
    std::complex<double> one = std::complex<double>(1., 0.);
    setConstant(outCheck, one);
    {
      verify_mddft::transform(inImpulse, outImpulse);
      double dd = absMaxDiffArray(outImpulse, outCheck);
      printf("Unit impulse low corner test on MDDFT max abs diff = %e\n", dd);
    }
    {
      verify_imddft::transform(inImpulse, outImpulse);
      double dd = absMaxDiffArray(outImpulse, outCheck);
      printf("Unit impulse low corner test on IMDDFT max abs diff = %e\n", dd);
    }

    for (int itn = 0; itn < verify::nx; itn++) // FIXME: itn < rounds
      {
        fftx::point_t<3> rpoint = fftx::point_t<3>({{itn+1, 1, 1}}); // FIXME: was unifPoint()
        setUnitImpulse(inImpulse, rpoint);
        verify_mddft::transform(inImpulse, outImpulse);
        setProductWaves(outCheck, rpoint);
        double dd = absMaxDiffArray(outImpulse, outCheck);
        std::cout << "Unit impulse test of MDDFT at " << rpoint
                  << " max abs diff = " << dd << std::endl;
        auto outImpulsePtr = outImpulse.m_data.local();
        auto outCheckPtr = outCheck.m_data.local();
        for (int i = verify::domain.lo[0]; i <= verify::domain.hi[0]; i++)
          for (int j = verify::domain.lo[1]; j <= verify::domain.hi[1]; j++)
            for (int k = verify::domain.lo[2]; k <= verify::domain.hi[2]; k++)
              {
                fftx::point_t<3> pt = fftx::point_t<3>({{i, j, k}});
                size_t pos = fftx::positionInBox(pt, verify::domain);
                std::complex<double> outImpulseHere = outImpulsePtr[pos];
                std::complex<double> outCheckHere = outCheckPtr[pos];
                std::cout << pt
                          << " outImpulse=" << outImpulseHere
                          << " outCheck=" << outCheckHere
                          << std::endl;
              }
      }
  }
  
  verify_mddft::destroy();
  verify_imddft::destroy();

  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
