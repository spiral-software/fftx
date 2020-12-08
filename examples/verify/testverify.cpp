
#include <cmath> // Without this, abs returns zero!
#include <random>
#include "mddft2.fftx.codegen.hpp"
#include "imddft2.fftx.codegen.hpp"
#include "mddft3.fftx.codegen.hpp"
#include "imddft3.fftx.codegen.hpp"
#include "verify.h"

// using namespace fftx;

std::mt19937 generator;
std::uniform_real_distribution<double> unifReal;
std::uniform_int_distribution<int> unifInt[3];

fftx::point_t<2> unifPoint2()
{
  return fftx::point_t<2>({{unifInt[0](generator),
                            unifInt[1](generator)}});
}

fftx::point_t<3> unifPoint3()
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

template<int DIM>
std::complex<double> unifComplexValPoint(std::complex<double>& a_v,
                                         fftx::point_t<DIM> a_pt)
{
  return unifComplex();
}

template<int DIM>
void unifComplexArray(fftx::array_t<DIM, std::complex<double>> a_arr)
{
  auto arrPtr = a_arr.m_data.local();
  auto dom = a_arr.m_domain;
  size_t npts = dom.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      arrPtr[ind] = unifComplex();
    }
}

template<int DIM>
void sumScaledArrays(fftx::array_t<DIM, std::complex<double>> a_sum,
                     const fftx::array_t<DIM, std::complex<double>> a_arr1,
                     std::complex<double> a_scale1,
                     const fftx::array_t<DIM, std::complex<double>> a_arr2,
                     std::complex<double> a_scale2)
{
  auto dom = a_sum.m_domain;
  assert(dom == a_arr1.m_domain);
  assert(dom == a_arr2.m_domain);
  auto sumPtr = a_sum.m_data.local();
  auto arr1Ptr = a_arr1.m_data.local();
  auto arr2Ptr = a_arr2.m_data.local();
  size_t npts = dom.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      sumPtr[ind] =
        a_scale1 * arr1Ptr[ind] +
        a_scale2 * arr2Ptr[ind];
    }
}

template<int DIM>
void setConstant(fftx::array_t<DIM, std::complex<double>>& a_arr,
                 std::complex<double> a_val)
{
  forall([a_val](std::complex<double>(&v), const fftx::point_t<DIM>& p)
         {
           v = a_val;
         }, a_arr);
}

template<int DIM>
void setUnitImpulse(fftx::array_t<DIM, std::complex<double>>& a_arr,
                    const fftx::point_t<DIM>& a_fixed)
{
  forall([a_fixed](std::complex<double>(&v), const fftx::point_t<DIM>& p)
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

template<int DIM>
void setProductWaves(fftx::array_t<DIM, std::complex<double>>& a_arr,
                     const fftx::point_t<DIM>& a_fixed)
{
  fftx::point_t<DIM> extent = a_arr.m_domain.extents();
  fftx::point_t<DIM> cornerLo = a_arr.m_domain.lo;
  std::complex<double> omega[DIM];
  for (int d = 0; d < DIM; d++)
    {
      double th = (-2 * (a_fixed[d] - cornerLo[d])) * M_PI / (extent[d] * 1.);
      omega[d] = std::complex<double>(cos(th), sin(th));
    }
  forall([omega, cornerLo](std::complex<double>(&v), const fftx::point_t<DIM>& p)
         {
           v = std::complex<double>(1., 0.);
           for (int d = 0; d < DIM; d++)
             {
               v *= pow(omega[d], p[d] - cornerLo[d]);
             }
         }, a_arr);
}

template<int DIM>
double absMaxDiffArray(fftx::array_t<DIM, std::complex<double>>& a_arr1,
                       fftx::array_t<DIM, std::complex<double>>& a_arr2)
{
  auto dom = a_arr1.m_domain;
  assert(dom == a_arr2.m_domain);
  auto arr1Ptr = a_arr1.m_data.local();
  auto arr2Ptr = a_arr2.m_data.local();
  size_t npts = dom.size();
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
  int rounds = 20;
  if (argc > 1)
    {
      rounds = atoi(argv[1]);
    }
  printf("Number of rounds: %d\n", rounds);
	
  printf("Call mddft2::init()\n");
  mddft2::init();
  printf("Call imddft2::init()\n");
  imddft2::init();

  printf("Call mddft3::init()\n");
  mddft3::init();
  printf("Call imddft3::init()\n");
  imddft3::init();

  // fftx::array_t<3, std::complex<double>> input(verify::domain3);
  // fftx::array_t<3, std::complex<double>> output(verify::domain3);

  /*
    Set up random number generator.
  */
  std::random_device rd;
  generator = std::mt19937(rd());
  unifReal = std::uniform_real_distribution<double>(-0.5, 0.5);
  for (int d = 0; d < 3; d++)
    {
      unifInt[d] = std::uniform_int_distribution<int>(verify::domain3.lo[d],
                                                      verify::domain3.hi[d]);
    }
  

  std::cout << "*******************************************" << std::endl;
  std::cout << "2D domain = " << verify::domain2 << std::endl;
  std::cout << "*******************************************" << std::endl;

  /*
    2D test 1: Check linearity.
   */
  {
    fftx::array_t<2, std::complex<double>> inA(verify::domain2);
    fftx::array_t<2, std::complex<double>> inB(verify::domain2);
    fftx::array_t<2, std::complex<double>> LCin(verify::domain2);

    fftx::array_t<2, std::complex<double>> outA(verify::domain2);
    fftx::array_t<2, std::complex<double>> outB(verify::domain2);
    fftx::array_t<2, std::complex<double>> LCout(verify::domain2);
    fftx::array_t<2, std::complex<double>> outLCin(verify::domain2);
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
          mddft2::transform(inA, outA);
          mddft2::transform(inB, outB);
          sumScaledArrays(LCout, outA, alpha, outB, beta);
          mddft2::transform(LCin, outLCin);

          double dd = absMaxDiffArray(outLCin, LCout);
          printf("2D MDDFT linearity test %d max abs diff = %e\n", itn+1, dd);
        }

        { // Test IMDDFT.
          imddft2::transform(inA, outA);
          imddft2::transform(inB, outB);
          sumScaledArrays(LCout, outA, alpha, outB, beta);
          imddft2::transform(LCin, outLCin);

          double dd = absMaxDiffArray(outLCin, LCout);
          printf("2D IMDDFT linearity test %d max abs diff = %e\n", itn+1, dd);
        }
      }
    }

  /*
    2D test 2: Check that unit impulse is transformed properly.
  */
  {
    fftx::array_t<2, std::complex<double>> inImpulse(verify::domain2);
    fftx::array_t<2, std::complex<double>> outImpulse(verify::domain2);
    fftx::array_t<2, std::complex<double>> outCheck(verify::domain2);

    // Unit impulse at low corner.
    setUnitImpulse(inImpulse, verify::domain2.lo);
    std::complex<double> one = std::complex<double>(1., 0.);
    setConstant(outCheck, one);
    {
      mddft2::transform(inImpulse, outImpulse);
      double dd = absMaxDiffArray(outImpulse, outCheck);
      printf("2D unit impulse low corner test on MDDFT max abs diff = %e\n", dd);
    }
    {
      imddft2::transform(inImpulse, outImpulse);
      double dd = absMaxDiffArray(outImpulse, outCheck);
      printf("2D unit impulse low corner test on IMDDFT max abs diff = %e\n", dd);
    }

    for (int itn = 0; itn < rounds; itn++)
        {
          fftx::point_t<2> rpoint = unifPoint2();
          setUnitImpulse(inImpulse, rpoint);
          mddft2::transform(inImpulse, outImpulse);
          setProductWaves(outCheck, rpoint);
          double dd = absMaxDiffArray(outImpulse, outCheck);
          std::cout << "2D unit impulse test of MDDFT at " << rpoint
                    << " max abs diff = " << dd << std::endl;
        }
  }
  


  std::cout << "*******************************************" << std::endl;
  std::cout << "2D domain = " << verify::domain3 << std::endl;
  std::cout << "*******************************************" << std::endl;

  /*
    3D test 1: Check linearity.
   */
  {
    fftx::array_t<3, std::complex<double>> inA(verify::domain3);
    fftx::array_t<3, std::complex<double>> inB(verify::domain3);
    fftx::array_t<3, std::complex<double>> LCin(verify::domain3);

    fftx::array_t<3, std::complex<double>> outA(verify::domain3);
    fftx::array_t<3, std::complex<double>> outB(verify::domain3);
    fftx::array_t<3, std::complex<double>> LCout(verify::domain3);
    fftx::array_t<3, std::complex<double>> outLCin(verify::domain3);
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
          mddft3::transform(inA, outA);
          mddft3::transform(inB, outB);
          sumScaledArrays(LCout, outA, alpha, outB, beta);
          mddft3::transform(LCin, outLCin);

          double dd = absMaxDiffArray(outLCin, LCout);
          printf("3D MDDFT linearity test %d max abs diff = %e\n", itn+1, dd);
        }

        { // Test IMDDFT.
          imddft3::transform(inA, outA);
          imddft3::transform(inB, outB);
          sumScaledArrays(LCout, outA, alpha, outB, beta);
          imddft3::transform(LCin, outLCin);

          double dd = absMaxDiffArray(outLCin, LCout);
          printf("3D IMDDFT linearity test %d max abs diff = %e\n", itn+1, dd);
        }
      }
    }

  /*
    3D test 2: Check that unit impulse is transformed properly.
  */
  {
    fftx::array_t<3, std::complex<double>> inImpulse(verify::domain3);
    fftx::array_t<3, std::complex<double>> outImpulse(verify::domain3);
    fftx::array_t<3, std::complex<double>> outCheck(verify::domain3);

    // Unit impulse at low corner.
    setUnitImpulse(inImpulse, verify::domain3.lo);
    std::complex<double> one = std::complex<double>(1., 0.);
    setConstant(outCheck, one);
    {
      mddft3::transform(inImpulse, outImpulse);
      double dd = absMaxDiffArray(outImpulse, outCheck);
      printf("3D unit impulse low corner test on MDDFT max abs diff = %e\n", dd);
    }
    {
      imddft3::transform(inImpulse, outImpulse);
      double dd = absMaxDiffArray(outImpulse, outCheck);
      printf("3D unit impulse low corner test on IMDDFT max abs diff = %e\n", dd);
    }

    for (int itn = 0; itn < rounds; itn++)
      {
        fftx::point_t<3> rpoint = unifPoint3();
        setUnitImpulse(inImpulse, rpoint);
        mddft3::transform(inImpulse, outImpulse);
        setProductWaves(outCheck, rpoint);
        double dd = absMaxDiffArray(outImpulse, outCheck);
        std::cout << "3D unit impulse test of MDDFT at " << rpoint
                  << " max abs diff = " << dd << std::endl;
      }
  }
  
  mddft2::destroy();
  imddft2::destroy();

  mddft3::destroy();
  imddft3::destroy();

  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
