#include "gethat.fftx.codegen.hpp"
#include "hockney.fftx.codegen.hpp"
#include "fftx3utilities.h"
#include "hockney.h"

using namespace fftx;

int main(int argc, char* argv[])
{
  int verbosity = 0;
  if (argc > 1)
    {
      verbosity = atoi(argv[1]);
    }
  else
    {
      printf("Usage:  %s [verbosity=0]\n", argv[0]);
    }
  printf("Running with verbosity %d\n", verbosity);
  printf("Set verbosity to 1 to check results, 2 to write all results.\n");
  
  // printf("%s: Entered test program.\n", argv[0]);
	
  array_t<3,double> G(hockney::rdomain);

  // G on (0:2*n-1)^3.
  int midG = hockney::n / 2;
  forall([midG](double(&v), const fftx::point_t<3>& p)
         {
           int sqsum = 0.;
           for (int d = 0; d < 3; d++)
             {
               int disp = p[d] - midG;
               sqsum += disp * disp;
             }
           if (sqsum > 0)
             {
               v = -1. / (sqsum * M_PI);
             }
           else
             {
               v = 0.;
             }
         }, G);

  array_t<3, std::complex<double>> GhatComplex(hockney::freq);
  printf("call gethat::init()\n");
  gethat::init();
  printf("call gethat::transform()\n");
  gethat::transform(G, GhatComplex);
  printf("gethat for size n=%d ns=%d nd=%d took  %.7e milliseconds\n",
         hockney::n, hockney::ns, hockney::nd, gethat::CPU_milliseconds);
  gethat::destroy();

  { // Check that the imaginary part of GhatComplex is zero.
      const std::complex<double>* GhatComplexPtr = GhatComplex.m_data.local();
      double totReal = 0.;
      double totImag = 0.;
      for (int pt = 0; pt < GhatComplex.m_domain.size(); pt++)
        {
          std::complex<double> GhatHere = GhatComplexPtr[pt];
          totReal += std::abs(GhatHere.real());
          totImag += std::abs(GhatHere.imag());
        }
      std::cout << "sum(|real(Ghat)|)=" << totReal
                << " sum(|imag(Ghat)|)=" << totImag
                << std::endl;
  }

  array_t<3, double> GhatReal(hockney::freq);
  {
    const std::complex<double>* GhatComplexPtr = GhatComplex.m_data.local();
    double* GhatRealPtr = GhatReal.m_data.local();
    for (int pt = 0; pt < hockney::freq.size(); pt++)
      {
        GhatRealPtr[pt] = GhatComplexPtr[pt].real() /
          (hockney::rdomain.size()*1.);
      }
  }
  
  array_t<3,double> input(hockney::sbox);
  array_t<3,double> output(hockney::dbox);

  // input has 1 inside sphere with this midpoint and radius
  int midinput = hockney::ns / 2;
  int radius = hockney::ns / 3;
  int radius2 = radius * radius;
  forall([midinput, radius2](double(&v), const fftx::point_t<3>& p)
         {
           // if(p==point_t<3>({{2,2,2}}))  v=1.0;
           // else  v=0.0;
           int sqsum = 0;
           for (int d = 0; d < 3; d++)
             {
               int disp = p[d] - midinput;
               sqsum += disp * disp;
             }
           if (sqsum < radius2)
             { v = 1.; }
           else
             { v = 0.; }
           
         }, input);

  
  printf("call hockney::init()\n");
  hockney::init();
  printf("call hockney::transform()\n");
  hockney::transform(input, output, GhatReal);
  printf("hockney for size n=%d ns=%d nd=%d took  %.7e milliseconds\n",
         hockney::n, hockney::ns, hockney::nd, hockney::CPU_milliseconds);
  hockney::destroy();

  if (verbosity >= 1)
    {
      // box_t<3> sbox({{0, 0, 0}}, {{ns-1, ns-1, ns-1}});
      int smin = 0;
      int smax = hockney::ns - 1;
      // box_t<3> dbox({{n-1-nd, n-1-nd, n-1-nd}}, {{n-1, n-1, n-1}});
      int dmin = hockney::n - 1 - hockney::nd;
      int dmax = hockney::n - 1;
      // box_t<3> rdomain({{0, 0, 0}}, {{n-1, n-1, n-1}});
      int Gmin = 0;
      int Gmax = hockney::n - 1;
      
      double* outputData = output.m_data.local();
      double* inputData = input.m_data.local();
      double* GData = G.m_data.local();

      double outputMax = 0.;
      for (size_t pt = 0; pt < hockney::dbox.size(); pt++)
        {
          updateMaxAbs(outputMax, outputData[pt]);
        }
      std::cout << "Max absolute value of output is "
                << outputMax << std::endl;

      double diffAbsMax = 0.;
      size_t ptOut = 0;
      for (int id = dmin; id <= dmax; id++)
        for (int jd = dmin; jd <= dmax; jd++)
          for (int kd = dmin; kd <= dmax; kd++)
            {
              // Set direct_{id,jd,kd} =
              // sum_{is,js,ks} ( input([is,js,ks]) * G([id-is,jd-js,kd-ks]) )
              // Note dest range [id,jd,kd] is [n-1-nd:n-1]^3,
              // and source range [is,js,ks] is [0:ns-1]^3,
              // so diff range [id-is,jd-js,kd-ks] is [n-nd-ns:n-1]^3.
              // Compare this exact answer with output of transform.
              double directAns = 0.;
              for (int is = smin; is <= smax; is++)
                for (int js = smin; js <= smax; js++)
                  for (int ks = smin; ks <= smax; ks++)
                    {
                      point_t<3> ps = point_t<3>({{is, js, ks}});
                      point_t<3> pdiff = point_t<3>({{id-is, jd-js, kd-ks}});
                      directAns +=
                        inputData[positionInBox(ps, hockney::sbox)] *
                        GData[positionInBox(pdiff, hockney::rdomain)];
                    }
              double calcAns = outputData[ptOut];
              double diffAns = calcAns - directAns;
              if (verbosity >= 2)
                {
                  printf("%3d%3d%3d exact=%15.7e calc=%15.7e diff=%15.7e\n",
                         id, jd, kd, directAns, calcAns, diffAns);
                }
              updateMaxAbs(diffAbsMax, diffAns);
              ptOut++;
            }
      std::cout << "Max absolute difference is " << diffAbsMax << std::endl;
    }
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
