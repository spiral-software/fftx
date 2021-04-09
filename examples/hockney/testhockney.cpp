#include "gethat.fftx.codegen.hpp"
#include "hockney.fftx.codegen.hpp"
#include "hockney.h"

using namespace fftx;

int main(int argc, char* argv[])
{
  printf("%s: Entered test program.\n", argv[0]);
	
  array_t<3,double> G(hockney::rdomain);

  // G on (0:2*n-1)^3.
  int Nall = hockney::n;
  forall([Nall](double(&v), const fftx::point_t<3>& p)
         {
           int sqsum = 0.;
           for (int d = 0; d < 3; d++)
             {
               int disp = p[d] - Nall;
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
      std::cout << "Ghat sum(|real|)=" << totReal
                << " sum(|imag|)=" << totImag
                << std::endl;
  }

  array_t<3, double> GhatReal(hockney::freq);
  { // This is an ugly hack.
    const std::complex<double>* GhatComplexPtr = GhatComplex.m_data.local();
    double* GhatRealPtr = GhatReal.m_data.local();
    for (int pt = 0; pt < hockney::freq.size(); pt++)
      {
        GhatRealPtr[pt] = GhatComplexPtr[pt].real();
      }
  }
  
  array_t<3,double> input(hockney::sbox);
  array_t<3,double> output(hockney::dbox);

  // input has 1 inside sphere of radius 4
  int Ninput = hockney::ns;
  forall([Ninput](double(&v), const fftx::point_t<3>& p)
         {
           // if(p==point_t<3>({{2,2,2}}))  v=1.0;
           // else  v=0.0;
           int sqsum = 0;
           for (int d = 0; d < 3; d++)
             {
               int disp = p[d] - Ninput;
               sqsum += disp * disp;
             }
           if (sqsum < 16)
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

  // BEGIN DEBUG
  if (true)
    {
      double* outputData = output.m_data.local();
      double* inputData = input.m_data.local();
      double* GData = G.m_data.local();
      double outputmax = 0.;
      for (size_t pt = 0; pt < hockney::dbox.size(); pt++)
        {
          double absval = std::abs(outputData[pt]);
          if (absval > outputmax)
            {
              outputmax = absval;
            }
        }
      std::cout << "Max absolute value of output is "
                << outputmax << std::endl;
      // std::cout << "Checking max error in rows from "
      // << (-hockney::nd) << " to " << hockney::nd << std::endl;
      double diffabsmax = 0.;
      size_t pt = 0;
      for (int i = -hockney::nd; i <= hockney::nd; i++)
        {
          for (int j = -hockney::nd; j <= hockney::nd; j++)
            for (int k = -hockney::nd; k <= hockney::nd; k++)
              {
                double Gr_ijk = 0.;
                // We loop through ALL of G, which is on
                // hockney::rdomain == (0:2*Nall-1)^3.
                size_t ptpt = 0;
                for (int ii = -Nall; ii < Nall; ii++)
                  for (int jj = -Nall; jj < Nall; jj++)
                    for (int kk = -Nall; kk < Nall; kk++)
                      {
                        int idiff = i - ii;
                        int jdiff = j - jj;
                        int kdiff = k - kk;
                        if ( (std::abs(idiff) <= hockney::ns) &&
                             (std::abs(jdiff) <= hockney::ns) &&
                             (std::abs(kdiff) <= hockney::ns) )
                          {
                            // Taking input on inputBox == (0:2*hockney::ns)^3.
                            point_t<3> p = point_t<3>({{idiff+hockney::ns, jdiff+hockney::ns, kdiff+hockney::ns}});
                            // Adding input(i-ii, j-jj, k-kk) * G(ii, jj, kk).
                            Gr_ijk += inputData[positionInBox(p, hockney::sbox)] *
                              GData[ptpt];
                          }
                        ptpt++;
                      }
                double Gcalc_ijk = outputData[pt] / (Nall*Nall*Nall*8.);
                double diff_ijk = Gcalc_ijk - Gr_ijk;
                if (false)
                  {
                    printf("%3d%3d%3d exact=%15.7e calc=%15.7e diff=%15.7e\n",
                           i, j, k, Gr_ijk, Gcalc_ijk, diff_ijk);
                  }
                if (std::abs(diff_ijk) > diffabsmax)
                  {
                    diffabsmax = std::abs(diff_ijk);
                  }
                pt++;
              }
          // std::cout << "row " << i << " |diff| <= " << diffabsmax << std::endl;
        }
      std::cout << "Max absolute difference is " << diffabsmax << std::endl;
    }
  // END DEBUG
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
