#include "gethat.fftx.codegen.hpp"
#include "hockney.fftx.codegen.hpp"
#include "fftx3utilities.h"
#include "hockney.h"

#include "device_macros.h"

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

  fftx::array_t<3, double> GHost(hockney::rdomain);
  const double* GHostPtr = GHost.m_data.local();

  // G on (0:n-1)^3.
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
         }, GHost);

  fftx::array_t<3, std::complex<double>> GhatComplexHost(hockney::freq);
  std::complex<double>* GhatComplexHostPtr =
    GhatComplexHost.m_data.local();

  double* GDevicePtr;
  auto Gbytes = hockney::rdomain.size() * sizeof(double);
  DEVICE_MALLOC(&GDevicePtr, Gbytes);
  fftx::array_t<3, double> GDevice
    (fftx::global_ptr<double>(GDevicePtr, 0, 1),
     hockney::rdomain);
  DEVICE_MEM_COPY(GDevicePtr, GHostPtr,
                  Gbytes,
                  MEM_COPY_HOST_TO_DEVICE);
  
  std::complex<double>* GhatComplexDevicePtr;
  auto GhatComplexbytes = hockney::freq.size() * sizeof(std::complex<double>);
  DEVICE_MALLOC(&GhatComplexDevicePtr, GhatComplexbytes);
  fftx::array_t<3, std::complex<double>> GhatComplexDevice
    (fftx::global_ptr<std::complex<double>>(GhatComplexDevicePtr, 0, 1),
     hockney::freq);

  fftx::array_t<3, double>& G = GDevice;
  fftx::array_t<3, std::complex<double>>& GhatComplex = GhatComplexDevice;

  printf("call gethat::init()\n");
  gethat::init();
  printf("call gethat::transform()\n");
  gethat::transform(G, GhatComplex);
  printf("gethat for size n=%d ns=%d nd=%d took  %.7e milliseconds\n",
         hockney::n, hockney::ns, hockney::nd, gethat::CPU_milliseconds);
  gethat::destroy();

  DEVICE_MEM_COPY(GhatComplexHostPtr, GhatComplexDevicePtr,
                  GhatComplexbytes,
                  MEM_COPY_DEVICE_TO_HOST);

  { // Check that the imaginary part of GhatComplexHost is zero.
      double totReal = 0.;
      double totImag = 0.;
      for (int pt = 0; pt < GhatComplexHost.m_domain.size(); pt++)
        {
          std::complex<double> GhatHere = GhatComplexHostPtr[pt];
          totReal += std::abs(GhatHere.real());
          totImag += std::abs(GhatHere.imag());
        }
      std::cout << "sum(|real(Ghat)|)=" << totReal
                << " sum(|imag(Ghat)|)=" << totImag
                << std::endl;
  }

  fftx::array_t<3, double> GhatRealHost(hockney::freq);
  double* GhatRealHostPtr = GhatRealHost.m_data.local();
  {
    for (int pt = 0; pt < hockney::freq.size(); pt++)
      {
        GhatRealHostPtr[pt] = GhatComplexHostPtr[pt].real() /
          (hockney::rdomain.size()*1.);
      }
  }

  double* GhatRealDevicePtr;
  auto GhatRealbytes = hockney::freq.size() * sizeof(double);
  DEVICE_MALLOC(&GhatRealDevicePtr, GhatRealbytes);
  fftx::array_t<3, double>
    GhatRealDevice(fftx::global_ptr<double>(GhatRealDevicePtr, 0, 1),
                   hockney::freq);
  DEVICE_MEM_COPY(GhatRealDevicePtr, GhatRealHostPtr,
                  GhatRealbytes,
                  MEM_COPY_HOST_TO_DEVICE);

  fftx::array_t<3,double> inputHost(hockney::sbox);
  fftx::array_t<3,double> outputHost(hockney::dbox);
  double* inputHostPtr = inputHost.m_data.local();
  double* outputHostPtr = outputHost.m_data.local();

  // input has 1 inside sphere with this midpoint and radius
  int midinput = hockney::ns / 2;
  int radius = hockney::ns / 3;
  int radius2 = radius * radius;
  forall([midinput, radius2](double(&v), const fftx::point_t<3>& p)
         {
           // if(p==fftx::point_t<3>({{2,2,2}}))  v=1.0;
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
           
         }, inputHost);

  double* inputDevicePtr;
  auto inputbytes = hockney::sbox.size() * sizeof(double);
  DEVICE_MALLOC(&inputDevicePtr, inputbytes);
  fftx::array_t<3, double> inputDevice
    (fftx::global_ptr<double>(inputDevicePtr, 0, 1),
     hockney::sbox);
  DEVICE_MEM_COPY(inputDevicePtr, inputHostPtr,
                  inputbytes,
                  MEM_COPY_HOST_TO_DEVICE);
  
  double* outputDevicePtr;
  auto outputbytes = hockney::dbox.size() * sizeof(double);
  DEVICE_MALLOC(&outputDevicePtr, outputbytes);
  fftx::array_t<3, double> outputDevice
    (fftx::global_ptr<double>(outputDevicePtr, 0, 1),
     hockney::dbox);
  
  fftx::array_t<3,double>& input = inputDevice;
  fftx::array_t<3,double>& output = outputDevice;

  printf("call hockney::init()\n");
  hockney::init();
  printf("call hockney::transform()\n");
  hockney::transform(input, output, GhatReal);
  printf("hockney for size n=%d ns=%d nd=%d took  %.7e milliseconds\n",
         hockney::n, hockney::ns, hockney::nd, hockney::CPU_milliseconds);
  hockney::destroy();

  DEVICE_MEM_COPY(outputHostPtr, outputDevicePtr,
                  outputbytes,
                  MEM_COPY_DEVICE_TO_HOST);
  if (verbosity >= 1)
    {
      // fftx::box_t<3> sbox({{0, 0, 0}}, {{ns-1, ns-1, ns-1}});
      int smin = 0;
      int smax = hockney::ns - 1;
      // fftx::box_t<3> dbox({{n-1-nd, n-1-nd, n-1-nd}}, {{n-1, n-1, n-1}});
      int dmin = hockney::n - 1 - hockney::nd;
      int dmax = hockney::n - 1;
      // fftx::box_t<3> rdomain({{0, 0, 0}}, {{n-1, n-1, n-1}});
      int Gmin = 0;
      int Gmax = hockney::n - 1;
      
      double outputMax = 0.;
      for (size_t pt = 0; pt < hockney::dbox.size(); pt++)
        {
          updateMaxAbs(outputMax, outputHostPtr[pt]);
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
                      fftx::point_t<3> ps =
                        fftx::point_t<3>({{is, js, ks}});
                      fftx::point_t<3> pdiff =
                        fftx::point_t<3>({{id-is, jd-js, kd-ks}});
                      directAns +=
                        inputHostPtr[positionInBox(ps, hockney::sbox)] *
                        GHostPtr[positionInBox(pdiff, hockney::rdomain)];
                    }
              double calcAns = outputHostPtr[ptOut];
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

  DEVICE_FREE(GDevicePtr);
  DEVICE_FREE(GhatComplexDevicePtr);
  DEVICE_FREE(GhatRealDevicePtr);
  DEVICE_FREE(inputDevicePtr);
  DEVICE_FREE(outputDevicePtr);
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
