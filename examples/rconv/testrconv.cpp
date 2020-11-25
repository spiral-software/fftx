
#include <math.h> // Without this, abs is the wrong function!
#include "rconv.fftx.codegen.hpp"

using namespace fftx;

int main(int argc, char* argv[])
{
  printf("%s: Entered test program\n call rconv::init()\n", argv[0]);
	
  rconv::init();
 
  const int nx=32;
  const int ny=32;
  const int nz=32;

  const int fx = (nx/2+1);
  const int fy = ny;
  const int fz = nz;
  
  box_t<3> rdomain(point_t<3>({{1,1,1}}), point_t<3>({{nx,ny,nz}}));
  box_t<3> freq(point_t<3>({{1,1,1}}), point_t<3>({{fx, fy, fz}}));
  
  std::array<array_t<3,std::complex<double>>,2> intermediates {freq, freq};
  array_t<3,double> input(rdomain);
  array_t<3,double> output(rdomain);
  array_t<3,double> symbol(freq);


  forall([](double(&v), const fftx::point_t<3>& p)
         {
           v = p[0]; // FIXME but v=p[1] or v=[2] is OK.  WAS: v=2.0;
         },input);

  double scaling = 1. / (nx*ny*nz*1.); // FIXME new
    forall([scaling](double(&v), const fftx::point_t<3>& p)
           {
           if(p==point_t<3>::Unit())
             v=1;
           else
             v=0;         
           v = scaling; // FIXME new, constant symbol
           },symbol);

  
  printf("call rconv::transform()\n");
  rconv::transform(input, output, symbol);

  // BEGIN DEBUG with constant symbol
  {
    auto outPtr = output.m_data.local();
    auto inPtr = input.m_data.local();
    double diffMax = 0.;
    for (int pt = 0; pt < nx*ny*nz; pt++)
      {
        double diff = outPtr[pt] - inPtr[pt];
        double diffAbs = abs(diff);
        if (diffAbs > diffMax)
          {
            diffMax = diffAbs;
          }
      }
    std::cout << "diffMax = " << diffMax << std::endl;
  }
  // END DEBUG
         
  rconv::destroy();

  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
