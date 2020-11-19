

#include "rconv.fftx.codegen.hpp"


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
  array_t<3,double> inputs(rdomain);
  array_t<3,double> outputs(rdomain);
  array_t<3,double> symbol(freq);


  forall([](double(&v), const fftx::point_t<3>& p)
         {
           v=2.0;
         },input);

    forall([](double(&v), const fftx::point_t<3>& p)
           {
           if(p==point_t<3>::Unit())
             v=1;
           else
             v=0;         
           },symbol);

  
  printf("call rconv::transform()\n");
  rconv::transform(input, output, symbol);

  rconv::destroy();


  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
