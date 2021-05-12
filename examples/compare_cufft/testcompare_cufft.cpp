

#include "spiralmddft.fftx.codegen.hpp"

int main(int argc, char* argv[])
{
  printf("%s: Entered test program\n call spiralmddft::init()\n", argv[0]);
	
  spiralmddft::init();

  fftx::box_t<3> domain(fftx::point_t<3>({{1,1,1}}),fftx::point_t<3>({{32,32,32}}));
  
  fftx::array_t<3,std::complex<double>> input(domain);
  fftx::array_t<3,std::complex<double>> output(domain);

  forall([](std::complex<double>(&v), const fftx::point_t<3>& p)
         {
           v=std::complex<double>(2.0,0.0);
         },input);

  printf("call spiralmddft::transform()\n");
  spiralmddft::transform(input, output);
  printf("mddft for size 32 32 32 took  %.7e milliseconds\n", spiralmddft::CPU_milliseconds);
  spiralmddft::destroy();

  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
