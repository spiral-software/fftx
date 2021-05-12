#include "mddft.fftx.codegen.hpp"
#include "imddft.fftx.codegen.hpp"
#include "test_plan.h"

int main(int argc, char* argv[])
{
  printf("%s: Entered test program\n call mddft::init()\n", argv[0]);
	
  mddft::init();

  fftx::array_t<3,std::complex<double>> input(test_plan::domain);
  fftx::array_t<3,std::complex<double>> output(test_plan::domain);

  forall([](std::complex<double>(&v), const fftx::point_t<3>& p)
         {
           v=std::complex<double>(2.0,0.0);
         },input);

  printf("call mddft::transform()\n");
  mddft::transform(input, output);
  printf("mddft for size %d %d %d took  %.7e milliseconds\n",
         fftx_nx, fftx_ny, fftx_nz, mddft::CPU_milliseconds);
  mddft::destroy();


  printf("call imddft::init()\n");
  imddft::init();

  printf("call imddft::transform()\n");
  imddft::transform(input, output);
  printf("imddft for size %d %d %d took  %.7e milliseconds\n",
         fftx_nx, fftx_ny, fftx_nz, imddft::CPU_milliseconds);
  imddft::destroy();
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
