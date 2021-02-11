

#include "mddft.fftx.codegen.hpp"
#include "imddft.fftx.codegen.hpp"

#include <chrono>

int main(int argc, char* argv[])
{
  printf("%s: Entered test program\n call mddft::init()\n", argv[0]);
	
  mddft::init();

  fftx::box_t<3> domain(fftx::point_t<3>({{1,1,1}}),fftx::point_t<3>({{32,32,32}}));
  
  fftx::array_t<3,std::complex<double>> inputH(domain);
  fftx::array_t<3,std::complex<double>> outputH(domain);

  forall([](std::complex<double>(&v), const fftx::point_t<3>& p)
         {
           v=std::complex<double>(2.0,0.0);
         },inputH);
  // additional code for GPU programs
  std::complex<double> * bufferPtr;
  std::complex<double> * inputPtr;
  std::complex<double> * outputPtr;
  cudaMalloc(&bufferPtr, domain.size()*sizeof(std::complex<double>)*2);
  inputPtr = bufferPtr;
  outputPtr = bufferPtr+domain.size();
  cudaMemcpy(inputPtr, inputH.m_data.local(), domain.size()*sizeof(std::complex<double>),
             cudaMemcpyHostToDevice);
  fftx::array_t<3,std::complex<double>> input(fftx::global_ptr<std::complex<double>>(inputPtr,0,1), domain);
  fftx::array_t<3,std::complex<double>> output(fftx::global_ptr<std::complex<double>>(outputPtr,0,1), domain);
  //  end special code for GPU
  
  printf("call mddft::transform()\n");

  mddft::transform(input, output);

  printf("mddft for size 32 32 32 took  %.7e milliseconds wrt CPU\n", mddft::CPU_milliseconds);
  printf("mddft for size 32 32 32 took  %.7e milliseconds wrt GPU\n", mddft::GPU_milliseconds);
  


  mddft::destroy();


  printf("call imddft::init()\n");
  imddft::init();

  printf("call imddft::transform()\n");
  imddft::transform(input, output);
  printf("imddft for size 32 32 32 took  %.7e milliseconds wrt CPU\n", imddft::CPU_milliseconds);
  printf("imddft for size 32 32 32 took  %.7e milliseconds wrt GPU\n", imddft::GPU_milliseconds);
  

  imddft::destroy();
  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
