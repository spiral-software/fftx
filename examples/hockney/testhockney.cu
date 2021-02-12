

#include "hockney.fftx.codegen.hpp"

using namespace fftx;

int main(int argc, char* argv[])
{
  printf("%s: Entered test program\n call hockney::init()\n", argv[0]);
	
  hockney::init();
 
  const int n  =8;
  const int ns =3;
  const int nd =5;

  box_t<3> sbox({{0,0,0}}, {{ns-1, ns-1, ns-1}});
  box_t<3> dbox({{n-nd,n-nd,n-nd}}, {{n-1,n-1,n-1}});
  box_t<3> freq({{0,0,0}}, {{(n-1)/2+1, n-1, n-1}});
  
  array_t<3,double> inputH(sbox);
  array_t<3,double> outputH(dbox);
  array_t<3,double> symbolH(freq);

  forall([](double(&v), const fftx::point_t<3>& p)
         {
           if(p==point_t<3>({{2,2,2}}))  v=1.0;
           else  v=0.0;
         },inputH);

  forall([](double(&v), const fftx::point_t<3>& p)
           {
           if(p==point_t<3>::Zero())
             v=0;
           else
             v=-1.0/(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);         
           },symbolH);

  double* bufferPtr;
  double* inputPtr;
  double* symbolPtr;
  double* outputPtr;
  cudaMalloc(&bufferPtr, (sbox.size()+2+dbox.size()+2+freq.size())*sizeof(double));
  inputPtr = bufferPtr;
  symbolPtr = bufferPtr+sbox.size()+2;
  outputPtr = symbolPtr+dbox.size()+2;
  cudaMemcpy(inputPtr, inputH.m_data.local(), sbox.size()*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(symbolPtr, symbolH.m_data.local(), freq.size()*sizeof(double),cudaMemcpyHostToDevice);

  array_t<3, double> input(global_ptr<double>(inputPtr,0,1), sbox);
  array_t<3, double> output(global_ptr<double>(outputPtr,0,1), dbox);
  array_t<3, double> symbol(global_ptr<double>(symbolPtr, 0, 1), freq);
  
  
  printf("call hockney::transform()\n");
  hockney::transform(input, output, symbol);
  printf("hockney for size n=%d ns=%d nd=%d took  %.7e milliseconds\n", n, ns, nd, hockney::CPU_milliseconds);
  printf("hockney for size n=%d ns=%d nd=%d took  %.7e milliseconds wrt GPU\n", n, ns, nd, hockney::GPU_milliseconds);
  hockney::destroy();


  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
