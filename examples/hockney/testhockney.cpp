

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
 
  array_t<3,double> inputs(sbox);
  array_t<3,double> outputs(dbox);
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

  
  printf("call hockney::transform()\n");
  hockney::transform(input, output, symbol);

  hockney::destroy();


  
  printf("%s: All done, exiting\n", argv[0]);
  return 0;
}
