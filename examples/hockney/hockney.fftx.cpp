// make

#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>

using namespace fftx;


int main(int argc, char* argv[])
{

  tracing=true;

  const int n  =8;
  const int ns =3;
  const int nd =5;

  box_t<3> sbox({{0,0,0}}, {{ns-1, ns-1, ns-1}});
  box_t<3> dbox({{n-nd,n-nd,n-nd}}, {{n-1,n-1,n-1}});
  box_t<3> rdomain({{0,0,0}}, {{n-1, n-1, n-1}});
  box_t<3> freq({{0,0,0}}, {{(n-1)/2+1, n-1, n-1}});

  std::array<array_t<3, double>,2>   realIntermediates {rdomain, rdomain};
  std::array<array_t<3,std::complex<double>>,2> intermediates {freq, freq};
  array_t<3,double> input(sbox);
  array_t<3,double> output(dbox);
  array_t<3,double> symbol(freq);


  setInputs(input);
  setOutputs(output);
 
  
  openScalarDAG();

  zeroEmbedBox(realIntermediates[0], input);
  PRDFT(rdomain.extents(), intermediates[0], realIntermediates[0]);
  kernel(symbol, intermediates[1], intermediates[0]);
  IPRDFT(rdomain.extents(), realIntermediates[1], intermediates[1]);
  extractBox(output, realIntermediates[1]);
  
  closeScalarDAG(intermediates, "rconv");
  
}
