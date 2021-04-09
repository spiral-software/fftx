#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include "hockney.h"

using namespace fftx;


int main(int argc, char* argv[])
{

  tracing=true;

  //  const int n  = 8; // 130;
  //  const int ns = 3; // 33;
  //  const int nd = 5; // 96;
  //
  //  box_t<3> sbox({{0,0,0}}, {{ns-1, ns-1, ns-1}});
  //  box_t<3> dbox({{n-nd,n-nd,n-nd}}, {{n-1,n-1,n-1}});
  //  box_t<3> rdomain({{0,0,0}}, {{n-1, n-1, n-1}});
  //  box_t<3> freq({{0,0,0}}, {{(n-1)/2+1, n-1, n-1}});

  std::array<array_t<3, double>, 2>   realIntermediates {hockney::rdomain, hockney::rdomain};
  std::array<array_t<3,std::complex<double>>, 2> complexIntermediates {hockney::freq, hockney::freq};
  array_t<3,double> input(hockney::sbox);
  array_t<3,double> output(hockney::dbox);
  array_t<3,double> symbol(hockney::freq);


  setInputs(input);
  setOutputs(output);
 
  openScalarDAG();

  // realIntermediates[0] := zeroEmbedBox(input);
  zeroEmbedBox(realIntermediates[0], input);
  // complexIntermediates[0] := PRDFT(realIntermediates[0]);
  PRDFT(hockney::rdomain.extents().flipped(), complexIntermediates[0], realIntermediates[0]);
  // complexIntermediates[1] := symbol .* complexIntermediates[0];
  kernel(symbol, complexIntermediates[1], complexIntermediates[0]);
  // realIntermediates[1] := IPRDFT(complexIntermediates[1]);
  IPRDFT(hockney::rdomain.extents().flipped(), realIntermediates[1], complexIntermediates[1]);
  // output := extractBox(realIntermediates[1]);
  extractBox(output, realIntermediates[1]);
  
  closeScalarDAG(complexIntermediates, realIntermediates, "hockney");
  
}
