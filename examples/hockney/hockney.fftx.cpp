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

  std::array<array_t<3, double>, 2>   realInter {hockney::rdomain, hockney::rdomain};
  std::array<array_t<3,std::complex<double>>, 2> complexInter {hockney::freq, hockney::freq};
  array_t<3,double> input(hockney::sbox);
  array_t<3,double> output(hockney::dbox);
  array_t<3,double> symbol(hockney::freq);


  setInputs(input);
  setOutputs(output);
 
  openScalarDAG();

  // on rdomain                   on sbox
  // realInter[0] := zeroEmbedBox(input);
  zeroEmbedBox(realInter[0], input);

  // on freq                  on rdomain
  // complexInter[0] := PRDFT(realInter[0]);
  PRDFT(hockney::rdomain.extents().flipped(), complexInter[0], realInter[0]);

  // on freq            on freq   on freq
  // complexInter[1] := symbol .* complexInter[0];
  kernel(symbol, complexInter[1], complexInter[0]);

  // on rdomain             on freq
  // realInter[1] := IPRDFT(complexInter[1]);
  IPRDFT(hockney::rdomain.extents().flipped(), realInter[1], complexInter[1]);

  // on dbox              on rdomain
  // output := extractBox(realInter[1]);
  extractBox(output, realInter[1]);
  
  closeScalarDAG(complexInter, realInter, "hockney");
  
}
