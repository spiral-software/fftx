// make

#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include "rconv.h"

using namespace fftx;

int main(int argc, char* argv[])
{

  tracing=true;
  
  std::array<array_t<2,std::complex<double>>,2> intermediates {rconv::fdomain2, rconv::fdomain2};
  array_t<2,double> inputs(rconv::domain2);
  array_t<2,double> outputs(rconv::domain2);
  array_t<2,double> symbol(rconv::fdomain2);


  setInputs(inputs);
  setOutputs(outputs);
 
  openScalarDAG();

  PRDFT(rconv::domain2.extents().flipped(), intermediates[0], inputs);
  kernel(symbol, intermediates[1], intermediates[0]);
  IPRDFT(rconv::domain2.extents().flipped(), outputs, intermediates[1]);
  
  closeScalarDAG(intermediates, "rconv2");
  
}
