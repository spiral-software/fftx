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
  
  std::array<array_t<3,std::complex<double>>,2> intermediates {rconv::fdomain3, rconv::fdomain3};
  array_t<3,double> inputs(rconv::domain3);
  array_t<3,double> outputs(rconv::domain3);
  array_t<3,double> symbol(rconv::fdomain3);


  setInputs(inputs);
  setOutputs(outputs);
 
  openScalarDAG();

  PRDFT(rconv::domain3.extents().flipped(), intermediates[0], inputs);
  kernel(symbol, intermediates[1], intermediates[0]);
  IPRDFT(rconv::domain3.extents().flipped(), outputs, intermediates[1]);
  
  closeScalarDAG(intermediates, "rconv3");
  
}
