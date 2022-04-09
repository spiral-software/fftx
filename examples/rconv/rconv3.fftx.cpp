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
  
  std::array<array_t<3,std::complex<double>>,2> intermediates {rconv_once::fdomain3, rconv_once::fdomain3};
  array_t<3,double> inputs(rconv_once::domain3);
  array_t<3,double> outputs(rconv_once::domain3);
  array_t<3,double> symbol(rconv_once::fdomain3);


  setInputs(inputs);
  setOutputs(outputs);
 
  openScalarDAG();

  PRDFT(rconv_once::domain3.extents(), intermediates[0], inputs);
  kernel(symbol, intermediates[1], intermediates[0]);
  IPRDFT(rconv_once::domain3.extents(), outputs, intermediates[1]);
  
  closeScalarDAG(intermediates, "rconv3");
  
}
