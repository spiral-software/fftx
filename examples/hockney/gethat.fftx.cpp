#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include "hockney.h"

using namespace fftx;


int main(int argc, char* argv[])
{

  tracing=true;

  std::array<array_t<3,std::complex<double>>, 0> intermediates; 
  array_t<3,double> input(hockney::rdomain);
  array_t<3,std::complex<double>> output(hockney::freq);

  setInputs(input);
  setOutputs(output);
  
  openScalarDAG();

  PRDFT(hockney::rdomain.extents().flipped(), output, input);
  
  closeScalarDAG(intermediates, "gethat");
  
}
