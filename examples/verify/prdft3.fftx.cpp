#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include "verify.h"

using namespace fftx;

int main(int argc, char* argv[])
{

  tracing=true;

  std::array<array_t<3,std::complex<double>>,1> intermediates {{verify::empty3}}; // in this case, empty
  array_t<3, double> inputs(verify::domain3);
  array_t<3, std::complex<double>> outputs(verify::fdomain3);


  setInputs(inputs);
  setOutputs(outputs);
  
  openScalarDAG();
  PRDFT(verify::domain3.extents().flipped(), outputs, inputs);

  closeScalarDAG(intermediates, "prdft3");
}
