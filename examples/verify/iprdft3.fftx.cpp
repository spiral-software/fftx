#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include "verify.h"

using namespace fftx;

int main(int argc, char* argv[])
{

  tracing=true;

  std::array<array_t<3, std::complex<double>>, 1> intermediates {{verify::empty3}}; // in this case, empty
  array_t<3, std::complex<double>> inputs(verify::fdomain3);
  array_t<3, double> outputs(verify::domain3);

  setInputs(inputs);
  setOutputs(outputs);
  
  openScalarDAG();
  IPRDFT(verify::domain3.extents().flipped(), outputs, inputs);

  closeScalarDAG(intermediates, "iprdft3");
}
