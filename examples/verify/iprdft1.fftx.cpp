#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include "verify.h"

using namespace fftx;

int main(int argc, char* argv[])
{

  tracing=true;

  std::array<array_t<1, std::complex<double>>, 1> intermediates {{verify::empty1}}; // in this case, empty
  array_t<1, std::complex<double>> inputs(verify::fdomain1);
  array_t<1, double> outputs(verify::domain1);

  setInputs(inputs);
  setOutputs(outputs);
  
  openScalarDAG();
  IPRDFT(verify::domain1.extents().flipped(), outputs, inputs);

  closeScalarDAG(intermediates, "iprdft1");
}
