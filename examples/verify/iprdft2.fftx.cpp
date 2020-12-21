#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include "verify.h"

using namespace fftx;

int main(int argc, char* argv[])
{

  tracing=true;

  std::array<array_t<2, std::complex<double>>, 1> intermediates {{verify::empty2}}; // in this case, empty
  array_t<2, std::complex<double>> inputs(verify::fdomain2);
  array_t<2, double> outputs(verify::domain2);

  setInputs(inputs);
  setOutputs(outputs);
  
  openScalarDAG();
  IPRDFT(verify::domain2.extents().flipped(), outputs, inputs);

  closeScalarDAG(intermediates, "iprdft2");
}
