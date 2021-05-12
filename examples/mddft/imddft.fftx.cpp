// make

#include "fftx3.hpp"
#include "test_plan.h"
#include <array>
#include <cstdio>
#include <cassert>

using namespace fftx;


int main(int argc, char* argv[])
{

  tracing=true;
  
  box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{fftx_nx, fftx_ny, fftx_nz}}));
  
  std::array<array_t<3,std::complex<double>>,1> intermediates {domain};
  array_t<3,std::complex<double>> inputs(domain);
  array_t<3,std::complex<double>> outputs(domain);


  setInputs(inputs);
  setOutputs(outputs);
  
  openScalarDAG();
  
  MDDFT(domain.extents(), 1, intermediates[0], inputs);
  IMDDFT(domain.extents(), 1, outputs, intermediates[0]);

  closeScalarDAG(intermediates, "imddft");
  
}
