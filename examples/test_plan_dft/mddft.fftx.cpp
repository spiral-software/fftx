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
  
  box_t<3> empty(point_t<3>({{1,1,1}}), point_t<3>({{0,0,0}}));
  box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{test_plan::nx,test_plan::ny,test_plan::nz}}));
  
  std::array<array_t<3,std::complex<double>>,1> intermediates {{empty}}; // in this case, empty
  array_t<3,std::complex<double>> inputs(domain);
  array_t<3,std::complex<double>> outputs(domain);


  setInputs(inputs);
  setOutputs(outputs);
  
  openScalarDAG();
  MDDFT(domain.extents(), 1, outputs, inputs);

  closeScalarDAG(intermediates, "mddft");
}
