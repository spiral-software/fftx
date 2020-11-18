// make

#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>

using namespace fftx;


int main(int argc, char* argv[])
{

  tracing=true;
  
  const int nx=32;
  const int ny=32;
  const int nz=32;

  if (argc < 2) {
	  printf("Usage: %s <prefix>\n<prefix> is a required argument\n", argv[0]);
	  exit (-1);
  }

  
  box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{nx,ny,nz}}));
  
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
