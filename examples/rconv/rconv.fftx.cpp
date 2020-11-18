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

  const int fx = (nx/2+1);
  const int fy = ny;
  const int fz = nz;
  
  box_t<3> rdomain(point_t<3>({{1,1,1}}), point_t<3>({{nx,ny,nz}}));
  box_t<3> freq(point_t<3>({{1,1,1}}), point_t<3>({{fx, fy, fz}}));
  
  std::array<array_t<3,std::complex<double>>,2> intermediates {freq, freq};
  array_t<3,double> inputs(rdomain);
  array_t<3,double> outputs(rdomain);
  array_t<3,double> symbol(freq);


  setInputs(inputs);
  setOutputs(outputs);
 
  
  openScalarDAG();

  PRDFT(rdomain.extents(), intermediates[0], inputs);
  kernel(symbol, intermediates[1], intermediates[0]);
  IPRDFT(rdomain.extents(), outputs, intermediates[1]);
  
  closeScalarDAG(intermediates, "rconv");
  
}
