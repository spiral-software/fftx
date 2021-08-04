#include "fftx3.hpp"
#include "fftx_mpi.hpp"
#include <mpi.h>
#include <array>
#include <cstdio>
#include <cassert>

using namespace fftx;
using namespace std;

int main(int argc, char* argv[])
{

  const int nx=32;
  const int ny=32;
  const int nz=32;

  const int p = 2;

  box_t<2> grid2D(point_t<2>({{0, 0}}),
	          point_t<2>({{p-1, p-1}}));
    
  //canonical 2D global distribution
  vector<FFTX_Distribution> inDist({FFTX_NO_DIST, FFTX_GRID_X, FFTX_GRID_Y});

  //rotated 2D global distribution
  vector<FFTX_Distribution> outDist({FFTX_NO_DIST, FFTX_GRID_X, FFTX_GRID_Y});

  box_t<3> empty(point_t<3>({{1,1,1}}), point_t<3>({{0,0,0}}));
  box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{nx,ny,nz}}));

  array<    d_array_t<3,complex<double>>,  1> intermediates;
  d_array_t<3,complex<double>> inputs(domain, inDist);    // by default use canonical local layout
  d_array_t<3,complex<double>> outputs(domain, outDist);
  
  //Describe Global Computation
  openScalarDAG(grid2D);

  MDDFT(domain.extents(), 1, outputs, inputs);   //polymorphic by type

  closeScalarDAG(intermediates, "mddft3d", grid2D);       

  return 0;
}
