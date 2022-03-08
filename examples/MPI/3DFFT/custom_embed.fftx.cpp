#include "fftx3.hpp"
#include "fftx_mpi.hpp"
#include <mpi.h>
#include <array>
#include <cstdio>
#include <cassert>

using namespace fftx;
using namespace std;

//user defined function to describe custom function

//number of batches of DFT for a given node in a particular dimension
int custom_dist_batch(int node, int dims)
{

  //hard coded to 32x32x32 3dfft on 2x2 grid
  return 32*32/4; 
}

//given (x, y, z) return the start of the local buffer for the DFT containing (x, y, z)
//assumes that at least one of the dimensions is not distributed.
//this is supposed to the global indexing but returns local starting address
void* custom_dist_data(int node, int x, int y, int z)
{
  return NULL;
}

//returns tuples of indices that have to be computed
//indices are global indexing
void* custom_dist_indices(int node)
{
  return NULL;
}


int main(int argc, char* argv[])
{

  const int nx = 64;
  const int ny = 64;
  const int nz = 64;
  
  const int sm_nx=32;
  const int sm_ny=32;
  const int sm_nz=32;

  const int off_x = (nx - sm_nx)/2;
  const int off_y = (ny - sm_ny)/2;
  const int off_z = (nz - sm_nz)/2;
  
  const int p = 2;

  box_t<2> grid2D(point_t<2>({{0, 0}}),
	          point_t<2>({{p-1, p-1}}));
    

  //rotated 2D global distribution
  vector<FFTX_Distribution> outDist({FFTX_NO_DIST, FFTX_GRID_X, FFTX_GRID_Y});

  box_t<3> empty(point_t<3>({{1,1,1}}), point_t<3>({{0,0,0}}));
  box_t<3> sm_domain(point_t<3>({{off_x,off_y,off_z}}),
		     point_t<3>({{off_x + sm_nx - 1,
			          off_y + sm_ny - 1,
			          off_z + sm_nz - 1}}));
  box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{nx,ny,nz}}));

  array<    d_array_t<3,complex<double>>,  1> intermediates;

  //Register custom distribution where the 2 fastest dims (assumed x, y ) are distributed.
  custom_distribution<3> cust_dist(vector<FFTX_Distribution>({FFTX_CUSTOM, FFTX_CUSTOM, FFTX_NO_DIST}),
				   custom_dist_batch,
				   custom_dist_indices,
				   custom_dist_data
				   );  

  d_array_t<3, complex<double>> inputs(sm_domain, cust_dist);    // by default use canonical local layout

  //  cout<<cust_dist.m_dist[0]<<endl;
  
  d_array_t<3, complex<double>> embedbox(domain, cust_dist);
  d_array_t<3, complex<double>> outputs(domain, outDist);

  //Describe Global Computation
  openScalarDAG(grid2D);

  zeroEmbedBox(embedbox, inputs); 
  MDDFT(domain.extents(), 1, outputs, embedbox);   //polymorphic by type  
  closeScalarDAG(intermediates, "mddft3d", grid2D);       

  cout<<"-----"<<endl;

  /*
  cout<<off_x<<endl;
  cout<<sm_nx<<endl;
  cout<<off_x + sm_nx<<endl;
  cout<<inputs.m_domain.extents()[0]<<endl;
  cout<<embedbox.is_embedding<<endl;
  */
  return 0;
}
