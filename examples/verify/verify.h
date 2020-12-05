using namespace fftx;

namespace verify
{
  const int nx=3;
  const int ny=4;
  const int nz=5;

  box_t<3> empty(point_t<3>({{1,1,1}}), point_t<3>({{0,0,0}}));
  box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{nx,ny,nz}}));
}
