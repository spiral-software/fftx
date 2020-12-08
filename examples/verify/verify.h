using namespace fftx;

template<int DIM> point_t<DIM> FLIP(point_t<DIM> p)
{
  point_t<DIM> pflip;
  for (int d = 0; d < DIM; d++)
    {
      pflip[d] = p[DIM-1 - d];
    }
  return pflip;
}

namespace verify
{
  const int nx = 24;
  const int ny = 32;
  const int nz = 40;

  box_t<2> empty2(point_t<2>({{1, 1}}),
                  point_t<2>({{0, 0}}));
  box_t<2> domain2(point_t<2>({{1, 1}}),
                   point_t<2>({{nx, ny}}));

  box_t<3> empty3(point_t<3>({{1, 1, 1}}),
                  point_t<3>({{0, 0, 0}}));
  box_t<3> domain3(point_t<3>({{1, 1, 1}}),
                   point_t<3>({{nx, ny, nz}}));
}
