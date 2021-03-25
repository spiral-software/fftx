using namespace fftx;

namespace test_plan
{
  const int nx = 4;
  const int ny = 4;
  const int nz = 4;

  box_t<3> domain(point_t<3>({{1, 1, 1}}),
                  point_t<3>({{nx, ny, nz}}));
}
