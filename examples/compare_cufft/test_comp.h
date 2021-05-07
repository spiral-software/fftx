using namespace fftx;

namespace test_comp
{
  const int nx = 32;
  const int ny = 40;
  const int nz = 48;

  // some arbitrary offsets
  const int offx = 3;
  const int offy = 5;
  const int offz = 11;

  box_t<3> domain(point_t<3>({{offx+1,  offy+1,  offz+1}}),
                  point_t<3>({{offx+nx, offy+ny, offz+nz}}));
}
