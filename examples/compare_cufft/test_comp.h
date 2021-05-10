using namespace fftx;

namespace test_comp
{
  #ifndef fftx_nx
  #define fftx_nx 32
  #endif

  #ifndef fftx_ny
  #define fftx_ny 40
  #endif

  #ifndef fftx_nz
  #define fftx_nz 48
  #endif

  // some arbitrary offsets
  const int offx = 3;
  const int offy = 5;
  const int offz = 11;

  box_t<3> domain(point_t<3>({{offx+1,  offy+1,  offz+1}}),
                  point_t<3>({{offx+fftx_nx, offy+fftx_ny, offz+fftx_nz}}));
}
