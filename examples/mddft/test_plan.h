using namespace fftx;

namespace test_plan
{
  #ifndef fftx_nx
  #define fftx_nx 32
  #endif

  #ifndef fftx_ny
  #define fftx_ny 32
  #endif

  #ifndef fftx_nz
  #define fftx_nz 32
  #endif

  box_t<3> domain(point_t<3>({{1, 1, 1}}),
                  point_t<3>({{fftx_nx, fftx_ny, fftx_nz}}));
}
