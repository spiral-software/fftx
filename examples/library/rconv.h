using namespace fftx;

namespace rconv_dims
{
  // some arbitrary offsets
  const int offx = 3;
  const int offy = 5;
  const int offz = 11;

  #ifndef fftx_nx
  #define fftx_nx 32
  #endif

  #ifndef fftx_ny
  #define fftx_ny 32
  #endif

  #ifndef fftx_nz
  #define fftx_nz 32
  #endif
  
  const int fx = fftx_nx/2 + 1;
  const int fy = fftx_ny;
  const int fz = fftx_nz;

  box_t<3> empty3(point_t<3>({{1, 1, 1}}),
                  point_t<3>({{0, 0, 0}}));
  box_t<3> domain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                   point_t<3>({{offx+fftx_nx, offy+fftx_ny, offz+fftx_nz}}));
  box_t<3> fdomain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                    point_t<3>({{offx+fx, offy+fy, offz+fz}}));
}
