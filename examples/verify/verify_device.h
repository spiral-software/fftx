using namespace fftx;

namespace verify
{
  const int offx = 3;
  const int offy = 5;
  const int offz = 11;
  
  #ifndef fftx_nx
  #define fftx_nx 80
  #endif

  #ifndef fftx_ny
  #define fftx_ny 80
  #endif

  #ifndef fftx_nz
  #define fftx_nz 374
  #endif

#if FFTX_COMPLEX_TRUNC_LAST
  const int fx = fftx_nx;
  const int fy = fftx_ny;
  const int fz = fftx_nz/2 + 1;
#else
  const int fx = fftx_nx/2 + 1;
  const int fy = fftx_ny;
  const int fz = fftx_nz;
#endif

  box_t<1> empty1(point_t<1>({{1}}),
                  point_t<1>({{0}}));
  box_t<1> domain1(point_t<1>({{offx+1}}),
                   point_t<1>({{offx+fftx_nx}}));
  box_t<1> fdomain1(point_t<1>({{offx+1}}),
                    point_t<1>({{offx+fx}}));

  box_t<2> empty2(point_t<2>({{1, 1}}),
                  point_t<2>({{0, 0}}));
  box_t<2> domain2(point_t<2>({{offx+1, offy+1}}),
                   point_t<2>({{offx+fftx_nx, offy+fftx_ny}}));
  box_t<2> fdomain2(point_t<2>({{offx+1, offy+1}}),
                    point_t<2>({{offx+fx, offy+fy}}));

  box_t<3> empty3(point_t<3>({{1, 1, 1}}),
                  point_t<3>({{0, 0, 0}}));
  box_t<3> domain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                   point_t<3>({{offx+fftx_nx, offy+fftx_ny, offz+fftx_nz}}));
  box_t<3> fdomain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                    point_t<3>({{offx+fx, offy+fy, offz+fz}}));
}
