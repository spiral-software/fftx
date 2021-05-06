using namespace fftx;

namespace rconv
{
  const int offx = 3;
  const int offy = 5;
  const int offz = 11;

  //  The following constants define the cube/problem size.  The values here are defaults
  //  and will be used if not over-ridden during the build process
  //  const int nx = 24;
  //  const int ny = 32;
  //  const int nz = 40;

  #ifndef nx
  #define nx 24
  #endif

  #ifndef ny
  #define ny 32
  #endif

  #ifndef nz
  #define nz 40
  #endif
  
  const int fx = nx/2 + 1;
  const int fy = ny;
  const int fz = nz;

  box_t<1> empty1(point_t<1>({{1}}),
                  point_t<1>({{0}}));
  box_t<1> domain1(point_t<1>({{offx+1}}),
                   point_t<1>({{offx+nx}}));
  box_t<1> fdomain1(point_t<1>({{offx+1}}),
                    point_t<1>({{offx+fx}}));

  box_t<2> empty2(point_t<2>({{1, 1}}),
                  point_t<2>({{0, 0}}));
  box_t<2> domain2(point_t<2>({{offx+1, offy+1}}),
                   point_t<2>({{offx+nx, offy+ny}}));
  box_t<2> fdomain2(point_t<2>({{offx+1, offy+1}}),
                    point_t<2>({{offx+fx, offy+fy}}));

  box_t<3> empty3(point_t<3>({{1, 1, 1}}),
                  point_t<3>({{0, 0, 0}}));
  box_t<3> domain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                   point_t<3>({{offx+nx, offy+ny, offz+nz}}));
  box_t<3> fdomain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                    point_t<3>({{offx+fx, offy+fy, offz+fz}}));
}
