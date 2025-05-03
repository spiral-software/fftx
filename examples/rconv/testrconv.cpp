#include <cmath> // Without this, abs is the wrong function!
#include <random>

#include "fftxRealConvolution.hpp"

// using namespace fftx;
// fftx::handle_t (a_transform)
template<int DIM>
int rconvDimension(std::vector<int> sizes,
                   fftx::box_t<DIM> a_domain,
                   fftx::box_t<DIM> a_fdomain,
                   int a_rounds,
                   int a_verbosity)
{
  fftx::OutStream() << "***** test " << DIM << "D real convolution on "
                    << a_domain << std::endl;
  RealConvolution<DIM> fun(sizes, a_domain, a_fdomain);
  return fun.testAll(a_rounds, a_verbosity);
}


int main(int argc, char* argv[])
{
  int mm = 24, nn = 32, kk = 40; // default cube dimensions
  int offx = 3, offy = 5, offz = 11; // offsets
  char *prog = argv[0];
  int baz = 0;
  int verbosity = 0;
  int rounds = 2;
  while ( argc > 1 && argv[1][0] == '-' ) {
      switch ( argv[1][1] ) {
      case 'i':
          if(strlen(argv[1]) > 2) {
            baz = 2;
          } else {
            baz = 0;
            argv++, argc--;
          }
          rounds = atoi ( & argv[1][baz] );
          break;
      case 's':
          if(strlen(argv[1]) > 2) {
            baz = 2;
          } else {
            baz = 0;
            argv++, argc--;
          }
          mm = atoi ( & argv[1][baz] );;
          while ( argv[1][baz] != 'x' ) baz++;
          baz++ ;
          nn = atoi ( & argv[1][baz] );
          while ( argv[1][baz] != 'x' ) baz++;
          baz++ ;
          kk = atoi ( & argv[1][baz] );
          break;
      case 'v':
          if(strlen(argv[1]) > 2) {
            baz = 2;
          } else {
            baz = 0;
            argv++, argc--;
          }
          verbosity = atoi ( & argv[1][baz] );
          break;
      case 'h':
          fftx::OutStream() << "Usage: " << argv[0]
                            << ": [ -i rounds ] [-v verbosity: 0 for summary, 1 for categories, 2 for subtests, 3 for all iterations] [ -s MMxNNxKK ] [ -h (print help message) ]"
                            << std::endl;
          exit (0);
      default:
        fftx::OutStream() << prog << ": unknown argument: "
                          << argv[1] << " ... ignored" << std::endl;
      }
      argv++, argc--;
  }

  // printf("Running size %dx%dx%d with verbosity %d, random %d rounds\n", mm, nn, kk, verbosity, rounds);
  fftx::OutStream() << "Running size " << mm << "x" << nn << "x" << kk
                    << " with verbosity " << verbosity
                    << ", random " << rounds << " rounds" << std::endl;
  std::vector<int> sizes{mm, nn, kk};

  /*
    Set up random number generator.
  */
  std::random_device rd;
  generator = std::mt19937(rd());
  unifRealDist = std::uniform_real_distribution<double>(-0.5, 0.5);

  /*
    2-dimensional tests.
  */
  //    rconv2::init();
  //    rconvDimension(rconv2::transform, domain2, fdomain2,
  //                   rounds, verbosity);
  //    rconv2::destroy();
  
  /*
    3-dimensional tests.
  */
  fftx::point_t<3> fullExtents({{mm, nn, kk}});
  fftx::point_t<3> truncExtents = truncatedComplexDimensions(fullExtents);
  fftx::point_t<3> offsets({{offx, offy, offz}});
  
  fftx::box_t<3> domain3 = domainFromSize(fullExtents, offsets);
  fftx::box_t<3> fdomain3 = domainFromSize(truncExtents, offsets);

  int status = rconvDimension(sizes, domain3, fdomain3, rounds, verbosity);
  // rconv3::destroy();
  
  fftx::OutStream() << prog << ": All done, exiting with status "
                    << status << std::endl;
  std::flush(fftx::OutStream());

  return status;
}
