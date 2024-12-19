#include "fftx.hpp"
#include "fftxutilities.hpp"
#include "fftxinterface.hpp"
#include "fftxhockneyconvObj.hpp"
#include <string>
#include <fstream>

#if defined FFTX_CUDA
#include "fftxcudabackend.hpp"
#elif defined FFTX_HIP
#include "fftxhipbackend.hpp"
#elif defined FFTX_SYCL
#include "fftxsyclbackend.hpp"
#else  
#include "fftxcpubackend.hpp"
#endif
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined (FFTX_SYCL)
#include "fftxdevice_macros.h"
#endif

#if defined(FFTX_HIP) || defined(FFTX_CUDA)
//  Build a random input buffer for Spiral and rocfft
//  host_X is the host buffer to setup -- it'll be copied to the device later
//  sizes is a vector with the X, Y, & Z dimensions

static void buildInputBuffer ( double *host_X, std::vector<int> sizes )
{
    srand(time(NULL));
    for ( int imm = 0; imm < sizes.at(0); imm++ ) {
        for ( int inn = 0; inn < sizes.at(1); inn++ ) {
            for ( int ikk = 0; ikk < sizes.at(2); ikk++ ) {
                int offset = (ikk + inn*sizes.at(2) + imm*sizes.at(1)*sizes.at(2));
                host_X[offset] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
            }
        }
    }
    return;
}

static void buildInputBuffer_complex ( double *host_X, std::vector<int> sizes )
{
    srand(time(NULL));
    for ( int imm = 0; imm < sizes.at(0); imm++ ) {
        for ( int inn = 0; inn < sizes.at(1); inn++ ) {
            for ( int ikk = 0; ikk < sizes.at(2); ikk++ ) {
                int offset = (ikk + inn * sizes.at(2) + imm * sizes.at(1) * sizes.at(2)) * 2;
                host_X[offset + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
                host_X[offset + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
            }
        }
    }
    return;
}
#else
static void buildInputBuffer ( double *host_X, std::vector<int> sizes )
{
    srand(time(NULL));
    for ( int imm = 0; imm < sizes.at(0); imm++ ) {
        for ( int inn = 0; inn < sizes.at(1); inn++ ) {
            for ( int ikk = 0; ikk < sizes.at(2); ikk++ ) {
                int offset = (ikk + inn*sizes.at(2) + imm*sizes.at(1)*sizes.at(2));
                host_X[offset] = 1;
            }
        }
    }
    return;
}

static void buildInputBuffer_complex( double *host_X, std::vector<int> sizes)
{
    for ( int imm = 0; imm < sizes.at(0); imm++ ) {
        for ( int inn = 0; inn < sizes.at(1); inn++ ) {
            for ( int ikk = 0; ikk < sizes.at(2); ikk++ ) {
                int offset = (ikk + inn * sizes.at(2) + imm * sizes.at(1) * sizes.at(2)) * 2;
                host_X[offset + 0] = 1;
                host_X[offset + 1] = 1;
            }
        }
    }
    return;
}
#endif
// Check that the buffer are identical (within roundoff)
// spiral_Y is the output buffer from the Spiral generated transform (result on GPU copied to host array spiral_Y)
// devfft_Y is the output buffer from the device equivalent transform (result on GPU copied to host array devfft_Y)
// arrsz is the size of each array
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
static void checkOutputBuffers ( FFTX_DEVICE_FFT_DOUBLECOMPLEX *spiral_Y, FFTX_DEVICE_FFT_DOUBLECOMPLEX *devfft_Y, long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int indx = 0; indx < arrsz; indx++ ) {
        FFTX_DEVICE_FFT_DOUBLECOMPLEX s = spiral_Y[indx];
        FFTX_DEVICE_FFT_DOUBLECOMPLEX c = devfft_Y[indx];

        bool elem_correct = ( (abs(s.x - c.x) < 1e-7) && (abs(s.y - c.y) < 1e-7) );
        maxdelta = maxdelta < (double)(abs(s.x -c.x)) ? (double)(abs(s.x -c.x)) : maxdelta ;
        maxdelta = maxdelta < (double)(abs(s.y -c.y)) ? (double)(abs(s.y -c.y)) : maxdelta ;
        correct &= elem_correct;
    }
    
    printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fflush ( stdout );

    return;
}
#endif

// static void checkOutputBuffers( FFTX_DEVICE_FFT_DOUBLEREAL *spiral_Y, FFTX_DEVICE_FFT_DOUBLEREAL *devfft_Y, long arrsz )
// {
//     bool correct = true;
//     double maxdelta = 0.0;

//     for ( int indx = 0; indx < arrsz; indx++ ) {
//         FFTX_DEVICE_FFT_DOUBLEREAL s = spiral_Y[indx];
//         FFTX_DEVICE_FFT_DOUBLEREAL c = devfft_Y[indx];

//         double deltar = abs ( s - c );
//         bool   elem_correct = ( deltar < 1e-7 );
//         maxdelta = maxdelta < deltar ? deltar : maxdelta ;
//         correct &= elem_correct;
//     }
    
//     printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
//     fflush ( stdout );

//     return;
// }

int main(int argc, char* argv[])
{
  int iterations = 2;
  int mm = 33, nn = 130, kk = 96; // default cube dimensions
  char *prog = argv[0];
  int baz = 0;

  while ( argc > 1 && argv[1][0] == '-' ) {
      switch ( argv[1][1] ) {
      case 'i':
          if(strlen(argv[1]) > 2) {
            baz = 2;
          } else {
            baz = 0;
            argv++, argc--;
          }
          iterations = atoi (& argv[1][baz] );
          break;
      case 's':
          if(strlen(argv[1]) > 2) {
            baz = 2;
          } else {
            baz = 0;
            argv++, argc--;
          }
          mm = atoi (& argv[1][baz] );
          while ( argv[1][baz] != 'x' ) baz++;
          baz++ ;
          nn = atoi ( & argv[1][baz] );
          while ( argv[1][baz] != 'x' ) baz++;
          baz++ ;
          kk = atoi ( & argv[1][baz] );
          break;
      case 'h':
          printf ( "Usage: %s: [ -i iterations ] [ -s <input cube length>x<padded cube length>x<output cube length> ] [ -h (print help message) ]\n", argv[0] );
          exit (0);
      default:
          printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
      }
      argv++, argc--;
  }

  fftx::OutStream() << "cube lengths:"
                    << " input " << mm
                    << ", padded " << nn
                    << ", output " << kk
                    << std::endl;
  std::vector<int> sizes{mm, nn, kk};
  fftx::box_t<3> domain ( fftx::point_t<3> ( { { 1, 1, 1 } } ),
                          fftx::point_t<3> ( { { mm, mm, mm } } ));
  fftx::box_t<3> padd ( fftx::point_t<3> ( { { 1, 1, 1 } } ),
                          fftx::point_t<3> ( { { nn, nn, nn } } ));                        
  fftx::box_t<3> outputd ( fftx::point_t<3> ( { { 1, 1, 1 } } ),
                          fftx::point_t<3> ( { { kk, kk, kk } } ));

  fftx::array_t<3,double> inputHost(domain);
  fftx::array_t<3,double> outputHost(outputd);
  fftx::array_t<3,std::complex<double>> symbolHost(padd);

  double *dX, *dY;
  std::complex<double> *dsym;

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  FFTX_DEVICE_MALLOC(&dX, inputHost.m_domain.size() * sizeof(double));

  FFTX_DEVICE_MALLOC(&dY, outputHost.m_domain.size() * sizeof(double));

  FFTX_DEVICE_MALLOC(&dsym,  symbolHost.m_domain.size() * sizeof(std::complex<double>));

#elif defined(FFTX_SYCL)
  sycl::buffer<double> buf_Y(outputHost.m_data.local(), outputHost.m_domain.size());
  sycl::buffer<double> buf_X(inputHost.m_data.local(), inputHost.m_domain.size());
  sycl::buffer<std::complex<double>> buf_sym(symbolHost.m_data.local(), symbolHost.m_domain.size());
#else
  dX = (double *) inputHost.m_data.local();
  dY = (double *) outputHost.m_data.local();
  dsym = (std::complex<double> *) symbolHost.m_data.local();
#endif
  if ( FFTX_DEBUGOUT ) fftx::OutStream() << "memory allocated" << std::endl;  

  float *hockneyconv_gpu = new float[iterations];

#if defined FFTX_CUDA
  std::vector<void*> args{&dY,&dX,&dsym};
  std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
  std::string devfft  = "cufft";
#elif defined FFTX_HIP
  std::vector<void*> args{dY,dX,dsym};
  std::string descrip = "AMD GPU";                //  "CPU and GPU";
  std::string devfft  = "rocfft";
#elif defined FFTX_SYCL
  std::vector<void*> args{(void*)&(buf_Y),(void*)&(buf_X),(void*)&(buf_sym)};
  std::string descrip = "Intel GPU";                //  "CPU and GPU";
  std::string devfft  = "mklfft";
#else
  std::vector<void*> args{(void*)dY,(void*)dX,(void*)dsym};
  std::string descrip = "CPU";                //  "CPU";
  std::string devfft = "fftw";
#endif

HOCKNEYCONVProblem hcp(args, sizes, "hockneyconv");

double *hostinp = (double *) inputHost.m_data.local();
std::vector<int> hostrange{mm,mm,mm};
std::complex<double> *symbp = (std::complex<double>*) symbolHost.m_data.local();
std::vector<int> symrange{nn,nn,nn};
for(int itn = 0; itn < iterations; itn++) 
{
  
  buildInputBuffer(hostinp, hostrange);
  buildInputBuffer_complex((double*)symbp, symrange);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  FFTX_DEVICE_MEM_COPY(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(double),
                  FFTX_MEM_COPY_HOST_TO_DEVICE);
  FFTX_DEVICE_MEM_COPY(dsym, symbolHost.m_data.local(),  symbolHost.m_domain.size() * sizeof(std::complex<double>),
                  FFTX_MEM_COPY_HOST_TO_DEVICE);
#endif
  if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied inputs\n";

  hcp.transform();
  hockneyconv_gpu[itn] = hcp.getTime();
#if defined(FFTX_SYCL)		
  {
    // fftx::OutStream() << "MKLFFT comparison not implemented printing first output element" << std::endl;
    // sycl::host_accessor h_acc(dY);
    // fftx::OutStream() << h_acc[0] << std::endl;
  }
#endif
}

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    printf ( "Times in milliseconds for %s on HOCKNEY CONVOLUTION for %d trials of size %d %d %d:\nTrial #\tSpiral\trocfft\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2) );        //  , devfft.c_str() );
    for (int itn = 0; itn < iterations; itn++) {
        printf ( "%d\t%.7e\n", itn, hockneyconv_gpu[itn]);
    }
#else
     printf ( "Times in milliseconds for %s on HOCKNEY CONVOLUTION for %d trials of size %d %d %d\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
    for (int itn = 0; itn < iterations; itn++) {
        printf ( "%d\t%.7e\n", itn, hockneyconv_gpu[itn]);
    }
#endif

}
