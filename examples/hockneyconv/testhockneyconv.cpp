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

static void buildInputBuffer ( double *host_X, fftx::point_t<3> extents )
{
    srand(time(NULL));
    for ( int imm = 0; imm < extents[0]; imm++ ) {
        for ( int inn = 0; inn < extents[1]; inn++ ) {
            for ( int ikk = 0; ikk < extents[2]; ikk++ ) {
                int offset = (ikk + inn*extents[2] + imm*extents[1]*extents[2]);
                host_X[offset] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
            }
        }
    }
    return;
}

static void buildInputBuffer_complex ( double *host_X, fftx::point_t<3> extents )
{
    srand(time(NULL));
    for ( int imm = 0; imm < extents[0]; imm++ ) {
        for ( int inn = 0; inn < extents[1]; inn++ ) {
            for ( int ikk = 0; ikk < extents[2]; ikk++ ) {
                int offset = (ikk + inn * extents[2] + imm * extents[1] * extents[2]) * 2;
                host_X[offset + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
                host_X[offset + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
            }
        }
    }
    return;
}
#else
static void buildInputBuffer ( double *host_X, fftx::point_t<3> extents )
{
    srand(time(NULL));
    for ( int imm = 0; imm < extents[0]; imm++ ) {
        for ( int inn = 0; inn < extents[1]; inn++ ) {
            for ( int ikk = 0; ikk < extents[2]; ikk++ ) {
                int offset = (ikk + inn*extents[2] + imm*extents[1]*extents[2]);
                host_X[offset] = 1;
            }
        }
    }
    return;
}

static void buildInputBuffer_complex( double *host_X, fftx::point_t<3> extents )
{
    for ( int imm = 0; imm < extents[0]; imm++ ) {
        for ( int inn = 0; inn < extents[1]; inn++ ) {
            for ( int ikk = 0; ikk < extents[2]; ikk++ ) {
                int offset = (ikk + inn * extents[2] + imm * extents[1] * extents[2]) * 2;
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
static bool checkOutputBuffers ( FFTX_DEVICE_FFT_DOUBLECOMPLEX *spiral_Y, FFTX_DEVICE_FFT_DOUBLECOMPLEX *devfft_Y, long arrsz )
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
    
    // printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fftx::OutStream() << "Correct: " << (correct ? "True" : "False") << "\t"
                      << "Max delta = " << std::scientific << maxdelta
                      << std::endl;
    // fflush ( stdout );
    std::flush(fftx::OutStream());
    

    return correct;
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
  int mm = 32, nn = 32, kk = 128; // default cube dimensions
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
          // printf ( "Usage: %s: [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]\n", argv[0] );
          fftx::OutStream() << "Usage: " << argv[0]
                            << ": [ -i iterations ] [ -s MMxNNxKK ] [ -h (pri\
nt help message) ]"
                            << std::endl;
          exit (0);
      default:
          // printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
          fftx::OutStream() << prog << ": unknown argument: "
                            << argv[1] << " ... ignored" << std::endl;
      }
      argv++, argc--;
  }
  fftx::OutStream() << mm << " " << nn << " " << kk << std::endl;
  int status = 0;
  std::vector<int> sizes{mm, nn, kk};
  fftx::box_t<3> inputd ( fftx::point_t<3> ( { { 1, 1, 1 } } ),
                          fftx::point_t<3> ( { { mm, nn, kk } } ));
  fftx::box_t<3> padd ( fftx::point_t<3> ( { { 1, 1, 1 } } ),
                        fftx::point_t<3> ( { { 2*mm, 2*nn, 2*kk } } ));
  fftx::box_t<3> outputd ( fftx::point_t<3> ( { { 1, 1, 1 } } ),
                          fftx::point_t<3> ( { { mm, nn, kk } } ));

  fftx::array_t<3,double> inputHost(inputd);
  fftx::array_t<3,double> outputHost(outputd);
  fftx::array_t<3,std::complex<double>> symbolHost(padd);

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  FFTX_DEVICE_PTR inputTfmPtr = fftxDeviceMallocForHostArray(inputHost);
  FFTX_DEVICE_PTR outputTfmPtr = fftxDeviceMallocForHostArray(outputHost);
  FFTX_DEVICE_PTR symbolTfmPtr = fftxDeviceMallocForHostArray(symbolHost);
#elif defined(FFTX_SYCL)
  size_t inputpts = inputHost.m_domain.size();
  size_t outputpts = outputHost.m_domain.size();
  size_t symbolpts = symbolHost.m_domain.size();
  sycl::buffer<double> inputTfmPtr(inputHost.m_data.local(), inputpts);
  sycl::buffer<double> outputTfmPtr(outputHost.m_data.local(), outputpts);
  sycl::buffer<std::complex<double>> symbolTfmPtr(symbolHost.m_data.local(), symbolpts);
#else
  double* inputTfmPtr = (double *) inputHost.m_data.local();
  double* outputTfmPtr = (double *) outputHost.m_data.local();
  std::complex<double>* symbolTfmPtr = (std::complex<double> *) symbolHost.m_data.local();
#endif
  if ( FFTX_DEBUGOUT ) fftx::OutStream() << "memory allocated" << std::endl;  

  float *hockneyconv_gpu = new float[iterations];

#if defined FFTX_CUDA
  std::vector<void*> args{&outputTfmPtr,&inputTfmPtr,&symbolTfmPtr};
  std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
  std::string devfft  = "cufft";
#elif defined FFTX_HIP
  std::vector<void*> args{outputTfmPtr,inputTfmPtr,symbolTfmPtr};
  std::string descrip = "AMD GPU";                //  "CPU and GPU";
  std::string devfft  = "rocfft";
#elif defined FFTX_SYCL
  std::vector<void*> args{(void*)&(outputTfmPtr),(void*)&(inputTfmPtr),(void*)&(symbolTfmPtr)};
  std::string descrip = "Intel GPU";                //  "CPU and GPU";
  std::string devfft  = "mklfft";
#else
  std::vector<void*> args{(void*)outputTfmPtr,(void*)inputTfmPtr,(void*)symbolTfmPtr};
  std::string descrip = "CPU";                //  "CPU";
  std::string devfft = "fftw";
#endif

  HOCKNEYCONVProblem hcp(args, sizes, "hockneyconv");

  double *hostinp = (double *) inputHost.m_data.local();
  fftx::point_t<3> inputextents = inputHost.m_domain.extents();
  fftx::point_t<3> symbolextents = symbolHost.m_domain.extents();
  std::complex<double> *symbp = (std::complex<double>*) symbolHost.m_data.local();
  for (int itn = 0; itn < iterations; itn++) 
    {
      buildInputBuffer(hostinp, inputextents);
      buildInputBuffer_complex((double*)symbp, symbolextents);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      fftxCopyHostArrayToDevice(inputTfmPtr, inputHost);
      fftxCopyHostArrayToDevice(symbolTfmPtr, symbolHost);
#endif
      if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied inputs\n";

      hcp.transform();
      hockneyconv_gpu[itn] = hcp.getTime();
#if defined(FFTX_SYCL)		
      {
        // fftx::OutStream() << "MKLFFT comparison not implemented printing first output element" << std::endl;
        // sycl::host_accessor h_acc(outputTfmPtr);
        // fftx::OutStream() << h_acc[0] << std::endl;
      }
#endif
    }

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  fftxDeviceFree(inputTfmPtr);
  fftxDeviceFree(outputTfmPtr);
  fftxDeviceFree(symbolTfmPtr);

  // printf ( "Times in milliseconds for %s on HOCKNEY CONVOLUTION for %d trials of size %d %d %d:\nTrial #\tSpiral\trocfft\n",
  // descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2) );        //  , devfft.c_str() );
#endif
  // printf ( "Times in milliseconds for %s on HOCKNEY CONVOLUTION for %d trials of size %d %d %d\n",
  // descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
  fftx::OutStream() << "Times in milliseconds for " << descrip
                    << " on HOCKNEY CONVOLUTION for "
                    << iterations << " trials of size "
                    << sizes.at(0) << " "
                    << sizes.at(1) << " "
                    << sizes.at(2) << ":" << std::endl;
  fftx::OutStream() << "Trial#    Spiral" << std::endl;
  for (int itn = 0; itn < iterations; itn++)
    {
      // printf ( "%d\t%.7e\n", itn, hockneyconv_gpu[itn]);
      fftx::OutStream() << std::setw(4) << (itn+1)
                        << std::scientific << std::setw(17)
                        << hockneyconv_gpu[itn] << std::endl;
    }

    delete[] hockneyconv_gpu;

    fftx::OutStream() << prog << ": All done, exiting" << std::endl;
    std::flush(fftx::OutStream());

    return status;
}
