#include "fftx.hpp"
#include "fftxinterface.hpp"
#include "fftxbatch1dprdftObj.hpp"
#include <math.h>  
#include "fftxibatch1dprdftObj.hpp"
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
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#include "fftxdevice_macros.h"
#endif

#define FFTX_CH_CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
      fftx::ErrStream() << "Cuda error in file '" << __FILE__                \
                        << "' in line " << __LINE__                          \
                        << " : " << cudaGetErrorString( err)                 \
                        << "." << std::endl;                                 \
        exit(EXIT_FAILURE);                                                  \
    } \
}

#define FFTX_CUDA_SAFE_CALL(call) FFTX_CH_CUDA_SAFE_CALL(call)

//  Build a random input buffer for Spiral and rocfft/cufft
//  host_X is the host buffer to setup -- it'll be copied to the device later
//  sizes is a vector with the X, Y, & Z dimensions

static void buildInputBuffer ( double *host_X, std::vector<int> sizes )
{
    for ( int imm = 0; imm < sizes.at(0)*sizes.at(1); imm++ ) {
        host_X[imm] = ((double) rand()) / (double) (RAND_MAX/2);
        // host_X[imm] = 1.0;
    }
    return;
}

// Check that the buffer are identical (within roundoff)
// spiral_Y is the output buffer from the Spiral generated transform (result on GPU copied to host array spiral_Y)
// devfft_Y is the output buffer from the device equivalent transform (result on GPU copied to host array devfft_Y)
// arrsz is the size of each array

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
static void checkOutputBuffers_fwd ( FFTX_DEVICE_FFT_DOUBLECOMPLEX *spiral_Y, FFTX_DEVICE_FFT_DOUBLECOMPLEX *devfft_Y, long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int indx = 0; indx < arrsz; indx++ ) {
        FFTX_DEVICE_FFT_DOUBLECOMPLEX s = spiral_Y[indx];
        FFTX_DEVICE_FFT_DOUBLECOMPLEX c = devfft_Y[indx];
        // fftx::OutStream() << s.x << ":" << s.y << " ";
        // fftx::OutStream() << c.x << ":" << c.y << std::endl;


        bool elem_correct = ( (abs(s.x - c.x) < 1e-7) && (abs(s.y - c.y) < 1e-7) );
        maxdelta = maxdelta < (double)(abs(s.x -c.x)) ? (double)(abs(s.x -c.x)) : maxdelta ;
        maxdelta = maxdelta < (double)(abs(s.y -c.y)) ? (double)(abs(s.y -c.y)) : maxdelta ;
        correct &= elem_correct;
    }
    
    fftx::OutStream() << "Correct: " << (correct ? "True" : "False") << "\t"
                      << "Max delta = "
                      << std::scientific << std::uppercase << maxdelta
                      << std::endl;
    std::flush(fftx::OutStream());

    return;
}

static void checkOutputBuffers_inv ( FFTX_DEVICE_FFT_DOUBLEREAL *spiral_Y, FFTX_DEVICE_FFT_DOUBLEREAL *devfft_Y, long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int indx = 0; indx < arrsz; indx++ ) {
        FFTX_DEVICE_FFT_DOUBLEREAL s = spiral_Y[indx];
        FFTX_DEVICE_FFT_DOUBLEREAL c = devfft_Y[indx];
        // fftx::OutStream() << s << " " << c << std::endl;

        double deltar = abs ( s - c );
        bool   elem_correct = ( deltar < 1e-7 );
        maxdelta = maxdelta < deltar ? deltar : maxdelta ;
        correct &= elem_correct;
    }
    
    fftx::OutStream() << "Correct: " << (correct ? "True" : "False") << "\t"
                      << "Max delta = "
                      << std::scientific << std::uppercase << maxdelta
                      << std::endl;
    std::flush(fftx::OutStream());

    return;
}


#endif

int main(int argc, char* argv[])
{
    int iterations = 2;
    int N = 64; // default cube dimensions
    int B = 4;
    int read = 0;
    std::string reads = "Sequential";
    std::string writes = "Sequential";
    int write = 0;
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
            N = atoi (& argv[1][baz] );
            while ( argv[1][baz] != 'x' ) baz++;
            baz++ ;
            B = atoi (& argv[1][baz]);
            break;
        case 'r':
            if(strlen(argv[1]) > 2) {
              baz = 2;
            } else {
              baz = 0;
              argv++, argc--;
            }
            read = atoi (& argv[1][baz] );
            while ( argv[1][baz] != 'x' ) baz++;
            baz++ ;
            write = atoi ( & argv[1][baz] );
            break;
        case 'h':
            fftx::OutStream() << "Usage: " << argv[0]
                            << " [ -i iterations ] [ -s NxB (DFT Length x Batch Size) ] [-r ReadxWrite (sequential = 0, strided = 1)] [ -h (print help message) ]"
                              << std::endl;
            exit (0);
        default:
            fftx::OutStream() << prog << ": unknown argument: "
                              << argv[1] << " ... ignored:" << std::endl;
        }
        argv++, argc--;
    }
    if(read == 0)
        reads = "Sequential";
    else
        reads = "Strided";
    if(write == 0)
        writes = "Sequential";
    else
        writes = "Strided";

     if ( FFTX_DEBUGOUT ) fftx::OutStream() << N << " " << B << " " << reads << " " << writes << std::endl;
    std::vector<int> sizes{N,B, read,write};
    // fftx::box_t<1> domain ( point_t<1> ( { { N } } ));

    std::vector<std::complex<double>> outDevfft1(N*B);
    std::vector<double> inputHost(N*B);
    std::vector<std::complex<double>> outputHost(N*B);
    std::vector<double> outDevfft2(N*B);
    std::vector<double> outputHost2(N*B);

     double *dX, *dY;
     std::complex<double> *dsym, *tempX;


#if defined (FFTX_CUDA) || defined(FFTX_HIP)
     if ( FFTX_DEBUGOUT ) fftx::OutStream() << "allocating memory" << std::endl;
    FFTX_DEVICE_MALLOC((void**)&dX, inputHost.size() * sizeof(double));
    FFTX_DEVICE_MALLOC((void **)&dY, outputHost2.size() * sizeof(double));
    FFTX_DEVICE_MALLOC((void **)&dsym,  outputHost.size() * sizeof(std::complex<double>));
    FFTX_DEVICE_MALLOC((void**)&tempX, outputHost.size()  * sizeof(std::complex<double>));
#elif defined FFTX_SYCL
  sycl::buffer<double> buf_Y(outputHost2.data(), outputHost2.size());
  sycl::buffer<double> buf_X(inputHost.data(), inputHost.size());
  sycl::buffer<std::complex<double>> buf_sym(outputHost.data(), outputHost.size());
  sycl::buffer<std::complex<double>> buf_tempX(outputHost.data(), outputHost.size());
#else
    dX = (double *) inputHost.data();
    dY = (double *) outputHost2.data();
    tempX = new std::complex<double>[outputHost.size()];
    dsym = new std::complex<double>[outputHost.size()];
#endif

    float *batch1dprdft_gpu = new float[iterations];
    float *ibatch1dprdft_gpu = new float[iterations];
#if defined FFTX_CUDA
    std::vector<void*> args{&tempX,&dX};
    std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
    std::string devfft  = "cufft";
#elif defined FFTX_HIP
    std::vector<void*> args{tempX,dX};
    std::string descrip = "AMD GPU";                //  "CPU and GPU";
    std::string devfft  = "rocfft";
#elif defined FFTX_SYCL
    std::vector<void*> args{(void*)&(buf_tempX),(void*)&(buf_X)};
    std::string descrip = "Intel GPU";                //  "CPU and GPU";
    std::string devfft  = "mklfft";
#else
    std::vector<void*> args{(void*)tempX,(void*)dX};
    std::string descrip = "CPU";                //  "CPU";
    std::string devfft = "fftw";
#endif

BATCH1DPRDFTProblem b1prdft(args, sizes, "b1prdft");


#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    //  Setup a plan to run the transform using cu or roc fft
    FFTX_DEVICE_FFT_HANDLE plan;
    FFTX_DEVICE_FFT_RESULT res;
    FFTX_DEVICE_FFT_TYPE   xfmtype = FFTX_DEVICE_FFT_D2Z ;
    FFTX_DEVICE_EVENT_T custart, custop;
    FFTX_DEVICE_EVENT_CREATE ( &custart );
    FFTX_DEVICE_EVENT_CREATE ( &custop );
    float *devmilliseconds = new float[iterations];
    float *invdevmilliseconds = new float[iterations];
    bool check_buff = true;                // compare results of spiral - RTC with device fft
    
    int xr = N;
    int xc = N/2 +1; 
    if(read == 0 && write == 0) {
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "APAR, APAR" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan, 1, &xr, //plan, rank, n,
                                    &xr,   1,  xr, // iembed, istride, idist,
                                    &xc,   1,  xc, // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    } else if(read == 0 && write == 1) { 
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "APAR, AVEC" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan, 1, &xr, //plan, rank, n,
                                    &xr,   1,  xr, // iembed, istride, idist,
                                    &xc,   B,  1, // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    }else if(read == 1 && write == 0) {
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "AVEC, APAR" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan, 1, &xr,  //plan, rank, n,
                                    &xr,   B,  1,  // iembed, istride, idist,
                                    &xc,   1,  xc,  // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    }
    else {
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "AVEC, AVEC" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan, 1, &xr,  //plan, rank, n,
                                    &xr,   B,  1,  // iembed, istride, idist,
                                    &xc,   B,  1,  // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    }

    if ( res != FFTX_DEVICE_FFT_SUCCESS ) {
        fftx::OutStream() << "Create FFTX_DEVICE_FFT_PLAN_MANY failed with error code "
                          << res << " ... skip buffer check" << std::endl;
        check_buff = false;
    }
#endif

    double *hostinp = (double *) inputHost.data();
    for (int itn = 0; itn < iterations; itn++)
    {
        // setup random data for input buffer (Use different randomized data each iteration)
        buildInputBuffer(hostinp, sizes);
    #if defined(FFTX_HIP) || defined(FFTX_CUDA)
        FFTX_DEVICE_MEM_COPY(dX, inputHost.data(),  inputHost.size() * sizeof(double),
                        FFTX_MEM_COPY_HOST_TO_DEVICE);
        if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied X" << std::endl;
    #endif
        
        
        b1prdft.transform();
        batch1dprdft_gpu[itn] = b1prdft.getTime();
    
    #if defined(FFTX_SYCL)		
	  {
            // fftx::OutStream() << "MKLFFT comparison not implemented printing first output element" << std::endl;
            // sycl::host_accessor h_acc(buf_tempX);
            // fftx::OutStream() << h_acc[0] << std::endl;
	  }
	  #endif
    
    #if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY ( outputHost.data(), tempX,
                          outputHost.size() * sizeof(std::complex<double>), FFTX_MEM_COPY_DEVICE_TO_HOST );
        //  Run the roc fft plan on the same input data
        if ( check_buff ) {
            FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECD2Z ( plan,
                                       (FFTX_DEVICE_FFT_DOUBLEREAL    *) dX,
                                       (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) tempX);
            if ( res != FFTX_DEVICE_FFT_SUCCESS) {
                fftx::OutStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
                check_buff = false;
                //  break;
            }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &devmilliseconds[itn], custart, custop );

            FFTX_DEVICE_MEM_COPY ( outDevfft1.data(), tempX,
                              outDevfft1.size() * sizeof(std::complex<double>), FFTX_MEM_COPY_DEVICE_TO_HOST );

            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tReal Batch 1D FFT (Forward)\t";
            checkOutputBuffers_fwd ( (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) outputHost.data(),
                                 (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) outDevfft1.data(),
                                 (long) outDevfft1.size() );
        }
    #endif
    }

#if defined FFTX_CUDA
    std::vector<void*> args2{&dY,&tempX};
#elif defined FFTX_HIP
    std::vector<void*> args2{dY,tempX};
#elif defined FFTX_SYCL
	std::vector<void*> args2{(void*)&(buf_Y), (void*)&(buf_tempX)};
#else
    std::vector<void*> args2{(void*)dY,(void*)tempX};
#endif

#if defined(FFTX_HIP) || defined(FFTX_CUDA)
FFTX_DEVICE_FFT_HANDLE plan2;
FFTX_DEVICE_FFT_TYPE xfmtype2 = FFTX_DEVICE_FFT_Z2D ;
if(read == 0 && write == 0) {
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "APAR, APAR" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan2, 1, &xr, //plan, rank, n,
                                    &xc,   1,  xc, // iembed, istride, idist,
                                    &xr,   1,  xr, // oembed, ostride, odist,
                                    xfmtype2, B); // type and batch
    } else if(read == 0 && write == 1) { 
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "APAR, AVEC" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan2, 1, &xr, //plan, rank, n,
                                    &xc,   1,  xc, // iembed, istride, idist,
                                    &xr,   B,  1, // oembed, ostride, odist,
                                    xfmtype2, B); // type and batch
    }else if(read == 1 && write == 0) {
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "AVEC, APAR" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan2, 1, &xr,  //plan, rank, n,
                                    &xc,   B,  1,  // iembed, istride, idist,
                                    &xr,   1,  xr,  // oembed, ostride, odist,
                                    xfmtype2, B); // type and batch
    }
    else {
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "AVEC, AVEC" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan2, 1, &xr,  //plan, rank, n,
                                    &xc,   B,  1,  // iembed, istride, idist,
                                    &xr,   B,  1,  // oembed, ostride, odist,
                                    xfmtype2, B); // type and batch
    }
#endif
    IBATCH1DPRDFTProblem ib1prdft(args2, sizes, "ib1prdft");

    for (int itn = 0; itn < iterations; itn++)
    {
        ib1prdft.transform();
        ibatch1dprdft_gpu[itn] = ib1prdft.getTime();
    
    #if defined (FFTX_SYCL)
	  {
            // fftx::OutStream() << "MKLFFT comparison not implemented printing first output element" << std::endl;
            // sycl::host_accessor h_acc(buf_Y);
            // fftx::OutStream() << h_acc[0] << std::endl;
	  }
    #endif

    #if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY ( outputHost2.data(), dY,
                          outputHost2.size() * sizeof(double), FFTX_MEM_COPY_DEVICE_TO_HOST );
        //  Run the roc fft plan on the same input data
        if ( check_buff ) {
            FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECZ2D ( plan2,
                                       (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) tempX,
                                       (FFTX_DEVICE_FFT_DOUBLEREAL *) dY);
            if ( res != FFTX_DEVICE_FFT_SUCCESS) {
                fftx::OutStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
                check_buff = false;
                // break;
            }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &invdevmilliseconds[itn], custart, custop );

            FFTX_DEVICE_MEM_COPY ( outDevfft2.data(), dY,
                              outDevfft2.size() * sizeof(double), FFTX_MEM_COPY_DEVICE_TO_HOST );
            
            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tReal Batch 1D FFT (Inverse)\t";
            checkOutputBuffers_inv ( (FFTX_DEVICE_FFT_DOUBLEREAL *) outputHost2.data(),
                                 (FFTX_DEVICE_FFT_DOUBLEREAL *) outDevfft2.data(),
                                 (long) outDevfft2.size() );
        }
    #endif
    }


#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on Real Batch 1D FFT (forward) for " << iterations
                      << " trials of size " << sizes.at(0)
                      << " and batch " << sizes.at(1) << ":"
                      << std::endl;
    fftx::OutStream() << "Trial #\tSpiral\t\t" << devfft << std::endl;
    for (int itn = 0; itn < iterations; itn++) {
      fftx::OutStream() << itn << "\t" << std::scientific << std::setprecision(7)
                        << batch1dprdft_gpu[itn] << "\t"
                        << devmilliseconds[itn] << std::endl;
    }

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on Real Batch 1D FFT (inverse) for " << iterations
                      << " trials of size " << sizes.at(0)
                      << " and batch " << sizes.at(1) << ":"
                      << std::endl;
    fftx::OutStream() << "Trial #\tSpiral\t\t" << devfft << std::endl;
    for (int itn = 0; itn < iterations; itn++) {
      fftx::OutStream() << itn << "\t" << std::scientific << std::setprecision(7)
                        << ibatch1dprdft_gpu[itn] << "\t"
                        << invdevmilliseconds[itn] << std::endl;
    }
#else
    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on Real Batch 1D FFT (forward) for " << iterations
                      << " trials of size " << sizes.at(0)
                      << " and batch " << sizes.at(1) << ":"
                      << std::endl;
    for (int itn = 0; itn < iterations; itn++) {
      fftx::OutStream() << itn << "\t" << std::scientific << std::setprecision(7)
                        << batch1dprdft_gpu[itn] << std::endl;
    }

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on Real Batch 1D FFT (inverse) for " << iterations
                      << " trials of size " << sizes.at(0)
                      << " and batch " << sizes.at(1) << ":"
                      << std::endl;
    for (int itn = 0; itn < iterations; itn++) {
      fftx::OutStream() << itn << "\t" << std::scientific << std::setprecision(7)
                        << ibatch1dprdft_gpu[itn] << std::endl;
    }
#endif

    fftx::OutStream() << prog << ": All done, exiting" << std::endl;
  
    return 0;
}
