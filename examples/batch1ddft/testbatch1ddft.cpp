#include "fftx3.hpp"
#include "interface.hpp"
#include "batch1ddftObj.hpp"
#include <math.h>  
#include "ibatch1ddftObj.hpp"
#include <string>
#include <fstream>

#if defined FFTX_CUDA
#include "cudabackend.hpp"
#elif defined FFTX_HIP
#include "hipbackend.hpp"
#else  
#include "cpubackend.hpp"
#endif
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#include "device_macros.h"
#endif

#define CH_CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } \
}

#define CUDA_SAFE_CALL(call) CH_CUDA_SAFE_CALL(call)

//  Build a random input buffer for Spiral and rocfft
//  host_X is the host buffer to setup -- it'll be copied to the device later
//  sizes is a vector with the X, Y, & Z dimensions

static void buildInputBuffer ( std::complex<double> *host_X, std::vector<int> sizes )
{
    for ( int imm = 0; imm < sizes.at(0)*sizes.at(1); imm++ ) {
        host_X[imm] = std::complex<double>(((double) rand()) / (double) (RAND_MAX/2), 0);
    }
    return;
}


// Check that the buffer are identical (within roundoff)
// spiral_Y is the output buffer from the Spiral generated transform (result on GPU copied to host array spiral_Y)
// devfft_Y is the output buffer from the device equivalent transform (result on GPU copied to host array devfft_Y)
// arrsz is the size of each array

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
static void checkOutputBuffers_fwd ( DEVICE_FFT_DOUBLECOMPLEX *spiral_Y, DEVICE_FFT_DOUBLECOMPLEX *devfft_Y, long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int indx = 0; indx < arrsz; indx++ ) {
        DEVICE_FFT_DOUBLECOMPLEX s = spiral_Y[indx];
        DEVICE_FFT_DOUBLECOMPLEX c = devfft_Y[indx];
        // std::cout << spiral_Y[indx] << ":" << devfft_Y[indx] << std::endl;


        bool elem_correct = ( (abs(s.x - c.x) < 1e-7) && (abs(s.y - c.y) < 1e-7) );
        maxdelta = maxdelta < (double)(abs(s.x -c.x)) ? (double)(abs(s.x -c.x)) : maxdelta ;
        maxdelta = maxdelta < (double)(abs(s.y -c.y)) ? (double)(abs(s.y -c.y)) : maxdelta ;
        correct &= elem_correct;
    }
    
    printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fflush ( stdout );

    return;
}

static void checkOutputBuffers_inv ( DEVICE_FFT_DOUBLEREAL *spiral_Y, DEVICE_FFT_DOUBLEREAL *devfft_Y, long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int indx = 0; indx < arrsz; indx++ ) {
        DEVICE_FFT_DOUBLEREAL s = spiral_Y[indx];
        DEVICE_FFT_DOUBLEREAL c = devfft_Y[indx];

        double deltar = abs ( s - c );
        bool   elem_correct = ( deltar < 1e-7 );
        maxdelta = maxdelta < deltar ? deltar : maxdelta ;
        correct &= elem_correct;
    }
    
    printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fflush ( stdout );

    return;
}


#endif

int main(int argc, char* argv[])
{
    int iterations = 2;
    int N = 32; // default cube dimensions
    int B = 2;
    int read = 0;
    std::string reads = "Sequential";
    std::string writes = "Strided";
    int write = 1;
    char *prog = argv[0];
    int baz = 0;

    while ( argc > 1 && argv[1][0] == '-' ) {
        switch ( argv[1][1] ) {
        case 'i':
            argv++, argc--;
            iterations = atoi ( argv[1] );
            break;
        case 's':
            baz = 0;
            argv++, argc--;
            N = atoi ( argv[1] );
            while ( argv[1][baz] != 'x' ) baz++;
            baz++ ;
            B = atoi (& argv[1][baz]);
            // nn = atoi ( & argv[1][baz] );
            // while ( argv[1][baz] != 'x' ) baz++;
            // baz++ ;
            // kk = atoi ( & argv[1][baz] );
            break;
        case 'r':
            baz = 0;
            argv++, argc--;
            read = atoi ( argv[1] );
            while ( argv[1][baz] != 'x' ) baz++;
            baz++ ;
            write = atoi ( & argv[1][baz] );
            // while ( argv[1][baz] != 'x' ) baz++;
            // baz++ ;
            // kk = atoi ( & argv[1][baz] );
            break;
        case 'h':
            printf ( "Usage: %s: [ -i iterations ] [ -s NxB (DFT Length x Batch Size) ] [-r ReadxWrite (sequential = 0, strided = 1)] [ -h (print help message) ]\n", argv[0] );
            exit (0);
        default:
            printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
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

    std::cout << N << " " << B << " " << reads << " " << writes << std::endl;
    std::vector<int> sizes{N,B, read,write};
    // fftx::box_t<1> domain ( point_t<1> ( { { N } } ));

    std::vector<std::complex<double>> outDevfft1(N*B*2);
    std::vector<double> inputHost(N*B*2);
    std::vector<std::complex<double>> outputHost(N*B*2);
    std::vector<std::complex<double>> outDevfft2(N*B*2);
    std::vector<std::complex<double>> outputHost2(N*B*2);

    std::complex<double> *dX, *dY, *dsym, *tempX;


#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    std::cout << "allocating memory" << std::endl;
    DEVICE_MALLOC((void**)&dX, inputHost.size() * sizeof(std::complex<double>));
    DEVICE_MALLOC((void **)&dY, outputHost.size() * sizeof(std::complex<double>));
    DEVICE_MALLOC((void **)&dsym,  outputHost.size() * sizeof(std::complex<double>));
    DEVICE_MALLOC((void**)&tempX, outputHost.size()  * sizeof(std::complex<double>));
#else
    dX = (std::complex<double> *) inputHost.data();
    dY = (std::complex<double> *) outputHost.data();
    tempX = new std::complex<double>[outputHost.size()];
    dsym = new std::complex<double>[outputHost.size()];
#endif

    float *batch1ddft_gpu = new float[iterations];
    float *ibatch1ddft_gpu = new float[iterations];
#if defined FFTX_CUDA
    std::vector<void*> args{&tempX,&dX};
    std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
    std::string devfft  = "cufft";
#elif defined FFTX_HIP
    std::vector<void*> args{tempX,dX};
    std::string descrip = "AMD GPU";                //  "CPU and GPU";
    std::string devfft  = "rocfft";
#else
    std::vector<void*> args{(void*)tempX,(void*)dX};
    std::string descrip = "CPU";                //  "CPU";
    std::string devfft = "fftw";
#endif

BATCH1DDFTProblem b1dft(args, sizes, "b1dft");


#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    //  Setup a plan to run the transform using cu or roc fft
    DEVICE_FFT_HANDLE plan;
    DEVICE_FFT_RESULT res;
    DEVICE_FFT_TYPE   xfmtype = DEVICE_FFT_Z2Z ;
    DEVICE_EVENT_T custart, custop;
    DEVICE_EVENT_CREATE ( &custart );
    DEVICE_EVENT_CREATE ( &custop );
    float *devmilliseconds = new float[iterations];
    float *invdevmilliseconds = new float[iterations];
    bool check_buff = true;                // compare results of spiral - RTC with device fft
    
    
    if(read == 0 && write == 0) {
        std::cout << "APAR, APAR" << std::endl;
        res = DEVICE_FFT_PLAN_MANY(&plan, 1, &N, //plan, rank, n,
                                    &N,   1,  N, // iembed, istride, idist,
                                    &N,   1,  N, // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    } else if(read == 0 && write == 1) { 
        std::cout << "APAR, AVEC" << std::endl;
        res = DEVICE_FFT_PLAN_MANY(&plan, 1, &N, //plan, rank, n,
                                    &N,   1,  N, // iembed, istride, idist,
                                    &N,   B,  1, // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    }else if(read == 1 && write == 0) {
        std::cout << "AVEC, APAR" << std::endl;
        res = DEVICE_FFT_PLAN_MANY(&plan, 1, &N,  //plan, rank, n,
                                    &N,   B,  1,  // iembed, istride, idist,
                                    &N,   1,  N,  // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    }
    else {
        std::cout << "AVEC, AVEC" << std::endl;
        res = DEVICE_FFT_PLAN_MANY(&plan, 1, &N,  //plan, rank, n,
                                    &N,   B,  1,  // iembed, istride, idist,
                                    &N,   B,  1,  // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    }

    if ( res != DEVICE_FFT_SUCCESS ) {
        printf ( "Create DEVICE_FFT_PLAN_MANY failed with error code %d ... skip buffer check\n", res );
        check_buff = false;
    }
#endif

    std::complex<double> *hostinp = (std::complex<double> *) inputHost.data();
    for (int itn = 0; itn < iterations; itn++)
    {
        // setup random data for input buffer (Use different randomized data each iteration)
        buildInputBuffer(hostinp, sizes);
    #if defined(FFTX_HIP)
        DEVICE_MEM_COPY(dX, inputHost.data(),  inputHost.size() * sizeof(std::complex<double>),
                        MEM_COPY_HOST_TO_DEVICE);
    #endif
    #if defined (FFTX_CUDA) 
         DEVICE_MEM_COPY((void*)dX, inputHost.data(),  inputHost.size() * sizeof(std::complex<double>),
                        MEM_COPY_HOST_TO_DEVICE);
    #endif
        if ( DEBUGOUT ) std::cout << "copied X" << std::endl;
        
        b1dft.transform();
        batch1ddft_gpu[itn] = b1dft.getTime();
        //gatherOutput(outputHost, args);
    #if defined (FFTX_CUDA) || defined(FFTX_HIP)
     #if defined(FFTX_HIP)
        DEVICE_MEM_COPY ( outputHost.data(), tempX,
                          outputHost.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
     #endif
     #if defined(FFTX_CUDA)
        DEVICE_MEM_COPY ( outputHost.data(), (void*)tempX,
                          outputHost.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
     #endif
        //  Run the roc fft plan on the same input data
        if ( check_buff ) {
            DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECZ2Z ( plan,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) dX,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) tempX,
                                      DEVICE_FFT_FORWARD);
            if ( res != DEVICE_FFT_SUCCESS) {
                printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
                check_buff = false;
                //  break;
            }
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &devmilliseconds[itn], custart, custop );

        #if defined FFTX_HIP
            DEVICE_MEM_COPY ( outDevfft1.data(), tempX,
                              outDevfft1.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
        #endif
        #if defined FFTX_CUDA
            DEVICE_MEM_COPY ( outDevfft1.data(), (void*)tempX,
                              outDevfft1.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );    
        #endif

            printf ( "DFT = %d Batch = %d Read = %s, Write = %s \tBatch 1D FFT (Forward)\t", N, B, reads.c_str(), writes.c_str());
            checkOutputBuffers_fwd ( (DEVICE_FFT_DOUBLECOMPLEX *) outputHost.data(),
                                 (DEVICE_FFT_DOUBLECOMPLEX *) outDevfft1.data(),
                                 (long) outDevfft1.size() );
        }
    #endif
    }


#if defined FFTX_CUDA
    std::vector<void*> args2{&dY,&tempX};
#elif defined FFTX_HIP
    std::vector<void*> args2{dY,tempX};
#else
    std::vector<void*> args2{(void*)dY,(void*)tempX};
    //std::string devfft  = "rocfft";
#endif


    IBATCH1DDFTProblem ib1dft(args2, sizes, "ib1dft");

    for (int itn = 0; itn < iterations; itn++)
    {
        ib1dft.transform();
        // batch1ddft_gpu[itn] = ib1prdft.getTime();
        //gatherOutput(outputHost, args);
    #if defined (FFTX_CUDA) || defined(FFTX_HIP)
     #if defined(FFTX_HIP)
        DEVICE_MEM_COPY ( outputHost2.data(), dY,
                          outputHost.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
     #endif
     #if defined(FFTX_CUDA)
        DEVICE_MEM_COPY ( outputHost.data(), (void*)dY,
                          outputHost.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
     #endif
        //  Run the roc fft plan on the same input data
        if ( check_buff ) {
            DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECZ2Z ( plan,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) tempX,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) dY,
                                      DEVICE_FFT_INVERSE);
            if ( res != DEVICE_FFT_SUCCESS) {
                printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
                check_buff = false;
                // break;
            }
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &devmilliseconds[itn], custart, custop );

        #if defined FFTX_HIP
            DEVICE_MEM_COPY ( outDevfft2.data(), dY,
                              outDevfft2.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
        #endif
        #if defined FFTX_CUDA
            DEVICE_MEM_COPY ( outDevfft2.data(), (void*)dY,
                              outDevfft2.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );    
        #endif

            
            printf ( "DFT = %d Batch = %d Read = %s Write = %s  \tBatch 1D FFT (Inverse)\t", N, B, reads.c_str(), writes.c_str());
            checkOutputBuffers_fwd ( (DEVICE_FFT_DOUBLECOMPLEX *) outputHost2.data(),
                                 (DEVICE_FFT_DOUBLECOMPLEX *) outDevfft2.data(),
                                 (long) outDevfft2.size() );
        }
    #endif
    }


    // for(int i = 0; i < outputHost2.size(); i++) {
    //     std::cout<< outputHost2[i] <<  " " << outDevfft2[i] << std::endl;
    // }

    // std::cout << "finished stuff" << std::endl;

    printf("%s: All done, exiting\n", prog);
  
    return 0;
}
