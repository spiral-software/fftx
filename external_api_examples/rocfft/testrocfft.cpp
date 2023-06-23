#include "fftx3.hpp"
#include <string>
#include <fstream>
#include <hip/hip_runtime.h> 
#include <hipfft.h>
#include "rocfft.h"
#include "shim.hpp"


//  Build a random input buffer for Spiral and rocfft
//  host_X is the host buffer to setup -- it'll be copied to the device later
//  sizes is a vector with the X, Y, & Z dimensions

static void buildInputBuffer ( double *host_X, std::vector<int> sizes )
{
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


// Check that the buffer are identical (within roundoff)
// spiral_Y is the output buffer from the Spiral generated transform (result on GPU copied to host array spiral_Y)
// devfft_Y is the output buffer from the device equivalent transform (result on GPU copied to host array devfft_Y)
// arrsz is the size of each array
static void checkOutputBuffers ( hipfftDoubleComplex *spiral_Y, hipfftDoubleComplex *devfft_Y, long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int indx = 0; indx < arrsz; indx++ ) {
        hipfftDoubleComplex s = spiral_Y[indx];
        hipfftDoubleComplex c = devfft_Y[indx];

        bool elem_correct = ( (abs(s.x - c.x) < 1e-7) && (abs(s.y - c.y) < 1e-7) );
        maxdelta = maxdelta < (double)(abs(s.x -c.x)) ? (double)(abs(s.x -c.x)) : maxdelta ;
        maxdelta = maxdelta < (double)(abs(s.y -c.y)) ? (double)(abs(s.y -c.y)) : maxdelta ;
        correct &= elem_correct;
    }
    
    printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fflush ( stdout );

    return;
}

int main(int argc, char* argv[])
{
    int iterations = 2;
    int mm = 24, nn = 32, kk = 40; // default cube dimensions
    char *prog = argv[0];
    int baz = 0;

    while ( argc > 1 && argv[1][0] == '-' ) {
        switch ( argv[1][1] ) {
        case 'i':
            argv++, argc--;
            iterations = atoi ( argv[1] );
            break;
        case 's':
            argv++, argc--;
            mm = atoi ( argv[1] );
            while ( argv[1][baz] != 'x' ) baz++;
            baz++ ;
            nn = atoi ( & argv[1][baz] );
            while ( argv[1][baz] != 'x' ) baz++;
            baz++ ;
            kk = atoi ( & argv[1][baz] );
            break;
        case 'h':
            printf ( "Usage: %s: [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]\n", argv[0] );
            exit (0);
        default:
            printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
        }
        argv++, argc--;
    }

    std::cout << mm << " " << nn << " " << kk << std::endl;
    std::vector<int> sizes{mm,nn,kk};
    fftx::box_t<3> domain ( fftx::point_t<3> ( { { 1, 1, 1 } } ),
                            fftx::point_t<3> ( { { mm, nn, kk } } ));

    fftx::array_t<3,std::complex<double>> inputHost(domain);
    fftx::array_t<3,std::complex<double>> outputHostfwd(domain);
    fftx::array_t<3,std::complex<double>> outputHostinv(domain);
    fftx::array_t<3,std::complex<double>> outDevfftfwd(domain);
    fftx::array_t<3,std::complex<double>> outDevfftinv(domain);
    std::complex<double> *dX, *dY;

    std::cout << "allocating memory\n";
    hipMalloc((void **)&dX, mm*nn*kk * sizeof(std::complex<double>));
    std::cout << "allocated X\n";

    hipMalloc((void **)&dY, mm*nn*kk * sizeof(std::complex<double>));
    std::cout << "allocated Y\n";

    float *mddft_gpu = new float[iterations];
    float *imddft_gpu = new float[iterations];

    //  Setup a plan to run the transform using cu or roc fft
    hipfftHandle plan;
    hipfftResult res;
    hipfftType xfmtype = HIPFFT_Z2Z ;
    hipEvent_t custart, custop;
    hipEventCreate ( &custart );
    hipEventCreate ( &custop );
    float *devmilliseconds = new float[iterations];
    float *invdevmilliseconds = new float[iterations];
    bool check_buff = true;                // compare results of spiral - RTC with device fft

    //Forward Plan FFTX
    res = hipfftPlan3d ( &plan, mm, nn, kk, xfmtype );
    if ( res != HIPFFT_SUCCESS ) {
        printf ( "Create hipfftPlan3d failed with error code %d ... skip buffer check\n", res );
        check_buff = false;
    }

    double *hostinp = (double *) inputHost.m_data.local();
    // setup random data for input buffer 
    buildInputBuffer ( hostinp, sizes );
    hipMemcpy(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(std::complex<double>),
                hipMemcpyHostToDevice);
    for (int itn = 0; itn < iterations; itn++)
    {
        hipEventRecord ( custart );
        res = hipfftExecZ2Z ( plan,
                                    (hipfftDoubleComplex *) dX,
                                    (hipfftDoubleComplex *) dY,
                                    HIPFFT_FORWARD );
        if ( res != HIPFFT_SUCCESS) {
            printf ( "Launch hipfftExecZ2Z failed with error code %d ... skip buffer check\n", res );
            check_buff = false;
            //  break;
        }

        hipEventRecord ( custop );
        hipEventSynchronize ( custop );
        hipEventElapsedTime ( &mddft_gpu[itn], custart, custop );

        hipMemcpy ( outputHostfwd.m_data.local(), dY,
                            outputHostfwd.m_domain.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost );
    }

    //Inverse Plan FFTX
    res = hipfftPlan3d ( &plan, mm, nn, kk, xfmtype );
    if ( res != HIPFFT_SUCCESS ) {
        printf ( "Create hipfftPlan3d failed with error code %d ... skip buffer check\n", res );
        check_buff = false;
    }

    //use output buffer of forward for inverse
    hipMemcpy(dX, outputHostfwd.m_data.local(),  outputHostfwd.m_domain.size() * sizeof(std::complex<double>),
                    hipMemcpyHostToDevice);

    for (int itn = 0; itn < iterations; itn++)
    {
        hipEventRecord ( custart );
        res = hipfftExecZ2Z ( plan,
                                    (hipfftDoubleComplex *) dX,
                                    (hipfftDoubleComplex *) dY,
                                    HIPFFT_BACKWARD );
        if ( res != HIPFFT_SUCCESS) {
            printf ( "Launch hipfftExecZ2Z failed with error code %d ... skip buffer check\n", res );
            check_buff = false;
            //  break;
        }

        hipEventRecord ( custop );
        hipEventSynchronize ( custop );
        hipEventElapsedTime ( &mddft_gpu[itn], custart, custop );

        hipMemcpy ( outputHostinv.m_data.local(), dY,
                            outputHostinv.m_domain.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost );
    }

    #undef hipfftPlan3d
    #undef hipfftExecZ2Z

    //Forward Plan rocFFT 
    res = hipfftPlan3d ( &plan, mm, nn, kk, xfmtype );
    if ( res != HIPFFT_SUCCESS ) {
        printf ( "Create hipfftPlan3d failed with error code %d ... skip buffer check\n", res );
        check_buff = false;
    }

    hipMemcpy(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(std::complex<double>),
                        hipMemcpyHostToDevice);

    for (int itn = 0; itn < iterations; itn++)
    {
        if ( check_buff ) {
            hipEventRecord ( custart );
            res = hipfftExecZ2Z ( plan,
                                    (hipfftDoubleComplex *) dX,
                                    (hipfftDoubleComplex *) dY,
                                    HIPFFT_FORWARD );
            if ( res != HIPFFT_SUCCESS) {
                printf ( "Launch hipfftExecZ2Z failed with error code %d ... skip buffer check\n", res );
                check_buff = false;
                //  break;
            }
            hipEventRecord ( custop );
            hipEventSynchronize ( custop );
            hipEventElapsedTime ( &devmilliseconds[itn], custart, custop );

            hipMemcpy ( outDevfftfwd.m_data.local(), dY,
                              outDevfftfwd.m_domain.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost );
            printf ( "cube = [ %d, %d, %d ]\tMDDFT (Forward)\t", mm, nn, kk );
            checkOutputBuffers ( (hipfftDoubleComplex *) outputHostfwd.m_data.local(),
                                 (hipfftDoubleComplex *) outDevfftfwd.m_data.local(),
                                 (long) outDevfftfwd.m_domain.size() );
        }
    }

    //use output buffer of forward for inverse
    hipMemcpy(dX, outDevfftfwd.m_data.local(),  outDevfftfwd.m_domain.size() * sizeof(std::complex<double>),
                        hipMemcpyHostToDevice);

    for (int itn = 0; itn < iterations; itn++)
    {
        if ( check_buff ) {
            hipEventRecord ( custart );
            res = hipfftExecZ2Z ( plan,
                                    (hipfftDoubleComplex *) dX,
                                    (hipfftDoubleComplex *) dY,
                                    HIPFFT_BACKWARD );
            if ( res != HIPFFT_SUCCESS) {
                printf ( "Launch hipfftExecZ2Z failed with error code %d ... skip buffer check\n", res );
                check_buff = false;
                //  break;
            }
            hipEventRecord ( custop );
            hipEventSynchronize ( custop );
            hipEventElapsedTime ( &devmilliseconds[itn], custart, custop );

            hipMemcpy ( outDevfftinv.m_data.local(), dY,
                              outDevfftinv.m_domain.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost );
            printf ( "cube = [ %d, %d, %d ]\tMDDFT (Backward)\t", mm, nn, kk );
            checkOutputBuffers ( (hipfftDoubleComplex *) outputHostinv.m_data.local(),
                                 (hipfftDoubleComplex *) outDevfftinv.m_data.local(),
                                 (long) outDevfftinv.m_domain.size() );
        }
    }

    // // setup the inverse transform (we'll reuse the device fft plan already created)
    // IMDDFTProblem imdp(args, sizes, "imddft");

    // for (int itn = 0; itn < iterations; itn++)
    // {
    //     // setup random data for input buffer (Use different randomized data each iteration)
    //     buildInputBuffer ( hostinp, sizes );
    // #if defined (FFTX_CUDA) || defined(FFTX_HIP)
    //     DEVICE_MEM_COPY(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(std::complex<double>),
    //                     MEM_COPY_HOST_TO_DEVICE);
    // #endif
    //     if ( DEBUGOUT ) std::cout << "copied X\n";
        
    //     imdp.transform();
    //     //gatherOutput(outputHost, args);
    // #if defined (FFTX_CUDA) || defined(FFTX_HIP)
    //     DEVICE_MEM_COPY ( outputHost.m_data.local(), dY,
    //                       outputHost.m_domain.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
    // #endif
    //     imddft_gpu[itn] = imdp.getTime();

    // #if defined (FFTX_CUDA) || defined(FFTX_HIP)
    //     //  Run the device fft plan on the same input data
    //     if ( check_buff ) {
    //         DEVICE_EVENT_RECORD ( custart );
    //         res = DEVICE_FFT_EXECZ2Z ( plan,
    //                                    (DEVICE_FFT_DOUBLECOMPLEX *) dX,
    //                                    (DEVICE_FFT_DOUBLECOMPLEX *) dY,
    //                                    DEVICE_FFT_INVERSE );
    //         if ( res != DEVICE_FFT_SUCCESS) {
    //             printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
    //             check_buff = false;
    //             //  break;
    //         }
    //         DEVICE_EVENT_RECORD ( custop );
    //         DEVICE_EVENT_SYNCHRONIZE ( custop );
    //         DEVICE_EVENT_ELAPSED_TIME ( &invdevmilliseconds[itn], custart, custop );

    //         DEVICE_MEM_COPY ( outDevfft.m_data.local(), dY,
    //                           outDevfft.m_domain.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
    //         printf ( "cube = [ %d, %d, %d ]\tMDDFT (Inverse)\t", mm, nn, kk );
    //         checkOutputBuffers ( (DEVICE_FFT_DOUBLECOMPLEX *) outputHost.m_data.local(),
    //                              (DEVICE_FFT_DOUBLECOMPLEX *) outDevfft.m_data.local(),
    //                              (long) outDevfft.m_domain.size() );
    //     }
    // #endif
    // }

// #if defined (FFTX_CUDA) || defined(FFTX_HIP)
//     printf ( "Times in milliseconds for %s on MDDFT (forward) for %d trials of size %d %d %d:\nTrial #\tSpiral\t%s\n",
//              descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2), devfft.c_str() );
//     for (int itn = 0; itn < iterations; itn++) {
//         printf ( "%d\t%.7e\t%.7e\n", itn, mddft_gpu[itn], devmilliseconds[itn] );
//     }

//     printf ( "Times in milliseconds for %s on MDDFT (inverse) for %d trials of size %d %d %d:\nTrial #\tSpiral\t%s\n",
//              descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2), devfft.c_str() );
//     for (int itn = 0; itn < iterations; itn++) {
//         printf ( "%d\t%.7e\t%.7e\n", itn, imddft_gpu[itn], invdevmilliseconds[itn] );
//     }
// #else
//      printf ( "Times in milliseconds for %s on MDDFT (forward) for %d trials of size %d %d %d\n",
//              descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
//     for (int itn = 0; itn < iterations; itn++) {
//         printf ( "%d\t%.7e\n", itn, mddft_gpu[itn]);
//     }

//     printf ( "Times in milliseconds for %s on MDDFT (inverse) for %d trials of size %d %d %d\n",
//              descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
//     for (int itn = 0; itn < iterations; itn++) {
//         printf ( "%d\t%.7e\n", itn, imddft_gpu[itn]);
//     }
// #endif


    // delete[] mddft_cpu;
    // delete[] imddft_cpu;
    // delete[] mddft_gpu;
    // delete[] imddft_gpu;

    printf("%s: All done, exiting\n", prog);
  
    return 0;
}
