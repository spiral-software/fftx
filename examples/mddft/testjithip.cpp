#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <memory>
// #include <cuda_runtime.h>
#include <complex>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include "fftx3.hpp"
#include "hipbackend.hpp"
#include "device_macros.h"

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

static void checkOutputBuffers ( DEVICE_FFT_DOUBLECOMPLEX *spiral_Y, DEVICE_FFT_DOUBLECOMPLEX *devfft_Y, long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int indx = 0; indx < arrsz; indx++ ) {
        DEVICE_FFT_DOUBLECOMPLEX s = spiral_Y[indx];
        DEVICE_FFT_DOUBLECOMPLEX c = devfft_Y[indx];

        bool elem_correct = ( (abs(s.x - c.x) < 1e-7) && (abs(s.y - c.y) < 1e-7) );
        maxdelta = maxdelta < (double)(abs(s.x -c.x)) ? (double)(abs(s.x -c.x)) : maxdelta ;
        maxdelta = maxdelta < (double)(abs(s.y -c.y)) ? (double)(abs(s.y -c.y)) : maxdelta ;
        correct &= elem_correct;
    }
    
    printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fflush ( stdout );

    return;
}


int main(int argc, char *argv[])
{
    int mm = 0, nn = 0, kk = 0; // cube dimensions
    char *prog = argv[0];
    int baz = 0;
    char *input_file = NULL;
    
    while ( argc > 1 && argv[1][0] == '-' ) {
        switch ( argv[1][1] ) {
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
        case 'f':
            argv++, argc--;
            input_file = argv[1];
            break;
        case 'h':
            printf ( "Usage: %s: -s MMxNNxKK -f jit_file [ -h (print help message) ]\n", argv[0] );
            exit (0);
        default:
            printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
        }
        argv++, argc--;
    }

    if ( mm == 0 || nn == 0 || kk == 0 || !input_file ) {
        printf ( "Arguments required\n" );
        printf ( "Usage: %s: -s MMxNNxKK -f jit_file [ -h (print help message) ]\n", argv[0] );
        exit ( -1 );
    }
    
    printf ( "Standalone test for size: %dx%dx%d using cached JIT from file: %s\n", mm, nn, kk, input_file );
    bool direction = true;
    if ( strstr ( input_file, "_imddft" ) != NULL )
        direction = false;

    long arrsz = mm * nn * kk;
    double *hostX = new double[arrsz * 2];
    double *hostY = new double[arrsz * 2];
    double *devY  = new double[arrsz * 2];

    std::vector<int> sizes{ mm, nn, kk };
    buildInputBuffer ( hostX, sizes );
    
    hipDeviceptr_t  dX, dY, dsym;
    std::cout << "allocating memory\n" << arrsz << "\n";
    HIP_SAFE_CALL(hipMalloc((void **)&dX, arrsz * sizeof(std::complex<double>)));
    std::cout << "allocated X\n";

    HIP_SAFE_CALL(hipMemcpy(dX, hostX,  arrsz * sizeof(std::complex<double>), hipMemcpyHostToDevice));
    std::cout << "copied hostX\n";

    HIP_SAFE_CALL(hipMalloc((void **)&dY, arrsz * sizeof(std::complex<double>)));
    std::cout << "allocated Y\n";

    // HIP_SAFE_CALL(cuMemcpyHtoD(dY, Y, 64* sizeof(double)));
    HIP_SAFE_CALL(hipMalloc((void **)&dsym, 64* sizeof(double)));

    std::vector<void*> inside;//{&dY, &dX, &dsym};
    inside.push_back(dY);
    inside.push_back(dX);
    inside.push_back(dsym);

    Executor e;
    e.execute ( input_file, inside );
    std::cout << e.getKernelTime() << std::endl;
    
    HIP_SAFE_CALL ( DEVICE_MEM_COPY ( hostY, dY,  arrsz * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST ) );
    std::cout << "copied Y to host\n";

    //  Setup a plan to run the transform using cu or roc fft
    DEVICE_FFT_HANDLE plan;
    DEVICE_FFT_RESULT res;
    DEVICE_FFT_TYPE   xfmtype = DEVICE_FFT_Z2Z ;
    
    res = DEVICE_FFT_PLAN3D ( &plan, mm, nn, kk, xfmtype );
    if ( res != DEVICE_FFT_SUCCESS ) {
        printf ( "Create DEVICE_FFT_PLAN_3D() failed with error code %d ... skip buffer check\n", res );
    }
    else {
        res = DEVICE_FFT_EXECZ2Z ( plan,
                                   (DEVICE_FFT_DOUBLECOMPLEX *) dX,
                                   (DEVICE_FFT_DOUBLECOMPLEX *) dY,
                                   direction ? DEVICE_FFT_FORWARD : DEVICE_FFT_INVERSE );
        if ( res != DEVICE_FFT_SUCCESS) {
            printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
        }
        else {
            HIP_SAFE_CALL ( DEVICE_MEM_COPY ( devY, dY, arrsz * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST ) );
            printf ( "cube = [ %d, %d, %d ]\tMDDFT (%s)\t", mm, nn, kk, direction ? "Forward" : "Inverse" );
            checkOutputBuffers ( (DEVICE_FFT_DOUBLECOMPLEX *) hostY, (DEVICE_FFT_DOUBLECOMPLEX *) devY, arrsz );
        }
    }

    delete[] hostX;
    delete[] hostY;
    delete[] devY;

    printf ( "Standalone test for size: %dx%dx%d completed successfully\n", mm, nn, kk );

    return 0;
}
