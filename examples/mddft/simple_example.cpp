//device runtimes
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

//stl
#include <complex>
#include <iostream>
#include <vector>

//fftx order matters
// #include "fftx3.hpp"
#include "device_macros.h"
#include "interface.hpp"
#include "mddftObj.hpp"
#include "imddftObj.hpp"

#define mm 24
#define nn 32
#define kk 40

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

int main(int argc, char *argv[])
{
    printf ( "Standalone test for size: %dx%dx%d\n", mm, nn, kk);

    long arrsz = mm * nn * kk;
    double *hostX = new double[arrsz * 2];
    double *hostY = new double[arrsz * 2];
    double *devY  = new double[arrsz * 2];

    std::vector<int> sizes{ mm, nn, kk };
    buildInputBuffer (hostX,sizes);
    hipDeviceptr_t  dY, dX, dsym;
    std::cout << "allocated Y" << std::endl;
    hipMalloc((void **)&dY, arrsz * sizeof(std::complex<double>));
    std::cout << "allocating memory\n" << arrsz << std::endl;
    hipMalloc((void **)&dX, arrsz * sizeof(std::complex<double>));
    std::cout << "allocated X" << std::endl;
    hipMemcpy(dX, hostX,  arrsz * sizeof(std::complex<double>), hipMemcpyHostToDevice);
    std::cout << "copied hostX" << std::endl;
    hipMalloc((void **)&dsym, 64* sizeof(double));
    MDDFTProblem mdp(std::vector<void*>{dY,dX,dsym}, sizes);
    mdp.transform(); 
    std::cout << mdp.getTime() << std::endl;
    IMDDFTProblem imdp(std::vector<void*>{dY,dX,dsym}, sizes);
    imdp.transform(); 
    std::cout << imdp.getTime() << std::endl;
    hipMemcpy( hostY, dY,  arrsz * sizeof(std::complex<double>), hipMemcpyDeviceToHost);
    std::cout << "copied Y to host" << std::endl;
    delete[] hostX;
    delete[] hostY;
    delete[] devY;

    // printf ( "Standalone test for size: %dx%dx%d completed successfully\n", mm, nn, kk );
    std::cout << "Standalone test for size: " << mm << "x" << nn << "x" << kk << " completed successfully" << std::endl;

    return 0;
}
