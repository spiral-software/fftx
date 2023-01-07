#include "fftx3.hpp"
#include <string>
#include <fstream>
#include "fftxfft.hpp"
#include "device_macros.h"

int main(int argc, char* argv[])
{
    int iterations = 2;
    int mm = 24, nn = 32, kk = 40; // cube dimensions
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
    fftx::box_t<3> domain ( point_t<3> ( { { 1, 1, 1 } } ),
    point_t<3> ( { { mm, nn, kk } } ));

    fftx::array_t<3,std::complex<double>> inputHost(domain);
    fftx::array_t<3,std::complex<double>> outputHost(domain);

    forall([](std::complex<double>(&v), const fftx::point_t<3>& p) {
            v=std::complex<double>(2.0,0.0);
        },inputHost);
        
    hipDeviceptr_t  dX, dY, dsym;

	std::cout << "allocating memory\n";
	hipMalloc((void **)&dX, inputHost.m_domain.size() * sizeof(std::complex<double>));
	std::cout << "allocated X\n";
	hipMemcpy(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(std::complex<double>), hipMemcpyHostToDevice);
	std::cout << "copied X\n";
	hipMalloc((void **)&dY, outputHost.m_domain.size() * sizeof(std::complex<double>));
	std::cout << "allocated Y\n";
	// //  HIP_SAFE_CALL(cuMemcpyHtoD(dY, Y, 64* sizeof(double)));
	// DEVICE_MALLOC((void **)&dsym,  outputHost.m_domain.size()*  sizeof(std::complex<double>));
    printf("call mddft::init()\n");
    // mddft::init();

    printf("call mddft::transform()\n");

    for (int itn = 0; itn < iterations; itn++) {
        mddft(mm, nn, kk, dY, dX);
        //gatherOutput(outputHost, args);
        hipMemcpy(outputHost.m_data.local(), &dY,
                            outputHost.m_domain.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost);
        
    }
    printf("finished the code\n");
    return 0;
}