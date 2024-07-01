#include "fftx3.hpp"
#include "fftx3utilities.h"
#include "interface.hpp"
#include "mdprdftObj.hpp"
#include "imdprdftObj.hpp"
#include <string>
#include <fstream>

#if defined FFTX_CUDA
#include "cudabackend.hpp"
#elif defined FFTX_HIP
#include "hipbackend.hpp"
#elif defined FFTX_SYCL
#include "syclbackend.hpp"
#else  
#include "cpubackend.hpp"
#endif
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined (FFTX_SYCL)
#include "device_macros.h"
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
static void checkOutputBuffers_fwd ( DEVICE_FFT_DOUBLECOMPLEX *spiral_Y, DEVICE_FFT_DOUBLECOMPLEX *devfft_Y, long arrsz )
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
    int mm = 24, nn = 32, kk = 40; // default cube dimensions
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
            printf ( "Usage: %s: [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]\n", argv[0] );
            exit (0);
        default:
            printf ( "%s: unknown argument: %s ... ignored\n", prog, argv[1] );
        }
        argv++, argc--;
    }
    int K_adj = (int) ( kk / 2 ) + 1;
    std::cout << mm << " " << nn << " " << kk << std::endl;
    std::vector<int> sizes{mm,nn,kk};
    fftx::box_t<3> domain ( point_t<3> ( { { 1, 1, 1 } } ),
                            point_t<3> ( { { mm, nn, kk } } ));
    fftx::box_t<3> outputd ( point_t<3> ( { { 1, 1, 1 } } ),
                            point_t<3> ( { { mm, nn, K_adj } } ));

    fftx::array_t<3,double> inputHost(domain);
    fftx::array_t<3,std::complex<double>> outputHost(outputd);
    fftx::array_t<3,double> outputHost2(domain);
    fftx::array_t<3,std::complex<double>> outDevfft1(outputd);
    fftx::array_t<3,double> outDevfft2(domain);

    double * dX, *dY, *dsym;
    std::complex<double> * tempX;

    if ( DEBUGOUT )std::cout << "allocating memory" << std::endl;

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    DEVICE_MALLOC(&dX, inputHost.m_domain.size() * sizeof(double));

    DEVICE_MALLOC(&dY, outputHost2.m_domain.size() * sizeof(double));

    DEVICE_MALLOC(&dsym,  outputHost.m_domain.size() * sizeof(double));

    DEVICE_MALLOC(&tempX, mm * nn * K_adj * sizeof(std::complex<double>));
#elif defined(FFTX_SYCL)
    sycl::buffer<double> buf_Y(outputHost2.m_data.local(), outputHost2.m_domain.size());
    sycl::buffer<double> buf_X(inputHost.m_data.local(), inputHost.m_domain.size());
    sycl::buffer<double> buf_sym(inputHost.m_data.local(), inputHost.m_domain.size());
    sycl::buffer<std::complex<double>> buf_tempX(outputHost.m_data.local(), outputHost.m_domain.size());
#else
    dX = (double *) inputHost.m_data.local();
    dY = (double *) outputHost2.m_data.local();
    tempX = new std::complex<double>[outputHost.m_domain.size()];
    dsym = new double[outputHost2.m_domain.size()];
#endif
    if ( DEBUGOUT ) std::cout << "memory allocated" << std::endl;

    float *mddft_gpu = new float[iterations];
    float *imddft_gpu = new float[iterations];
#if defined FFTX_CUDA
    std::vector<void*> args{&tempX,&dX,&dsym};
    std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
    std::string devfft  = "cufft";
#elif defined FFTX_HIP
    std::vector<void*> args{tempX,dX,dsym};
    std::string descrip = "AMD GPU";                //  "CPU and GPU";
    std::string devfft  = "rocfft";
#elif defined FFTX_SYCL
    std::vector<void*> args{(void*)&(buf_tempX),(void*)&(buf_X),(void*)&(buf_sym)};
    std::string descrip = "Intel GPU";                //  "CPU and GPU";
    std::string devfft  = "mklfft";
#else
    std::vector<void*> args{(void*)tempX,(void*)dX,(void*)dsym};
    std::string descrip = "CPU";                //  "CPU";
    std::string devfft = "fftw";
    //std::string devfft  = "rocfft";
#endif

    MDPRDFTProblem mdp(args, sizes, "mdprdft");
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    //  Setup a plan to run the transform using cu or roc fft
    DEVICE_FFT_HANDLE plan;
    DEVICE_FFT_RESULT res;
    DEVICE_FFT_TYPE   xfmtype = DEVICE_FFT_D2Z ;
    DEVICE_EVENT_T custart, custop;
    DEVICE_EVENT_CREATE ( &custart );
    DEVICE_EVENT_CREATE ( &custop );
    float *devmilliseconds = new float[iterations];
    float *invdevmilliseconds = new float[iterations];
    bool check_buff = true;                // compare results of spiral - RTC with device fft
    
    res = DEVICE_FFT_PLAN3D ( &plan, mm, nn, kk, xfmtype );
    if ( res != DEVICE_FFT_SUCCESS ) {
        printf ( "Create DEVICE_FFT_PLAN3D failed with error code %d ... skip buffer check\n", res );
        check_buff = false;
    }
#endif

    double *hostinp = (double *) inputHost.m_data.local();
    for (int itn = 0; itn < iterations; itn++)
    {
        // setup random data for input buffer (Use different randomized data each iteration)
        buildInputBuffer ( hostinp, sizes );
    #if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY(dX, inputHost.m_data.local(),  inputHost.m_domain.size() * sizeof(double),
                        MEM_COPY_HOST_TO_DEVICE);
    #endif
        if ( DEBUGOUT ) std::cout << "copied X\n";
        
        mdp.transform();
        mddft_gpu[itn] = mdp.getTime();

    #if defined(FFTX_SYCL)		
	{
    std::cout << "MKLFFT comparison not implemented printing first output element" << std::endl;
		sycl::host_accessor h_acc(buf_tempX);
		std::cout << h_acc[0] << std::endl;
	}
	#endif

    #if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY ( outputHost.m_data.local(), tempX,
                            outputHost.m_domain.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
        if ( check_buff ) {
                DEVICE_EVENT_RECORD ( custart );
                res = DEVICE_FFT_EXECD2Z ( plan,
                                        (DEVICE_FFT_DOUBLEREAL *) dX,
                                        (DEVICE_FFT_DOUBLECOMPLEX *) tempX
                                        );
                if ( res != DEVICE_FFT_SUCCESS) {
                    printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
                    check_buff = false;
                    //  break;
                }
                DEVICE_EVENT_RECORD ( custop );
                DEVICE_EVENT_SYNCHRONIZE ( custop );
                DEVICE_EVENT_ELAPSED_TIME ( &devmilliseconds[itn], custart, custop );

                DEVICE_MEM_COPY ( outDevfft1.m_data.local(), tempX,
                                outDevfft1.m_domain.size() * sizeof(std::complex<double>), MEM_COPY_DEVICE_TO_HOST );
                printf ( "cube = [ %d, %d, %d ]\tMDPRDFT (Forward)\t", mm, nn, kk );
                checkOutputBuffers_fwd ( (DEVICE_FFT_DOUBLECOMPLEX *) outputHost.m_data.local(),
                                    (DEVICE_FFT_DOUBLECOMPLEX *) outDevfft1.m_data.local(),
                                    (long) outDevfft1.m_domain.size() );
            }
    #endif
    }

    // setup the inverse transform (we'll reuse the device fft plan already created)
#if defined FFTX_CUDA
    std::vector<void*> args1{&dY,&tempX,&dsym};
#elif defined FFTX_HIP
    std::vector<void*> args1{dY,tempX,dsym};
#elif defined FFTX_SYCL
    std::vector<void*> args1{(void*)&(buf_Y),(void*)&(buf_tempX),(void*)&(buf_sym)};	
#else
    std::vector<void*> args1{(void*)dY,(void*)tempX,(void*)dsym};
#endif

    IMDPRDFTProblem imdp("imdprdft");
    imdp.setArgs(args1);
    imdp.setSizes(sizes);

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    DEVICE_FFT_HANDLE plan2;     
    DEVICE_FFT_TYPE xfmtype2 = DEVICE_FFT_Z2D ;
    res = DEVICE_FFT_PLAN3D ( &plan2, mm, nn, kk, xfmtype2 );
    if ( res != DEVICE_FFT_SUCCESS ) {
        printf ( "Create DEVICE_FFT_PLAN3D failed with error code %d ... skip buffer check\n", res );
        check_buff = false;
    }
#endif

    std::vector<int> sizes2{mm,nn,K_adj};
    std::complex<double> *hostinp_complex = (std::complex<double> *) outputHost.m_data.local();
    for (int itn = 0; itn < iterations; itn++)
    {
        buildInputBuffer_complex((double*)hostinp_complex, sizes2);
        symmetrizeHermitian(outputHost, outputHost2);
    #if defined (FFTX_CUDA) || defined(FFTX_HIP)    
        DEVICE_MEM_COPY (tempX, outputHost.m_data.local(), (  mm * nn * K_adj ) * sizeof(std::complex<double>), MEM_COPY_HOST_TO_DEVICE );
    #endif

        if ( DEBUGOUT ) std::cout << "copied tempX" << std::endl;
        imdp.transform();
        imddft_gpu[itn] = imdp.getTime();

	#if defined (FFTX_SYCL)
	{
    std::cout << "MKLFFT comparison not implemented printing first output element" << std::endl;
		sycl::host_accessor h_acc(buf_Y);
		std::cout << h_acc[0] << std::endl;
	}
    #endif
    
	#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY ( outputHost2.m_data.local(), dY,
                          outputHost2.m_domain.size() * sizeof(double), MEM_COPY_DEVICE_TO_HOST );
    //  Run the device fft plan on the same input data
        if ( check_buff ) {
            DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECZ2D ( plan2,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) tempX,
                                       (DEVICE_FFT_DOUBLEREAL *) dY);
            if ( res != DEVICE_FFT_SUCCESS) {
                printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
                check_buff = false;
                //  break;
            }
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &invdevmilliseconds[itn], custart, custop );

            DEVICE_MEM_COPY ( outDevfft2.m_data.local(), dY,
                              outDevfft2.m_domain.size() * sizeof(double), MEM_COPY_DEVICE_TO_HOST );
            printf ( "cube = [ %d, %d, %d ]\tIMDPRDFT (Inverse)\t", mm, nn, kk );
            checkOutputBuffers_inv ( (DEVICE_FFT_DOUBLEREAL *) outputHost2.m_data.local(),
                                 (DEVICE_FFT_DOUBLEREAL *) outDevfft2.m_data.local(),
                                 (long) outDevfft2.m_domain.size() );
        }
    #endif
    }
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    printf ( "Times in milliseconds for %s on MDPRDFT (forward) for %d trials of size %d %d %d:\nTrial #\tSpiral\trocfft\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2) );        //  , devfft.c_str() );
    for (int itn = 0; itn < iterations; itn++) {
        printf ( "%d\t%.7e\t%.7e\n", itn, mddft_gpu[itn], devmilliseconds[itn] );
    }

    printf ( "Times in milliseconds for %s on MDPRDFT (inverse) for %d trials of size %d %d %d:\nTrial #\tSpiral\trocfft\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2) );
    for (int itn = 0; itn < iterations; itn++) {
        printf ( "%d\t%.7e\t%.7e\n", itn, imddft_gpu[itn], invdevmilliseconds[itn] );
    }
#else
     printf ( "Times in milliseconds for %s on MDPRDFT (forward) for %d trials of size %d %d %d\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
    for (int itn = 0; itn < iterations; itn++) {
        printf ( "%d\t%.7e\n", itn, mddft_gpu[itn]);
    }

    printf ( "Times in milliseconds for %s on MDPRDFT (inverse) for %d trials of size %d %d %d\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
    for (int itn = 0; itn < iterations; itn++) {
        printf ( "%d\t%.7e\n", itn, imddft_gpu[itn]);
    }
#endif
    // for(int i = 0; i < outDevfft1.m_domain.size(); i++) {
    //     std::cout << outputHost.m_data.local()[i] << " " << outDevfft1.m_data.local()[i] << std::endl;
    // }

    delete[] mddft_gpu;
    delete[] imddft_gpu;

    printf("%s: All done, exiting\n", prog);
  
    return 0;

}