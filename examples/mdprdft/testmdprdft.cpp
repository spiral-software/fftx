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
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#include "device_macros.h"
#elif defined (FFTX_SYCL)
// #include "mkl_dfti.h"
// #include <CL/sycl.hpp>
#include <oneapi/mkl/dfti.hpp>
// #include <oneapi/mkl/vm.hpp>
#endif

#if defined(FFTX_HIP) || defined(FFTX_CUDA) || defined(FFTX_SYCL)
//  Build a random input buffer for Spiral and rocfft
//  host_X is the host buffer to setup -- it'll be copied to the device later
//  sizes is a vector with the X, Y, & Z dimensions

static void setInput ( double *host_X, std::vector<int> sizes )
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

static void setInput_complex ( double *host_X, std::vector<int> sizes )
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
static void setInput ( double *host_X, std::vector<int> sizes )
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

static void setInput_complex( double *host_X, std::vector<int> sizes)
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

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#define DOUBLECOMPLEX DEVICE_FFT_DOUBLECOMPLEX
#define DOUBLEREAL DEVICE_FFT_DOUBLEREAL
#define REALPART(z) z.x
#define IMAGPART(z) z.y
#elif defined (FFTX_SYCL)
#define DOUBLECOMPLEX std::complex<double>
#define DOUBLEREAL double
#define REALPART(z) z.real()
#define IMAGPART(z) z.imag()
#endif

// Check that the buffer are identical (within roundoff)
// outputFFTXPtr is the output buffer from the Spiral-generated transform
// (result on GPU copied to host array outputFFTXPtr);
// outputVendorPtr is the output buffer from the vendor transform
// (result on GPU copied to host array outputVendorPtr).
// arrsz is the size of each array

static void checkOutputs_R2C ( DOUBLECOMPLEX *outputFFTXPtr,
                               DOUBLECOMPLEX *outputVendorPtr,
                               long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int ind = 0; ind < arrsz; ind++ )
      {
        double sreal = REALPART(outputFFTXPtr[ind]);
        double simag = IMAGPART(outputFFTXPtr[ind]);
        double creal = REALPART(outputVendorPtr[ind]);
        double cimag = IMAGPART(outputVendorPtr[ind]);

        double diffreal = sreal - creal;
        double diffimag = simag - cimag;
        
        bool elem_correct =
          ( (abs(diffreal) < 1.e-7) && (abs(diffimag) < 1.e-7) );
	updateMaxAbs(maxdelta, diffreal);
	updateMaxAbs(maxdelta, diffimag);
        correct &= elem_correct;
      }
    
    printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fflush ( stdout );

    return;
}

static void checkOutputs_C2R ( DOUBLEREAL *outputFFTXPtr,
                               DOUBLEREAL *outputVendorPtr,
                               long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int ind = 0; ind < arrsz; ind++ )
      {
        DOUBLEREAL s = outputFFTXPtr[ind];
        DOUBLEREAL c = outputVendorPtr[ind];

        double deltar = s - c;
        bool   elem_correct = ( abs(deltar) < 1e-7 );
        updateMaxAbs(maxdelta, deltar);
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

    fftx::array_t<3,double> inputHostArray(domain);
    fftx::array_t<3,std::complex<double>> outputComplexFFTXHostArray(outputd);
    fftx::array_t<3,double> outputRealFFTXHostArray(domain);
    fftx::array_t<3,std::complex<double>> outputComplexVendorHostArray(outputd);
    fftx::array_t<3,double> outputRealVendorHostArray(domain);

    size_t npts = domain.size();
    size_t nptsTrunc = outputd.size();

    auto inputHostPtr = inputHostArray.m_data.local();
    auto outputComplexFFTXHostPtr = outputComplexFFTXHostArray.m_data.local();
    auto outputRealFFTXHostPtr = outputRealFFTXHostArray.m_data.local();
    auto outputComplexVendorHostPtr = outputComplexVendorHostArray.m_data.local();
    auto outputRealVendorHostPtr = outputRealVendorHostArray.m_data.local();
  
    double * inputTfmPtr, *outputTfmPtr, *symbolTfmPtr;
    std::complex<double> * tempTfmPtr;

    if ( DEBUGOUT )std::cout << "allocating memory" << std::endl;

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    DEVICE_MALLOC(&inputTfmPtr, npts * sizeof(double));
    DEVICE_MALLOC(&outputTfmPtr, npts * sizeof(double));
    DEVICE_MALLOC(&symbolTfmPtr,  nptsTrunc * sizeof(double));
    DEVICE_MALLOC(&tempTfmPtr, nptsTrunc * sizeof(std::complex<double>));
#elif defined(FFTX_SYCL)
    sycl::buffer<double> outputBufTfmPtr(outputRealFFTXHostPtr, npts);
    sycl::buffer<double> inputBufTmpPtr(inputHostPtr, npts);
    sycl::buffer<double> symbolBufTmpPtr(inputHostPtr, npts);
    // FIXME: Double size?
    sycl::buffer<std::complex<double>> tempBufTmpPtr(outputComplexFFTXHostPtr, nptsTrunc * 2);
#else
    inputTfmPtr = (double *) inputHostPtr;
    outputTfmPtr = (double *) outputRealFFTXHostPtr;
    tempTfmPtr = new std::complex<double>[nptsTrunc];
    symbolTfmPtr = new double[npts];
#endif
    if ( DEBUGOUT ) std::cout << "memory allocated" << std::endl;

#if defined FFTX_CUDA
    std::vector<void*> argsR2C{&tempTfmPtr,&inputTfmPtr,&symbolTfmPtr};
    std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
    std::string vendorfft  = "cufft";
#elif defined FFTX_HIP
    std::vector<void*> argsR2C{tempTfmPtr,inputTfmPtr,symbolTfmPtr};
    std::string descrip = "AMD GPU";                //  "CPU and GPU";
    std::string vendorfft  = "rocfft";
#elif defined FFTX_SYCL
    std::vector<void*> argsR2C{(void*)&(tempBufTmpPtr), (void*)&(inputBufTmpPtr), (void*)&(symbolBufTmpPtr)};
    std::string descrip = "Intel GPU";                //  "CPU and GPU";
    std::string vendorfft  = "mklfft";
#else
    std::vector<void*> argsR2C{(void*)tempTfmPtr, (void*)inputTfmPtr, (void*)symbolTfmPtr};
    std::string descrip = "CPU";                //  "CPU";
    // std::string vendorfft = "fftw";
#endif

    float *mdprdft_gpu = new float[iterations];
    float *imdprdft_gpu = new float[iterations];
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
    float *mdprdft_vendor_millisec = new float[iterations];
    float *imdprdft_vendor_millisec = new float[iterations];
    bool check_output = true; // compare results of Spiral-RTC with vendor FFT
#endif

    MDPRDFTProblem mdp(argsR2C, sizes, "mdprdft");
    
    //  Set up a plan to run the transform using vendor FFT.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    DEVICE_FFT_HANDLE planR2C;
    DEVICE_FFT_RESULT res;
    DEVICE_FFT_TYPE   xfmtypeR2C = DEVICE_FFT_D2Z ;
    DEVICE_EVENT_T custart, custop;
    DEVICE_EVENT_CREATE ( &custart );
    DEVICE_EVENT_CREATE ( &custop );
    
    res = DEVICE_FFT_PLAN3D ( &planR2C, mm, nn, kk, xfmtypeR2C );
    if ( res != DEVICE_FFT_SUCCESS )
      {
        printf ( "Create DEVICE_FFT_PLAN3D failed with error code %d ... skip buffer check\n", res );
        check_output = false;
      }
#elif defined (FFTX_SYCL)
    sycl::device dev;
    try
      {
	dev = sycl::device(sycl::gpu_selector_v);
      }
    catch (sycl::exception const &e)
      {
	std::cout << "You are running on a system without a GPU. For best results please use a GPU." << std::endl;
	std::cout << "Program terminating." << std::endl;
	exit(-1);
	// dev = sycl::device(sycl::cpu_selector_v);
      }
    sycl::context ctx = sycl::context(dev);
    cl_device_id ocl_dev =
      sycl::get_native<cl::sycl::backend::opencl, sycl::device>(dev);
    cl_context   ocl_ctx =
      sycl::get_native<cl::sycl::backend::opencl, sycl::context>(ctx);
    cl_int err = CL_SUCCESS;
    cl_command_queue ocl_queue =
      clCreateCommandQueueWithProperties(ocl_ctx, ocl_dev,0,&err);
    sycl::queue Q = sycl::make_queue<sycl::backend::opencl>(ocl_queue,ctx);

    // Initialize SYCL queue
    //	  sycl::queue Q(sycl::default_selector{});
    auto sycl_device = Q.get_device();
    auto sycl_context = Q.get_context();
    std::cout << "Running on: "
	      << Q.get_device().get_info<sycl::info::device::name>() << std::endl;
	  
    auto inputRealVendorPtr = sycl::malloc_shared< double >
      (npts, sycl_device, sycl_context);
    auto outputComplexVendorPtr = sycl::malloc_shared< std::complex<double> >
      (nptsTrunc, sycl_device, sycl_context);
    
    auto inputComplexVendorPtr = sycl::malloc_shared< std::complex<double> >
      (nptsTrunc, sycl_device, sycl_context);
    auto outputRealVendorPtr = sycl::malloc_shared< double >
      (npts, sycl_device, sycl_context);
    
    // Initialize 3D FFT descriptor
    std::vector<std::int64_t> Nvec{mm, nn, kk};
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
				 oneapi::mkl::dft::domain::REAL>
      transform_plan_3d(Nvec);
    transform_plan_3d.commit(Q);
#endif

    double *hostinp = (double *) inputHostPtr;
    for (int itn = 0; itn < iterations; itn++)
    {
        // setup random data for input buffer (Use different randomized data each iteration)
        setInput ( hostinp, sizes );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY(inputTfmPtr, inputHostPtr, npts * sizeof(double),
                        MEM_COPY_HOST_TO_DEVICE);
#endif
 	if ( DEBUGOUT ) std::cout << "copied input from host to device\n";
        
        // Run transform: input inputTfmPtr, output tempTfmPtr.
        mdp.transform();
        mdprdft_gpu[itn] = mdp.getTime();
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY ( outputComplexFFTXHostPtr, tempTfmPtr,
                          nptsTrunc * sizeof(std::complex<double>),
                          MEM_COPY_DEVICE_TO_HOST );
        if ( check_output )
          { //  Run the vendor FFT plan on the same input data.
            DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECD2Z ( planR2C,
                                       (DEVICE_FFT_DOUBLEREAL *) inputTfmPtr,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) tempTfmPtr
                                       );
            if ( res != DEVICE_FFT_SUCCESS)
              {
                printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
                check_output = false;
                //  break;
              }
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &mdprdft_vendor_millisec[itn], custart, custop );
            DEVICE_MEM_COPY ( outputComplexVendorHostPtr, tempTfmPtr,
                              nptsTrunc * sizeof(std::complex<double>),
                              MEM_COPY_DEVICE_TO_HOST );
            printf ( "cube = [ %d, %d, %d ]\tMDPRDFT (Forward)\t", mm, nn, kk );
            checkOutputs_R2C ( (DEVICE_FFT_DOUBLECOMPLEX *) outputComplexFFTXHostPtr,
                               (DEVICE_FFT_DOUBLECOMPLEX *) outputComplexVendorHostPtr,
                               (long) nptsTrunc);
            }
#elif defined ( FFTX_SYCL)
 	// If this is absent then iterations after the first aren't correct.
	sycl::host_accessor inputHostAcc(inputButTfmPtr);

	// FIXME: tempHostAcc is double* if tempBufTfmPtr is sycl::buffer<double>.
	sycl::host_accessor tempHostAcc(tempBufTfmPtr);
	for (int ind = 0; ind < npts; ind++)
	  {
	    inputVendorPtr[ind] = inputHostPtr[ind];
          }
        for (int ind = 0; ind < nptsTrunc; ind++)
          {
	    outputComplexFFTXHostPtr[ind] = tempHostAcc[ind];
	  }
	
	auto start_time = std::chrono::high_resolution_clock::now();
        //  Run the vendor FFT plan on the same input data.
	// Perform forward transform on real array
	oneapi::mkl::dft::compute_forward(transform_plan_3d, inputRealVendorPtr, outputComplexVendorPtr).wait();
	auto end_time = std::chrono::high_resolution_clock::now();

	std::chrono::duration<float, std::milli> duration = end_time - start_time;
	mdprdft_vendor_millisec[itn] = duration.count();
      
	for (int ind = 0; ind < nptsTrunc; ind++)
	  {
	    outputComplexVendorHostPtr[ind] = outputComplexVendorPtr[ind];
	  }
      
	printf ( "cube = [ %d, %d, %d ]\tMDDFT (Forward)\t", mm, nn, kk );
	if (check_output)
	  {
	    checkOutputs_R2C ( (std::complex<double>*) outputComplexFFTXHostPtr,
                               (std::complex<double>*) outputComplexVendorHostPtr,
                               nptsTrunc );
	  }
       
#endif
    }

    // setup the inverse transform (we'll reuse the vendor FFT plan already created)
#if defined FFTX_CUDA
    std::vector<void*> argsC2R{&outputTfmPtr,&tempTfmPtr,&symbolTfmPtr};
#elif defined FFTX_HIP
    std::vector<void*> argsC2R{outputTfmPtr,tempTfmPtr,symbolTfmPtr};
#elif defined FFTX_SYCL
    std::vector<void*> argsC2R{(void*)&(outputBufTfmPtr), (void*)&(tempBufTmpPtr), (void*)&(symbolBufTmpPtr)};	
#else
    std::vector<void*> argsC2R{(void*)outputTfmPtr, (void*)tempTfmPtr, (void*)symbolTfmPtr};
#endif

    IMDPRDFTProblem imdp("imdprdft");
    imdp.setArgs(argsC2R);
    imdp.setSizes(sizes);

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    DEVICE_FFT_HANDLE planC2R;     
    DEVICE_FFT_TYPE xfmtypeC2R = DEVICE_FFT_Z2D ;
    res = DEVICE_FFT_PLAN3D ( &planC2R, mm, nn, kk, xfmtypeC2R );
    if ( res != DEVICE_FFT_SUCCESS )
      {
        printf ( "Create DEVICE_FFT_PLAN3D failed with error code %d ... skip buffer check\n", res );
        check_output = false;
      }
#endif

    std::vector<int> sizesTrunc{mm,nn,K_adj};
    std::complex<double> *hostinp_complex = (std::complex<double> *) outputComplexFFTXHostPtr;
    for (int itn = 0; itn < iterations; itn++)
      {
        setInput_complex((double*)hostinp_complex, sizesTrunc);
        symmetrizeHermitian(outputComplexFFTXHostArray, outputRealFFTXHostArray);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)    
        DEVICE_MEM_COPY (tempTfmPtr, outputComplexFFTXHostPtr,
                         nptsTrunc * sizeof(std::complex<double>),
                         MEM_COPY_HOST_TO_DEVICE );
#endif
        if ( DEBUGOUT ) std::cout << "copied tempTfmPtr" << std::endl;

        // Run transform: input tempTfmPtr, output outputTfmPtr.
        imdp.transform();
        imdprdft_gpu[itn] = imdp.getTime();

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY ( outputRealFFTXHostPtr, outputTfmPtr,
                          npts * sizeof(double), MEM_COPY_DEVICE_TO_HOST );
        //  Run the vendor FFT plan on the same input data.
        if ( check_output )
          {
            DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECZ2D ( planC2R,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) tempTfmPtr,
                                       (DEVICE_FFT_DOUBLEREAL *) outputTfmPtr);
            if ( res != DEVICE_FFT_SUCCESS)
              {
                printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
                check_output = false;
                //  break;
              }
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &imdprdft_vendor_millisec[itn], custart, custop );
            
            DEVICE_MEM_COPY ( outputRealVendorHostPtr, outputTfmPtr,
                              npts * sizeof(double), MEM_COPY_DEVICE_TO_HOST );
            printf ( "cube = [ %d, %d, %d ]\tIMDPRDFT (Inverse)\t", mm, nn, kk );
            checkOutputs_C2R ( (DEVICE_FFT_DOUBLEREAL *) outputRealFFTXHostPtr,
                               (DEVICE_FFT_DOUBLEREAL *) outputRealVendorHostPtr,
                               (long) npts );
        }

  #elif defined (FFTX_SYCL)
	// If this is absent then iterations after the first aren't correct.
        sycl::host_accessor tempHostAcc(tempBufTfmPtr);
	for (int ind = 0; ind < nptsTrunc; ind++)
	  {
	    inputComplexVendorPtr[ind] = outputComplexFFTXHostPtr[ind];
          }
	for (int ind = 0; ind < npts; ind++)
	  {
	    outputRealFFTXHostPtr[ind] = outputHostAcc[ind];
	  }

   	auto start_time = std::chrono::high_resolution_clock::now();
         // Run the vendor FFT plan on the same input data.
	// Perform backward transform on complex array
	oneapi::mkl::dft::compute_backward(transform_plan_3d, inputComplexVendorPtr, outputRealVendorPtr).wait();
        auto end_time = std::chrono::high_resolution_clock::now();

    	std::chrono::duration<float, std::milli> duration = end_time - start_time;
	imdprdft_vendor_millisec[itn] = duration.count();

  	for (int ind = 0; ind < npts; ind++)
	  {
	    outputRealVendorHostPtr[ind] = outputRealVendorPtr[ind];
	  }

        printf ( "cube = [ %d, %d, %d ]\tIMDPRDFT (Inverse)\t", mm, nn, kk );

        checkOutputs_C2R ( (double*) outputRealFFTXHostPtr,
                           (double*) outputRealVendorHostPtr,
                           (long) npts );
      
#endif
      }

#if defined(FFTX_SYCL)
    // Clean up.
    // sycl::free(inbuf, sycl_context);
    // sycl::free(outbuf, sycl_context);
    sycl::free(inputRealVendorPtr, sycl_context);
    sycl::free(outputComplexVendorPtr, sycl_context);
    sycl::free(inputComplexVendorPtr, sycl_context);
    sycl::free(outputRealVendorPtr, sycl_context);
#endif
    

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
    printf ( "Times in milliseconds for %s on MDPRDFT (forward) for %d trials of size %d %d %d:\n",
	     descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2) );
    printf ( "Trial#    Spiral           %s\n", vendorfft.c_str() );
    for (int itn = 0; itn < iterations; itn++)
      {
        printf ( "%4d%17.7e%17.7e\n", itn+1, mdprdft_gpu[itn], mdprdft_vendor_millisec[itn] );
      }

    printf ( "Times in milliseconds for %s on MDPRDFT (inverse) for %d trials of size %d %d %d:\n",
	     descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2) );
    printf ( "Trial#    Spiral           %s\n", vendorfft.c_str() );
    for (int itn = 0; itn < iterations; itn++)
      {
        printf ( "%4d%17.7e%17.7e\n", itn+1, imdprdft_gpu[itn], imdprdft_vendor_millisec[itn] );
      }
#else
    printf ( "Times in milliseconds for %s on MDPRDFT (forward) for %d trials of size %d %d %d\n",
	     descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
    printf ( "Trial#    Spiral\n");
    for (int itn = 0; itn < iterations; itn++)
      {
        printf ( "%4d%17.7e\n", itn+1, mdprdft_gpu[itn] );
      }

    printf ( "Times in milliseconds for %s on MDPRDFT (inverse) for %d trials of size %d %d %d\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
    printf ( "Trial#    Spiral\n");
    for (int itn = 0; itn < iterations; itn++)
      {
	printf ( "%4d%17.7e\n", itn+1, imdprdft_gpu[itn] );
      }
#endif

    // for(int i = 0; i < outputComplexVendorHostArray.m_domain.size(); i++) {
    //     std::cout << outputComplexFFTXHostPtr[i] << " " << outputComplexVendorHostPtr[i] << std::endl;
    // }

    delete[] mdprdft_gpu;
    delete[] imdprdft_gpu;
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
    delete[] mdprdft_vendor_millisec;
    delete[] imdprdft_vendor_millisec;
#endif
    printf("%s: All done, exiting\n", prog);
  
    return 0;

}
