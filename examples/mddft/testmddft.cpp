#include "fftx3.hpp"
#include "fftx3utilities.h"
#include "interface.hpp"
#include "mddftObj.hpp"
#include "imddftObj.hpp"
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

//  Build a random input buffer for Spiral and vendor FFT
//  host_X is the host buffer to setup -- it'll be copied to the device later
//  sizes is a vector with the X, Y, & Z dimensions
static void setInput ( double *host_X, std::vector<int> sizes )
{
    for ( int imm = 0; imm < sizes.at(0); imm++ ) {
        for ( int inn = 0; inn < sizes.at(1); inn++ ) {
            for ( int ikk = 0; ikk < sizes.at(2); ikk++ ) {
                int offset = (ikk + inn * sizes.at(2) + imm * sizes.at(1) * sizes.at(2)) * 2;
#if defined(FFTX_HIP) || defined(FFTX_CUDA) || defined(FFTX_SYCL)
                host_X[offset + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
                host_X[offset + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
#else // CPU
                host_X[offset + 0] = 1.;
                host_X[offset + 1] = 1.;
#endif
            }
        }
    }
    return;
}

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#define DOUBLECOMPLEX DEVICE_FFT_DOUBLECOMPLEX
#define REALPART(z) z.x
#define IMAGPART(z) z.y
#elif defined (FFTX_SYCL)
#define DOUBLECOMPLEX std::complex<double>
#define REALPART(z) z.real()
#define IMAGPART(z) z.imag()
#endif

// Check that the buffer are identical (within roundoff)
// outputFFTXPtr is the output buffer from the Spiral-generated transform
// (result on GPU copied to host array outputFFTXPtr);
// outputVendorPtr is the output buffer from the vendor transform
// (result on GPU copied to host array outputVendorPtr).
// arrsz is the size of each array
static void checkOutputs ( DOUBLECOMPLEX *outputFFTXPtr,
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

    std::cout << mm << " " << nn << " " << kk << std::endl;
    std::vector<int> sizes{mm,nn,kk};
    fftx::box_t<3> domain ( point_t<3> ( { { 1, 1, 1 } } ),
                            point_t<3> ( { { mm, nn, kk } } ));

    fftx::array_t<3,std::complex<double>> inputHostArray(domain);
    fftx::array_t<3,std::complex<double>> outputFFTXHostArray(domain);
    fftx::array_t<3,std::complex<double>> symbolHostArray(domain);
    fftx::array_t<3,std::complex<double>> outputVendorHostArray(domain);

    size_t npts = domain.size();
    size_t bytes = npts * sizeof(std::complex<double>);

    auto inputHostPtr = inputHostArray.m_data.local();
    auto outputFFTXHostPtr = outputFFTXHostArray.m_data.local();
    auto symbolHostPtr = symbolHostArray.m_data.local();
    auto outputVendorHostPtr = outputVendorHostArray.m_data.local();

#if defined FFTX_CUDA
    CUdevice cuDevice;
    CUcontext context;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    //  CUdeviceptr  inputTfmPtr, outputTfmPtr, symbolTfmPtr;
    std::complex<double> *inputTfmPtr, *outputTfmPtr, *symbolTfmPtr;
#elif defined FFTX_HIP
    hipDeviceptr_t  inputTfmPtr, outputTfmPtr, symbolTfmPtr;
#elif defined FFTX_SYCL
#else  
    double * inputTfmPtr, *outputTfmPtr, *symbolTfmPtr;
#endif

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    std::cout << "allocating memory\n";
    DEVICE_MALLOC((void **)&inputTfmPtr, bytes);
    if ( DEBUGOUT ) std::cout << "allocated inputTfmPtr on device\n";

    DEVICE_MALLOC((void **)&outputTfmPtr, bytes);
    if ( DEBUGOUT ) std::cout << "allocated outputTfmPtr on device\n";

    DEVICE_MALLOC((void **)&symbolTfmPtr, bytes);
#elif defined (FFTX_SYCL)
    // If you do sycl::buffer<std::complex<double>> then you need npts * 2.
    sycl::buffer<double> outputTfmPtr((double*) outputFFTXHostPtr, npts * 2);
    sycl::buffer<double> inputTfmPtr((double*) inputHostPtr, npts * 2);
    sycl::buffer<double> symbolTfmPtr((double*) symbolHostPtr, npts * 2);
#else // CPU
    inputTfmPtr = (double *) inputHostPtr;
    outputTfmPtr = (double *) outputFFTXHostPtr;
    symbolTfmPtr = new double[npts];
#endif

#if defined FFTX_CUDA
    std::vector<void*> args{&outputTfmPtr, &inputTfmPtr, &symbolTfmPtr};
    std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
    std::string vendorfft  = "cufft";
#elif defined FFTX_HIP
    std::vector<void*> args{outputTfmPtr, inputTfmPtr, symbolTfmPtr};
    std::string descrip = "AMD GPU";                //  "CPU and GPU";
    std::string vendorfft  = "rocfft";
#elif defined FFTX_SYCL
    std::vector<void*> args{(void*)&(outputTfmPtr), (void*)&(inputTfmPtr), (void*)&(symbolTfmPtr)};
    std::string descrip = "Intel GPU";                //  "CPU and GPU";
    std::string vendorfft  = "mklfft";
#else // CPU
    std::vector<void*> args{(void*)outputTfmPtr, (void*)inputTfmPtr, (void*)symbolTfmPtr};
    std::string descrip = "CPU";                //  "CPU";
    // std::string vendorfft = "fftw";
#endif

    // double* mddft_cpu = new double[iterations];
    // double* imddft_cpu = new double[iterations];
    // #if defined (FFTX_CUDA) || defined(FFTX_HIP)
    // additional code for GPU programs
    float *mddft_gpu = new float[iterations];
    float *imddft_gpu = new float[iterations];
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
    float *mddft_vendor_millisec = new float[iterations];
    float *imddft_vendor_millisec = new float[iterations];
    bool check_output = true; // compare results of Spiral-RTC with vendor FFT
#endif
    
    MDDFTProblem mdp(args, sizes, "mddft");

    //  Set up a plan to run the transform using vendor FFT.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    DEVICE_FFT_HANDLE plan;
    DEVICE_FFT_RESULT res;
    DEVICE_FFT_TYPE   xfmtype = DEVICE_FFT_Z2Z ;
    DEVICE_EVENT_T custart, custop;
    DEVICE_EVENT_CREATE ( &custart );
    DEVICE_EVENT_CREATE ( &custop );
    
    res = DEVICE_FFT_PLAN3D ( &plan, mm, nn, kk, xfmtype );
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
	  
    auto inputVendorPtr = sycl::malloc_shared< std::complex<double> >
      (npts, sycl_device, sycl_context);
    auto outputVendorPtr = sycl::malloc_shared< std::complex<double> >
      (npts, sycl_device, sycl_context);
    
    // Initialize 3D FFT descriptor
    std::vector<std::int64_t> Nvec{mm, nn, kk};
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
				 oneapi::mkl::dft::domain::COMPLEX>
      transform_plan_3d(Nvec);
    transform_plan_3d.commit(Q);
#endif

    // double *hostinp = (double *) inputHostPtr;
    for (int itn = 0; itn < iterations; itn++)
      {
        // Set up random data for input buffer.
	// (Use different randomized data each iteration.)

	// setInput ( hostinp, sizes );
	setInput ( (double*) inputHostPtr, sizes );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
	DEVICE_MEM_COPY(inputTfmPtr, inputHostPtr,
                        bytes, MEM_COPY_HOST_TO_DEVICE);
#endif
	if ( DEBUGOUT ) std::cout << "copied input from host to device\n";

	// Run transform: input inputTfmPtr, output outputTfmPtr.
	mdp.transform();
	mddft_gpu[itn] = mdp.getTime();
	// gatherOutput(outputFFTXHostArray, args);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
	DEVICE_MEM_COPY ( outputFFTXHostPtr, outputTfmPtr,
                          bytes, MEM_COPY_DEVICE_TO_HOST );
        if ( check_output )
	  { //  Run the vendor FFT plan on the same input data.
	    DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECZ2Z ( plan,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) inputTfmPtr,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) outputTfmPtr,
                                       DEVICE_FFT_FORWARD );
            if ( res != DEVICE_FFT_SUCCESS)
	      {
		printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
		check_output = false;
		//  break;
	      }
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &mddft_vendor_millisec[itn], custart, custop );

            DEVICE_MEM_COPY ( outputVendorHostPtr, outputTfmPtr,
                              bytes, MEM_COPY_DEVICE_TO_HOST );
            printf ( "cube = [ %d, %d, %d ]\tMDDFT (Forward)\t", mm, nn, kk );
            checkOutputs ( (DEVICE_FFT_DOUBLECOMPLEX *) outputFFTXHostPtr,
			   (DEVICE_FFT_DOUBLECOMPLEX *) outputVendorHostPtr,
			   (long) npts );
	  }
#elif defined (FFTX_SYCL)
	// If this is absent then iterations after the first aren't correct.
	sycl::host_accessor inputHostAcc(inputTfmPtr);

	// outputHostAcc is double* because outputTfmPtr is sycl::buffer<double>.
	sycl::host_accessor outputHostAcc(outputTfmPtr);
	for (int ind = 0; ind < npts; ind++)
	  {
	    inputVendorPtr[ind] = inputHostPtr[ind];
	    // outputFFTXHostPtr[ind] = outputHostAcc[ind];
	    outputFFTXHostPtr[ind] =
	      std::complex(outputHostAcc[2*ind], outputHostAcc[2*ind+1]);
	  }
	
	auto start_time = std::chrono::high_resolution_clock::now();
        //  Run the vendor FFT plan on the same input data.
	// Perform forward transform on complex array
	oneapi::mkl::dft::compute_forward(transform_plan_3d, inputVendorPtr, outputVendorPtr).wait();
	auto end_time = std::chrono::high_resolution_clock::now();

	std::chrono::duration<float, std::milli> duration = end_time - start_time;
	mddft_vendor_millisec[itn] = duration.count();
      
	for (int ind = 0; ind < npts; ind++)
	  {
	    outputVendorHostPtr[ind] = outputVendorPtr[ind];
	  }
      
	printf ( "cube = [ %d, %d, %d ]\tMDDFT (Forward)\t", mm, nn, kk );
	if (check_output)
	  {
	    checkOutputs ( (std::complex<double>*) outputFFTXHostPtr,
			   (std::complex<double>*) outputVendorHostPtr,
			   npts );
	  }
#endif
      }

    // setup the inverse transform (we'll reuse the vendor FFT plan already created)
    IMDDFTProblem imdp(args, sizes, "imddft");

    for (int itn = 0; itn < iterations; itn++)
      {
        // Set up random data for input buffer.
	// (Use different randomized data each iteration.)

	setInput ( (double*) inputHostPtr, sizes );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY(inputTfmPtr, inputHostPtr,
                        bytes, MEM_COPY_HOST_TO_DEVICE);
#endif
        if ( DEBUGOUT ) std::cout << "copied input from host to device\n";
        
	// Run transform: input inputTfmPtr, output outputTfmPtr.
        imdp.transform();
        imddft_gpu[itn] = imdp.getTime();
        //gatherOutput(outputFFTXHostArray, args);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY ( outputFFTXHostPtr, outputTfmPtr,
                          bytes, MEM_COPY_DEVICE_TO_HOST );
        if ( check_output )
	  { // Run the vendor FFT plan on the same input data.
            DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECZ2Z ( plan,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) inputTfmPtr,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) outputTfmPtr,
                                       DEVICE_FFT_INVERSE );
            if ( res != DEVICE_FFT_SUCCESS)
	      {
		printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
                check_output = false;
                //  break;
	      }
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &imddft_vendor_millisec[itn], custart, custop );

            DEVICE_MEM_COPY ( outputVendorHostPtr, outputTfmPtr,
                              bytes, MEM_COPY_DEVICE_TO_HOST );
            printf ( "cube = [ %d, %d, %d ]\tMDDFT (Inverse)\t", mm, nn, kk );
            checkOutputs ( (DEVICE_FFT_DOUBLECOMPLEX *) outputFFTXHostPtr,
			   (DEVICE_FFT_DOUBLECOMPLEX *) outputVendorHostPtr,
			   (long) npts );
        }
#elif defined (FFTX_SYCL)
	// If this is absent then iterations after the first aren't correct.
	sycl::host_accessor inputHostAcc(inputTfmPtr);

	// outputHostAcc is double* because outputTfmPtr is sycl::buffer<double>.
	sycl::host_accessor outputHostAcc(outputTfmPtr);
	for (int ind = 0; ind < npts; ind++)
	  {
	    inputVendorPtr[ind] = inputHostPtr[ind];
	    // outputFFTXHostPtr[ind] = outputHostAcc[ind];
	    outputFFTXHostPtr[ind] =
	      std::complex(outputHostAcc[2*ind], outputHostAcc[2*ind+1]);
	  }
	
	auto start_time = std::chrono::high_resolution_clock::now();
        // Run the vendor FFT plan on the same input data.
	// Perform backward transform on complex array
	oneapi::mkl::dft::compute_backward(transform_plan_3d, inputVendorPtr, outputVendorPtr).wait();
	auto end_time = std::chrono::high_resolution_clock::now();

	std::chrono::duration<float, std::milli> duration = end_time - start_time;
	imddft_vendor_millisec[itn] = duration.count();

	for (int ind = 0; ind < npts; ind++)
	  {
	    outputVendorHostPtr[ind] = outputVendorPtr[ind];
	  }
      
	printf ( "cube = [ %d, %d, %d ]\tMDDFT (Inverse)\t", mm, nn, kk );
	if (check_output)
	  {
	    checkOutputs ( (std::complex<double>*) outputFFTXHostPtr,
			   (std::complex<double>*) outputVendorHostPtr,
			   (long) npts );
	  }
#endif
    }

#if defined(FFTX_SYCL)
    // Clean up.
    // sycl::free(inbuf, sycl_context);
    // sycl::free(outbuf, sycl_context);
    sycl::free(inputVendorPtr, sycl_context);
    sycl::free(outputVendorPtr, sycl_context);
#endif
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
    printf ( "Times in milliseconds for %s on MDDFT (forward) for %d trials of size %d %d %d:\n",
	     descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2) );
    printf ( "Trial#    Spiral           %s\n", vendorfft.c_str() );
    for (int itn = 0; itn < iterations; itn++)
      {
        printf ( "%4d%17.7e%17.7e\n", itn+1, mddft_gpu[itn], mddft_vendor_millisec[itn] );
      }

    printf ( "Times in milliseconds for %s on MDDFT (inverse) for %d trials of size %d %d %d:\n",
	     descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2) );
    printf ( "Trial#    Spiral           %s\n", vendorfft.c_str() );
    for (int itn = 0; itn < iterations; itn++)
      {
        printf ( "%4d%17.7e%17.7e\n", itn+1, imddft_gpu[itn], imddft_vendor_millisec[itn] );
      }
#else
    printf ( "Times in milliseconds for %s on MDDFT (forward) for %d trials of size %d %d %d\n",
	     descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
    printf ( "Trial#    Spiral\n");
    for (int itn = 0; itn < iterations; itn++)
      {
        printf ( "%4d%17.7e\n", itn+1, mddft_gpu[itn] );
      }

    printf ( "Times in milliseconds for %s on MDDFT (inverse) for %d trials of size %d %d %d\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
    printf ( "Trial#    Spiral\n");
    for (int itn = 0; itn < iterations; itn++)
      {
	printf ( "%4d%17.7e\n", itn+1, imddft_gpu[itn] );
      }
#endif

    // delete[] mddft_cpu;
    // delete[] imddft_cpu;
    delete[] mddft_gpu;
    delete[] imddft_gpu;
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
    delete[] mddft_vendor_millisec;
    delete[] imddft_vendor_millisec;
#endif
    printf("%s: All done, exiting\n", prog);
  
    return 0;
}
