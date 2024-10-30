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
//  inputPtr is the host buffer to setup -- it'll be copied to the device later
//  sizes is a vector with the X, Y, & Z dimensions
static void setInput ( double *inputPtr, std::vector<int> sizes )
{
    for ( int imm = 0; imm < sizes.at(0); imm++ ) {
        for ( int inn = 0; inn < sizes.at(1); inn++ ) {
            for ( int ikk = 0; ikk < sizes.at(2); ikk++ ) {
                int offset = (ikk + inn * sizes.at(2) + imm * sizes.at(1) * sizes.at(2)) * 2;
#if defined(FFTX_HIP) || defined(FFTX_CUDA) || defined(FFTX_SYCL)
                inputPtr[offset + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
                inputPtr[offset + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
#else // CPU
                inputPtr[offset + 0] = 1.;
                inputPtr[offset + 1] = 1.;
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

    fftx::OutStream() << "Correct: " << (correct ? "True" : "False") << "\t"
                      << "Max delta = " << std::scientific << maxdelta
                      << std::endl;
    std::flush(fftx::OutStream());

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
            fftx::OutStream() << "Usage: " << argv[0]
                              << ": [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]"
                              << std::endl;
            exit (0);
        default:
            fftx::OutStream() << prog << ": unknown argument: "
                              << argv[1] << " ... ignored" << std::endl;
        }
        argv++, argc--;
    }

    fftx::OutStream() << mm << " " << nn << " " << kk << std::endl;
    std::vector<int> sizes{mm,nn,kk};
    fftx::box_t<3> domain ( point_t<3> ( { { 1, 1, 1 } } ),
                            point_t<3> ( { { mm, nn, kk } } ));

    fftx::array_t<3,std::complex<double>> inputHostArray(domain);
    fftx::array_t<3,std::complex<double>> outputFFTXHostArray(domain);
    // fftx::array_t<3,std::complex<double>> symbolHostArray(domain);
    fftx::array_t<3,std::complex<double>> outputVendorHostArray(domain);
    /*
    fftx::box_t<2> domain ( point_t<2> ( { { 1, 1 } } ),
                            point_t<2> ( { { mm, nn } } ));

    fftx::array_t<2,std::complex<double>> inputHostArray(domain);
    fftx::array_t<2,std::complex<double>> outputFFTXHostArray(domain);
    // fftx::array_t<2,std::complex<double>> symbolHostArray(domain);
    fftx::array_t<2,std::complex<double>> outputVendorHostArray(domain);
    */

    size_t npts = domain.size();
    size_t bytes = npts * sizeof(std::complex<double>);

    auto inputHostPtr = inputHostArray.m_data.local();
    auto outputFFTXHostPtr = outputFFTXHostArray.m_data.local();
    // auto symbolHostPtr = symbolHostArray.m_data.local();
    auto outputVendorHostPtr = outputVendorHostArray.m_data.local();

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    DEVICE_PTR inputTfmPtr, outputTfmPtr, symbolTfmPtr;
    fftx::OutStream() << "allocating memory\n";
    DEVICE_MALLOC((void **)&inputTfmPtr, bytes);
    if ( DEBUGOUT ) fftx::OutStream() << "allocated inputTfmPtr on device\n";

    DEVICE_MALLOC((void **)&outputTfmPtr, bytes);
    if ( DEBUGOUT ) fftx::OutStream() << "allocated outputTfmPtr on device\n";

    // DEVICE_MALLOC((void **)&symbolTfmPtr, bytes);
    symbolTfmPtr = (DEVICE_PTR) NULL;
#elif defined (FFTX_SYCL)
    // If you do sycl::buffer<std::complex<double>> then you need npts * 2.
    sycl::buffer<double> inputTfmPtr((double*) inputHostPtr, npts * 2);
    sycl::buffer<double> outputTfmPtr((double*) outputFFTXHostPtr, npts * 2);
    sycl::buffer<double> symbolTfmPtr((double*) NULL, 0); // not needed
#else // CPU
    double* inputTfmPtr = (double *) inputHostPtr;
    double* outputTfmPtr = (double *) outputFFTXHostPtr;
    // double* symbolTfmPtr = new double[npts];
    double* symbolTfmPtr = (double *) NULL;
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

    fftx::OutStream() << std::scientific
                      << std::uppercase
                      << std::setprecision(7);
    
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
        fftx::OutStream() << "Create DEVICE_FFT_PLAN3D failed with error code "
                          << res << " ... skip buffer check" << std::endl;
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
	fftx::OutStream() << "You are running on a system without a GPU. For best results please use a GPU." << std::endl;
	fftx::OutStream() << "Program terminating." << std::endl;
	exit(-1);
	// dev = sycl::device(sycl::cpu_selector_v);
      }
    sycl::context ctx = sycl::context(dev);
    /*
    cl_device_id ocl_dev =
      sycl::get_native<cl::sycl::backend::opencl, sycl::device>(dev);
    cl_context   ocl_ctx =
      sycl::get_native<cl::sycl::backend::opencl, sycl::context>(ctx);
    cl_int err = CL_SUCCESS;
    cl_command_queue ocl_queue =
      clCreateCommandQueueWithProperties(ocl_ctx, ocl_dev,0,&err);
    sycl::queue Q = sycl::make_queue<sycl::backend::opencl>(ocl_queue,ctx);
    */
    sycl::property_list props{sycl::property::queue::enable_profiling()};
    sycl::queue Q = sycl::queue(ctx, dev, props);

    // Initialize SYCL queue
    //	  sycl::queue Q(sycl::default_selector{});
    auto sycl_device = Q.get_device();
    auto sycl_context = Q.get_context();
    fftx::OutStream() << "Running on: "
                      << Q.get_device().get_info<sycl::info::device::name>()
                      << std::endl;
	  
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
	DEVICE_MEM_COPY((void*)inputTfmPtr, inputHostPtr,
                        bytes, MEM_COPY_HOST_TO_DEVICE);
#endif
	if ( DEBUGOUT ) fftx::OutStream() << "copied input from host to device\n";

	// Run transform: input inputTfmPtr, output outputTfmPtr.
	mdp.transform();
	mddft_gpu[itn] = mdp.getTime();
	// gatherOutput(outputFFTXHostArray, args);

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
        if ( check_output )
	  { //  Run the vendor FFT plan on the same input data.	
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against cuFFT or rocFFT.
	    DEVICE_MEM_COPY ( outputFFTXHostPtr, (void*)outputTfmPtr,
			      bytes, MEM_COPY_DEVICE_TO_HOST );

	    // Run cuFFT or rocFFT.
	    DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECZ2Z ( plan,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) inputTfmPtr,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) outputTfmPtr,
                                       DEVICE_FFT_FORWARD );
            if ( res != DEVICE_FFT_SUCCESS)
	      {
                fftx::OutStream() << "Launch DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
		check_output = false;
		//  break;
	      }
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &mddft_vendor_millisec[itn], custart, custop );

            DEVICE_MEM_COPY ( outputVendorHostPtr, (void*)outputTfmPtr,
                              bytes, MEM_COPY_DEVICE_TO_HOST );
#elif defined (FFTX_SYCL)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against MKL FFT.

	    // If this is absent then iterations after the first aren't correct.
	    sycl::host_accessor inputHostAcc(inputTfmPtr);

	    // outputHostAcc is double* because outputTfmPtr is sycl::buffer<double>.
	    sycl::host_accessor outputHostAcc(outputTfmPtr);
	    for (int ind = 0; ind < npts; ind++)
	      {
		// outputFFTXHostPtr[ind] = outputHostAcc[ind];
		outputFFTXHostPtr[ind] =
		  std::complex(outputHostAcc[2*ind], outputHostAcc[2*ind+1]);
	      }
	
	    // Run MKL FFT plan on the same input data.
	    for (int ind = 0; ind < npts; ind++)
	      { // These are both complex.
		inputVendorPtr[ind] = inputHostPtr[ind];
	      }
	    auto start_time = std::chrono::high_resolution_clock::now();
	    // Perform forward transform on complex array
            // oneapi::mkl::dft::compute_forward(transform_plan_3d, inputVendorPtr, outputVendorPtr).wait();
            sycl::event e = oneapi::mkl::dft::compute_forward(transform_plan_3d,
                                                              inputVendorPtr,
                                                              outputVendorPtr);
            Q.wait();
	    auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t profile_nanosec = e_end - e_start;
            // std::chrono::duration<float, std::milli> duration = end_time - start_time;
	    // mddft_vendor_millisec[itn] = duration.count();
            mddft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds
      
	    for (int ind = 0; ind < npts; ind++)
	      { // These are both complex.
		outputVendorHostPtr[ind] = outputVendorPtr[ind];
	      }
#endif
	    // printf ( "cube = [ %d, %d, %d ]\tMDDFT (Inverse)\t", mm, nn, kk );
            fftx::OutStream() << "cube = [ "
                              << mm << ", " << nn << ", " << kk << " ]\t"
                              << "MDDFT (Forward) \t";
	    checkOutputs ( (DOUBLECOMPLEX*) outputFFTXHostPtr,
			   (DOUBLECOMPLEX*) outputVendorHostPtr,
			   (long) npts );
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
        DEVICE_MEM_COPY((void*)inputTfmPtr, inputHostPtr,
                        bytes, MEM_COPY_HOST_TO_DEVICE);
#endif
        if ( DEBUGOUT ) fftx::OutStream() << "copied input from host to device\n";
        
	// Run transform: input inputTfmPtr, output outputTfmPtr.
        imdp.transform();
        imddft_gpu[itn] = imdp.getTime();
        //gatherOutput(outputFFTXHostArray, args);

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)	
        if ( check_output )
	  { // Run the vendor FFT plan on the same input data.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against cuFFT or rocFFT.
	    DEVICE_MEM_COPY ( outputFFTXHostPtr, (void*)outputTfmPtr,
			      bytes, MEM_COPY_DEVICE_TO_HOST );

	    // Run cuFFT or rocFFT.	    
	    DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECZ2Z ( plan,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) inputTfmPtr,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) outputTfmPtr,
                                       DEVICE_FFT_INVERSE );
            if ( res != DEVICE_FFT_SUCCESS)
	      {
                fftx::OutStream() << "Launch DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
                check_output = false;
                //  break;
	      }

            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &imddft_vendor_millisec[itn], custart, custop );

            DEVICE_MEM_COPY ( outputVendorHostPtr, (void*)outputTfmPtr,
                              bytes, MEM_COPY_DEVICE_TO_HOST );
#elif defined (FFTX_SYCL)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against MKL FFT.

	    // If this is absent then iterations after the first aren't correct.
	    sycl::host_accessor inputHostAcc(inputTfmPtr);

	    // outputHostAcc is double* because outputTfmPtr is sycl::buffer<double>.
	    sycl::host_accessor outputHostAcc(outputTfmPtr);
	    for (int ind = 0; ind < npts; ind++)
	      {
		// outputFFTXHostPtr[ind] = outputHostAcc[ind];
		outputFFTXHostPtr[ind] =
		  std::complex(outputHostAcc[2*ind], outputHostAcc[2*ind+1]);
	      }

	    // Run MKL FFT plan on the same input data.
	    for (int ind = 0; ind < npts; ind++)
	      { // These are both complex.
		inputVendorPtr[ind] = inputHostPtr[ind];
	      }
	    auto start_time = std::chrono::high_resolution_clock::now();
	    // Perform backward transform on complex array
            // oneapi::mkl::dft::compute_backward(transform_plan_3d, inputVendorPtr, outputVendorPtr).wait();
            sycl::event e = oneapi::mkl::dft::compute_backward(transform_plan_3d,
                                                               inputVendorPtr,
                                                               outputVendorPtr);
            Q.wait();
	    auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t profile_nanosec = e_end - e_start;
	    // std::chrono::duration<float, std::milli> duration = end_time - start_time;
	    // imddft_vendor_millisec[itn] = duration.count();
            imddft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

	    for (int ind = 0; ind < npts; ind++)
	      { // These are both complex.
		outputVendorHostPtr[ind] = outputVendorPtr[ind];
	      }
#endif
	    // printf ( "cube = [ %d, %d, %d ]\tIMDDFT (Inverse)\t", mm, nn, kk );
            fftx::OutStream() << "cube = [ "
                              << mm << ", " << nn << ", " << kk << " ]\t"
                              << "IMDDFT (Inverse)\t";
	    checkOutputs ( (DOUBLECOMPLEX*) outputFFTXHostPtr,
			   (DOUBLECOMPLEX*) outputVendorHostPtr,
			   (long) npts );
	  }
#endif
      }

    // Clean up.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    DEVICE_FREE((void*)inputTfmPtr);
    DEVICE_FREE((void*)outputTfmPtr);
    DEVICE_FREE((void*)symbolTfmPtr);
#elif defined(FFTX_SYCL)
    sycl::free(inputVendorPtr, sycl_context);
    sycl::free(outputVendorPtr, sycl_context);
#else
    delete[] symbolTfmPtr;
#endif
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on MDDFT (forward) for " << iterations
                      << " trials of size "
                      << sizes.at(0) << " "
                      << sizes.at(1) << " "
                      << sizes.at(2) << ":" << std::endl;
    fftx::OutStream() << "Trial#    Spiral           " << vendorfft << std::endl;
    for (int itn = 0; itn < iterations; itn++)
      {
        // printf ( "%4d%17.7e%17.7e\n", itn+1, mddft_gpu[itn], mddft_vendor_millisec[itn] );
        fftx::OutStream() << std::setw(4) << (itn+1)
                          << std::setw(17) << mddft_gpu[itn]
                          << std::setw(17) << mddft_vendor_millisec[itn]
                          << std::endl;
      }

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on MDDFT (inverse) for " << iterations
                      << " trials of size "
                      << sizes.at(0) << " "
                      << sizes.at(1) << " "
                      << sizes.at(2) << ":" << std::endl;
    // printf ( "Trial#    Spiral           %s\n", vendorfft.c_str() );
    fftx::OutStream() << "Trial#    Spiral           " << vendorfft
                      << std::endl;
    for (int itn = 0; itn < iterations; itn++)
      {
        // / printf ( "%4d%17.7e%17.7e\n", itn+1, imddft_gpu[itn], imddft_vendor_millisec[itn] );
        fftx::OutStream() << std::setw(4) << (itn+1)
                          << std::setw(17) << imddft_gpu[itn]
                          << std::setw(17) << imddft_vendor_millisec[itn]
                          << std::endl;
      }

    delete[] mddft_vendor_millisec;
    delete[] imddft_vendor_millisec;
#else
    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on MDDFT (forward) for "
                      << iterations << " trials of size "
                      << sizes.at(0) << " "
                      << sizes.at(1) << " "
                      << sizes.at(2) << std::endl;
    fftx::OutStream() << "Trial#    Spiral" << std::endl;
    for (int itn = 0; itn < iterations; itn++)
      {
        // printf ( "%4d%17.7e\n", itn+1, mddft_gpu[itn] );
        fftx::OutStream() << std::setw(4) << (itn+1)
                          << std::setw(17) << mddft_gpu[itn]
                          << std::endl;
      }

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on MDDFT (inverse) for "
                      << iterations << " trials of size "
                      << sizes.at(0) << " "
                      << sizes.at(1) << " "
                      << sizes.at(2) << std::endl;
    fftx::OutStream() << "Trial#    Spiral" << std::endl;
    for (int itn = 0; itn < iterations; itn++)
      {
	// printf ( "%4d%17.7e\n", itn+1, imddft_gpu[itn] );
        fftx::OutStream() << std::setw(4) << (itn+1)
                          << std::setw(17) << imddft_gpu[itn]
                          << std::endl;
      }
#endif
    delete[] mddft_gpu;
    delete[] imddft_gpu;

    // printf("%s: All done, exiting\n", prog);
    fftx::OutStream() << prog << ": All done, exiting" << std::endl;
    std::flush(fftx::OutStream());

    return 0;
}
