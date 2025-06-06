//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.
//

#include "fftx.hpp"
#include "fftxutilities.hpp"
#include "fftxinterface.hpp"
#include "fftxmddftObj.hpp"
#include "fftximddftObj.hpp"
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
#elif defined (FFTX_SYCL)
// #include "mkl_dfti.h"
// #include <sycl/sycl.hpp>
#include <oneapi/mkl/dft.hpp>
// #include <oneapi/mkl/vm.hpp>
#elif defined (FFTX_USE_FFTW)
#include "fftw3.h"
#endif

static void setRandomData ( double *data, long arrsz)
{
  for ( int ind = 0; ind < arrsz; ind++)
    {
      data[ind] = ((double) rand()) / (double) (RAND_MAX/2);
    }
  return;
}

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#define FFTX_DOUBLECOMPLEX FFTX_DEVICE_FFT_DOUBLECOMPLEX
#define FFTX_REALPART(z) z.x
#define FFTX_IMAGPART(z) z.y
#elif defined (FFTX_SYCL)
#define FFTX_DOUBLECOMPLEX std::complex<double>
#define FFTX_REALPART(z) z.real()
#define FFTX_IMAGPART(z) z.imag()
#elif defined (FFTX_USE_FFTW)
#define FFTX_DOUBLECOMPLEX fftw_complex
#define FFTX_REALPART(z) z[0]
#define FFTX_IMAGPART(z) z[1]
#else // need these #defines here, or else #ifdef around checkOutputs.
#define FFTX_DOUBLECOMPLEX std::complex<double>
#define FFTX_REALPART(z) z.real()
#define FFTX_IMAGPART(z) z.imag()
#endif

// Check that the buffers are identical (within roundoff)
// outputFFTXPtr is the output buffer from the Spiral-generated transform
// (result on GPU copied to host array outputFFTXPtr);
// outputVendorPtr is the output buffer from the vendor transform
// (result on GPU copied to host array outputVendorPtr).
// arrsz is the size of each array
static bool checkOutputs ( FFTX_DOUBLECOMPLEX *outputFFTXPtr,
			   FFTX_DOUBLECOMPLEX *outputVendorPtr,
			   long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int ind = 0; ind < arrsz; ind++ )
      {
        double sreal = FFTX_REALPART(outputFFTXPtr[ind]);
        double simag = FFTX_IMAGPART(outputFFTXPtr[ind]);
        double creal = FFTX_REALPART(outputVendorPtr[ind]);
        double cimag = FFTX_IMAGPART(outputVendorPtr[ind]);
        
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

    return correct;
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
    int status = 0;
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

    auto inputHostPtr = inputHostArray.m_data.local();
    auto outputFFTXHostPtr = outputFFTXHostArray.m_data.local();
    // auto symbolHostPtr = symbolHostArray.m_data.local();
    auto outputVendorHostPtr = outputVendorHostArray.m_data.local();

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_PTR inputTfmPtr = fftxDeviceMallocForHostArray(inputHostArray);
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "allocated inputTfmPtr on device\n";

    FFTX_DEVICE_PTR outputTfmPtr = fftxDeviceMallocForHostArray(outputFFTXHostArray);
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "allocated outputTfmPtr on device\n";

    FFTX_DEVICE_PTR symbolTfmPtr = (FFTX_DEVICE_PTR) NULL;
#elif defined (FFTX_SYCL)
    // If you do sycl::buffer<std::complex<double>> then you need npts * 2.
    sycl::buffer<double> inputTfmPtr((double*) inputHostPtr, npts * 2);
    sycl::buffer<double> outputTfmPtr((double*) outputFFTXHostPtr, npts * 2);
    sycl::buffer<double> symbolTfmPtr((double*) NULL, 0); // not needed
#else // CPU
    double* inputTfmPtr = (double *) inputHostPtr;
    double* outputTfmPtr = (double *) outputFFTXHostPtr;
    double* symbolTfmPtr = (double *) NULL;
#endif

    // Order within args:  output, input, symbol.
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
    std::string vendorfft = "FFTW";
#endif

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL) || defined(FFTX_USE_FFTW)
    // compare results of Spiral-RTC with vendor FFT or FFTW
    bool check_output = true;
#else
    bool check_output = false;
#endif

    fftx::OutStream() << std::scientific
                      << std::uppercase
                      << std::setprecision(7);
    
    MDDFTProblem mdp(args, sizes, "mddft");

    //  Set up a plan to run the transform using vendor FFT.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_HANDLE plan;
    FFTX_DEVICE_FFT_RESULT res;
    FFTX_DEVICE_FFT_TYPE   xfmtype = FFTX_DEVICE_FFT_Z2Z ;
    FFTX_DEVICE_EVENT_T custart, custop;
    FFTX_DEVICE_EVENT_CREATE ( &custart );
    FFTX_DEVICE_EVENT_CREATE ( &custop );
    
    res = FFTX_DEVICE_FFT_PLAN3D ( &plan, mm, nn, kk, xfmtype );
    if ( res != FFTX_DEVICE_FFT_SUCCESS )
      {
        fftx::OutStream() << "Create FFTX_DEVICE_FFT_PLAN3D failed with error code "
                          << res << " ... skip buffer check" << std::endl;
        check_output = false;
        status++;
      }
#elif defined (FFTX_SYCL)
    sycl::device dev;
    try
      {
	dev = sycl::device(sycl::gpu_selector_v);
      }
    catch (sycl::exception const &e)
      {
	fftx::ErrStream() << "You are running on a system without a GPU. For best results please use a GPU." << std::endl;
	fftx::ErrStream() << "Program terminating." << std::endl;
	exit(-1);
	// dev = sycl::device(sycl::cpu_selector_v);
      }
    sycl::context ctx = sycl::context(dev);
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
#elif defined (FFTX_USE_FFTW)
    fftw_plan plan = fftw_plan_dft_3d(mm, nn, kk,
                                      (fftw_complex*) inputHostPtr,
                                      (fftw_complex*) outputVendorHostPtr,
                                      FFTW_FORWARD, FFTW_ESTIMATE);
#endif

    float *mddft_gpu = new float[iterations];
    float *mddft_vendor_millisec = new float[iterations];

    // double *hostinp = (double *) inputHostPtr;
    for (int itn = 0; itn < iterations; itn++)
      {
        // Fill input buffer with random data, different each iteration.
        setRandomData( (double*) inputHostPtr, 2 * npts );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        fftxCopyHostArrayToDevice(inputTfmPtr, inputHostArray);
	if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied input from host to device\n";
#endif

	// Run transform: input inputTfmPtr, output outputTfmPtr.
	mdp.transform();
	mddft_gpu[itn] = mdp.getTime();
	// gatherOutput(outputFFTXHostArray, args);

        if ( check_output )
	  { // Run the vendor FFT plan, or FFTW, on the same input data.	
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against cuFFT or rocFFT.
            fftxCopyDeviceToHostArray(outputFFTXHostArray, outputTfmPtr);

	    // Run cuFFT or rocFFT.
	    FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECZ2Z ( plan,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) inputTfmPtr,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) outputTfmPtr,
                                            FFTX_DEVICE_FFT_FORWARD );
            if ( res != FFTX_DEVICE_FFT_SUCCESS)
	      {
                fftx::OutStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
		check_output = false;
                status++;
		//  break;
	      }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &mddft_vendor_millisec[itn], custart, custop );

            fftxCopyDeviceToHostArray(outputVendorHostArray, outputTfmPtr);
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
#elif defined (FFTX_USE_FFTW)
            auto start = std::chrono::high_resolution_clock::now();
            fftw_execute(plan);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            mddft_vendor_millisec[itn] = duration.count();
#endif
            fftx::OutStream() << "cube = [ "
                              << mm << ", " << nn << ", " << kk << " ]\t"
                              << "MDDFT (Forward) \t";
            bool chk = checkOutputs ( (FFTX_DOUBLECOMPLEX*) outputFFTXHostPtr,
                                      (FFTX_DOUBLECOMPLEX*) outputVendorHostPtr,
                                      (long) npts );
            if (chk == false) status++;
	  } // end check_output
      } // end iteration

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on MDDFT (forward) for "
                      << iterations << " trials of size "
                      << sizes.at(0) << " "
                      << sizes.at(1) << " "
                      << sizes.at(2) << ":" << std::endl;
    fftx::OutStream() << "Trial#    Spiral";
    if (check_output)
      {
        fftx::OutStream() << "           " << vendorfft;
      }
    fftx::OutStream() << std::endl;
    for (int itn = 0; itn < iterations; itn++)
      {
        fftx::OutStream() << std::setw(4) << (itn+1)
                          << std::setw(17) << mddft_gpu[itn];
        if (check_output)
          {
            fftx::OutStream() << std::setw(17) << mddft_vendor_millisec[itn];
          }
        fftx::OutStream() << std::endl;
      }
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#elif defined(FFTX_SYCL)
#elif defined(FFTX_USE_FFTW)
    fftw_destroy_plan(plan);
#endif

    delete[] mddft_gpu;
    delete[] mddft_vendor_millisec;


    // Set up the inverse transform.
    // (We'll reuse the vendor FFT plan already created.)
    IMDDFTProblem imdp(args, sizes, "imddft");

#if defined(FFTX_USE_FFTW)
    plan = fftw_plan_dft_3d(mm, nn, kk,
                            (fftw_complex*) inputHostPtr,
                            (fftw_complex*) outputVendorHostPtr,
                            FFTW_BACKWARD, FFTW_ESTIMATE);
#endif

    float *imddft_gpu = new float[iterations];
    float *imddft_vendor_millisec = new float[iterations];

    for (int itn = 0; itn < iterations; itn++)
      {
        // Fill input buffer with random data, different each iteration.
        setRandomData( (double*) inputHostPtr, 2 * npts );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        fftxCopyHostArrayToDevice(inputTfmPtr, inputHostArray);
#endif
        if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied input from host to device\n";
        
	// Run transform: input inputTfmPtr, output outputTfmPtr.
        imdp.transform();
        imddft_gpu[itn] = imdp.getTime();
        // gatherOutput(outputFFTXHostArray, args);

        if ( check_output )
	  { // Run the vendor FFT plan, or FFTW, on the same input data.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against cuFFT or rocFFT.
            fftxCopyDeviceToHostArray(outputFFTXHostArray, outputTfmPtr);

	    // Run cuFFT or rocFFT.	    
	    FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECZ2Z ( plan,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) inputTfmPtr,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) outputTfmPtr,
                                            FFTX_DEVICE_FFT_INVERSE );
            if ( res != FFTX_DEVICE_FFT_SUCCESS)
	      {
                fftx::OutStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
                check_output = false;
                status++;
                //  break;
	      }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &imddft_vendor_millisec[itn], custart, custop );

            fftxCopyDeviceToHostArray(outputVendorHostArray, outputTfmPtr);
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
#elif defined(FFTX_USE_FFTW)
            auto start = std::chrono::high_resolution_clock::now();
            fftw_execute(plan);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            imddft_vendor_millisec[itn] = duration.count();
#endif
	    // printf ( "cube = [ %d, %d, %d ]\tIMDDFT (Inverse)\t", mm, nn, kk );
            fftx::OutStream() << "cube = [ "
                              << mm << ", " << nn << ", " << kk << " ]\t"
                              << "IMDDFT (Inverse)\t";
	    bool chk = checkOutputs ( (FFTX_DOUBLECOMPLEX*) outputFFTXHostPtr,
                                      (FFTX_DOUBLECOMPLEX*) outputVendorHostPtr,
                                      (long) npts );
            if (chk == false) status++;
	  }
      } // end iteration

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on MDDFT (inverse) for "
                      << iterations << " trials of size "
                      << sizes.at(0) << " "
                      << sizes.at(1) << " "
                      << sizes.at(2) << ":" << std::endl;
    fftx::OutStream() << "Trial#    Spiral";
    if (check_output)
      {
        fftx::OutStream() << "           " << vendorfft;
      }
    fftx::OutStream() << std::endl;
    for (int itn = 0; itn < iterations; itn++)
      {
        fftx::OutStream() << std::setw(4) << (itn+1)
                          << std::setw(17) << imddft_gpu[itn];
        if (check_output)
          {
            fftx::OutStream() << std::setw(17) << imddft_vendor_millisec[itn];
          }
        fftx::OutStream() << std::endl;
      }
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_DESTROY(plan);
#elif defined(FFTX_SYCL)
#elif defined(FFTX_USE_FFTW)
    fftw_destroy_plan(plan);
#endif

    delete[] imddft_gpu;
    delete[] imddft_vendor_millisec;

    // Clean up.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    fftxDeviceFree(inputTfmPtr);
    fftxDeviceFree(outputTfmPtr);
    // fftxDeviceFree(symbolTfmPtr);
#elif defined(FFTX_SYCL)
    sycl::free(inputVendorPtr, sycl_context);
    sycl::free(outputVendorPtr, sycl_context);
#else
    // delete[] symbolTfmPtr;
#endif

    // printf("%s: All done, exiting\n", prog);
    fftx::OutStream() << prog << ": All done, exiting with status "
                      << status << std::endl;
    std::flush(fftx::OutStream());

    return status;
}
