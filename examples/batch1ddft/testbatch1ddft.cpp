//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.
//

#include <math.h>
#include "fftx.hpp"
#include "fftxutilities.hpp"
#include "fftxinterface.hpp"
#include "fftxbatch1ddftObj.hpp"
#include "fftxibatch1ddftObj.hpp"
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
#include <oneapi/mkl/dft.hpp>
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

#define FFTX_BATCH_SEQUENTIAL 0
#define FFTX_BATCH_STRIDED 1

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
    int N = 32; // default array length
    int B = 2; // default batch size
    int read = FFTX_BATCH_SEQUENTIAL;
    int write = FFTX_BATCH_SEQUENTIAL;
    std::string reads = "Sequential";
    std::string writes = "Sequential";

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
            N = atoi (& argv[1][baz] );
            while ( argv[1][baz] != 'x' ) baz++;
            baz++ ;
            B = atoi ( & argv[1][baz] );
            break;
        case 'r':
            if(strlen(argv[1]) > 2) {
              baz = 2;
            } else {
              baz = 0;
              argv++, argc--;
            }
            read = atoi (& argv[1][baz] );
            while ( argv[1][baz] != 'x' ) baz++;
            baz++ ;
            write = atoi ( & argv[1][baz] );
            break;
        case 'h':
            fftx::OutStream() << "Usage: " << argv[0]
                              << " [ -i iterations ] [ -s NxB (DFT Length x Batch Size) ] [-r ReadxWrite (sequential = 0, strided = 1)] [ -h (print help message) ]"
                              << std::endl;
            exit (0);
        default:
            fftx::ErrStream() << prog << ": unknown argument: "
                              << argv[1] << " ... ignored" << std::endl;
        }
        argv++, argc--;
    }

    if (read == FFTX_BATCH_SEQUENTIAL)
        reads = "Sequential";
    else if (read == FFTX_BATCH_STRIDED)
        reads = "Strided";
    else
      {
            fftx::ErrStream() << "Usage: " << argv[0]
                              << " [ -i iterations ] [ -s NxB (DFT Length x Batch Size) ] [-r ReadxWrite (sequential = 0, strided = 1)] [ -h (print help message) ]"
                              << std::endl;
            exit (-1);
      }
    
    if (write == FFTX_BATCH_SEQUENTIAL)
        writes = "Sequential";
    else if (write == FFTX_BATCH_STRIDED)
        writes = "Strided";
    else
      {
            fftx::ErrStream() << "Usage: " << argv[0]
                              << " [ -i iterations ] [ -s NxB (DFT Length x Batch Size) ] [-r ReadxWrite (sequential = 0, strided = 1)] [ -h (print help message) ]"
                              << std::endl;
            exit (-1);
      }

    if ( FFTX_DEBUGOUT ) fftx::OutStream() << N << " " << B << " " << reads << " " << writes << std::endl;
    int status = 0;

    std::vector<int> sizes{N,B, read,write};

    long npts = B * N;

    std::vector<std::complex<double>> inputHostVector(npts);
    std::vector<std::complex<double>> outputFFTXHostVector(npts);
    std::vector<std::complex<double>> outputVendorHostVector(npts);

    std::complex<double> *inputHostPtr = (std::complex<double> *) inputHostVector.data();
    std::complex<double> *outputFFTXHostPtr = (std::complex<double> *) outputFFTXHostVector.data();
    std::complex<double> *outputVendorHostPtr = (std::complex<double> *) outputVendorHostVector.data();

    std::complex<double> *inputTfmPtr, *outputFFTXTfmPtr, *outputVendorTfmPtr;

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "allocating memory" << std::endl;
    // Allocate memory on device.
    size_t bytes = npts * sizeof(std::complex<double>);
    FFTX_DEVICE_MALLOC((void**)&inputTfmPtr, bytes);
    FFTX_DEVICE_MALLOC((void**)&outputFFTXTfmPtr, bytes);
    FFTX_DEVICE_MALLOC((void**)&outputVendorTfmPtr, bytes);
#else // SYCL or CPU
    // FIXME: should these be (double *)?
    inputTfmPtr = inputHostVector.data();
    outputFFTXTfmPtr = outputFFTXHostVector.data();
    outputVendorTfmPtr = outputVendorHostVector.data();
#endif
    
#ifdef FFTX_SYCL
    // These will store complex data, so need double the array size.
    sycl::buffer<double> inputBuffer((double*) inputTfmPtr, 2*npts);
    sycl::buffer<double> outputFFTXBuffer((double*) outputFFTXTfmPtr, 2*npts);
    sycl::buffer<double> outputVendorBuffer((double*) outputVendorTfmPtr, 2*npts);
#endif

    // Order within args:  output, input.
#if defined FFTX_CUDA
    std::vector<void*> args{&outputFFTXTfmPtr, &inputTfmPtr};
    std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
    std::string vendorfft  = "cufft";
#elif defined FFTX_HIP
    std::vector<void*> args{outputFFTXTfmPtr, inputTfmPtr};
    std::string descrip = "AMD GPU";                //  "CPU and GPU";
    std::string vendorfft  = "rocfft";
#elif defined FFTX_SYCL
    std::vector<void*> args{(void*)&(outputFFTXBuffer), (void*)&(inputBuffer)};
    std::string descrip = "Intel GPU";                //  "CPU and GPU";
    std::string vendorfft  = "mklfft";
#else // CPU
    std::vector<void*> args{(void*)outputFFTXTfmPtr, (void*)inputTfmPtr};
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
    
    BATCH1DDFTProblem b1dft(args, sizes, "b1dft");
    
    //  Set up a plan to run the transform using vendor FFT.
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL) || defined(FFTX_USE_FFTW)
    int read_stride = (read == FFTX_BATCH_SEQUENTIAL) ? 1 : B;
    int read_dist = (read == FFTX_BATCH_SEQUENTIAL) ? N : 1;

    int write_stride = (write == FFTX_BATCH_SEQUENTIAL) ? 1 : B;
    int write_dist = (write == FFTX_BATCH_SEQUENTIAL) ? N : 1;
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_HANDLE plan;
    FFTX_DEVICE_FFT_RESULT res;
    FFTX_DEVICE_FFT_TYPE   xfmtype = FFTX_DEVICE_FFT_Z2Z ;
    FFTX_DEVICE_EVENT_T custart, custop;
    FFTX_DEVICE_EVENT_CREATE ( &custart );
    FFTX_DEVICE_EVENT_CREATE ( &custop );

    std::string read_str = (read == FFTX_BATCH_SEQUENTIAL) ? "APAR" : "AVEC";
    std::string write_str = (write == FFTX_BATCH_SEQUENTIAL) ? "APAR" : "AVEC";

    if ( FFTX_DEBUGOUT ) fftx::OutStream() << read_str << ", "
                                           << write_str << std::endl;
    res = FFTX_DEVICE_FFT_PLAN_MANY( &plan, 1, &N, // plan, rank, length
                                     &N, read_stride, read_dist,
                                     &N, write_stride, write_dist,
                                     xfmtype, B); // type, batch
    if ( res != FFTX_DEVICE_FFT_SUCCESS )
      {
        fftx::ErrStream() << "Create FFTX_DEVICE_FFT_PLAN_MANY failed with error code "
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
    auto sycl_device = Q.get_device();
    auto sycl_context = Q.get_context();
    fftx::OutStream() << "Running on: "
                      << Q.get_device().get_info<sycl::info::device::name>()
                      << std::endl;

    FFTX_DOUBLECOMPLEX* inputVendorPtr =
      sycl::malloc_shared< std::complex<double> >
      (npts, sycl_device, sycl_context);
    FFTX_DOUBLECOMPLEX* outputVendorPtr =
      sycl::malloc_shared< std::complex<double> >
      (npts, sycl_device, sycl_context);

    // Set strides and distance for reading and writing MKL transforms.
    // N.B. Our application specifies either sequential or strided input,
    // and either sequential or strided output,
    // and these specifications apply to both the forward transform
    // and the inverse transform.

    // If we have specified sequential reads & strided writes, or vice versa,
    // then we need separate mkl::dft plans for compute_forward and
    // compute_backward, in order for the output ordering to be correct.
    std::vector<std::int64_t> read_strides = {0, read_stride};
    std::vector<std::int64_t> write_strides = {0, write_stride};

    // Initialize batch 1D FFT descriptor
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
				 oneapi::mkl::dft::domain::COMPLEX> plan(N);
    plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, B);
    plan.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, read_strides);
    plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, read_dist);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, write_strides);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, write_dist);
    plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
		   oneapi::mkl::dft::config_value::NOT_INPLACE);
    plan.commit(Q);
#elif defined (FFTX_USE_FFTW)
    fftw_plan plan = fftw_plan_many_dft(1, &N, B,
                                        (fftw_complex*) inputHostPtr,
                                        &N, read_stride, read_dist,
                                        (fftw_complex*) outputVendorHostPtr,
                                        &N, write_stride, write_dist,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
#endif
#endif

    float *batch1ddft_gpu = new float[iterations];
    float *batch1ddft_vendor_millisec = new float[iterations];

    for (int itn = 0; itn < iterations; itn++)
      {
        // Fill input buffer with random data, different each iteration.
	setRandomData ( (double*) inputHostPtr, 2 * npts );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY(inputTfmPtr,
                             inputHostPtr,
                             bytes,
                             FFTX_MEM_COPY_HOST_TO_DEVICE);
	if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied input from host to device\n";
#endif

	// Run transform: input inputTfmPtr, output outputFFTXTfmPtr.
	b1dft.transform();
	batch1ddft_gpu[itn] = b1dft.getTime();

        if ( check_output )
	  { // Run the vendor FFT plan on the same input data.	
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against cuFFT or rocFFT.
            FFTX_DEVICE_MEM_COPY(outputFFTXHostPtr,
                                 outputFFTXTfmPtr,
                                 bytes,
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);

	    // Run cuFFT or rocFFT.
	    FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECZ2Z ( plan,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) inputTfmPtr,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) outputVendorTfmPtr,
                                            FFTX_DEVICE_FFT_FORWARD );
            if ( res != FFTX_DEVICE_FFT_SUCCESS)
	      {
                fftx::ErrStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
		check_output = false;
                status++;
		//  break;
	      }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &batch1ddft_vendor_millisec[itn], custart, custop );

            FFTX_DEVICE_MEM_COPY(outputVendorHostVector.data(),
                                 outputVendorTfmPtr,
                                 bytes,
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);
            
#elif defined (FFTX_SYCL)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against MKL FFT.

	    // If this is absent then iterations after the first aren't correct.
	    sycl::host_accessor inputAcc(inputBuffer);

	    // outputFFTXAcc is double* because outputFFTXBuffer is sycl::buffer<double>.
	    sycl::host_accessor outputFFTXAcc(outputFFTXBuffer);
	    for (int ind = 0; ind < npts; ind++)
	      {
		outputFFTXHostVector[ind] =
		  std::complex(outputFFTXAcc[2*ind], outputFFTXAcc[2*ind+1]);
	      }
	
	    // Run MKL FFT plan on the same input data.
	    for (int ind = 0; ind < npts; ind++)
	      { // These are both complex.
		inputVendorPtr[ind] = inputHostPtr[ind];
	      }
	    auto start_time = std::chrono::high_resolution_clock::now();
	    // Perform forward transform on complex array
            // oneapi::mkl::dft::compute_forward(plan, inputVendorPtr, outputVendorPtr).wait();
            sycl::event e = oneapi::mkl::dft::compute_forward(plan,
                                                              inputVendorPtr,
                                                              outputVendorPtr);
            Q.wait();
	    auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t profile_nanosec = e_end - e_start;
            batch1ddft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

            for (int ind = 0; ind < npts; ind++)
              {
                outputVendorHostPtr[ind] = outputVendorPtr[ind];
              }
#elif defined (FFTX_USE_FFTW)
            auto start = std::chrono::high_resolution_clock::now();
            fftw_execute(plan);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            batch1ddft_vendor_millisec[itn] = duration.count();
#endif
            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tBatch 1D FFT (Forward)\t";
            bool chk = checkOutputs ( (FFTX_DOUBLECOMPLEX*) outputFFTXHostPtr,
                                      (FFTX_DOUBLECOMPLEX*) outputVendorHostPtr,
                                      npts);
            if (chk == false) status++;
	  } // end check_output
      } // end iteration

    fftx::OutStream() << "Times in milliseconds for 1D FFT (forward)"
                      << " of length " << N << " batch " << B
                      << " for " << iterations << " trials " << std::endl;
    fftx::OutStream() << "Trial#    Spiral";
    if (check_output)
      {
        fftx::OutStream() << "           " << vendorfft;
      }
    fftx::OutStream() << std::endl;
    for (int itn = 0; itn < iterations; itn++)
      {
        fftx::OutStream() << std::setw(4) << (itn+1)
                          << std::setw(17) << batch1ddft_gpu[itn];
        if (check_output)
          {
            fftx::OutStream() << std::setw(17) << batch1ddft_vendor_millisec[itn];
          }
        fftx::OutStream() << std::endl;
      }

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#elif defined(FFTX_SYCL)
#elif defined(FFTX_USE_FFTW)
    fftw_destroy_plan(plan);
#endif
    
    delete[] batch1ddft_gpu;
    delete[] batch1ddft_vendor_millisec;


    // Set up the inverse transform.
    // (We'll reuse the vendor FFT plan already created.)
    IBATCH1DDFTProblem ib1dft(args, sizes, "ib1dft");

#if defined (FFTX_SYCL)
    // Need a separate plan for compute_backward if and only if read != write.
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
				 oneapi::mkl::dft::domain::COMPLEX> planInv(N);
    if (read != write)
      {
        planInv.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, B);
        planInv.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, write_strides);
        planInv.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, write_dist);
        planInv.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, read_strides);
        planInv.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, read_dist);
        planInv.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
			  oneapi::mkl::dft::config_value::NOT_INPLACE);
        planInv.commit(Q);
      }
#elif defined(FFTX_USE_FFTW)
    plan = fftw_plan_many_dft(1, &N, B,
                              (fftw_complex*) inputHostPtr,
                              &N, read_stride, read_dist,
                              (fftw_complex*) outputVendorHostPtr,
                              &N, write_stride, write_dist,
                              FFTW_BACKWARD, FFTW_ESTIMATE);
#endif

    float *ibatch1ddft_gpu = new float[iterations];
    float *ibatch1ddft_vendor_millisec = new float[iterations];

    for (int itn = 0; itn < iterations; itn++)
      {
        // Fill input buffer with random data, different each iteration.
	setRandomData ( (double*) inputHostPtr, 2 * npts );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY(inputTfmPtr,
                             inputHostPtr,
                             bytes,
                             FFTX_MEM_COPY_HOST_TO_DEVICE);
        if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied input from host to device\n";
#endif
        
	// Run transform: input inputTfmPtr, output outputFFTXTfmPtr.
        ib1dft.transform();
        ibatch1ddft_gpu[itn] = ib1dft.getTime();
        // gatherOutput(outputFFTXHostArray, args);

        if ( check_output )
	  { // Run the vendor FFT plan on the same input data.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against cuFFT or rocFFT.
            FFTX_DEVICE_MEM_COPY(outputFFTXHostPtr,
                                 outputFFTXTfmPtr,
                                 bytes,
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);

	    // Run cuFFT or rocFFT.	    
	    FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECZ2Z ( plan,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) inputTfmPtr,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) outputVendorTfmPtr,
                                            FFTX_DEVICE_FFT_INVERSE );
            if ( res != FFTX_DEVICE_FFT_SUCCESS)
	      {
                fftx::ErrStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
                check_output = false;
                status++;
                //  break;
	      }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &ibatch1ddft_vendor_millisec[itn], custart, custop );

            FFTX_DEVICE_MEM_COPY(outputVendorHostVector.data(),
                                 outputVendorTfmPtr,
                                 bytes,
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);
#elif defined (FFTX_SYCL)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against MKL FFT.

	    // If this is absent then iterations after the first aren't correct.
	    sycl::host_accessor inputAcc(inputBuffer);

	    // outputFFTXAcc is double* because outputFFTXBuffer is sycl::buffer<double>.
	    sycl::host_accessor outputFFTXAcc(outputFFTXBuffer);
	    for (int ind = 0; ind < npts; ind++)
	      {
		outputFFTXHostVector[ind] =
		  std::complex(outputFFTXAcc[2*ind],
                               outputFFTXAcc[2*ind+1]);
	      }

	    // Run MKL FFT plan on the same input data.
	    for (int ind = 0; ind < npts; ind++)
	      { // These are both complex.
		inputVendorPtr[ind] = inputHostPtr[ind];
	      }
	    auto start_time = std::chrono::high_resolution_clock::now();
	    // Perform backward transform on complex array
            // oneapi::mkl::dft::compute_backward(plan, inputVendorPtr, outputVendorPtr).wait();
            sycl::event e = oneapi::mkl::dft::compute_backward((read == write) ? plan : planInv,
                                                               inputVendorPtr,
                                                               outputVendorPtr);
            Q.wait();
	    auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t profile_nanosec = e_end - e_start;
            ibatch1ddft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

            for (int ind = 0; ind < npts; ind++)
              {
                outputVendorHostPtr[ind] = outputVendorPtr[ind];
              }
#elif defined (FFTX_USE_FFTW)
            auto start = std::chrono::high_resolution_clock::now();
            fftw_execute(plan);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            ibatch1ddft_vendor_millisec[itn] = duration.count();
#endif
            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tBatch 1D FFT (Inverse)\t";
	    bool chk = checkOutputs ( (FFTX_DOUBLECOMPLEX*) outputFFTXHostPtr,
                                      (FFTX_DOUBLECOMPLEX*) outputVendorHostPtr,
                                      npts );
            if (chk == false) status++;
	  } // end check_output
      } // end iteration
            
    fftx::OutStream() << "Times in milliseconds for 1D FFT (inverse)"
                      << " of length " << N << " batch " << B
                      << " for " << iterations << " trials " << std::endl;
    fftx::OutStream() << "Trial#    Spiral";
    if (check_output)
      {
        fftx::OutStream() << "           " << vendorfft;
      }
    fftx::OutStream() << std::endl;
    for (int itn = 0; itn < iterations; itn++)
      {
        fftx::OutStream() << std::setw(4) << (itn+1)
                          << std::setw(17) << ibatch1ddft_gpu[itn];
        if (check_output)
          {
            fftx::OutStream() << std::setw(17) << ibatch1ddft_vendor_millisec[itn];
          }
        fftx::OutStream() << std::endl;
      }

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_DESTROY(plan);
#elif defined(FFTX_SYCL)
#elif defined(FFTX_USE_FFTW)
    fftw_destroy_plan(plan);
#endif

    delete[] ibatch1ddft_gpu;
    delete[] ibatch1ddft_vendor_millisec;

    // Clean up.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    fftxDeviceFree(inputTfmPtr);
    fftxDeviceFree(outputFFTXTfmPtr);
    fftxDeviceFree(outputVendorTfmPtr);
#elif defined(FFTX_SYCL)
    sycl::free(inputVendorPtr, sycl_context);
    sycl::free(outputVendorPtr, sycl_context);
#endif

    // printf("%s: All done, exiting\n", prog);
    fftx::OutStream() << prog << ": All done, exiting with status "
                      << status << std::endl;
    std::flush(fftx::OutStream());

    return status;
}
