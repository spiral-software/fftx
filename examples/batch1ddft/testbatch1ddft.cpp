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
#include <oneapi/mkl/dfti.hpp>
#endif

//  Build a random input buffer for Spiral and vendor FFT
//  inputPtr is the host buffer to setup -- it'll be copied to the device later.
//  sizes is the vector {N, B, read, write}.
static void setInput ( double *inputPtr, std::vector<int> sizes )
{
    for ( int imm = 0; imm < 2 * sizes.at(0) * sizes.at(1); imm++ ) {
      inputPtr[imm] = 1.0 - ((double) rand()) / (double) (RAND_MAX/2);
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
static void checkOutputs ( FFTX_DOUBLECOMPLEX *outputFFTXPtr,
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

    return;
}


int main(int argc, char* argv[])
{
    int iterations = 2;
    int N = 32; // default array length
    int B = 2; // default batch size
    int read = 0;
    int write = 0;
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
            fftx::ErrStream() << "Usage: " << argv[0]
                              << " [ -i iterations ] [ -s NxB (DFT Length x Batch Size) ] [-r ReadxWrite (sequential = 0, strided = 1)] [ -h (print help message) ]"
                              << std::endl;
            exit (0);
        default:
            fftx::ErrStream() << prog << ": unknown argument: "
                              << argv[1] << " ... ignored" << std::endl;
        }
        argv++, argc--;
    }
    if (read == 0)
        reads = "Sequential";
    else
        reads = "Strided";
    if (write == 0)
        writes = "Sequential";
    else
        writes = "Strided";

    if ( FFTX_DEBUGOUT ) fftx::OutStream() << N << " " << B << " " << reads << " " << writes << std::endl;
    std::vector<int> sizes{N,B, read,write};

    std::vector<std::complex<double>> inputHostVector(B * N);
    std::vector<std::complex<double>> outputFFTXHostVector(B * N);
    std::vector<std::complex<double>> outputVendorHostVector(B * N);

    std::complex<double> *inputHostPtr = (std::complex<double> *) inputHostVector.data();
    std::complex<double> *outputFFTXHostPtr = (std::complex<double> *) outputFFTXHostVector.data();
    std::complex<double> *outputVendorHostPtr = (std::complex<double> *) outputVendorHostVector.data();

    std::complex<double> *inputTfmPtr, *outputFFTXTfmPtr, *outputVendorTfmPtr;

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "allocating memory" << std::endl;
    // Allocate memory on device.
    FFTX_DEVICE_MALLOC((void**)&inputTfmPtr, inputHostVector.size() * sizeof(std::complex<double>));
    FFTX_DEVICE_MALLOC((void**)&outputFFTXTfmPtr, outputFFTXHostVector.size() * sizeof(std::complex<double>));
    FFTX_DEVICE_MALLOC((void**)&outputVendorTfmPtr, outputVendorHostVector.size() * sizeof(std::complex<double>));
#elif defined (FFTX_SYCL)
    // These will store complex data, so need double the array size.
    sycl::buffer<double> inputBuffer((double*) inputHostVector.data(), 2 * inputHostVector.size());
    sycl::buffer<double> outputFFTXBuffer((double*) outputFFTXHostVector.data(), 2 * outputFFTXHostVector.size());
    sycl::buffer<double> outputVendorBuffer((double*) outputVendorHostVector.data(), 2 * outputVendorHostVector.size());
#else // CPU
    // FIXME: should these be (double *)?
    inputTfmPtr = inputHostVector.data();
    outputFFTXTfmPtr = outputFFTXHostVector.data();
    outputVendorTfmPtr = outputVendorHostVector.data();
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

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
    // compare results of Spiral-RTC with vendor FFT
    bool check_output = true;
#else
    bool check_output = false;
#endif

    BATCH1DDFTProblem b1dft(args, sizes, "b1dft");
    
    fftx::OutStream() << std::scientific
                      << std::uppercase
                      << std::setprecision(7);
    
    //  Set up a plan to run the transform using vendor FFT.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_HANDLE plan;
    FFTX_DEVICE_FFT_RESULT res;
    FFTX_DEVICE_FFT_TYPE   xfmtype = FFTX_DEVICE_FFT_Z2Z ;
    FFTX_DEVICE_EVENT_T custart, custop;
    FFTX_DEVICE_EVENT_CREATE ( &custart );
    FFTX_DEVICE_EVENT_CREATE ( &custop );

    std::string read_str;
    int istride, idist;
    if (read == 0)
      {
        read_str = "APAR";
        istride = 1;
        idist = N;
      }
    else
      {
        read_str = "AVEC";
        istride = B;
        idist = 1;
      }

    std::string write_str;
    int ostride, odist;
    if (write == 0)
      {
        write_str = "APAR";
        ostride = 1;
        odist = N;
      }
    else
      {
        write_str = "AVEC";
        ostride = B;
        odist = 1;
      }

    if ( FFTX_DEBUGOUT ) fftx::OutStream() << read_str << ", "
                                           << write_str << std::endl;
    res = FFTX_DEVICE_FFT_PLAN_MANY( &plan, 1, &N, // plan, rank, length
                                     &N, istride, idist,
                                     &N, ostride, odist,
                                     xfmtype, B); // type, batch
    if ( res != FFTX_DEVICE_FFT_SUCCESS )
      {
        fftx::ErrStream() << "Create FFTX_DEVICE_FFT_PLAN_MANY failed with error code "
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

    auto inputVendorPtr = sycl::malloc_shared< std::complex<double> >
      (B * N, sycl_device, sycl_context);
    auto outputVendorPtr = sycl::malloc_shared< std::complex<double> >
      (B * N, sycl_device, sycl_context);

    int fwd_dist, bwd_dist;
    std::int64_t fwd_strides[2];
    std::int64_t bwd_strides[2];
    fwd_strides[0] = 0;
    bwd_strides[0] = 0;
    if (read == 0)
      {
        fwd_dist = N;
        fwd_strides[1] = 1;
      }
    else
      {
        fwd_dist = 1;
        fwd_strides[1] = B;
      }

    if (write == 0)
      {
        bwd_dist = N;
        bwd_strides[1] = 1;
      }
    else
      {
        bwd_dist = 1;
        bwd_strides[1] = B;
      }
    //    double bwd_scale = (double) 1.0 / (double) N;
    //    std::cout << "bwd_scale = " << bwd_scale << std::endl;

    // Initialize batch 1D FFT descriptor
    std::vector<std::int64_t> Nvec{mm, nn, kk};
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
				 oneapi::mkl::dft::domain::COMPLEX> plan(N);
    // plan.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, bwd_scale);
    plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, B);
    plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fwd_dist);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, bwd_dist);
    plan.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, fwd_strides);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, bwd_strides);
    plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    plan.commit(Q);
#endif

    float *batch1ddft_gpu = new float[iterations];
    float *batch1ddft_vendor_millisec = new float[iterations];

    for (int itn = 0; itn < iterations; itn++)
      {
        // Set up random data for input buffer.
	// (Use different randomized data each iteration.)

	setInput ( (double*) inputHostPtr, sizes );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY(inputTfmPtr,
                             inputHostPtr,
                             inputHostVector.size() * sizeof(std::complex<double>),
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
                                 outputFFTXHostVector.size() * sizeof(std::complex<double>),
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
		//  break;
	      }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &batch1ddft_vendor_millisec[itn], custart, custop );

            FFTX_DEVICE_MEM_COPY(outputVendorHostVector.data(),
                                 outputVendorTfmPtr,
                                 outputVendorHostVector.size() * sizeof(std::complex<double>),
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);
            
#elif defined (FFTX_SYCL)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against MKL FFT.

	    // If this is absent then iterations after the first aren't correct.
	    sycl::host_accessor inputAcc(inputBuffer);

	    // outputFFTXAcc is double* because outputFFTXBuffer is sycl::buffer<double>.
	    sycl::host_accessor outputFFTXAcc(outputFFTXBuffer);
	    for (int ind = 0; ind < outputFFTXHostVector.size(); ind++)
	      {
		outputFFTXHostVector[ind] =
		  std::complex(outputFFTXAcc[2*ind], outputFFTXAcc[2*ind+1]);
	      }
	
	    // Run MKL FFT plan on the same input data.
	    for (int ind = 0; ind < inputHostVector.size(); ind++)
	      { // These are both complex.
		inputVendorPtr[ind] = inputHostPtr[ind];
	      }
	    auto start_time = std::chrono::high_resolution_clock::now();
	    // Perform forward transform on complex array
            // oneapi::mkl::dft::compute_forward(transform_plan_3d, inputVendorPtr, outputVendorPtr).wait();
            sycl::event e = oneapi::mkl::dft::compute_forward(plan,
                                                              inputVendorPtr,
                                                              outputVendorPtr);
            Q.wait();
	    auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t profile_nanosec = e_end - e_start;
            batch1ddft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds
      
	    for (int ind = 0; ind < outputVendorHostVector.size(); ind++)
	      { // These are both complex.
		outputVendorHostPtr[ind] = outputVendorPtr[ind];
	      }
#endif
            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tBatch 1D FFT (Forward)\t";
	    checkOutputs ( (FFTX_DOUBLECOMPLEX*) outputFFTXHostPtr,
			   (FFTX_DOUBLECOMPLEX*) outputVendorHostPtr,
			   (long) outputFFTXHostVector.size() );
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
    
    delete[] batch1ddft_gpu;
    delete[] batch1ddft_vendor_millisec;

    // Set up the inverse transform.
    // (We'll reuse the vendor FFT plan already created.)
    IBATCH1DDFTProblem ib1dft(args, sizes, "ib1dft");

    float *ibatch1ddft_gpu = new float[iterations];
    float *ibatch1ddft_vendor_millisec = new float[iterations];

    for (int itn = 0; itn < iterations; itn++)
      {
        // Set up random data for input buffer.
	// (Use different randomized data each iteration.)

	setInput ( (double*) inputHostPtr, sizes );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY(inputTfmPtr,
                             inputHostPtr,
                             inputHostVector.size() * sizeof(std::complex<double>),
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
                                 outputFFTXHostVector.size() * sizeof(std::complex<double>),
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
                //  break;
	      }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &ibatch1ddft_vendor_millisec[itn], custart, custop );

            FFTX_DEVICE_MEM_COPY(outputVendorHostVector.data(),
                                 outputVendorTfmPtr,
                                 outputVendorHostVector.size() * sizeof(std::complex<double>),
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);
#elif defined (FFTX_SYCL)
	    // Copy output of FFTX transform from device to host
	    // in order to check it against MKL FFT.

	    // If this is absent then iterations after the first aren't correct.
	    sycl::host_accessor inputAcc(inputBuffer);

	    // outputFFTXAcc is double* because outputFFTXBuffer is sycl::buffer<double>.
	    sycl::host_accessor outputFFTXAcc(outputFFTXBuffer);
	    for (int ind = 0; ind < outputFFTXHostVector.size(); ind++)
	      {
		outputFFTXHostVector[ind] =
		  std::complex(outputFFTXAcc[2*ind], outputFFTXAcc[2*ind+1]);
	      }

	    // Run MKL FFT plan on the same input data.
	    for (int ind = 0; ind < inputHostVector.size(); ind++)
	      { // These are both complex.
		inputVendorPtr[ind] = inputHostPtr[ind];
	      }
	    auto start_time = std::chrono::high_resolution_clock::now();
	    // Perform backward transform on complex array
            // oneapi::mkl::dft::compute_backward(transform_plan_3d, inputVendorPtr, outputVendorPtr).wait();
            sycl::event e = oneapi::mkl::dft::compute_backward(plan,
                                                               inputVendorPtr,
                                                               outputVendorPtr);
            Q.wait();
	    auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t profile_nanosec = e_end - e_start;
            ibatch1ddft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

	    for (int ind = 0; ind < outputVendorHostVector.size(); ind++)
	      { // These are both complex.
		outputVendorHostPtr[ind] = outputVendorPtr[ind];
	      }
#endif
            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tBatch 1D FFT (Inverse)\t";
	    checkOutputs ( (FFTX_DOUBLECOMPLEX*) outputFFTXHostPtr,
			   (FFTX_DOUBLECOMPLEX*) outputVendorHostPtr,
			   (long) outputFFTXHostVector.size() );
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
    fftx::OutStream() << prog << ": All done, exiting" << std::endl;
    std::flush(fftx::OutStream());

    return 0;
}
