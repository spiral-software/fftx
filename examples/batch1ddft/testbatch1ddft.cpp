#include "fftx.hpp"
#include "fftxutilities.hpp"
#include "fftxinterface.hpp"
#include "fftxbatch1ddftObj.hpp"
#include <math.h>  
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

#define FFTX_CH_CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
      fftx::ErrStream() << "Cuda error in file '" << __FILE__                \
                        << "' in line " << __LINE__                          \
                        << " : " << cudaGetErrorString( err)                 \
                        << "." << std::endl;                                 \
        exit(EXIT_FAILURE);                                                  \
    } \
}

#define FFTX_CUDA_SAFE_CALL(call) FFTX_CH_CUDA_SAFE_CALL(call)

//  Build a random input buffer for Spiral and rocfft/cufft.
//  host_X is the host buffer to setup -- it'll be copied to the device later.
//  sizes is a vector with the X, Y, & Z dimensions.
static void buildInputBuffer ( std::complex<double> *host_X, std::vector<int> sizes )
{
    for ( int imm = 0; imm < sizes.at(0)*sizes.at(1); imm++ ) {
        host_X[imm] = std::complex<double>(((double) rand()) / (double) (RAND_MAX/2), 0);
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

// Check that the buffer are identical (within roundoff)
// spiral_Y is the output buffer from the Spiral generated transform (result on GPU copied to host array spiral_Y)
// devfft_Y is the output buffer from the device equivalent transform (result on GPU copied to host array devfft_Y)
// arrsz is the size of each array

static void checkOutputBuffers_fwd ( FFTX_DOUBLECOMPLEX *spiral_Y, FFTX_DOUBLECOMPLEX *devfft_Y, long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int indx = 0; indx < arrsz; indx++ ) {
        FFTX_DOUBLECOMPLEX s = spiral_Y[indx];
        FFTX_DOUBLECOMPLEX c = devfft_Y[indx];

        double sreal = FFTX_REALPART(spiral_Y[indx]);
        double simag = FFTX_IMAGPART(spiral_Y[indx]);
        double creal = FFTX_REALPART(devfft_Y[indx]);
        double cimag = FFTX_IMAGPART(devfft_Y[indx]);

        double diffreal = sreal - creal;
        double diffimag = simag - cimag;

        bool elem_correct = ( (abs(diffreal) < 1e-7) &&
                              (abs(diffimag) < 1e-7) );
        updateMaxAbs(maxdelta, diffreal);
        updateMaxAbs(maxdelta, diffimag);
        correct &= elem_correct;
    }

    fftx::OutStream() << "Correct: " << (correct ? "True" : "False") << "\t"
                      << "Max delta = "
                      << std::scientific << std::uppercase << maxdelta
                      << std::endl;
    std::flush(fftx::OutStream());

    return;
}


int main(int argc, char* argv[])
{
    int iterations = 2;
    int N = 32; // default cube dimensions
    int B = 2;
    int read = 0;
    std::string reads = "Sequential";
    std::string writes = "Sequential";
    int write = 0;
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
            B = atoi (& argv[1][baz]);
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
            fftx::OutStream() << prog << ": unknown argument: "
                              << argv[1] << " ... ignored:" << std::endl;
        }
        argv++, argc--;
    }
    if(read == 0)
        reads = "Sequential";
    else
        reads = "Strided";
    if(write == 0)
        writes = "Sequential";
    else
        writes = "Strided";

     if ( FFTX_DEBUGOUT ) fftx::OutStream() << N << " " << B << " " << reads << " " << writes << std::endl;
    std::vector<int> sizes{N,B, read,write};
    // fftx::box_t<1> domain ( point_t<1> ( { { N } } ));

    // FIXME:  Why are these N*B*2 and not just N*B?
    std::vector<std::complex<double>> outDevfft1(N*B);
    std::vector<std::complex<double>> inputHost(N*B);
    std::vector<std::complex<double>> outputHost(N*B);
    std::vector<std::complex<double>> outDevfft2(N*B);
    std::vector<std::complex<double>> outputHost2(N*B);

    std::complex<double> *dX, *dY, *tempX;


#if defined (FFTX_CUDA) || defined(FFTX_HIP)
     if ( FFTX_DEBUGOUT ) fftx::OutStream() << "allocating memory" << std::endl;
    FFTX_DEVICE_MALLOC((void**)&dX, inputHost.size() * sizeof(std::complex<double>));
    FFTX_DEVICE_MALLOC((void **)&dY, outputHost.size() * sizeof(std::complex<double>));
    FFTX_DEVICE_MALLOC((void**)&tempX, outputHost.size()  * sizeof(std::complex<double>));
#elif defined(FFTX_SYCL)
    //    sycl::buffer<std::complex<double>> buf_Y(outputHost2.data(), outputHost2.size());
    //    sycl::buffer<std::complex<double>> buf_X(inputHost.data(), inputHost.size());
    //    sycl::buffer<std::complex<double>> buf_tempX(outputHost.data(), outputHost.size());
    // These will be complex, so need double the array size.
    sycl::buffer<double> buf_Y((double*) outputHost2.data(), 2 * outputHost2.size());
    sycl::buffer<double> buf_X((double*) inputHost.data(), 2 * inputHost.size());
    sycl::buffer<double> buf_tempX((double*) outputHost.data(), 2 * outputHost.size());
#else
    dX = (std::complex<double> *) inputHost.data();
    dY = (std::complex<double> *) outputHost.data();
    tempX = new std::complex<double>[outputHost.size()];
#endif

    float *batch1ddft_gpu = new float[iterations];
    float *ibatch1ddft_gpu = new float[iterations];
#if defined FFTX_CUDA
    std::vector<void*> args{&tempX,&dX};
    std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
    std::string devfft  = "cufft";
#elif defined FFTX_HIP
    std::vector<void*> args{tempX,dX};
    std::string descrip = "AMD GPU";                //  "CPU and GPU";
    std::string devfft  = "rocfft";
#elif defined FFTX_SYCL
    std::vector<void*> args{(void*)&(buf_tempX),(void*)&(buf_X)};
    std::string descrip = "Intel GPU";                //  "CPU and GPU";
    std::string devfft  = "mklfft";
#else
    std::vector<void*> args{(void*)tempX,(void*)dX};
    std::string descrip = "CPU";                //  "CPU";
    std::string devfft = "fftw";
#endif

BATCH1DDFTProblem b1dft(args, sizes, "b1dft");


#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    //  Setup a plan to run the transform using cu or roc fft
    FFTX_DEVICE_FFT_HANDLE plan;
    FFTX_DEVICE_FFT_RESULT res;
    FFTX_DEVICE_FFT_TYPE   xfmtype = FFTX_DEVICE_FFT_Z2Z ;
    FFTX_DEVICE_EVENT_T custart, custop;
    FFTX_DEVICE_EVENT_CREATE ( &custart );
    FFTX_DEVICE_EVENT_CREATE ( &custop );
    float *devmilliseconds = new float[iterations];
    float *invdevmilliseconds = new float[iterations];
    bool check_buff = true;                // compare results of spiral - RTC with device fft
    
    
    if(read == 0 && write == 0) {
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "APAR, APAR" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan, 1, &N, //plan, rank, n,
                                    &N,   1,  N, // iembed, istride, idist,
                                    &N,   1,  N, // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    } else if(read == 0 && write == 1) { 
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "APAR, AVEC" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan, 1, &N, //plan, rank, n,
                                    &N,   1,  N, // iembed, istride, idist,
                                    &N,   B,  1, // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    }else if(read == 1 && write == 0) {
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "AVEC, APAR" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan, 1, &N,  //plan, rank, n,
                                    &N,   B,  1,  // iembed, istride, idist,
                                    &N,   1,  N,  // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    }
    else {
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "AVEC, AVEC" << std::endl;
        res = FFTX_DEVICE_FFT_PLAN_MANY(&plan, 1, &N,  //plan, rank, n,
                                    &N,   B,  1,  // iembed, istride, idist,
                                    &N,   B,  1,  // oembed, ostride, odist,
                                    xfmtype, B); // type and batch
    }

    if ( res != FFTX_DEVICE_FFT_SUCCESS ) {
        fftx::OutStream() << "Create FFTX_DEVICE_FFT_PLAN_MANY failed with error code "
                          << res << " ... skip buffer check" << std::endl;
        check_buff = false;
    }
#elif defined (FFTX_SYCL)
    // These are repeats from CUDA/HIP.
    float *devmilliseconds = new float[iterations];
    float *invdevmilliseconds = new float[iterations];
    bool check_buff = true;                // compare results of spiral - RTC with device fft

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
      (B*N, sycl_device, sycl_context);
    auto outputVendorPtr = sycl::malloc_shared< std::complex<double> >
      (B*N, sycl_device, sycl_context);

    int fwd_dist, bwd_dist;
    std::int64_t fwd_strides[4];
    std::int64_t bwd_strides[4];
    if (read == 0)
      {
        fwd_dist = N;
        fwd_strides[0] = 0;
        fwd_strides[1] = 1;
      }
    else
      {
        fwd_dist = 1;
        // not sure about these
        fwd_strides[0] = 0;
        fwd_strides[1] = B;
      }

    if (write == 0)
      {
        bwd_dist = N;
        bwd_strides[0] = 0;
        bwd_strides[1] = 1;
      }
    else
      {
        bwd_dist = 1;
        // not sure about these
        bwd_strides[0] = 0;
        bwd_strides[1] = B;
      }
    double bwd_scale = (double) 1.0 / (double) N;
    std::cout << "bwd_scale = " << bwd_scale << std::endl;
    
    // Initialize batch 1D FFT descriptor
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

    std::complex<double> *hostinp = (std::complex<double> *) inputHost.data();
    for (int itn = 0; itn < iterations; itn++)
    {
        // setup random data for input buffer (Use different randomized data each iteration)
        buildInputBuffer(hostinp, sizes);
    #if defined(FFTX_HIP) || defined(FFTX_CUDA)
        FFTX_DEVICE_MEM_COPY(dX, inputHost.data(),  inputHost.size() * sizeof(std::complex<double>),
                        FFTX_MEM_COPY_HOST_TO_DEVICE);
    #endif
        if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied X" << std::endl;
        
        b1dft.transform();
        batch1ddft_gpu[itn] = b1dft.getTime();
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY ( outputHost.data(), tempX,
                          outputHost.size() * sizeof(std::complex<double>), FFTX_MEM_COPY_DEVICE_TO_HOST );
        //  Run the roc fft plan on the same input data
        if ( check_buff ) {
            FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECZ2Z ( plan,
                                       (FFTX_DOUBLECOMPLEX *) dX,
                                       (FFTX_DOUBLECOMPLEX *) tempX,
                                      FFTX_DEVICE_FFT_FORWARD);
            if ( res != FFTX_DEVICE_FFT_SUCCESS) {
                fftx::OutStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
                check_buff = false;
                //  break;
            }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &devmilliseconds[itn], custart, custop );

            FFTX_DEVICE_MEM_COPY ( outDevfft1.data(),
                                   tempX,
                                   outDevfft1.size() * sizeof(std::complex<double>),
                                   FFTX_MEM_COPY_DEVICE_TO_HOST );

            // printf ( "DFT = %d Batch = %d Read = %s Write = %s \tBatch 1D FFT (Forward)\t", N, B, reads.c_str(), writes.c_str());
            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tBatch 1D FFT (Forward)\t";
            checkOutputBuffers_fwd ( (FFTX_DOUBLECOMPLEX *) outputHost.data(),
                                     (FFTX_DOUBLECOMPLEX *) outDevfft1.data(),
                                     (long) outDevfft1.size() );
        }
    #elif defined(FFTX_SYCL)
        if ( check_buff ) {
        // Copy output of FFTX transform from device to host
        // in order to check it against MKL FFT.

        // If this is absent then iterations after the first aren't correct.
        sycl::host_accessor XAcc(buf_X);
        // tempXAcc is double* because buf_tempX is sycl::buffer<double>.
        sycl::host_accessor tempXAcc(buf_tempX);
        for (int ind = 0; ind < N*B; ind++)
          {
            outputHost[ind] =
              std::complex<double>(tempXAcc[2*ind], tempXAcc[2*ind+1]);
          }

        // Run MKL FFT plan on the same input data.
        for (int ind = 0; ind < N*B; ind++)
          { // These are both complex.
            inputVendorPtr[ind] = hostinp[ind];
          }
        auto start_time = std::chrono::high_resolution_clock::now();
        // Perform forward transform on complex array
        sycl::event e = oneapi::mkl::dft::compute_forward(plan,
                                                          inputVendorPtr,
                                                          outputVendorPtr);
        Q.wait();
        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        uint64_t profile_nanosec = e_end - e_start;
        devmilliseconds[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

        for (int ind = 0; ind < B*N; ind++)
          { // These are both complex.
            outDevfft1[ind] = outputVendorPtr[ind];
          }
        checkOutputBuffers_fwd ( (FFTX_DOUBLECOMPLEX *) outputHost.data(),
                                 (FFTX_DOUBLECOMPLEX *) outDevfft1.data(),
                                 (long) outDevfft1.size() );
        }
#endif
    }

#if defined FFTX_CUDA
    std::vector<void*> args2{&dY,&tempX};
#elif defined FFTX_HIP
    std::vector<void*> args2{dY,tempX};
#elif defined FFTX_SYCL
    std::vector<void*> args2{(void*)&(buf_Y), (void*)&(buf_tempX)};
#else
    std::vector<void*> args2{(void*)dY,(void*)tempX};
#endif

    IBATCH1DDFTProblem ib1dft(args2, sizes, "ib1dft");

    for (int itn = 0; itn < iterations; itn++)
    {
        ib1dft.transform();
        ibatch1ddft_gpu[itn] = ib1dft.getTime();
    	
	#if defined (FFTX_SYCL)
	{
          // fftx::OutStream() << "MKLFFT comparison not implemented printing first output element" << std::endl;
          // sycl::host_accessor h_acc(buf_Y);
          // fftx::OutStream() << h_acc[0] << std::endl;
	}
    #endif

	#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY ( outputHost2.data(), dY,
                          outputHost.size() * sizeof(std::complex<double>), FFTX_MEM_COPY_DEVICE_TO_HOST );
    
        //  Run the roc fft plan on the same input data
        if ( check_buff ) {
            FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECZ2Z ( plan,
                                       (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) tempX,
                                       (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) dY,
                                      FFTX_DEVICE_FFT_INVERSE);
            if ( res != FFTX_DEVICE_FFT_SUCCESS) {
                fftx::OutStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check" << std::endl;
                check_buff = false;
                // break;
            }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &invdevmilliseconds[itn], custart, custop );

            FFTX_DEVICE_MEM_COPY ( outDevfft2.data(), dY,
                              outDevfft2.size() * sizeof(std::complex<double>), FFTX_MEM_COPY_DEVICE_TO_HOST );

            // printf ( "DFT = %d Batch = %d Read = %s Write = %s  \tBatch 1D FFT (Inverse)\t", N, B, reads.c_str(), writes.c_str());
            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tBatch 1D FFT (Inverse)\t";
            checkOutputBuffers_fwd ( (FFTX_DOUBLECOMPLEX *) outputHost2.data(),
                                     (FFTX_DOUBLECOMPLEX *) outDevfft2.data(),
                                     (long) outDevfft2.size() );
        }
    #elif defined(FFTX_SYCL)
        if ( check_buff ) {
        // Copy output of FFTX transform from device to host
        // in order to check it against MKL FFT.

        // If this is absent then iterations after the first aren't correct.
        sycl::host_accessor tempXAcc(buf_tempX);

        // YAcc is double* because buf_Y is sycl::buffer<double>.
        sycl::host_accessor YAcc(buf_Y);
        for (int ind = 0; ind < N*B; ind++)
          {
            outputHost2[ind] =
              std::complex<double>(YAcc[2*ind], YAcc[2*ind+1]);
          }

        // Run MKL FFT plan on the same input data.
        for (int ind = 0; ind < N*B; ind++)
          { // These are both complex.
            inputVendorPtr[ind] = tempX[ind];
          }
        std::cout << "inputVendorPtr[0] = " << inputVendorPtr[0] << std::endl;
        std::cout << "inputVendorPtr[1] = " << inputVendorPtr[1] << std::endl;
        // std::cout << "tempX[0] = " << tempX[0] << std::endl;
        // std::cout << "tempX[1] = " << tempX[1] << std::endl;
        std::cout << "tempXAcc[0] = " << tempXAcc[0] << "," << tempXAcc[1] << std::endl;
        std::cout << "tempXAcc[1] = " << tempXAcc[2] << "," << tempXAcc[3] << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        // Perform forward transform on complex array
        sycl::event e = oneapi::mkl::dft::compute_backward(plan,
                                                           inputVendorPtr,
                                                           outputVendorPtr);
        Q.wait();
        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        uint64_t profile_nanosec = e_end - e_start;
        invdevmilliseconds[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

        std::cout << "output[0] FFTX " << outputHost2[0] << " MKL " << outputVendorPtr[0] << std::endl;
        std::cout << "output[1] FFTX " << outputHost2[1] << " MKL " << outputVendorPtr[1] << std::endl;
        for (int ind = 0; ind < B*N; ind++)
          { // These are both complex.
            outDevfft2[ind] = outputVendorPtr[ind];
          }
        checkOutputBuffers_fwd ( (FFTX_DOUBLECOMPLEX *) outputHost2.data(),
                                 (FFTX_DOUBLECOMPLEX *) outDevfft2.data(),
                                 (long) outDevfft2.size() );
        }
#endif
    }


#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on Batch 1D FFT (forward) for " << iterations
                      << " trials of size " << sizes.at(0)
                      << " and batch " << sizes.at(1) << ":"
                      << std::endl;
    fftx::OutStream() << "Trial #\tSpiral\t\t" << devfft << std::endl;
    for (int itn = 0; itn < iterations; itn++) {
      fftx::OutStream() << itn << "\t" << std::scientific << std::setprecision(7)
                        << batch1ddft_gpu[itn] << "\t"
                        << devmilliseconds[itn] << std::endl;
    }

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on Batch 1D FFT (inverse) for " << iterations
                      << " trials of size " << sizes.at(0)
                      << " and batch " << sizes.at(1) << ":"
                      << std::endl;
    fftx::OutStream() << "Trial #\tSpiral\t\t" << devfft << std::endl;
    for (int itn = 0; itn < iterations; itn++) {
      fftx::OutStream() << itn << "\t" << std::scientific << std::setprecision(7)
                        << ibatch1ddft_gpu[itn] << "\t"
                        << invdevmilliseconds[itn] << std::endl;
    }
#else
    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on Batch 1D FFT (forward) for " << iterations
                      << " trials of size " << sizes.at(0)
                      << " and batch " << sizes.at(1) << ":"
                      << std::endl;
    for (int itn = 0; itn < iterations; itn++) {
      fftx::OutStream() << itn << "\t" << std::scientific << std::setprecision(7)
                        << batch1ddft_gpu[itn] << std::endl;
    }

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on Batch 1D FFT (inverse) for " << iterations
                      << " trials of size " << sizes.at(0)
                      << " and batch " << sizes.at(1) << ":"
                      << std::endl;
    for (int itn = 0; itn < iterations; itn++) {
      fftx::OutStream() << itn << "\t" << std::scientific << std::setprecision(7)
                        << ibatch1ddft_gpu[itn] << std::endl;
    }
#endif

    fftx::OutStream() << prog << ": All done, exiting" << std::endl;

    return 0;
}
