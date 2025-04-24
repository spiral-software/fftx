#include <math.h>
#include "fftx.hpp"
#include "fftxutilities.hpp"
#include "fftxinterface.hpp"
#include "fftxbatch1dprdftObj.hpp"
#include "fftxibatch1dprdftObj.hpp"
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
#elif defined (FFTX_USE_FFTW)
#include "fftw3.h"
#endif

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#define FFTX_DOUBLECOMPLEX FFTX_DEVICE_FFT_DOUBLECOMPLEX
#define FFTX_DOUBLEREAL FFTX_DEVICE_FFT_DOUBLEREAL
#define FFTX_REALPART(z) z.x
#define FFTX_IMAGPART(z) z.y
#elif defined (FFTX_SYCL)
#define FFTX_DOUBLECOMPLEX std::complex<double>
#define FFTX_DOUBLEREAL double
#define FFTX_REALPART(z) z.real()
#define FFTX_IMAGPART(z) z.imag()
#elif defined (FFTX_USE_FFTW)
#define FFTX_DOUBLECOMPLEX fftw_complex
#define FFTX_DOUBLEREAL double
#define FFTX_REALPART(z) z[0]
#define FFTX_IMAGPART(z) z[1]
#else // need these #defines here, or else #ifdef around checkOutputs.
#define FFTX_DOUBLECOMPLEX std::complex<double>
#define FFTX_DOUBLEREAL double
#define FFTX_REALPART(z) z.real()
#define FFTX_IMAGPART(z) z.imag()
#endif

#define FFTX_BATCH_SEQUENTIAL 0
#define FFTX_BATCH_STRIDED 1

static void setRandomData ( double *data, long arrsz)
{
  for ( int ind = 0; ind < arrsz; ind++)
    {
      data[ind] = ((double) rand()) / (double) (RAND_MAX/2);
    }
  return;
}

// Set the imaginary part of the first element of each vector to 0.
static void setFirstImag0 ( double *data, std::vector<int> sizesTrunc )
{
    int N_adj = sizesTrunc.at(0);
    int B = sizesTrunc.at(1);
    int read = sizesTrunc.at(2);
    int write = sizesTrunc.at(3);
    int lenvec = 2*N_adj;
    int lenfull = B * lenvec;
    // There are B input vectors, each containing lenvec doubles.
    // First vector element has real part at [0] and imaginary part at [1].
    int first = 1;
    int stride = (read == FFTX_BATCH_SEQUENTIAL) ? lenvec : 2;
    for (int ind = first; ind < lenfull; ind += stride)
      {
        data[ind] = 0.;
      }
}

// Set the imaginary part of the last element of each vector to 0.
static void setLastImag0 ( double *data, std::vector<int> sizesTrunc )
{
    int N_adj = sizesTrunc.at(0);
    int B = sizesTrunc.at(1);
    int read = sizesTrunc.at(2);
    int write = sizesTrunc.at(3);
    int lenvec = 2*N_adj;
    int lenfull = B * lenvec;
    // There are B input vectors, each containing lenvec doubles.
    int first = (read == FFTX_BATCH_SEQUENTIAL) ? (lenvec-1) : ((B-1)*lenvec + 1);
    int stride = (read == FFTX_BATCH_SEQUENTIAL) ? lenvec : 2;
    for (int ind = first; ind < lenfull; ind += stride)
      {
        data[ind] = 0.;
      }
}


// Check that the buffers are identical (within roundoff)
// outputFFTXPtr is the output buffer from the Spiral-generated transform
// (result on GPU copied to host array outputFFTXPtr);
// outputVendorPtr is the output buffer from the vendor transform
// (result on GPU copied to host array outputVendorPtr).
// arrsz is the size of each array
static void checkOutputs_R2C ( FFTX_DOUBLECOMPLEX *outputFFTXPtr,
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
    
    // printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fftx::OutStream() << "Correct: " << (correct ? "True" : "False") << "\t"
                      << "Max delta = " << maxdelta << std::endl;
    std::flush(fftx::OutStream());

    return;
}

static void checkOutputs_C2R ( FFTX_DOUBLEREAL *outputFFTXPtr,
                               FFTX_DOUBLEREAL *outputVendorPtr,
                               long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int ind = 0; ind < arrsz; ind++ )
      {
        FFTX_DOUBLEREAL s = outputFFTXPtr[ind];
        FFTX_DOUBLEREAL c = outputVendorPtr[ind];

        double deltar = s - c;
        bool   elem_correct = ( abs(deltar) < 1e-7 );
        updateMaxAbs(maxdelta, deltar);
        correct &= elem_correct;
    }
    
    // printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fftx::OutStream() << "Correct: " << (correct ? "True" : "False") << "\t"
                      << "Max delta = " << maxdelta << std::endl;
    std::flush(fftx::OutStream());

    return;
}


int main(int argc, char* argv[])
{
    int iterations = 2;
    int N = 64; // default cube dimensions
    int B = 4;
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

    int N_adj = (int) ( N / 2 ) + 1;
    std::vector<int> sizes{N,B, read,write};
    std::vector<int> sizesTrunc{N_adj,B, read,write};

    long npts = B * N;
    long bytes = npts * sizeof(double);

    long nptsTrunc = B * N_adj;
    long bytesTrunc = nptsTrunc * sizeof(std::complex<double>);

    fftx::OutStream() << std::scientific << std::uppercase;

    std::vector<double> realFFTXHostVector(npts);
    std::vector<double> realVendorHostVector(npts);
    std::vector<std::complex<double>> complexFFTXHostVector(npts);
    std::vector<std::complex<double>> complexVendorHostVector(npts);

    double* realFFTXHostPtr = (double*) realFFTXHostVector.data();
    double* realVendorHostPtr = (double*) realVendorHostVector.data();
    std::complex<double>* complexFFTXHostPtr = (std::complex<double>*) complexFFTXHostVector.data();
    std::complex<double>* complexVendorHostPtr = (std::complex<double>*) complexVendorHostVector.data();
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    if ( FFTX_DEBUGOUT )fftx::OutStream() << "allocating memory" << std::endl;
    double* realFFTXTfmPtr;
    double* realVendorTfmPtr;
    std::complex<double>* complexFFTXTfmPtr;
    std::complex<double>* complexVendorTfmPtr;
    FFTX_DEVICE_MALLOC((void**)&realFFTXTfmPtr, bytes);
    FFTX_DEVICE_MALLOC((void**)&realVendorTfmPtr, bytes);
    FFTX_DEVICE_MALLOC((void**)&complexFFTXTfmPtr, bytesTrunc);
    FFTX_DEVICE_MALLOC((void**)&complexVendorTfmPtr, bytesTrunc);
#elif defined (FFTX_SYCL)
    sycl::buffer<double> realFFTXBuffer(realFFTXHostPtr, npts);
    sycl::buffer<double> realVendorBuffer(realVendorHostPtr, npts);
    // Use sycl::buffer on double because of problems if on complex.
    sycl::buffer<double> complexFFTXBuffer((double*) complexFFTXHostPtr, nptsTrunc * 2);
    sycl::buffer<double> complexVendorBuffer((double*) complexVendorHostPtr, nptsTrunc * 2);
#else // CPU
    double* realFFTXTfmPtr = (double *) realFFTXHostPtr;
    double* complexFFTXTfmPtr = (double *) complexFFTXHostPtr;
#endif

    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "memory allocated" << std::endl;

    // Order within args for FFTX:  output, input.
#if defined FFTX_CUDA
    std::vector<void*> argsR2C{&complexFFTXTfmPtr, &realFFTXTfmPtr};
    std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
    std::string vendorfft  = "cufft";
#elif defined FFTX_HIP
    std::vector<void*> argsR2C{complexFFTXTfmPtr, realFFTXTfmPtr};
    std::string descrip = "AMD GPU";                //  "CPU and GPU";
    std::string vendorfft  = "rocfft";
#elif defined FFTX_SYCL
    std::vector<void*> argsR2C{(void*)&(complexFFTXBuffer), (void*)&(realFFTXBuffer)};
    std::string descrip = "Intel GPU";                //  "CPU and GPU";
    std::string vendorfft  = "mklfft";
#else // CPU
    std::vector<void*> argsR2C{(void*)complexFFTXTfmPtr, (void*)realFFTXTfmPtr};
    std::string descrip = "CPU";                //  "CPU";
    std::string vendorfft = "FFTW";
#endif

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL) || defined(FFTX_USE_FFTW)
    // compare results of Spiral-RTC with vendor FFT or FFTW
    bool check_output = true;
#else
    bool check_output = false;
#endif

    BATCH1DPRDFTProblem b1prdft(argsR2C, sizes, "b1prdft");

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL) || defined(FFTX_USE_FFTW)
    int read_stride = (read == FFTX_BATCH_SEQUENTIAL) ? 1 : B;
    int read_dist = (read == FFTX_BATCH_SEQUENTIAL) ? N : 1;

    int write_stride = (write == FFTX_BATCH_SEQUENTIAL) ? 1 : B;
    int write_dist = (write == FFTX_BATCH_SEQUENTIAL) ? N_adj : 1;
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    //  Set up a plan to run the transform using cufft or rocfft.
    FFTX_DEVICE_FFT_HANDLE planR2C;
    FFTX_DEVICE_FFT_RESULT res;
    FFTX_DEVICE_FFT_TYPE   xfmtypeR2C = FFTX_DEVICE_FFT_D2Z ;
    FFTX_DEVICE_EVENT_T custart, custop;
    FFTX_DEVICE_EVENT_CREATE ( &custart );
    FFTX_DEVICE_EVENT_CREATE ( &custop );

    std::string read_str = (read == FFTX_BATCH_SEQUENTIAL) ? "APAR" : "AVEC";
    std::string write_str = (write == FFTX_BATCH_SEQUENTIAL) ? "APAR" : "AVEC";
  
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << read_str << ", "
                                           << write_str << std::endl;
    
    res = FFTX_DEVICE_FFT_PLAN_MANY( &planR2C, 1, &N, // plan, rank, length
                                     &N, read_stride, read_dist,
                                     &N_adj, write_stride, write_dist,
                                     xfmtypeR2C, B); // type, batch
    if ( res != FFTX_DEVICE_FFT_SUCCESS )
      {
        fftx::OutStream() << "Create FFTX_DEVICE_FFT_PLAN_MANY failed with error code "
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

    auto realVendorPtr =
      sycl::malloc_shared< double >
      (npts, sycl_device, sycl_context);
    // Should this be <complex> with nptsTrunc?
    auto complexVendorPtr =
      sycl::malloc_shared< double >
      (nptsTrunc * 2, sycl_device, sycl_context);

    // Set strides and distance for reading and writing MKL transforms.
    // N.B. Our application specifies either sequential or strided input,
    // and either sequential or strided output,
    // and these specifications apply to both the forward transform
    // and the inverse transform.

    // If we have specified sequential reads & strided writes, or vice versa,
    // then we need separate mkl::dft plans for compute_forward and
    // compute_backward, in order for the output ordering to be correct.
    std::int64_t read_strides[] = {0, read_stride};
    std::int64_t write_strides[] = {0, write_stride};
    
    // Initialize batch 1D FFT descriptor
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                 oneapi::mkl::dft::domain::REAL> plan(N);
    plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, B);
    plan.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, read_strides);
    plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, read_dist);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, write_strides);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, write_dist);
    plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    plan.commit(Q);
#elif defined (FFTX_USE_FFTW)
    fftw_plan planR2C =
      fftw_plan_many_dft_r2c(1, &N, B,
                             realFFTXHostPtr,
                             &N, read_stride, read_dist,
                             (fftw_complex*) complexVendorHostPtr,
                             &N_adj, write_stride, write_dist,
                             FFTW_ESTIMATE);
#endif
#endif
    
    float *batch1dprdft_gpu = new float[iterations];
    float *batch1dprdft_vendor_millisec = new float[iterations];

    for (int itn = 0; itn < iterations; itn++)
      {
        // Fill input buffer with random data, different each iteration.
        setRandomData( (double*) realFFTXHostPtr, npts );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY(realFFTXTfmPtr,
                             realFFTXHostPtr,
                             bytes,
                             FFTX_MEM_COPY_HOST_TO_DEVICE);
 	if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied real input from host to device\n";
#endif
        
        // Run transform on GPU: input realFFTXTfmPtr, output complexFFTXTfmPtr.
        b1prdft.transform();
        batch1dprdft_gpu[itn] = b1prdft.getTime();

        if ( check_output )
          { //  Run the vendor FFT plan on the same input data.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
            // Copy output of FFTX R2C transform from device to host
	    // in order to check it against cuFFT or rocFFT.
            FFTX_DEVICE_MEM_COPY(complexFFTXHostPtr,
                                 complexFFTXTfmPtr,
                                 bytesTrunc,
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);

            // Run cuFFT or rocFFT.
            FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECD2Z ( planR2C,
                                            (FFTX_DEVICE_FFT_DOUBLEREAL *) realFFTXTfmPtr,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) complexVendorTfmPtr
                                            );
            if ( res != FFTX_DEVICE_FFT_SUCCESS)
              {
                fftx::OutStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check"
                                  << std::endl;
                check_output = false;
                //  break;
              }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &batch1dprdft_vendor_millisec[itn], custart, custop );

            FFTX_DEVICE_MEM_COPY(complexVendorHostPtr,
                                 complexVendorTfmPtr,
                                 bytesTrunc,
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);
#elif defined (FFTX_SYCL)
            // Copy output of FFTX R2C transform from device to host
	    // in order to check it against MKL FFT.
            // Need both of these accessors, else errors can occur after first iteration.
            sycl::host_accessor realFFTXAcc(realFFTXBuffer);
            // N.B. complexFFTXAcc is double* because complexFFTXBuffer is sycl::buffer<double>.
            sycl::host_accessor complexFFTXAcc(complexFFTXBuffer);
	    for (int ind = 0; ind < nptsTrunc; ind++)
	      {
		complexFFTXHostPtr[ind] =
		  std::complex(complexFFTXAcc[2*ind + 0],
                               complexFFTXAcc[2*ind + 1]);
	      }

            // Run MKL FFT plan on the same input data.
	    for (int ind = 0; ind < npts; ind++)
	      {
		realVendorPtr[ind] = realFFTXHostPtr[ind];
	      }
	
	    auto start_time = std::chrono::high_resolution_clock::now();
	    //  Run the vendor FFT plan on the same input data.
	    // Perform forward transform on real array
	    // oneapi::mkl::dft::compute_forward(plan, realVendorPtr, complexVendorPtr).wait();
            sycl::event e = oneapi::mkl::dft::compute_forward(plan,
                                                              realVendorPtr,
                                                              complexVendorPtr);
            Q.wait();
	    auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t profile_nanosec = e_end - e_start;
            batch1dprdft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

            for (int ind = 0; ind < nptsTrunc; ind++)
              {
                complexVendorHostPtr[ind] =
                  std::complex<double>(complexVendorPtr[2*ind],
                                       complexVendorPtr[2*ind + 1]);
              }
#elif defined (FFTX_USE_FFTW)
            auto start = std::chrono::high_resolution_clock::now();
            fftw_execute(planR2C);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            batch1dprdft_vendor_millisec[itn] = duration.count();
#endif
            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tBatch 1D FFT (R2C)\t";
	    checkOutputs_R2C ( (FFTX_DOUBLECOMPLEX *) complexFFTXHostPtr,
                               (FFTX_DOUBLECOMPLEX *) complexVendorHostPtr,
                               nptsTrunc);
	  } // end check_output
      } // end iteration

    fftx::OutStream() << "Times in milliseconds for 1D FFT (R2C)"
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
                          << std::setw(17) << batch1dprdft_gpu[itn];
        if (check_output)
          {
            fftx::OutStream() << std::setw(17) << batch1dprdft_vendor_millisec[itn];
          }
        fftx::OutStream() << std::endl;
      }
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_DESTROY(planR2C);
#elif defined(FFTX_SYCL)
#elif defined(FFTX_USE_FFTW)
    fftw_destroy_plan(planR2C);
#endif

    delete[] batch1dprdft_gpu;
    delete[] batch1dprdft_vendor_millisec;

    // Set up the inverse transform.
    
    // Order within args for FFTX:  output, input.
#if defined FFTX_CUDA
    std::vector<void*> argsC2R{&realFFTXTfmPtr, &complexFFTXTfmPtr};
#elif defined FFTX_HIP
    std::vector<void*> argsC2R{realFFTXTfmPtr, complexFFTXTfmPtr};
#elif defined FFTX_SYCL
    std::vector<void*> argsC2R{(void*)&(realFFTXBuffer), (void*)&(complexFFTXBuffer)};
#else // CPU
    std::vector<void*> argsC2R{(void*)realFFTXTfmPtr, (void*)complexFFTXTfmPtr};
#endif

    IBATCH1DPRDFTProblem ib1prdft(argsC2R, sizes, "ib1prdft");

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_HANDLE planC2R;     
    FFTX_DEVICE_FFT_TYPE xfmtypeC2R = FFTX_DEVICE_FFT_Z2D ;

    int read_dist_inv = (read == FFTX_BATCH_SEQUENTIAL) ? N_adj : 1;
    int write_dist_inv = (write == FFTX_BATCH_SEQUENTIAL) ? N : 1;
    res = FFTX_DEVICE_FFT_PLAN_MANY ( &planC2R, 1, &N, // plan, rank, length
                                      &N_adj, read_stride, read_dist_inv,
                                      &N, write_stride, write_dist_inv,
                                      xfmtypeC2R, B); // type, batch
    if ( res != FFTX_DEVICE_FFT_SUCCESS )
      {
        fftx::OutStream() << "Create FFTX_DEVICE_FFT_PLAN_MANY failed with error code "
                          << res << " ... skip buffer check" << std::endl;
        check_output = false;
      }
#elif defined (FFTX_SYCL)
    // Need a separate plan for compute_backward if and only if read != write.
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                 oneapi::mkl::dft::domain::REAL> planInv(N);
    if (read != write)
      {
        // Use read_dist_inv and write_dist_inv for C2R transform.
        int read_dist_inv = (read == FFTX_BATCH_SEQUENTIAL) ? N_adj : 1;
        int write_dist_inv = (write == FFTX_BATCH_SEQUENTIAL) ? N : 1;
        planInv.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, B);
        planInv.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, write_strides);
        planInv.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, write_dist_inv);
        planInv.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, read_strides);
        planInv.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, read_dist_inv);
        planInv.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
        planInv.commit(Q);
      }
#elif defined(FFTX_USE_FFTW)
    int read_dist_inv = (read == FFTX_BATCH_SEQUENTIAL) ? N_adj : 1;
    int write_dist_inv = (write == FFTX_BATCH_SEQUENTIAL) ? N : 1;
    fftw_plan planC2R =
      fftw_plan_many_dft_c2r(1, &N, B,
                             (fftw_complex*) complexFFTXHostPtr,
                             &N_adj, read_stride, read_dist_inv,
                             realVendorHostPtr,
                             &N, write_stride, write_dist_inv,
                             FFTW_ESTIMATE);
#endif

    float *ibatch1dprdft_gpu = new float[iterations];
    float *ibatch1dprdft_vendor_millisec = new float[iterations];
    for (int itn = 0; itn < iterations; itn++)
      {
        // Fill input buffer with random data, different each iteration.
        setRandomData( (double*) complexFFTXHostPtr, nptsTrunc * 2);
        // For complex vector v of length n to have the right symmetry:
        // v[0].imag == 0;
        // if N is even then v[N/2].imag == 0;
        // for k = N_adj + 1 to  N - 1, v[k] = conj(v[N - k]).
        setFirstImag0( (double*) complexFFTXHostPtr, sizesTrunc );
        if (N % 2 == 0) setLastImag0( (double*) complexFFTXHostPtr, sizesTrunc );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        FFTX_DEVICE_MEM_COPY(complexFFTXTfmPtr,
                             complexFFTXHostPtr,
                             bytesTrunc,
                             FFTX_MEM_COPY_HOST_TO_DEVICE);
        if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied IBATCH1DPRDFT input from host to device" << std::endl;
#endif

        // Run transform on GPU: input complexFFTXTfmPtr, output realFFTXTfmPtr.
        ib1prdft.transform();
        ibatch1dprdft_gpu[itn] = ib1prdft.getTime();

        if ( check_output )
          { //  Run the vendor FFT plan on the same input data.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
            // Copy output of FFTX C2R transform from device to host
	    // in order to check it against cuFFT or rocFFT.
            FFTX_DEVICE_MEM_COPY(realFFTXHostPtr,
                                 realFFTXTfmPtr,
                                 bytes,
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);
            
            // Run cuFFT or rocFFT.
	    FFTX_DEVICE_EVENT_RECORD ( custart );
            res = FFTX_DEVICE_FFT_EXECZ2D ( planC2R,
                                            (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) complexFFTXTfmPtr,
                                            (FFTX_DEVICE_FFT_DOUBLEREAL *) realVendorTfmPtr
                                            );
            if ( res != FFTX_DEVICE_FFT_SUCCESS)
              {
                fftx::OutStream() << "Launch FFTX_DEVICE_FFT_EXEC failed with error code "
                                  << res << " ... skip buffer check"
                                  << std::endl;
                check_output = false;
                //  break;
              }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &ibatch1dprdft_vendor_millisec[itn], custart, custop );
            
            FFTX_DEVICE_MEM_COPY(realVendorHostPtr,
                                 realVendorTfmPtr,
                                 bytes,
                                 FFTX_MEM_COPY_DEVICE_TO_HOST);
#elif defined (FFTX_SYCL)
            // Copy output of FFTX C2R transform from device to host
	    // in order to check it against MKL FFT.
            // Need both of these accessors, else errors can occur after first iteration.
            sycl::host_accessor realFFTXAcc(realFFTXBuffer);
            // N.B. complexFFTXAcc is double* because complexFFTXBuffer is sycl::buffer<double>.
            sycl::host_accessor complexFFTXAcc(complexFFTXBuffer);
	    for (int ind = 0; ind < npts; ind++)
	      {
                realVendorPtr[ind] = realFFTXAcc[ind];
	      }

            // Run MKL FFT plan on the same input data.
            for (int ind = 0; ind < nptsTrunc; ind++)
              {
                std::complex<double> v = complexFFTXHostPtr[ind];
                complexVendorPtr[2*ind + 0] = FFTX_REALPART(v);
                complexVendorPtr[2*ind + 1] = FFTX_IMAGPART(v);
              }

	    auto start_time = std::chrono::high_resolution_clock::now();
	    // Run the vendor FFT plan on the same input data.
	    // Perform backward transform on complex array
	    // oneapi::mkl::dft::compute_backward(plan, complexVendorPtr, realVendorPtr).wait();
            sycl::event e = oneapi::mkl::dft::compute_backward((read == write) ? plan : planInv,
                                                               complexVendorPtr,
                                                               realVendorPtr);
            Q.wait();
	    auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t profile_nanosec = e_end - e_start;
	    // std::chrono::duration<float, std::milli> duration = end_time - start_time;
	    // ibatch1dprdft_vendor_millisec[itn] = duration.count();
            ibatch1dprdft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

            for (int ind = 0; ind < npts; ind++)
              {
                realVendorHostPtr[ind] = realVendorPtr[ind];
              }
#elif defined (FFTX_USE_FFTW)
            auto start = std::chrono::high_resolution_clock::now();
            fftw_execute(planC2R);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            ibatch1dprdft_vendor_millisec[itn] = duration.count();
#endif
            fftx::OutStream() << "DFT = " << N
                              << " Batch = " << B
                              << " Read = " << reads
                              << " Write = " << writes
                              << " \tBatch 1D FFT (C2R)\t";
	    checkOutputs_C2R ((FFTX_DOUBLEREAL*) realFFTXHostPtr,
                              (FFTX_DOUBLEREAL*) realVendorHostPtr,
                              npts );
	  } // end check_output
      } // end iteration

    fftx::OutStream() << "Times in milliseconds for 1D FFT (C2R)"
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
                          << std::setw(17) << ibatch1dprdft_gpu[itn];
        if (check_output)
          {
            fftx::OutStream() << std::setw(17) << ibatch1dprdft_vendor_millisec[itn];
          }
        fftx::OutStream() << std::endl;
      }
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_DESTROY(planC2R);
#elif defined(FFTX_SYCL)
#elif defined(FFTX_USE_FFTW)
    fftw_destroy_plan(planC2R);
#endif

    delete[] ibatch1dprdft_gpu;
    delete[] ibatch1dprdft_vendor_millisec;
    
    // Clean up.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    fftxDeviceFree(realFFTXTfmPtr);
    fftxDeviceFree(realVendorTfmPtr);
    fftxDeviceFree(complexFFTXTfmPtr);
    fftxDeviceFree(complexVendorTfmPtr);
#elif defined(FFTX_SYCL)
    sycl::free(realVendorPtr, sycl_context);
    sycl::free(complexVendorPtr, sycl_context);
#else
    // delete[] symbolTfmPtr;
#endif

    fftx::OutStream() << prog << ": All done, exiting" << std::endl;
    std::flush(fftx::OutStream());

    return 0;
}
