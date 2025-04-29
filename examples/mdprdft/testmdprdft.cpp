#include "fftx.hpp"
#include "fftxutilities.hpp"
#include "fftxinterface.hpp"
#include "fftxmdprdftObj.hpp"
#include "fftximdprdftObj.hpp"
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

static void setRandomData ( double *data, long arrsz)
{
  for ( int ind = 0; ind < arrsz; ind++)
    {
      data[ind] = ((double) rand()) / (double) (RAND_MAX/2);
    }
  return;
}

// Check that the buffers are identical (within roundoff)
// outputFFTXPtr is the output buffer from the Spiral-generated transform
// (result on GPU copied to host array outputFFTXPtr);
// outputVendorPtr is the output buffer from the vendor transform
// (result on GPU copied to host array outputVendorPtr).
// arrsz is the size of each array
static bool checkOutputs_R2C ( FFTX_DOUBLECOMPLEX *outputFFTXPtr,
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

    return correct;
}

static bool checkOutputs_C2R ( FFTX_DOUBLEREAL *outputFFTXPtr,
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
            // printf ( "Usage: %s: [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]\n", argv[0] );
            fftx::OutStream() << "Usage: " << argv[0]
                              << ": [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]"
                              << std::endl;
            exit (0);
        default:
            fftx::OutStream() << prog
                              << ": unknown argument: " << argv[1]
                              << " ... ignored" << std::endl;
        }
        argv++, argc--;
    }

    fftx::OutStream() << std::scientific << std::uppercase;
    
    int K_adj = (int) ( kk / 2 ) + 1;
    fftx::OutStream() << mm << " " << nn << " " << kk << std::endl;
    int status = 0;
    std::vector<int> sizes{mm,nn,kk};
    std::vector<int> sizesTrunc{mm,nn,K_adj};
    fftx::box_t<3> domain ( point_t<3> ( { { 1, 1, 1 } } ),
                            point_t<3> ( { { mm, nn, kk } } ));
    fftx::box_t<3> outputd ( point_t<3> ( { { 1, 1, 1 } } ),
                            point_t<3> ( { { mm, nn, K_adj } } ));

    fftx::array_t<3,double> realFFTXHostArray(domain);
    fftx::array_t<3,double> realVendorHostArray(domain);
    fftx::array_t<3,std::complex<double>> complexFFTXHostArray(outputd);
    fftx::array_t<3,std::complex<double>> complexVendorHostArray(outputd);

    size_t npts = domain.size();
    size_t nptsTrunc = outputd.size();

    auto realFFTXHostPtr = realFFTXHostArray.m_data.local();
    auto realVendorHostPtr = realVendorHostArray.m_data.local();
    auto complexFFTXHostPtr = complexFFTXHostArray.m_data.local();
    auto complexVendorHostPtr = complexVendorHostArray.m_data.local();

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    if ( FFTX_DEBUGOUT )fftx::OutStream() << "allocating memory" << std::endl;
    FFTX_DEVICE_PTR realFFTXTfmPtr = fftxDeviceMallocForHostArray(realFFTXHostArray);
    FFTX_DEVICE_PTR realVendorTfmPtr = fftxDeviceMallocForHostArray(realVendorHostArray);
    FFTX_DEVICE_PTR complexFFTXTfmPtr = fftxDeviceMallocForHostArray(complexFFTXHostArray);
    FFTX_DEVICE_PTR complexVendorTfmPtr = fftxDeviceMallocForHostArray(complexVendorHostArray);
    FFTX_DEVICE_PTR symbolTfmPtr = (FFTX_DEVICE_PTR) NULL;
#elif defined(FFTX_SYCL)
    sycl::buffer<double> realFFTXTfmPtr(realFFTXHostPtr, npts);
    sycl::buffer<double> realVendorTfmPtr(realVendorHostPtr, npts);
    sycl::buffer<double> symbolTfmPtr((double*) NULL, 0);
    // Use sycl::buffer on double because of problems if on complex.
    sycl::buffer<double> complexFFTXTfmPtr((double*) complexFFTXHostPtr, nptsTrunc * 2);
    sycl::buffer<double> complexVendorTfmPtr((double*) complexVendorHostPtr, nptsTrunc * 2);
#else // CPU
    double* realFFTXTfmPtr = (double *) realFFTXHostPtr;
    double* complexFFTXTfmPtr = (double *) complexFFTXHostPtr;
    double* symbolTfmPtr = (double *) NULL;
#endif
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "memory allocated" << std::endl;

    // Order within args for FFTX:  output, input, symbol.
#if defined FFTX_CUDA
    std::vector<void*> argsR2C{&complexFFTXTfmPtr, &realFFTXTfmPtr, &symbolTfmPtr};
    std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
    std::string vendorfft  = "cufft";
#elif defined FFTX_HIP
    std::vector<void*> argsR2C{complexFFTXTfmPtr, realFFTXTfmPtr, symbolTfmPtr};
    std::string descrip = "AMD GPU";                //  "CPU and GPU";
    std::string vendorfft  = "rocfft";
#elif defined FFTX_SYCL
    std::vector<void*> argsR2C{(void*)&(complexFFTXTfmPtr), (void*)&(realFFTXTfmPtr), (void*)&(symbolTfmPtr)};
    std::string descrip = "Intel GPU";                //  "CPU and GPU";
    std::string vendorfft  = "mklfft";
#else // CPU
    std::vector<void*> argsR2C{(void*)complexFFTXTfmPtr, (void*)realFFTXTfmPtr, (void*)symbolTfmPtr};
    std::string descrip = "CPU";                //  "CPU";
    std::string vendorfft = "FFTW";
#endif

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL) || defined(FFTX_USE_FFTW)
    // compare results of Spiral-RTC with vendor FFT or FFTW
    bool check_output = true;
#else
    bool check_output = false;
#endif

    float *mdprdft_gpu = new float[iterations];
    float *mdprdft_vendor_millisec = new float[iterations];
    MDPRDFTProblem mdp(argsR2C, sizes, "mdprdft");
    
    //  Set up a plan to run the transform using vendor FFT.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_HANDLE planR2C;
    FFTX_DEVICE_FFT_RESULT res;
    FFTX_DEVICE_FFT_TYPE   xfmtypeR2C = FFTX_DEVICE_FFT_D2Z ;
    FFTX_DEVICE_EVENT_T custart, custop;
    FFTX_DEVICE_EVENT_CREATE ( &custart );
    FFTX_DEVICE_EVENT_CREATE ( &custop );
    
    res = FFTX_DEVICE_FFT_PLAN3D ( &planR2C, mm, nn, kk, xfmtypeR2C );
    if ( res != FFTX_DEVICE_FFT_SUCCESS )
      {
        fftx::OutStream() << "Create FFTX_DEVICE_FFT_PLAN3D failed with error code "
                          << res << " ... skip buffer check" << std::endl;
        check_output = false;
        status++;
      }
#elif defined (FFTX_SYCL)
    // Problems with using MKLFFT now.
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

    auto realVendorPtr = sycl::malloc_shared< double >
      (npts, sycl_device, sycl_context);
    auto complexVendorPtr = sycl::malloc_shared< double >
      (nptsTrunc * 2, sycl_device, sycl_context);

    // Initialize 3D FFT descriptor
    std::vector<std::int64_t> Nvec{mm, nn, kk};
    std::int64_t fwd_strides[] = {0, nn*kk, kk, 1};
    std::int64_t bwd_strides[] = {0, nn*K_adj, K_adj, 1};
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                 oneapi::mkl::dft::domain::REAL>
      plan(Nvec);
    plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, npts);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, nptsTrunc);
    plan.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, fwd_strides);
    plan.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, bwd_strides);
    plan.commit(Q);
#elif defined (FFTX_USE_FFTW)
    fftw_plan planR2C = fftw_plan_dft_r2c_3d(mm, nn, kk,
                                             realFFTXHostPtr,
                                             (fftw_complex*) complexVendorHostPtr,
                                             FFTW_ESTIMATE);
#endif

    for (int itn = 0; itn < iterations; itn++)
      {
        // Fill input buffer with random data, different each iteration.
	setRandomData( (double*) realFFTXHostPtr, npts);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        fftxCopyHostArrayToDevice(realFFTXTfmPtr, realFFTXHostArray);
 	if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied MDPRDFT input from host to device\n";
#endif
        
        // Run transform on GPU: input realFFTXTfmPtr, output complexFFTXTfmPtr.
        mdp.transform();
        mdprdft_gpu[itn] = mdp.getTime();

        if ( check_output )
          { //  Run the vendor FFT plan, or FFTW, on the same input data.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
            // Copy output of FFTX R2C transform from device to host
	    // in order to check it against cuFFT or rocFFT.
            fftxCopyDeviceToHostArray(complexFFTXHostArray, complexFFTXTfmPtr);

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
                status++;
                //  break;
              }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &mdprdft_vendor_millisec[itn], custart, custop );

            fftxCopyDeviceToHostArray(complexVendorHostArray, complexVendorTfmPtr);
#elif defined (FFTX_SYCL)
            // Copy output of FFTX R2C transform from device to host
	    // in order to check it against MKL FFT.
            // Need both of these accessors, else errors can occur after first iteration.
            sycl::host_accessor realFFTXHostAcc(realFFTXTfmPtr);
            // N.B. complexFFTXHostAcc is double* because complexFFTXTfmPtr is sycl::buffer<double>.
            sycl::host_accessor complexFFTXHostAcc(complexFFTXTfmPtr);
	    for (int ind = 0; ind < nptsTrunc; ind++)
	      {
		complexFFTXHostPtr[ind] =
		  std::complex(complexFFTXHostAcc[2*ind + 0],
                               complexFFTXHostAcc[2*ind + 1]);
	      }

            // Copy input to use in MKL call.
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
	    // std::chrono::duration<float, std::milli> duration = end_time - start_time;
	    // mdprdft_vendor_millisec[itn] = duration.count();
            mdprdft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

            for (int ind = 0; ind < nptsTrunc; ind++)
              {
                complexVendorHostPtr[ind] =
                  std::complex<double>(complexVendorPtr[2*ind + 0],
                                       complexVendorPtr[2*ind + 1]);
              }
#elif defined(FFTX_USE_FFTW)
            auto start = std::chrono::high_resolution_clock::now();
            fftw_execute(planR2C);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            mdprdft_vendor_millisec[itn] = duration.count();
#endif
            fftx::OutStream() << "cube = [ "
                              << mm << ", " << nn << ", " << kk << " ]\t"
                              << "MDPRDFT (Forward) \t";
	    bool chk = checkOutputs_R2C ( (FFTX_DOUBLECOMPLEX *) complexFFTXHostPtr,
                                          (FFTX_DOUBLECOMPLEX *) complexVendorHostPtr,
                                          (long) nptsTrunc);
            if (chk == false) status++;
	  } // end check_output
      } // end iteration

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on MDPRDFT (forward) for "
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
                          << std::setw(17) << mdprdft_gpu[itn];
        if (check_output)
          {
            fftx::OutStream() << std::setw(17) << mdprdft_vendor_millisec[itn];
          }
        fftx::OutStream() << std::endl;
      }

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_DESTROY(planR2C);
#elif defined(FFTX_SYCL)
#elif defined(FFTX_USE_FFTW)
    fftw_destroy_plan(planR2C);
#endif

    delete[] mdprdft_gpu;
    delete[] mdprdft_vendor_millisec;

    // Set up the inverse transform.
    
    // Order within args for FFTX:  output, input, symbol.
#if defined FFTX_CUDA
    std::vector<void*> argsC2R{&realFFTXTfmPtr, &complexFFTXTfmPtr, &symbolTfmPtr};
#elif defined FFTX_HIP
    std::vector<void*> argsC2R{realFFTXTfmPtr, complexFFTXTfmPtr, symbolTfmPtr};
#elif defined FFTX_SYCL
    std::vector<void*> argsC2R{(void*)&(realFFTXTfmPtr), (void*)&(complexFFTXTfmPtr), (void*)&(symbolTfmPtr)};	
#else // CPU
    std::vector<void*> argsC2R{(void*)realFFTXTfmPtr, (void*)complexFFTXTfmPtr, (void*)symbolTfmPtr};
#endif

    float *imdprdft_gpu = new float[iterations];
    float *imdprdft_vendor_millisec = new float[iterations];
    IMDPRDFTProblem imdp(argsC2R, sizes, "imdprdft");

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_HANDLE planC2R;     
    FFTX_DEVICE_FFT_TYPE xfmtypeC2R = FFTX_DEVICE_FFT_Z2D ;
    res = FFTX_DEVICE_FFT_PLAN3D ( &planC2R, mm, nn, kk, xfmtypeC2R );
    if ( res != FFTX_DEVICE_FFT_SUCCESS )
      {
        fftx::OutStream() << "Create FFTX_DEVICE_FFT_PLAN3D failed with error code "
                          << res << " ... skip buffer check" << std::endl;
        check_output = false;
        status++;
      }
#elif defined FFTX_SYCL
#elif defined (FFTX_USE_FFTW)
    fftw_plan planC2R = fftw_plan_dft_c2r_3d(mm, nn, kk,
                                             (fftw_complex*) complexFFTXHostPtr,
                                             realVendorHostPtr,
                                             FFTW_ESTIMATE);
#endif

    for (int itn = 0; itn < iterations; itn++)
      {
        // Fill input buffer with random data, different each iteration.
        setRandomData( (double*) complexFFTXHostPtr, 2 * nptsTrunc);
        symmetrizeHermitian(complexFFTXHostArray, realFFTXHostArray);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)    
        fftxCopyHostArrayToDevice(complexFFTXTfmPtr, complexFFTXHostArray);
        if ( FFTX_DEBUGOUT ) fftx::OutStream() << "copied IMDPRDFT input from host to device" << std::endl;
#endif

        // Run transform on GPU: input complexFFTXTfmPtr, output realFFTXTfmPtr.
        imdp.transform();
        imdprdft_gpu[itn] = imdp.getTime();

        if ( check_output )
          { //  Run the vendor FFT plan, or FFTW, on the same input data.
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
            // Copy output of FFTX C2R transform from device to host
	    // in order to check it against cuFFT or rocFFT.
            fftxCopyDeviceToHostArray(realFFTXHostArray, realFFTXTfmPtr);

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
                status++;
                //  break;
              }
            FFTX_DEVICE_EVENT_RECORD ( custop );
            FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
            FFTX_DEVICE_EVENT_ELAPSED_TIME ( &imdprdft_vendor_millisec[itn], custart, custop );
            
            fftxCopyDeviceToHostArray(realVendorHostArray, realVendorTfmPtr);
#elif defined (FFTX_SYCL)
            // Copy output of FFTX C2R transform from device to host
	    // in order to check it against MKL FFT.
            // Need both of these accessors, else errors can occur after first iteration.
            sycl::host_accessor realFFTXHostAcc(realFFTXTfmPtr);
            // N.B. complexFFTXHostAcc is double* because complexFFTXTfmPtr is sycl::buffer<double>.
            sycl::host_accessor complexFFTXHostAcc(complexFFTXTfmPtr);
	    for (int ind = 0; ind < npts; ind++)
	      {
                realVendorPtr[ind] = realFFTXHostAcc[ind];
	      }

            // Copy input to use in MKL call.
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
            sycl::event e = oneapi::mkl::dft::compute_backward(plan,
                                                               complexVendorPtr,
                                                               realVendorPtr);
            Q.wait();
	    auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t e_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t profile_nanosec = e_end - e_start;
	    // std::chrono::duration<float, std::milli> duration = end_time - start_time;
	    // imdprdft_vendor_millisec[itn] = duration.count();
            imdprdft_vendor_millisec[itn] = profile_nanosec * 1.e-6; // convert nanoseconds to milliseconds

	    for (int ind = 0; ind < npts; ind++)
	      {
		realVendorHostPtr[ind] = realVendorPtr[ind];
	      }
#elif defined(FFTX_USE_FFTW)
            auto start = std::chrono::high_resolution_clock::now();
            fftw_execute(planC2R);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            imdprdft_vendor_millisec[itn] = duration.count();
#endif
            fftx::OutStream() << "cube = [ "
                              << mm << ", " << nn << ", " << kk << " ]\t"
                              << "IMDPRDFT (Inverse)\t";
	    bool chk = checkOutputs_C2R ((FFTX_DOUBLEREAL*) realFFTXHostPtr,
                                         (FFTX_DOUBLEREAL*) realVendorHostPtr,
                                         (long) npts );
            if (chk == false) status++;
	  } // end check_output
      } // end iteration

    fftx::OutStream() << "Times in milliseconds for " << descrip
                      << " on MDPRDFT (inverse) for "
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
                          << std::setw(17) << imdprdft_gpu[itn];
        if (check_output)
          {
            fftx::OutStream() << std::setw(17) << imdprdft_vendor_millisec[itn];
          }
        fftx::OutStream() << std::endl;
      }
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    FFTX_DEVICE_FFT_DESTROY(planC2R);
#elif defined(FFTX_SYCL)
#elif defined(FFTX_USE_FFTW)
    fftw_destroy_plan(planC2R);
#endif

    delete[] imdprdft_gpu;
    delete[] imdprdft_vendor_millisec;
    
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

    return status;
}
