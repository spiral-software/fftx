#include "fftx3.hpp"
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
// #include "fftx3mkl.hpp"
#include <CL/sycl.hpp>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/vm.hpp>
#else  
#include "cpubackend.hpp"
#endif
#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined (FFTX_SYCL)
#include "device_macros.h"
#endif

#define CHECK_TOLERANCE 1e-7

#define WITHIN_TOLERANCE(x) ( abs(x) < CHECK_TOLERANCE )

inline void update_max_abs(double& valmax, double val)
{
  double absval = abs(val);
  if (absval > valmax)
    {
      valmax = absval;
    }
}
		       
//  Build a random input buffer for Spiral and rocfft
//  host_in is the host buffer to setup -- it'll be copied to the device later
//  sizes is a vector with the X, Y, & Z dimensions
static void buildInputBuffer ( double *host_in, std::vector<int> sizes )
{
    for ( int imm = 0; imm < sizes.at(0); imm++ ) {
        for ( int inn = 0; inn < sizes.at(1); inn++ ) {
            for ( int ikk = 0; ikk < sizes.at(2); ikk++ ) {
                int offset = (ikk + inn * sizes.at(2) + imm * sizes.at(1) * sizes.at(2)) * 2;
                host_in[offset + 0] =
		  1. - ((double) rand()) / (double) (RAND_MAX/2);
                host_in[offset + 1] =
		  1. - ((double) rand()) / (double) (RAND_MAX/2);
            }
        }
    }
    printf("buildInputBuffer host_in[0] = %.5e %.5e\n",
	   host_in[0], host_in[1]);
    return;
}

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
// Check that the buffers are identical (within roundoff).
// spiral_Y is the output buffer from the Spiral-generated transform
// (result on GPU copied to host array spiral_Y).
// devfft_Y is the output buffer from the device-equivalent transform
// (result on GPU copied to host array devfft_Y).
// arrsz is the number of points in each array.

static void checkOutputBuffers (
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
				DEVICE_FFT_DOUBLECOMPLEX *spiral_Y,
				DEVICE_FFT_DOUBLECOMPLEX *devfft_Y,
#elif defined (FFTX_SYCL)
				std::complex<double> *spiral_Y,
				std::complex<double> *devfft_Y,
#endif
				size_t arrsz )
{
  bool correct = true;
  double maxdelta = 0.0;
  for ( int indx = 0; indx < arrsz; indx++ )
    {
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      DEVICE_FFT_DOUBLECOMPLEX spiral_elem = spiral_Y[indx];
      DEVICE_FFT_DOUBLECOMPLEX dev_elem = devfft_Y[indx];
      double diff_real = spiral_elem.x - dev_elem.x;
      double diff_imag = spiral_elem.y - dev_elem.y;
#elif defined (FFTX_SYCL)
      std::complex<double> spiral_elem = spiral_Y[indx];
      std::complex<double> dev_elem = devfft_Y[indx];
      double diff_real = std::real(spiral_elem) - std::real(dev_elem);
      double diff_imag = std::imag(spiral_elem) - std::imag(dev_elem);
#endif
      update_max_abs(maxdelta, diff_real);
      update_max_abs(maxdelta, diff_imag);
      correct &= WITHIN_TOLERANCE(diff_real) && WITHIN_TOLERANCE(diff_imag);
    }
  
  printf ( "Correct: %s\tMax delta = %E\n",
	   (correct ? "True" : "False"), maxdelta );
  fflush ( stdout );
  
  return;
}
#endif

int main(int argc, char* argv[])
{
  /*
    Define input arguments and set defaults.
  */
  int iterations = 2;
  int mm = 24, nn = 32, kk = 40; // default cube dimensions
  char *prog = argv[0];
  int baz = 0;

  /*
    Read input arguments from command line.
  */
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

  /*
    Define input and output domains, arrays, sizes.
   */
  std::vector<int> sizes{mm,nn,kk};
  fftx::box_t<3> domain ( point_t<3> ( { { 1, 1, 1 } } ),
			  point_t<3> ( { { mm, nn, kk } } ));
    
  fftx::array_t<3,std::complex<double>> inputHost(domain);
  fftx::array_t<3,std::complex<double>> outputHost(domain);
  fftx::array_t<3,std::complex<double>> outDevfft(domain);

  size_t inputPoints = inputHost.m_domain.size();
  size_t outputPoints = outputHost.m_domain.size();

  size_t inputBytes = inputPoints * sizeof(std::complex<double>);
  size_t outputBytes = outputPoints * sizeof(std::complex<double>);

  auto inputHostData = inputHost.m_data.local();
  auto outputHostData = outputHost.m_data.local();
  auto outputDevfftData = outDevfft.m_data.local();

  /*
    Define buffers din, dout, dsym for data.
   */
#if defined FFTX_CUDA
  CUdevice cuDevice;
  CUcontext context;
  cuInit(0);
  cuDeviceGet(&cuDevice, 0);
  cuCtxCreate(&context, 0, cuDevice);
  //  CUdeviceptr  din, dout, dsym;
  std::complex<double> *din, *dout, *dsym;
#elif defined FFTX_HIP
  hipDeviceptr_t  din, dout, dsym;
#else
  double * din, *dout, *dsym;
  din = (double *) inputHostData;
  dout = (double *) outputHostData;
  dsym = new double[outputPoints];
#endif

  /*
    Allocate on CUDA or HIP or SYCL device.
  */
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  std::cout << "allocating memory\n";
  DEVICE_MALLOC((void **)&din, inputBytes);
  if ( DEBUGOUT ) std::cout << "allocated X\n";

  DEVICE_MALLOC((void **)&dout, outputBytes);
  if ( DEBUGOUT ) std::cout << "allocated Y\n";

  DEVICE_MALLOC((void **)&dsym,  outputBytes);
#elif defined(FFTX_SYCL)
  sycl::buffer<std::complex<double>> buf_out(outputHostData, outputPoints);
  sycl::buffer<std::complex<double>> buf_in(inputHostData, sycl::range<1>(inputPoints));
  sycl::buffer<std::complex<double>> buf_sym(inputHostData, inputPoints);
#endif

  /*
    Additional code for GPU programs.
  */
  // double* mddft_cpu = new double[iterations];
  // double* imddft_cpu = new double[iterations];
  // #if defined (FFTX_CUDA) || defined(FFTX_HIP)
  // additional code for GPU programs
  float *mddft_gpu = new float[iterations];
  float *imddft_gpu = new float[iterations];
#if defined FFTX_CUDA
  std::vector<void*> args{&dout,&din,&dsym};
  std::string descrip = "NVIDIA GPU";                //  "CPU and GPU";
  std::string devfft  = "cufft";
#elif defined FFTX_HIP
  std::vector<void*> args{dout,din,dsym};
  std::string descrip = "AMD GPU";                //  "CPU and GPU";
  std::string devfft  = "rocfft";
#elif defined FFTX_SYCL
  std::vector<void*> args{(void*)&(buf_out),(void*)&(buf_in),(void*)&(buf_sym)};
  std::string descrip = "Intel GPU";                //  "CPU and GPU";
  std::string devfft  = "mklfft";
#else
  std::vector<void*> args{(void*)dout,(void*)din,(void*)dsym};
  std::string descrip = "CPU";                //  "CPU";
  std::string devfft = "fftw";
  //std::string devfft  = "rocfft";
#endif

  /*
    Define MDDFT.
  */
  //MDDFTProblem mdp(inList, outList);
  //std::cout << *((int*)args.at(3)) << std::endl;
  MDDFTProblem mdp(args, sizes, "mddft");

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
  float *devmilliseconds = new float[iterations];
  float *invdevmilliseconds = new float[iterations];
  bool check_buff = true; // compare results of Spiral-RTC with device FFT
#endif
  
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  //  Set up a plan to run the transform using cu or roc fft
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
      check_buff = false;
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
	  
  // auto inbuf = sycl::malloc_shared< std::complex<double> >
  // (N + 2, sycl_device, sycl_context);
  // auto outbuf = sycl::malloc_shared< std::complex<double> >
  // (N + 2, sycl_device, sycl_context);

  auto inarray3d = sycl::malloc_shared< std::complex<double> >
    (inputPoints, sycl_device, sycl_context);
  auto outarray3d = sycl::malloc_shared< std::complex<double> >
    (outputPoints, sycl_device, sycl_context);
  sycl::buffer<std::complex<double>> outbuf3d(outarray3d, outputPoints);
    
  // Initialize 3D FFT descriptor
  std::vector<std::int64_t> Nvec{mm, nn, kk};
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
			       oneapi::mkl::dft::domain::COMPLEX>
    transform_plan_3d(Nvec);
  transform_plan_3d.commit(Q);
  
  //    std::cout << "defining mklFwdPtr and mklInvPtr" << std::endl;
  //    auto mklFwdPtr =
  //      new mklTransform3d<std::complex<double>, std::complex<double>>
  //      (mddft3dMKL, domain, domain);

  //    auto mklInvPtr =
  //      new mklTransform3d<std::complex<double>, std::complex<double>>
  //      (imddft3dMKL, domain, domain);
  //    std::cout << "defined mklFwdPtr and mklInvPtr" << std::endl;
#endif

  // Need pointer to double for buildInputBuffer().
  double *hostinp = (double *) inputHostData;

  /*
    Run iterations of MDDFT.
   */
  for (int itn = 0; itn < iterations; itn++)
    {
      // Set up random data for input buffer.
      // (Use different randomized data each iteration.)
      buildInputBuffer ( hostinp, sizes );

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      // Copy inputBytes bytes from inputHostData to din on device.
      DEVICE_MEM_COPY(din, inputHostData, inputBytes,
		      MEM_COPY_HOST_TO_DEVICE);
#elif defined(FFTX_SYCL)
      // For all of these, we have double *hostinp = (double *) inputHostData;
      
      // For SYCL, we have sycl::buffer<std::complex<double>> buf_in(inputHostData, inputPoints);
      // and std::vector<void*> args{(void*)&(buf_out),(void*)&(buf_in),(void*)&(buf_sym)};
      // and MDDFTProblem mdp(args, sizes, "mddft");
      std::cout << "Q.memcpy(&buf_in, &inputHostData, inputBytes);" << std::endl;
      Q.memcpy(&buf_in, &inputHostData, inputBytes);
      std::cout << "Q.memcpy(&buf_in, &inputHostData, inputBytes); done" << std::endl;
      Q.wait();
#endif
      if ( DEBUGOUT ) std::cout << "copied X\n";
      
      mdp.transform();
      //gatherOutput(outputHost, args);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      // Copy outputBytes bytes from dout on device to outputHostData.
      DEVICE_MEM_COPY(outputHostData, dout, outputBytes,
		      MEM_COPY_DEVICE_TO_HOST);
      
#elif defined(FFTX_SYCL)
      //	std::cout << "Q.memcpy(outputHostData, buf_out, inputBytes);" << std::endl;
      //	Q.memcpy(outputHostData, buf_out, inputBytes);
      //	Q.wait();
      //	std::cout << "Q.memcpy(outputHostData, buf_out, inputBytes); done" << std::endl;
#endif
      mddft_gpu[itn] = mdp.getTime();
      
#if defined (FFTX_SYCL)
      {
	sycl::host_accessor h_acc(buf_out);
	std::cout << "FFTX first output element " << h_acc[0] << std::endl;
	// ugh
	auto outputHostDataLocal = outputHostData;
	for (int ind = 0; ind < outputPoints; ind++)
	  {
	    outputHostDataLocal[ind] = h_acc[ind];
	  }
	//std::cout << "Q.memcpy(outputHostData, h_acc, inputBytes);" << std::endl;
	// Q.memcpy(outputHostData, &h_acc, inputBytes);
	// Q.wait();
	// std::cout << "Q.memcpy(outputHostData, h_acc, inputBytes); done" << std::endl;
      }
#endif
      
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      //  Run the roc fft plan on the same input data
      if ( check_buff )
	{
	  DEVICE_EVENT_RECORD ( custart );
	  res = DEVICE_FFT_EXECZ2Z ( plan,
				     (DEVICE_FFT_DOUBLECOMPLEX *) din,
				     (DEVICE_FFT_DOUBLECOMPLEX *) dout,
				     DEVICE_FFT_FORWARD );
	  if ( res != DEVICE_FFT_SUCCESS)
	    {
	      printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
	      check_buff = false;
	      //  break;
	    }
	  DEVICE_EVENT_RECORD ( custop );
	  DEVICE_EVENT_SYNCHRONIZE ( custop );
	  DEVICE_EVENT_ELAPSED_TIME ( &devmilliseconds[itn], custart, custop );
	  
	  DEVICE_MEM_COPY ( outputDevfftData, dout, outputBytes,
			    MEM_COPY_DEVICE_TO_HOST );
	  printf ( "cube = [ %d, %d, %d ]\tMDDFT (Forward)\t", mm, nn, kk );
	  checkOutputBuffers ( (DEVICE_FFT_DOUBLECOMPLEX *) outputHostData,
			       (DEVICE_FFT_DOUBLECOMPLEX *) outputDevfftData,
			       outputPoints );
	}
#elif defined (FFTX_SYCL)
      // TODO: FILL IN WITH CALL TO mklFwdPtr.
      // mklFwdPtr->exec(inputHost, outputHost);
      // int N = 32;
      
      // Q.single_task<>([=]() {
      // inbuf[N - N / 4 - 1] = 1.0;
      // inbuf[N - N / 4] = 1.0;
      // inbuf[N - N / 4 + 1] = 1.0; // Signal
      // }).wait();
      
      //	Q.single_task<>([=]() {
      //	  inarray3d[N - N / 4 - 1] = 1.0;
      //	  inarray3d[N - N / 4] = 1.0;
      //	  inarray3d[N - N / 4 + 1] = 1.0; // Signal
      //	}).wait();
      
      
      // // Initialize FFT descriptor
      // oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
      // oneapi::mkl::dft::domain::COMPLEX>
      // transform_plan(N);
      // transform_plan.commit(Q);
      
      std::cout << "Q.memcpy(inarray3d, &buf_in, inputBytes);" << std::endl;
      // This fails at runtime.
      //     auto inarray3d = sycl::malloc_shared< std::complex<double> > ...
      //     sycl::buffer<std::complex<double>> buf_in ...
      // Q.memcpy(inarray3d, &buf_in, inputBytes);
      //      Q.wait();
      std::cout << "Q.memcpy(inarray3d, &buf_in, inputBytes); done" << std::endl;
      
      // clock_t start_time = clock(); // Start timer
      auto start_time = std::chrono::high_resolution_clock::now();
      
      // Perform forward transform on complex array
      // auto evt1 =
      // oneapi::mkl::dft::compute_forward(transform_plan, inbuf, outbuf).wait();
      // oneapi::mkl::dft::compute_forward(transform_plan_3d, buf_in, outbuf3d);
      oneapi::mkl::dft::compute_forward(transform_plan_3d, inarray3d, outarray3d).wait();
      
      // clock_t end_time = clock(); // Stop timer
      
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> duration = end_time - start_time;
      devmilliseconds[itn] = duration.count();
      
      std::cout << "The MKL 3D forward FFT on " << domain << " took "
		<< devmilliseconds[itn] << " milliseconds."
	// << float(end_time - start_time) / CLOCKS_PER_SEC << " seconds."
		<< std::endl;
      
      std::cout << "Q.memcpy(outputDevfftData, &outarray3d, outputBytes);" << std::endl;
      // Copy outputBytes bytes to outputDevfftData from &outarray3d,
      Q.memcpy(outputDevfftData, &outarray3d, outputBytes);
      std::cout << "Q.wait()" << std::endl;
      Q.wait();
      std::cout << "Q.memcpy(outputDevfftData, &outarray3d, outputBytes); done" << std::endl;
      printf ( "cube = [ %d, %d, %d ]\tMDDFT (Forward)\t", mm, nn, kk );
      if (check_buff)
	{
	  checkOutputBuffers ( (std::complex<double>*) outputHostData,
			       (std::complex<double>*) outputDevfftData,
			       outputPoints );
	}
#endif
    }
  
#if defined(FFTX_SYCL)
  // Cleanup
  // sycl::free(inbuf, sycl_context);
  // sycl::free(outbuf, sycl_context);
  sycl::free(inarray3d, sycl_context);
  sycl::free(outarray3d, sycl_context);
#endif

    // setup the inverse transform (we'll reuse the device fft plan already created)
  IMDDFTProblem imdp(args, sizes, "imddft");

  /*
    Run iterations of IMDDFT.
  */
  for (int itn = 0; itn < iterations; itn++)
      {
        // Set up random data for input buffer.
	// (Use different randomized data each iteration.)
        buildInputBuffer ( hostinp, sizes );
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY(din, inputHostData, inputBytes,
                        MEM_COPY_HOST_TO_DEVICE);
#endif
        if ( DEBUGOUT ) std::cout << "copied X\n";
        
        imdp.transform();
        //gatherOutput(outputHost, args);
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        DEVICE_MEM_COPY ( outputHostData, dout, outputBytes,
			  MEM_COPY_DEVICE_TO_HOST );
                          
#endif
        imddft_gpu[itn] = imdp.getTime();
	
#if defined (FFTX_SYCL)
	{
	  std::cout << "MKLFFT comparison not implemented printing first output element" << std::endl;
	  sycl::host_accessor h_acc(buf_out);
	  std::cout << h_acc[0] << std::endl;
	}
#endif

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
        //  Run the device fft plan on the same input data.
        if ( check_buff )
	  {
            DEVICE_EVENT_RECORD ( custart );
            res = DEVICE_FFT_EXECZ2Z ( plan,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) din,
                                       (DEVICE_FFT_DOUBLECOMPLEX *) dout,
                                       DEVICE_FFT_INVERSE );
            if ( res != DEVICE_FFT_SUCCESS)
	      {
                printf ( "Launch DEVICE_FFT_EXEC failed with error code %d ... skip buffer check\n", res );
                check_buff = false;
                //  break;
	      }
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &invdevmilliseconds[itn], custart, custop );

            DEVICE_MEM_COPY ( outputDevfftData, dout, outputBytes,
			      MEM_COPY_DEVICE_TO_HOST );
            printf ( "cube = [ %d, %d, %d ]\tMDDFT (Inverse)\t", mm, nn, kk );
            checkOutputBuffers ( (DEVICE_FFT_DOUBLECOMPLEX *) outputHostData,
                                 (DEVICE_FFT_DOUBLECOMPLEX *) outputDevfftData,
                                 (long) outputPoints );
        }
#elif defined (FFTX_SYCL)
	// TODO: FILL IN WITH CALL TO mklInvPtr.
	// mklInvPtr->exec(inputHost, outputHost);
#endif
    }

#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
    printf ( "Times in milliseconds for %s on MDDFT (forward) for %d trials of size %d %d %d:\nTrial #\tSpiral\t%s\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2), devfft.c_str() );
    for (int itn = 0; itn < iterations; itn++) {
        printf ( "%d\t%.7e\t%.7e\n", itn, mddft_gpu[itn], devmilliseconds[itn] );
    }

    printf ( "Times in milliseconds for %s on MDDFT (inverse) for %d trials of size %d %d %d:\nTrial #\tSpiral\t%s\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2), devfft.c_str() );
    for (int itn = 0; itn < iterations; itn++)
      {
        printf ( "%d\t%.7e\t%.7e\n", itn, imddft_gpu[itn], invdevmilliseconds[itn] );
      }
    // TODO: #elif defined (FFTX_SYCL)
	
#else
     printf ( "Times in milliseconds for %s on MDDFT (forward) for %d trials of size %d %d %d\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
    for (int itn = 0; itn < iterations; itn++)
      {
        printf ( "%d\t%.7e\n", itn, mddft_gpu[itn]);
      }

    printf ( "Times in milliseconds for %s on MDDFT (inverse) for %d trials of size %d %d %d\n",
             descrip.c_str(), iterations, sizes.at(0), sizes.at(1), sizes.at(2));
    for (int itn = 0; itn < iterations; itn++)
      {
        printf ( "%d\t%.7e\n", itn, imddft_gpu[itn]);
      }
#endif

    // delete[] mddft_cpu;
    // delete[] imddft_cpu;
    delete[] mddft_gpu;
    delete[] imddft_gpu;

    printf("%s: All done, exiting\n", prog);
  
    return 0;
}
