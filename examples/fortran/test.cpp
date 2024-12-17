#include "stdlib.h"
// #include "stdio.h"
#include <complex>

#include "fftxinterface.hpp"
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#include "fftx_mpi.hpp"
#endif
#include "fftxmddftObj.hpp"
#include "fftximddftObj.hpp"
#include "fftxmdprdftObj.hpp"
#include "fftximdprdftObj.hpp"

//#if defined (FFTX_CUDA) || defined(FFTX_HIP) || defined (FFTX_SYCL)
// #include "fftxdevice_macros.h"
// #endif

// NOTE: assuming only 1 plan going at a time? otherwise, need to figure out a way to pass new GPU
//       buffers to host code and attach to the op that's calling this plan.

// Pointers to memory that will be allocated on the device.
double *dist_dev_out = NULL, *dist_dev_in = NULL, *dist_dev_sym = NULL;

// Allocate space on device, for type T, if pointer is NULL.
#define FFTX_DEVICE_MALLOC_TYPE_IFNULL(ptr, T, npts) ( { if (ptr == NULL) FFTX_DEVICE_MALLOC_TYPE(ptr, T, npts); })

// Allocate space on device, for type T.
#define FFTX_DEVICE_MALLOC_TYPE(ptr, T, npts) ( { FFTX_DEVICE_MALLOC(&ptr, npts * sizeof(T)); } )

// Allocate space on device, for type T.
#define HOST_MALLOC_TYPE(ptr, T, npts) ( { ptr = (T*) malloc(npts * sizeof(T)); } )

// Set SYCL buffer, for type T.
#define SYCL_BUFFER_TYPE(buf, ptr, T, npts) sycl::buffer<double> buf((double *) ptr, npts * sizeof(T)/sizeof(double))

#define CUDA_HOLDER_ARGS(holder) {&(holder.dev_out), &(holder.dev_in), &(holder.dev_sym)}

#define HIP_HOLDER_ARGS(holder) {holder.dev_out, holder.dev_in, holder.dev_sym}

// #define SYCL_HOLDER_ARGS(holder) {(void*)&(holder.dev_out), (void*)&(holder.dev_in), (void*)&(holder.dev_sym)}

#define SYCL_HOLDER_SET_ARGS(holder, outBuf, inBuf, symBuf) { std::vector<void*> args{(void*)&(outBuf), (void*)&(inBuf), (void*)&(symBuf)}; holder.problem->setArgs(args); }

#define HOST_HOLDER_ARGS(holder) {(void*)(holder.dev_out), (void*)(holder.dev_in), (void*)(holder.dev_sym)}

#define FFTX_DEVICE_COPY_INPUT(holder, buffer, bytes) ( { FFTX_DEVICE_MEM_COPY(holder.dev_in, buffer, bytes, FFTX_MEM_COPY_HOST_TO_DEVICE); } )

#define FFTX_DEVICE_COPY_OUTPUT(holder, buffer, bytes) ( { FFTX_DEVICE_MEM_COPY(buffer, holder.dev_out, bytes, FFTX_MEM_COPY_DEVICE_TO_HOST); } )

#define HOST_COPY_INPUT(holder, buffer, bytes) ( { memcpy(holder.dev_in, buffer, bytes); } )

#define HOST_COPY_OUTPUT(holder, buffer, bytes) ( { memcpy(buffer, holder.dev_out, bytes); } )

// Free up space on device, if pointer is not NULL.
#define FFTX_DEVICE_FREE_NONNULL(ptr) ( { if (ptr != NULL) FFTX_DEVICE_FREE(ptr); } )

#define HOST_FREE_NONNULL(ptr) ( { if (ptr != NULL) free(ptr); } )

// Macros set different for device or host.
#if defined (FFTX_CUDA)
#define HOST_OR_DEVICE_HOLDER_ARGS   CUDA_HOLDER_ARGS
#define HOST_OR_DEVICE_MALLOC_TYPE   FFTX_DEVICE_MALLOC_TYPE
#define HOST_OR_DEVICE_COPY_INPUT    FFTX_DEVICE_COPY_INPUT
#define HOST_OR_DEVICE_COPY_OUTPUT   FFTX_DEVICE_COPY_OUTPUT
#define HOST_OR_DEVICE_FREE_NONNULL  FFTX_DEVICE_FREE_NONNULL
#elif defined (FFTX_HIP)
#define HOST_OR_DEVICE_HOLDER_ARGS   HIP_HOLDER_ARGS
#define HOST_OR_DEVICE_MALLOC_TYPE   FFTX_DEVICE_MALLOC_TYPE
#define HOST_OR_DEVICE_COPY_INPUT    FFTX_DEVICE_COPY_INPUT
#define HOST_OR_DEVICE_COPY_OUTPUT   FFTX_DEVICE_COPY_OUTPUT
#define HOST_OR_DEVICE_FREE_NONNULL  FFTX_DEVICE_FREE_NONNULL
#else // CPU
#define HOST_OR_DEVICE_HOLDER_ARGS   HOST_HOLDER_ARGS
#define HOST_OR_DEVICE_MALLOC_TYPE   HOST_MALLOC_TYPE
#define HOST_OR_DEVICE_COPY_INPUT    HOST_COPY_INPUT
#define HOST_OR_DEVICE_COPY_OUTPUT   HOST_COPY_OUTPUT
#define HOST_OR_DEVICE_FREE_NONNULL  HOST_FREE_NONNULL
#endif

// These are indices of arguments in FFTXProblemPtr.
// #define ARG_OUT 0
// #define ARG_IN 1
// #define ARG_SYM 2

extern "C"
{
  struct mddft_holder
  {
    std::complex<double>* dev_out;
    std::complex<double>* dev_in;
    std::complex<double>* dev_sym;
    MDDFTProblem* problem;
  };

  void fftx_plan_mddft_shim(mddft_holder& holder, int M, int N, int K)
  {
    //    printf("fftx_plan_mddft M=%d, N=%d, K=%d, size=%d\n", 
      //           M, N, K, M*N*K);
    std::vector<int> sizes{M, N, K};
#ifdef FFTX_SYCL
    std::vector<void*> args; // fill in at execute time
#else
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_in,  std::complex<double>, M * N * K);
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_out, std::complex<double>, M * N * K);
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_sym, std::complex<double>, M * N * K);
    std::vector<void*> args(HOST_OR_DEVICE_HOLDER_ARGS(holder));
#endif
    holder.problem = new MDDFTProblem(args, sizes, "mddft");
  }
    
  void fftx_execute_mddft_shim(
                               mddft_holder& holder,
                               std::complex<double>* out_buffer,
                               std::complex<double>* in_buffer
                               )
  {
    std::vector<int>& sizes = holder.problem->sizes;
    size_t npts = sizes[0] * sizes[1] * sizes[2];
#ifdef FFTX_SYCL
    SYCL_BUFFER_TYPE(inputBuffer, in_buffer, std::complex<double>, npts);
    SYCL_BUFFER_TYPE(outputBuffer, out_buffer, std::complex<double>, npts);
    double* sym_buffer = new double[npts];
    SYCL_BUFFER_TYPE(symbolBuffer, sym_buffer, double, npts);
    SYCL_HOLDER_SET_ARGS(holder, outputBuffer, inputBuffer, symbolBuffer);
    holder.problem->transform();
    delete[] sym_buffer;
#else
    size_t in_bytes  = sizeof(std::complex<double>) * npts;
    size_t out_bytes = sizeof(std::complex<double>) * npts;
    HOST_OR_DEVICE_COPY_INPUT(holder, in_buffer, in_bytes);
    holder.problem->transform();
    HOST_OR_DEVICE_COPY_OUTPUT(holder, out_buffer, out_bytes);
#endif
  }

  void fftx_plan_destroy_mddft_shim(mddft_holder& holder)
  {
#ifndef FFTX_SYCL
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_in);
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_out);
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_sym);
#endif
    delete holder.problem;
  }
}

extern "C"
{
  struct imddft_holder
  {
    std::complex<double>* dev_out;
    std::complex<double>* dev_in;
    std::complex<double>* dev_sym;
    IMDDFTProblem* problem;
  };

  void fftx_plan_imddft_shim(imddft_holder& holder, int M, int N, int K)
  {
    //    printf("fftx_plan_imddft M=%d, N=%d, K=%d, size=%d\n", 
      //           M, N, K, M*N*K);
    std::vector<int> sizes{M, N, K};
#ifdef FFTX_SYCL
    std::vector<void*> args; // fill in at execute time
#else
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_in,  std::complex<double>, M * N * K);
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_out, std::complex<double>, M * N * K);
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_sym, std::complex<double>, M * N * K);
    std::vector<void*> args(HOST_OR_DEVICE_HOLDER_ARGS(holder));
#endif
    holder.problem = new IMDDFTProblem(args, sizes, "imddft");
  }
  
  void fftx_execute_imddft_shim(
                                imddft_holder& holder,
                                std::complex<double>* out_buffer,
                                std::complex<double>* in_buffer
                                )
  {
    std::vector<int>& sizes = holder.problem->sizes;
    size_t npts = sizes[0] * sizes[1] * sizes[2];
#ifdef FFTX_SYCL
    SYCL_BUFFER_TYPE(inputBuffer, in_buffer, std::complex<double>, npts);
    SYCL_BUFFER_TYPE(outputBuffer, out_buffer, std::complex<double>, npts);
    double* sym_buffer = new double[npts];
    SYCL_BUFFER_TYPE(symbolBuffer, sym_buffer, double, npts);
    SYCL_HOLDER_SET_ARGS(holder, outputBuffer, inputBuffer, symbolBuffer);
    holder.problem->transform();
    delete[] sym_buffer;
#else
    size_t in_bytes  = sizeof(std::complex<double>) * npts;
    size_t out_bytes = sizeof(std::complex<double>) * npts;
    HOST_OR_DEVICE_COPY_INPUT(holder, in_buffer, in_bytes);
    holder.problem->transform();
    HOST_OR_DEVICE_COPY_OUTPUT(holder, out_buffer, out_bytes);
#endif
  }

  void fftx_plan_destroy_imddft_shim(imddft_holder& holder)
  {
#ifndef FFTX_SYCL
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_in);
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_out);
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_sym);
#endif
    delete holder.problem;
  }
}

extern "C"
{
  struct mdprdft_holder
  {
    int npts;
    int nptsTrunc;
    std::complex<double>* dev_out;
    double* dev_in;
    double* dev_sym;
    MDPRDFTProblem* problem;
  };

  void fftx_plan_mdprdft_shim(mdprdft_holder& holder,
                              int M, int N, int K,
                              int npts, int nptsTrunc)
  {
    holder.npts = npts;
    holder.nptsTrunc = nptsTrunc;
    //    printf("fftx_plan_mdprdft M=%d, N=%d, K=%d, size=%d\n", 
      //           M, N, K, M*N*K);
    // fftx::point_t<3> sizesF({M, N, K});
    // fftx::point_t<3> sizesFtrunc = truncatedComplexDimensions(sizesF);
    // int npts = sizesF.product();
    // int nptsTrunc = sizesFtrunc.product();
    // printf("MDPRDFT 1 init npts=%d nptsTrunc=%d\n", holder.npts, holder.nptsTrunc);
    //    int K_adj = (K/2) + 1;
    //    printf("fftx_plan_mdprdft M=%d, N=%d, K=%d, K_adj=%d\n", 
    //           M, N, K, K_adj);
    // input, real, full dimensions
    std::vector<int> sizes{M, N, K};
#ifdef FFTX_SYCL
    std::vector<void*> args; // fill in at execute time
#else
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_in, double, holder.npts);
    // output, complex, one dimension truncated
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_out, std::complex<double>, holder.nptsTrunc);
    // symbol, real, one dimension truncated
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_sym, double, holder.nptsTrunc);
    std::vector<void*> args(HOST_OR_DEVICE_HOLDER_ARGS(holder));
#endif
    holder.problem = new MDPRDFTProblem(args, sizes, "mdprdft");
  }

  void fftx_execute_mdprdft_shim(
                                 mdprdft_holder& holder,
                                 std::complex<double>* out_buffer,
                                 double* in_buffer
                                 )
  {
    //    std::vector<int>& sizes = holder.problem->sizes;
    //    fftx::point_t<3> sizesF({sizes[0], sizes[1], sizes[2]});
    //    fftx::point_t<3> sizesFtrunc = truncatedComplexDimensions(sizesF);
    //    int npts = sizesF.product();
    //    int nptsTrunc = sizesFtrunc.product();
#ifdef FFTX_SYCL
    SYCL_BUFFER_TYPE(inputBuffer, in_buffer, double, holder.npts);
    SYCL_BUFFER_TYPE(outputBuffer, out_buffer, std::complex<double>, holder.nptsTrunc);
    double* sym_buffer = new double[holder.npts];
    SYCL_BUFFER_TYPE(symbolBuffer, sym_buffer, double, holder.npts);
    SYCL_HOLDER_SET_ARGS(holder, outputBuffer, inputBuffer, symbolBuffer);
    holder.problem->transform();
    delete[] sym_buffer;
#else
    size_t in_bytes  = sizeof(double) * holder.npts;
    size_t out_bytes = sizeof(std::complex<double>) * holder.nptsTrunc;
    HOST_OR_DEVICE_COPY_INPUT(holder, in_buffer, in_bytes);
    holder.problem->transform();
    HOST_OR_DEVICE_COPY_OUTPUT(holder, out_buffer, out_bytes);
#endif
  }
  
  void fftx_plan_destroy_mdprdft_shim(mdprdft_holder& holder)
  {
#ifndef FFTX_SYCL
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_in);
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_out);
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_sym);
#endif
    delete holder.problem;
  }
}

extern "C"
{
  struct imdprdft_holder
  {
    int npts;
    int nptsTrunc;
    double* dev_out;
    std::complex<double>* dev_in;
    double* dev_sym;
    IMDPRDFTProblem* problem;
  };

  void fftx_plan_imdprdft_shim(imdprdft_holder& holder,
                               int M, int N, int K,
                               int npts, int nptsTrunc)    
  {
    holder.npts = npts;
    holder.nptsTrunc = nptsTrunc;
    //    printf("fftx_plan_imdprdft M=%d, N=%d, K=%d, size=%d\n", 
      //           M, N, K, M*N*K);
    // fftx::point_t<3> sizesF({M, N, K});
    // fftx::point_t<3> sizesFtrunc = truncatedComplexDimensions(sizesF);
    // int npts = sizesF.product();
    // int nptsTrunc = sizesFtrunc.product();
    // input, complex, one dimension truncated
    std::vector<int> sizes{M, N, K};
#ifdef FFTX_SYCL
    std::vector<void*> args; // fill in at execute time
#else
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_in, std::complex<double>, holder.nptsTrunc);
    // output, real, full dimensions
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_out, double, holder.npts);
    // symbol, real, one dimension truncated
    HOST_OR_DEVICE_MALLOC_TYPE(holder.dev_sym, double, holder.nptsTrunc);
    std::vector<void*> args(HOST_OR_DEVICE_HOLDER_ARGS(holder));
#endif
    holder.problem = new IMDPRDFTProblem(args, sizes, "imdprdft");
  }

  void fftx_execute_imdprdft_shim(
                                  imdprdft_holder& holder,
				  double* out_buffer,
                                  std::complex<double>* in_buffer
                                  )
  {
    //    std::vector<int>& sizes = holder.problem->sizes;
    //    fftx::point_t<3> sizesF({sizes[0], sizes[1], sizes[2]});
    //    fftx::point_t<3> sizesFtrunc = truncatedComplexDimensions(sizesF);
    //    int npts = sizesF.product();
    //    int nptsTrunc = sizesFtrunc.product();
    // int K_adj = (sizes[2]/2) + 1;
    // for input, complex, one dimension truncated
#ifdef FFTX_SYCL
    SYCL_BUFFER_TYPE(inputBuffer, in_buffer, std::complex<double>, holder.nptsTrunc);
    SYCL_BUFFER_TYPE(outputBuffer, out_buffer, double, holder.npts);
    double* sym_buffer = new double[holder.npts];
    SYCL_BUFFER_TYPE(symbolBuffer, sym_buffer, double, holder.npts);
    SYCL_HOLDER_SET_ARGS(holder, outputBuffer, inputBuffer, symbolBuffer);
    holder.problem->transform();
    delete[] sym_buffer;
#else
    size_t in_bytes  = sizeof(std::complex<double>) * holder.nptsTrunc;
    // sizes[0] * sizes[1] * K_adj;
    // for output, real, full dimensions
    size_t out_bytes = sizeof(double) * holder.npts;
    HOST_OR_DEVICE_COPY_INPUT(holder, in_buffer, in_bytes);
    holder.problem->transform();
    HOST_OR_DEVICE_COPY_OUTPUT(holder, out_buffer, out_bytes);
#endif
  }
  
  void fftx_plan_destroy_imdprdft_shim(imdprdft_holder& holder)
  {
#ifndef FFTX_SYCL
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_in);
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_out);
    HOST_OR_DEVICE_FREE_NONNULL(holder.dev_sym);
#endif
    delete holder.problem;
  }
}

#if defined (FFTX_CUDA) || defined(FFTX_HIP)

extern "C"
{
  fftx_plan fftx_plan_distributed_shim(
    int p, int M, int N, int K, int batch, bool is_embedded, bool is_complex
  )
  {
    //    printf("fftx_plan_distributed p=%d, M=%d, N=%d, K=%d, batch=%d, is_embedded=%d, is_complex=%d\n", 
    //           p, M, N, K, batch,  is_embedded, is_complex);
    FFTX_DEVICE_MALLOC_TYPE_IFNULL(dist_dev_in, double, 2 * M * N * K * batch / p);
    FFTX_DEVICE_MALLOC_TYPE_IFNULL(dist_dev_out, double, 2 * M * N * K * batch / p);
    // This is fftx_plan_distributed_1d_spiral in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_1d_mpi_spiral.cpp
    // where struct fftx_plan is defined in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_mpi_spiral.cpp
    fftx_plan plan = fftx_plan_distributed_1d(MPI_COMM_WORLD, p, M, N, K, batch, is_embedded, is_complex);
    return plan;
  }
}

extern "C"
{
  struct mddft_dist_holder
  { // Note need double*, not std::complex<double>*, for FFTX.
    double* dev_out;
    double* dev_in;
    fftx_plan plan;
  };

  void fftx_plan_mddft_dist_shim(mddft_dist_holder& holder,
                                 int p, int M, int N, int K, int npts)
  {
    FFTX_DEVICE_MALLOC_TYPE(holder.dev_in, double, 2 * npts);
    FFTX_DEVICE_MALLOC_TYPE(holder.dev_out, double, 2 * npts);
    // This is fftx_plan_distributed_1d_spiral in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_1d_mpi_spiral.cpp
    // where struct fftx_plan is defined in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_mpi_spiral.cpp
    int batch = 1;
    bool is_embedded = false;
    bool is_complex = true;
    holder.plan = fftx_plan_distributed_1d(MPI_COMM_WORLD, p, M, N, K, batch, is_embedded, is_complex);
  }
  
  void fftx_execute_mddft_dist_shim(
                                    mddft_dist_holder& holder,
                                    std::complex<double>* out_buffer,
                                    std::complex<double>* in_buffer
                                    )
  {
    size_t in_pts = holder.plan->M * holder.plan->N * holder.plan->K *
      holder.plan->b / holder.plan->r;
    size_t in_bytes = in_pts * sizeof(std::complex<double>);
    FFTX_DEVICE_COPY_INPUT(holder, in_buffer, in_bytes);
    
    // This is fftx_plan_distributed_1d_spiral in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_1d_mpi_spiral.cpp
    // where struct fftx_plan is defined in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_mpi_spiral.cpp
    fftx_execute_1d(holder.plan, holder.dev_out, holder.dev_in, FFTX_DEVICE_FFT_FORWARD);

    size_t out_pts = holder.plan->M * holder.plan->N * holder.plan->K *
      holder.plan->b / holder.plan->r;
    size_t out_bytes = out_pts * sizeof(std::complex<double>);
    FFTX_DEVICE_COPY_OUTPUT(holder, out_buffer, out_bytes);
  }

  void fftx_plan_destroy_mddft_dist_shim(mddft_dist_holder& holder)
  {
    FFTX_DEVICE_FREE_NONNULL(holder.dev_in);
    FFTX_DEVICE_FREE_NONNULL(holder.dev_out);
    delete holder.plan;
  }
}

extern "C"
{
  struct imddft_dist_holder
  { // Note need double*, not std::complex<double>*, for FFTX.
    double* dev_out;
    double* dev_in;
    fftx_plan plan;
  };

  void fftx_plan_imddft_dist_shim(imddft_dist_holder& holder,
                                  int p, int M, int N, int K, int npts)
  {
    FFTX_DEVICE_MALLOC_TYPE(holder.dev_in, double, 2 * npts);
    FFTX_DEVICE_MALLOC_TYPE(holder.dev_out, double, 2 * npts);
    // This is fftx_plan_distributed_1d_spiral in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_1d_mpi_spiral.cpp
    // where struct fftx_plan is defined in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_mpi_spiral.cpp
    int batch = 1;
    bool is_embedded = false;
    bool is_complex = true;
    holder.plan = fftx_plan_distributed_1d(MPI_COMM_WORLD, p, M, N, K, batch, is_embedded, is_complex);
  }
  
  void fftx_execute_imddft_dist_shim(
                                     imddft_dist_holder& holder,
                                     std::complex<double>* out_buffer,
                                     std::complex<double>* in_buffer
                                     )
  {
    size_t in_pts = holder.plan->M * holder.plan->N * holder.plan->K *
      holder.plan->b / holder.plan->r;
    size_t in_bytes = in_pts * sizeof(std::complex<double>);
    FFTX_DEVICE_COPY_INPUT(holder, in_buffer, in_bytes);

    // This is fftx_execute_1d_spiral in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_1d_mpi_spiral.cpp
    fftx_execute_1d(holder.plan, holder.dev_out, holder.dev_in, FFTX_DEVICE_FFT_INVERSE);

    size_t out_pts = holder.plan->M * holder.plan->N * holder.plan->K *
      holder.plan->b / holder.plan->r;
    size_t out_bytes = out_pts * sizeof(std::complex<double>);
    FFTX_DEVICE_COPY_OUTPUT(holder, out_buffer, out_bytes);
  }

  void fftx_plan_destroy_imddft_dist_shim(imddft_dist_holder& holder)
  {
    FFTX_DEVICE_FREE_NONNULL(holder.dev_in);
    FFTX_DEVICE_FREE_NONNULL(holder.dev_out);
    delete holder.plan;
  }
}

extern "C"
{
  struct mdprdft_dist_holder
  { // Note need double*, not std::complex<double>*, for FFTX.
    int npts;
    int nptsTrunc;
    double* dev_out;
    double* dev_in;
    fftx_plan plan;
  };

  void fftx_plan_mdprdft_dist_shim(mdprdft_dist_holder& holder,
                                   int p, int M, int N, int K,
                                   int npts, int nptsTrunc)
  {
    holder.npts = npts;
    holder.nptsTrunc = nptsTrunc;
    FFTX_DEVICE_MALLOC_TYPE(holder.dev_in, double, 2 * holder.npts);
    FFTX_DEVICE_MALLOC_TYPE(holder.dev_out, double, 2 * holder.nptsTrunc);
    // This is fftx_plan_distributed_1d_spiral in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_1d_mpi_spiral.cpp
    // where struct fftx_plan is defined in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_mpi_spiral.cpp
    int batch = 1;
    bool is_embedded = false;
    bool is_complex = false;
    holder.plan = fftx_plan_distributed_1d(MPI_COMM_WORLD, p, M, N, K, batch, is_embedded, is_complex);
  }
  
  void fftx_execute_mdprdft_dist_shim(
                                      mdprdft_dist_holder& holder,
                                      std::complex<double>* out_buffer,
                                      double* in_buffer
                                      )
  {
    // N.B. reversal.
    // fftx::point_t<3> sizesF({holder.plan->K, holder.plan->N, holder.plan->M});
    // fftx::point_t<3> sizesFtrunc = truncatedComplexDimensions(sizesF);
    // int npts = sizesF.product();
    // int nptsTrunc = sizesFtrunc.product();

    // Do not divide by holder.plan->r, because holder.npts is already LOCAL.
    size_t in_pts = holder.npts * holder.plan->b;
    size_t in_bytes = in_pts * sizeof(double);
    FFTX_DEVICE_COPY_INPUT(holder, in_buffer, in_bytes);
    
    // This is fftx_execute_1d_spiral in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_1d_mpi_spiral.cpp
    fftx_execute_1d(holder.plan, holder.dev_out, holder.dev_in, FFTX_DEVICE_FFT_FORWARD);

    // Do not divide by holder.plan->r, because holder.nptsTrunc is already LOCAL.
    size_t out_pts = holder.nptsTrunc * holder.plan->b;
    size_t out_bytes = out_pts * sizeof(std::complex<double>);
    FFTX_DEVICE_COPY_OUTPUT(holder, out_buffer, out_bytes);
  }

  void fftx_plan_destroy_mdprdft_dist_shim(mdprdft_dist_holder& holder)
  {
    FFTX_DEVICE_FREE_NONNULL(holder.dev_in);
    FFTX_DEVICE_FREE_NONNULL(holder.dev_out);
    delete holder.plan;
  }
}


extern "C"
{
  struct imdprdft_dist_holder
  { // Note need double*, not std::complex<double>*, for FFTX.
    int npts;
    int nptsTrunc;
    double* dev_out;
    double* dev_in;
    fftx_plan plan;
  };

  void fftx_plan_imdprdft_dist_shim(imdprdft_dist_holder& holder,
                                    int p, int M, int N, int K,
                                    int npts, int nptsTrunc)
  {
    holder.npts = npts;
    holder.nptsTrunc = nptsTrunc;
    FFTX_DEVICE_MALLOC_TYPE(holder.dev_in, double, 2 * holder.npts);
    FFTX_DEVICE_MALLOC_TYPE(holder.dev_out, double, 2 * holder.nptsTrunc);
    // This is fftx_plan_distributed_1d_spiral in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_1d_mpi_spiral.cpp
    // where struct fftx_plan is defined in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_mpi_spiral.cpp
    int batch = 1;
    bool is_embedded = false;
    bool is_complex = false;
    holder.plan = fftx_plan_distributed_1d(MPI_COMM_WORLD, p, M, N, K, batch, is_embedded, is_complex);
  }
  
  void fftx_execute_imdprdft_dist_shim(
                                       imdprdft_dist_holder& holder,
                                       double* out_buffer,
                                       std::complex<double>* in_buffer
                                       )
  {
    // N.B. reversal.
    // fftx::point_t<3> sizesF({holder.plan->K, holder.plan->N, holder.plan->M});
    // fftx::point_t<3> sizesFtrunc = truncatedComplexDimensions(sizesF);
    // int npts = sizesF.product();
    // int nptsTrunc = sizesFtrunc.product();
    
    // Do not divide by holder.plan->r, because holder.nptsTrunc is already LOCAL.
    size_t in_pts = holder.nptsTrunc * holder.plan->b;
    size_t in_bytes = in_pts * sizeof(std::complex<double>);
    FFTX_DEVICE_COPY_INPUT(holder, in_buffer, in_bytes);
    
    // This is fftx_execute_1d_spiral in
    // $FFTX_HOME/src/library/lib_fftx_mpi/fftx_1d_mpi_spiral.cpp
    fftx_execute_1d(holder.plan, holder.dev_out, holder.dev_in, FFTX_DEVICE_FFT_INVERSE);

    // Do not divide by holder.plan->r, because holder.npts is already LOCAL.
    size_t out_pts = holder.npts * holder.plan->b;
    size_t out_bytes = out_pts * sizeof(double);
    FFTX_DEVICE_COPY_OUTPUT(holder, out_buffer, out_bytes);
  }

  void fftx_plan_destroy_imdprdft_dist_shim(imdprdft_dist_holder& holder)
  {
    FFTX_DEVICE_FREE_NONNULL(holder.dev_in);
    FFTX_DEVICE_FREE_NONNULL(holder.dev_out);
    delete holder.plan;
  }
}

#endif
