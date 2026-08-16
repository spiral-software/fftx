#ifndef FFTX_DEVICE_MACROS_HEADER
#define FFTX_DEVICE_MACROS_HEADER

//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.
//

// Need this for ErrStream().
#include "fftx.hpp"

#if defined(FFTX_HIP)
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <rocfft/rocfft.h>

#define FFTX_DEVICE_SUCCESS hipSuccess
#define FFTX_DEVICE_EVENT_T hipEvent_t
#define FFTX_DEVICE_EVENT_CREATE hipEventCreate
#define FFTX_DEVICE_SET hipSetDevice
#define FFTX_DEVICE_MALLOC hipMalloc
#define FFTX_DEVICE_EVENT_RECORD hipEventRecord
#define FFTX_DEVICE_EVENT_ELAPSED_TIME hipEventElapsedTime
#define FFTX_DEVICE_SYNCHRONIZE hipDeviceSynchronize
#define FFTX_DEVICE_EVENT_SYNCHRONIZE hipEventSynchronize
#define FFTX_DEVICE_FREE hipFree
#define FFTX_DEVICE_MEM_COPY hipMemcpy
#define FFTX_DEVICE_MEM_SET hipMemset
#define FFTX_MEM_COPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice
#define FFTX_MEM_COPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define FFTX_MEM_COPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define FFTX_DEVICE_ERROR_T hipError_t
#define FFTX_DEVICE_GET_LAST_ERROR hipGetLastError
#define FFTX_DEVICE_GET_ERROR_STRING hipGetErrorString
#define FFTX_DEVICE_PTR hipDeviceptr_t
#define FFTX_DEVICE_FFT_TYPE hipfftType
#define FFTX_DEVICE_FFT_RESULT hipfftResult
#define FFTX_DEVICE_FFT_HANDLE hipfftHandle
#define FFTX_DEVICE_FFT_CREATE hipfftCreate
#define FFTX_DEVICE_FFT_MAKE_PLAN_3D hipfftMakePlan3d
#define FFTX_DEVICE_FFT_PLAN3D hipfftPlan3d
#define FFTX_DEVICE_FFT_PLAN2D hipfftPlan2d
#define FFTX_DEVICE_FFT_PLAN_MANY hipfftPlanMany
#define FFTX_DEVICE_FFT_EXECZ2Z hipfftExecZ2Z
#define FFTX_DEVICE_FFT_EXECD2Z hipfftExecD2Z
#define FFTX_DEVICE_FFT_EXECZ2D hipfftExecZ2D
#define FFTX_DEVICE_FFT_DESTROY hipfftDestroy
#define FFTX_DEVICE_FFT_DOUBLEREAL hipfftDoubleReal
#define FFTX_DEVICE_FFT_DOUBLECOMPLEX hipfftDoubleComplex
#define FFTX_DEVICE_FFT_Z2Z HIPFFT_Z2Z
#define FFTX_DEVICE_FFT_D2Z HIPFFT_D2Z
#define FFTX_DEVICE_FFT_Z2D HIPFFT_Z2D
#define FFTX_DEVICE_FFT_SUCCESS HIPFFT_SUCCESS
#define FFTX_DEVICE_FFT_FORWARD HIPFFT_FORWARD
#define FFTX_DEVICE_FFT_INVERSE HIPFFT_BACKWARD

#define FFTX_DEVICE_RTC_SAFE_CALL(x)                                     \
    do {                                                            \
        hiprtcResult result = (x);									\
        if ( result != HIPRTC_SUCCESS ) {                           \
          fftx::ErrStream() << "\nrtc error: " #x " failed with error "	\
                            << hiprtcGetErrorString(result) << '\n';    \
            exit ( 1 );                                             \
        }                                                           \
    } while (0)

#define FFTX_DEVICE_SAFE_CALL(x)                                         \
    do {                                                            \
        hipError_t result = (x);									\
        if (result != hipSuccess ) {                                \
            fftx::ErrStream() << "\nmain error: " <<  hipGetErrorName(result) << " failed with error " \
                              << hipGetErrorString(result) << '\n';     \
            exit ( 1 );                                             \
        }                                                           \
    } while(0)

#elif defined(__CUDACC__) || defined(FFTX_CUDA)

#include <nvrtc.h>
#include <cufft.h>
#include "cuda_runtime.h"

#define FFTX_DEVICE_SUCCESS cudaSuccess
#define FFTX_DEVICE_EVENT_T cudaEvent_t
#define FFTX_DEVICE_EVENT_CREATE cudaEventCreate
#define FFTX_DEVICE_SET cudaSetDevice
#define FFTX_DEVICE_MALLOC cudaMalloc
#define FFTX_DEVICE_EVENT_RECORD cudaEventRecord
#define FFTX_DEVICE_EVENT_ELAPSED_TIME cudaEventElapsedTime
#define FFTX_DEVICE_SYNCHRONIZE cudaDeviceSynchronize
#define FFTX_DEVICE_EVENT_SYNCHRONIZE cudaEventSynchronize
#define FFTX_DEVICE_FREE cudaFree
#define FFTX_DEVICE_MEM_COPY cudaMemcpy
#define FFTX_DEVICE_MEM_SET cudaMemset
#define FFTX_MEM_COPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
#define FFTX_MEM_COPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define FFTX_MEM_COPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define FFTX_DEVICE_ERROR_T cudaError_t
#define FFTX_DEVICE_GET_LAST_ERROR cudaGetLastError
#define FFTX_DEVICE_GET_ERROR_STRING cudaGetErrorString
// #define FFTX_DEVICE_PTR CUdeviceptr
#define FFTX_DEVICE_PTR void **
#define FFTX_DEVICE_FFT_TYPE cufftType
#define FFTX_DEVICE_FFT_RESULT cufftResult
#define FFTX_DEVICE_FFT_HANDLE cufftHandle
#define FFTX_DEVICE_FFT_CREATE cufftCreate
#define FFTX_DEVICE_FFT_MAKE_PLAN_3D cufftMakePlan3d
#define FFTX_DEVICE_FFT_PLAN3D cufftPlan3d
#define FFTX_DEVICE_FFT_PLAN2D cufftPlan2d
#define FFTX_DEVICE_FFT_PLAN_MANY cufftPlanMany
#define FFTX_DEVICE_FFT_EXECZ2Z cufftExecZ2Z
#define FFTX_DEVICE_FFT_EXECD2Z cufftExecD2Z
#define FFTX_DEVICE_FFT_EXECZ2D cufftExecZ2D
#define FFTX_DEVICE_FFT_DESTROY cufftDestroy
#define FFTX_DEVICE_FFT_DOUBLEREAL cufftDoubleReal
#define FFTX_DEVICE_FFT_DOUBLECOMPLEX cufftDoubleComplex
#define FFTX_DEVICE_FFT_Z2Z CUFFT_Z2Z
#define FFTX_DEVICE_FFT_D2Z CUFFT_D2Z
#define FFTX_DEVICE_FFT_Z2D CUFFT_Z2D
#define FFTX_DEVICE_FFT_SUCCESS CUFFT_SUCCESS
#define FFTX_DEVICE_FFT_FORWARD CUFFT_FORWARD
#define FFTX_DEVICE_FFT_INVERSE CUFFT_INVERSE

#define FFTX_DEVICE_RTC_SAFE_CALL(x)                                 \
    do {                                                        \
        nvrtcResult result = (x);								\
        if (result != NVRTC_SUCCESS) {                          \
            fftx::ErrStream() << "\nerror: " #x " failed with error "	\
                              << nvrtcGetErrorString(result) << '\n';   \
            exit ( 1 );                                         \
        }                                                       \
    } while(0)

#define FFTX_DEVICE_SAFE_CALL(x)                                     \
    do {                                                        \
        CUresult result = (x);									\
        if (result != CUDA_SUCCESS) {                           \
            const char *msg;                                    \
            cuGetErrorName(result, &msg);                       \
            fftx::ErrStream() << "\nerror: " #x " failed with error "	\
                              << msg << '\n';                           \
            exit(1);                                            \
        }                                                       \
    } while(0)

#else
// neither CUDA nor HIP
#define FFTX_DEVICE_SUCCESS 0
#endif

// Functions that are defined if and only if either CUDA or HIP.
#if defined(__CUDACC__) || defined(FFTX_HIP) || defined(FFTX_CUDA)
#include <iostream>

inline void FFTX_DEVICE_CHECK_ERROR_Impl(FFTX_DEVICE_ERROR_T a_rc, const char* file, int line)
{
  if (a_rc != FFTX_DEVICE_SUCCESS)
    {
        fftx::ErrStream() << "FFTX Device Failure in " << file << " at line " << line << "\n"
                          << "Error code: " << a_rc 
                          << " (meaning:" << FFTX_DEVICE_GET_ERROR_STRING(a_rc) << ")"
                          << std::endl;
      exit(-1);
    }
}

//  Example of use: FFTX_DEVICE_CHECK_ERROR(FFTX_DEVICE_MEM_COPY(...));
//  Use this macro wrapper to capture the exact location of the call (macro expansion gives the file/line in the actual source file)
#define FFTX_DEVICE_CHECK_ERROR(a_rc) \
    FFTX_DEVICE_CHECK_ERROR_Impl((a_rc), __FILE__, __LINE__)


inline void FFTX_DEVICE_CHECK_Impl(FFTX_DEVICE_ERROR_T a_rc, const std::string& a_name, const char* file, int line)
{
   if (a_rc != FFTX_DEVICE_SUCCESS)
     {
        fftx::ErrStream() << a_name << " error in " << file << " at line " << line << "\n"
                        << "Error code: " << a_rc 
                        << " (meaning:" << FFTX_DEVICE_GET_ERROR_STRING(a_rc) << ")"
                        << std::endl;
        exit(-1);
     }
}

// Example of use: FFTX_DEVICE_CHECK(FFTX_DEVICE_MEM_COPY(...), "memcpy at step 2");
#define FFTX_DEVICE_CHECK(a_rc, a_name) \
    FFTX_DEVICE_CHECK_Impl((a_rc), (a_name), __FILE__, __LINE__)


inline void FFTX_DEVICE_FFT_CHECK_Impl(FFTX_DEVICE_FFT_RESULT a_rc, const std::string& a_name, const char* file, int line)
{
   if (a_rc != FFTX_DEVICE_FFT_SUCCESS)
     {
        fftx::ErrStream() << a_name << " error in " << file << " at line " << line << "\n"
                          << "Error code: " << a_rc  << std::endl;
        exit(-1);
     }
}

// Example of use: FFTX_DEVICE_FFT_CHECK(DEVICE_FFT_PLAN3D(...), "fftplan at step 3");
#define FFTX_DEVICE_FFT_CHECK(a_rc, a_name) \
    FFTX_DEVICE_FFT_CHECK_Impl((a_rc), (a_name), __FILE__, __LINE__)

// For allocating an array on device that is same size as array.
template<int DIM, typename Thost>
inline FFTX_DEVICE_PTR fftxDeviceMallocForHostArray(fftx::array_t<DIM, Thost>& a_hostArray)
{
  size_t npts = a_hostArray.m_domain.size();
  size_t bytes = npts * sizeof(Thost);
  FFTX_DEVICE_PTR devicePtr;
  FFTX_DEVICE_MALLOC((void **)&devicePtr, bytes);
  return devicePtr;
}

// Simpler function to free device memory.
template<typename T>
inline FFTX_DEVICE_ERROR_T fftxDeviceFree(T* a_devicePtr)
{
  return FFTX_DEVICE_FREE((void*) a_devicePtr);
}

// For copying an FFTX array on host to an array on device.
template<int DIM, typename Thost, typename Tdevice>
inline FFTX_DEVICE_ERROR_T fftxCopyHostArrayToDevice(Tdevice* a_devicePtr,
                                                     fftx::array_t<DIM, Thost>& a_hostArray)
{
  auto hostDataPtr = a_hostArray.m_data.local();
  size_t npts = a_hostArray.m_domain.size();
  size_t bytes = npts * sizeof(Thost);
  return FFTX_DEVICE_MEM_COPY((void*) a_devicePtr,
                              hostDataPtr,
                              bytes,
                              FFTX_MEM_COPY_HOST_TO_DEVICE);
}

// For copying an array on device to an FFTX array on host.
template<int DIM, typename Thost, typename Tdevice>
inline FFTX_DEVICE_ERROR_T fftxCopyDeviceToHostArray(fftx::array_t<DIM, Thost>& a_hostArray,
                                                     Tdevice* a_devicePtr)
{
  auto hostDataPtr = a_hostArray.m_data.local();
  size_t npts = a_hostArray.m_domain.size();
  size_t bytes = npts * sizeof(Thost);
  return FFTX_DEVICE_MEM_COPY(hostDataPtr,
                              (void*) a_devicePtr,
                              bytes,
                              FFTX_MEM_COPY_DEVICE_TO_HOST);
}
#endif                    // defined(__CUDACC__) || defined(FFTX_HIP)

#endif                    // FFTX_DEVICE_MACROS_HEADER
