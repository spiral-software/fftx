#ifdef FFTX_HIP
#include <hip/hip_runtime.h>
#include <hipfft.h>
#include "rocfft.h"
#define DEVICE_EVENT_T hipEvent_t
#define DEVICE_EVENT_CREATE hipEventCreate
#define DEVICE_MALLOC hipMalloc
#define DEVICE_EVENT_RECORD hipEventRecord
#define DEVICE_EVENT_ELAPSED_TIME hipEventElapsedTime
#define DEVICE_SYNCHRONIZE hipDeviceSynchronize
#define DEVICE_EVENT_SYNCHRONIZE hipEventSynchronize
#define DEVICE_FREE hipFree
#define DEVICE_MEM_COPY hipMemcpy
#define MEM_COPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define MEM_COPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define DEVICE_FFT_TYPE hipfftType
#define DEVICE_FFT_RESULT hipfftResult
#define DEVICE_FFT_HANDLE hipfftHandle
#define DEVICE_FFT_PLAN3D hipfftPlan3d
#define DEVICE_FFT_EXECZ2Z hipfftExecZ2Z
#define DEVICE_FFT_EXECD2Z hipfftExecD2Z
#define DEVICE_FFT_EXECZ2D hipfftExecZ2D
#define DEVICE_FFT_DESTROY hipfftDestroy
#define DEVICE_FFT_DOUBLEREAL hipfftDoubleReal
#define DEVICE_FFT_DOUBLECOMPLEX hipfftDoubleComplex
#define DEVICE_FFT_Z2Z HIPFFT_Z2Z
#define DEVICE_FFT_D2Z HIPFFT_D2Z
#define DEVICE_FFT_Z2D HIPFFT_Z2D
#define DEVICE_FFT_SUCCESS HIPFFT_SUCCESS
#define DEVICE_FFT_FORWARD HIPFFT_FORWARD
#define DEVICE_FFT_INVERSE HIPFFT_BACKWARD
#else
#include <cufft.h>
#include <helper_cuda.h>
#define DEVICE_EVENT_T cudaEvent_t
#define DEVICE_EVENT_CREATE cudaEventCreate
#define DEVICE_MALLOC cudaMalloc
#define DEVICE_EVENT_RECORD cudaEventRecord
#define DEVICE_EVENT_ELAPSED_TIME cudaEventElapsedTime
#define DEVICE_SYNCHRONIZE cudaDeviceSynchronize
#define DEVICE_EVENT_SYNCHRONIZE cudaEventSynchronize
#define DEVICE_FREE cudaFree
#define DEVICE_MEM_COPY cudaMemcpy
#define MEM_COPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define MEM_COPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define DEVICE_FFT_TYPE cufftType
#define DEVICE_FFT_RESULT cufftResult
#define DEVICE_FFT_HANDLE cufftHandle
#define DEVICE_FFT_PLAN3D cufftPlan3d
#define DEVICE_FFT_EXECZ2Z cufftExecZ2Z
#define DEVICE_FFT_EXECD2Z cufftExecD2Z
#define DEVICE_FFT_EXECZ2D cufftExecZ2D
#define DEVICE_FFT_DESTROY cufftDestroy
#define DEVICE_FFT_DOUBLEREAL cufftDoubleReal
#define DEVICE_FFT_DOUBLECOMPLEX cufftDoubleComplex
#define DEVICE_FFT_Z2Z CUFFT_Z2Z
#define DEVICE_FFT_D2Z CUFFT_D2Z
#define DEVICE_FFT_Z2D CUFFT_Z2D
#define DEVICE_FFT_SUCCESS CUFFT_SUCCESS
#define DEVICE_FFT_FORWARD CUFFT_FORWARD
#define DEVICE_FFT_INVERSE CUFFT_INVERSE
#endif
