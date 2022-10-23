#include <nvrtc.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include "fftx3.hpp"
#pragma once
#define NVRTC_SAFE_CALL(x) \
 do { \
 nvrtcResult result = x; \
 if (result != NVRTC_SUCCESS) { \
 std::cerr << "\nerror: " #x " failed with error " \
 << nvrtcGetErrorString(result) << '\n'; \
 exit(1); \
 } \
 } while(0)
#define CUDA_SAFE_CALL(x) \
 do { \
 CUresult result = x; \
 if (result != CUDA_SUCCESS) { \
 const char *msg; \
 cuGetErrorName(result, &msg); \
 std::cerr << "\nerror: " #x " failed with error " \
 << msg << '\n'; \
 exit(1); \
 } \
 } while(0)

void initDevice() {
    CUdevice cuDevice;
    CUcontext context;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    std::cout << "initialized gpu environment\n";
}

void initGPUPointer(CUdeviceptr& d, fftx::array_t<3,std::complex<double>> in) {
    std::cout << "allocating memory\n" << in.m_domain.size() << "\n";
    CUDA_SAFE_CALL(cuMemAlloc(&d, in.m_domain.size() * sizeof(std::complex<double>)));
    std::cout << "allocated X\n";
}

void copyData(CUdeviceptr& input, fftx::array_t<3,std::complex<double>>& in) {
    std::cout << "allocating memory\n" << in.m_domain.size() << "\n";
    CUDA_SAFE_CALL(cuMemcpyHtoD(input, in.m_data.local(),  in.m_domain.size() * sizeof(std::complex<double>)));
    std::cout << "copied X\n";
}


std::vector<void*> initGPUData(fftx::array_t<3,std::complex<double>> in,
                fftx::array_t<3,std::complex<double>> out) {
    CUdevice cuDevice;
    CUcontext context;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUdeviceptr *dX, *dY, *dsym;
    dX = new CUdeviceptr;
    dY = new CUdeviceptr;
    dsym = new CUdeviceptr;
    std::cout << "allocating memory\n" << in.m_domain.size() << "\n";
    CUDA_SAFE_CALL(cuMemAlloc(dX, in.m_domain.size() * sizeof(std::complex<double>)));
    std::cout << "allocated X\n";
    CUDA_SAFE_CALL(cuMemcpyHtoD(*dX, in.m_data.local(),  in.m_domain.size() * sizeof(std::complex<double>)));
    std::cout << "copied X\n";
    CUDA_SAFE_CALL(cuMemAlloc(dY, out.m_domain.size() * sizeof(std::complex<double>)));
    std::cout << "allocated Y\n";
    // //CUDA_SAFE_CALL(cuMemcpyHtoD(dY, Y, 64* sizeof(double)));
    CUDA_SAFE_CALL(cuMemAlloc(dsym, 64* sizeof(double)));

    std::vector<void*> inside;//{&dY, &dX, &dsym};
    inside.push_back(dY);
    inside.push_back(dX);
    inside.push_back(dsym);
    return inside;
}

void gatherOutput(fftx::array_t<3,std::complex<double>> &out, std::vector<void*>& params) {
     std::cout << out.m_domain.size() << std::endl;
     //out.m_data.local() = new std::complex<double>[out.m_domain.size()];
     CUDA_SAFE_CALL(cuMemcpyDtoH(out.m_data.local(), *((CUdeviceptr*)params.at(0)), out.m_domain.size()*sizeof(std::complex<double>)));
}