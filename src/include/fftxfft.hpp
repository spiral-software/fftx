#include <cstdlib>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <tuple>
#include <utility>
#include "interface.hpp"
#include "mddftlib.hpp"
#include "mdprdftlib.hpp"
#include "dftbatlib.hpp"
// #include "cudabackend.hpp"
#if defined FFTX_HIP
#include "hipbackend.hpp"
#elif defined FFTX_CUDA
#include "hipbackend.hpp"
#else
#include "cpubackend.hpp"
#endif
#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>

#if defined(_WIN32) || defined (_WIN64)
  #include <io.h>
  #define popen _popen
  #define pclose _pclose
#else
  #include <unistd.h>    // dup2
#endif

#include <sys/types.h> // rest for open/close
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <array>
#pragma once

#if defined ( PRINTDEBUG )
#define DEBUGOUT 1
#else
#define DEBUGOUT 0
#endif

namespace fftx_cuFFT {

#define CUFFT_FORWARD -1
#define CUFFT_INVERSE 1

typedef struct {
    int x;
    int y;
    int z;
    int batch;

} cufftHandle;

typedef enum cufftType_t {
    CUFFT_R2C = 0x2a,  // Real to complex (interleaved)
    CUFFT_C2R = 0x2c,  // Complex (interleaved) to real
    CUFFT_C2C = 0x29,  // Complex to complex (interleaved)
    CUFFT_D2Z = 0x6a,  // Double to double-complex (interleaved)
    CUFFT_Z2D = 0x6c,  // Double-complex (interleaved) to double
    CUFFT_Z2Z = 0x69   // Double-complex to double-complex (interleaved)
} cufftType;

typedef enum cufftResult_t {
    CUFFT_SUCCESS        = 0,  //  The cuFFT operation was successful
    CUFFT_INVALID_PLAN   = 1,  //  cuFFT was passed an invalid plan handle
    CUFFT_ALLOC_FAILED   = 2,  //  cuFFT failed to allocate GPU or CPU memory
    CUFFT_INVALID_TYPE   = 3,  //  No longer used
    CUFFT_INVALID_VALUE  = 4,  //  User specified an invalid pointer or parameter
    CUFFT_INTERNAL_ERROR = 5,  //  Driver or internal cuFFT library error
    CUFFT_EXEC_FAILED    = 6,  //  Failed to execute an FFT on the GPU
    CUFFT_SETUP_FAILED   = 7,  //  The cuFFT library failed to initialize
    CUFFT_INVALID_SIZE   = 8,  //  User specified an invalid transform size
    CUFFT_UNALIGNED_DATA = 9,  //  No longer used
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10, //  Missing parameters in call
    CUFFT_INVALID_DEVICE = 11, //  Execution of a plan was on different GPU than plan creation
    CUFFT_PARSE_ERROR    = 12, //  Internal plan database error
    CUFFT_NO_WORKSPACE   = 13,  //  No workspace has been provided prior to plan execution
    CUFFT_NOT_IMPLEMENTED = 14, // Function does not implement functionality for parameters given.
    CUFFT_LICENSE_ERROR  = 15, // Used in previous versions.
    CUFFT_NOT_SUPPORTED  = 16  // Operation is not supported for parameters given.
} cufftResult;

typedef std::complex<double> cufftComplex;

typedef std::tuple<int, int, int, int> keys_t;
 
struct key_hash : public std::unary_function<keys_t, std::size_t>
{
std::size_t operator()(const keys_t& k) const
{
return std::get<0>(k) ^ std::get<1>(k) ^ std::get<2>(k) ^ std::get<3>(k);
}
};
 
struct key_equal : public std::binary_function<keys_t, keys_t, bool>
{
bool operator()(const keys_t& v0, const keys_t& v1) const
{
return (
std::get<0>(v0) == std::get<0>(v1) &&
std::get<1>(v0) == std::get<1>(v1) &&
std::get<2>(v0) == std::get<2>(v1) &&
std::get<3>(v0) == std::get<3>(v1) 
);
}
};

typedef std::unordered_map<const keys_t,std::string,key_hash,key_equal> map_t;

map_t stored_mddft_jit;
map_t stored_mdprdft_jit;


#if defined FFTX_HIP
void mddft(int x, int y, int z, int sign, hipDeviceptr_t Y, hipDeviceptr_t X) {
    if ( DEBUGOUT) std::cout << "Entered mddft fftx hip api call" << std::endl;
    hipDeviceptr_t dsym;
    hipMalloc((void **)&dsym,  1* sizeof(std::complex<double>));
    std::vector<void*> args{Y,X,dsym};
    std::vector<int> sizes{x,y,z};
    if(stored_mddft_jit.find(std::make_tuple(x,y,z,sign)) != stored_mddft_jit.end()) {
        if ( DEBUGOUT) std::cout << "running cached instance" << std::endl;
        Executor e;
        e.execute(stored_mddft_jit.at(std::make_tuple(x,y,z,sign)));
    }
    else {
        if(sign == -1)
            MDDFTProblem mdp(args, sizes, "mddft");
        else 
            IMDDFTProblem mdp(args, sizes, "imddft");
        mdp.transform();
        stored_mddft_jit[std::make_tuple(x,y,z,sign)] = mdp.returnJIT();
    }
}

void mdprdft(int x, int y, int z, int sign, hipDeviceptr_t Y, hipDeviceptr_t X) {
    if ( DEBUGOUT) std::cout << "Entered mdprdft fftx hip api call" << std::endl;
    hipDeviceptr_t dsym;
    hipMalloc((void **)&dsym,  1* sizeof(std::complex<double>));
    std::vector<void*> args{Y,X,dsym};
    std::vector<int> sizes{x,y,z};
    if(stored_mdprdft_jit.find(std::make_tuple(x,y,z,sign)) != stored_mdprdft_jit.end()) {
        if ( DEBUGOUT) std::cout << "running cached instance" << std::endl;
        Executor e;
        e.execute(stored_mdprdft_jit.at(std::make_tuple(x,y,z,sign)));
    }
    else {
        if(sign == -1)
            MDPRDFTProblem mdp(args, sizes, "mdprdft");
        else 
            IMDPRDFTProblem mdp(args, sizes, "imdprdft");
        mdp.transform();
        stored_mdprdft_jit[std::make_tuple(x,y,z,sign)] = mdp.returnJIT();
    }
}

#else
void mddft(int x, int y, int z, int sign, double * Y, double * X) {
    if ( DEBUGOUT) std::cout << "Entered mddft fftx cpu api call" << std::endl;
    std::complex<double> * dsym = new std::complex<double>[1];
    // hipMalloc((void **)&dsym,  1* sizeof(std::complex<double>));
    std::vector<void*> args{(void*)Y,(void*)X,(void*)dsym};
    std::vector<int> sizes{x,y,z};
    if(stored_mddft_jit.find(std::make_tuple(x,y,z,sign)) != stored_mddft_jit.end()) {
        if ( DEBUGOUT) std::cout << "running cached instance" << std::endl;
        Executor e;
        e.execute(stored_mddft_jit.at(std::make_tuple(x,y,z,sign)));
    }
    else {
        if(sign == -1)
            MDDFTProblem mdp(args, sizes, "mddft");
        else 
            IMDDFTProblem mdp(args, sizes, "imddft");
        mdp.transform();
        stored_mddft_jit[std::make_tuple(x,y,z,sign)] = mdp.returnJIT();
    }
}
void mdprdft(int x, int y, int z, int sign, double * Y, double * X) {
    if ( DEBUGOUT) std::cout << "Entered mddft fftx cpu api call" << std::endl;
    std::complex<double> * dsym = new std::complex<double>[1];
    // hipMalloc((void **)&dsym,  1* sizeof(std::complex<double>));
    std::vector<void*> args{(void*)Y,(void*)X,(void*)dsym};
    std::vector<int> sizes{x,y,z};
    if(stored_mdprdft_jit.find(std::make_tuple(x,y,z,sign)) != stored_mdprdft_jit.end()) {
        if ( DEBUGOUT) std::cout << "running cached instance" << std::endl;
        Executor e;
        e.execute(stored_mdprdft_jit.at(std::make_tuple(x,y,z,sign)));
    }
    else {
        if(sign == -1)
            MDPRDFTProblem mdp(args, sizes, "mdprdft");
        else 
            IMDPRDFTProblem mdp(args, sizes, "imdprdft");
        mdp.transform();
        stored_mdprdft_jit[std::make_tuple(x,y,z,sign)] = mdp.returnJIT();
    }
}
#endif


#if defined FFTX_HIP 

cufftResult cufftCreate(cufftHandle * plan) {
    return CUFFT_SUCCESS;
}

cufftResult cufftDestroy(cufftHandle plan) {
    return CUFFT_SUCCESS;
}


void cudaMalloc(void** ptr, size_t size) {
    hipMalloc(ptr, size);
}

cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
        int istride, int idist, int *onembed, int ostride,
        int odist, cufftType type, int batch) {
            if(rank != 3) {
                std::cout << "only supports 3d ffts" << std::endl;
                return CUFFT_SETUP_FAILED;
            }
            plan->x = n[0];
            plan->y = n[1];
            plan->z = n[2];
            return CUFFT_SUCCESS;
        }
// cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata,
//         cufftComplex *odata, int direction) {
cufftResult cufftExecC2C(cufftHandle plan, hipDeviceptr_t Y,
         hipDeviceptr_t X, int sign) {
    if ( DEBUGOUT) std::cout << "Entered mddft cuapi call for hip" << std::endl;
    hipDeviceptr_t dsym;
    hipMalloc((void **)&dsym,  1* sizeof(std::complex<double>));
    std::vector<void*> args{Y,X,dsym};
    std::vector<int> sizes{plan.x,plan.y,plan.z};
    if(stored_jit.find(std::make_tuple(plan.x,plan.y,plan.z,sign)) != stored_jit.end()) {
        if ( DEBUGOUT) std::cout << "running cached instance cuapi call for hip" << std::endl;
        Executor e;
        e.execute(stored_jit.at(std::make_tuple(plan.x,plan.y,plan.z,sign)));
    }
    else {
        if(sign == -1)
            MDDFTProblem mdp(args, sizes);
        else 
            IMDDFTProblem mdp(args, sizes);
        mdp.transform();
        stored_jit[std::make_tuple(plan.x,plan.y,plan.z,sign)] = mdp.returnJIT();
    }
    return CUFFT_SUCCESS; 
}
#endif
}
