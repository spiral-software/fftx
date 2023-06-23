#include <iostream>
#include <algorithm>
#include <any>
#include <string>
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
#if defined FFTX_HIP
#include "hipbackend.hpp"
#elif defined FFTX_CUDA
#include "hipbackend.hpp"
#else
#include "cpubackend.hpp"
#endif
#pragma once

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
std::vector<int> internal_sizes;


namespace fftx_external {

    #if defined FFTX_HIP
    hipfftResult hipfftPlan3d(hipfftHandle *plan, int nx, int ny, int nz, hipfftType type) {
        std::cout << "inside hijacked hip plan" << std::endl;
        internal_sizes.push_back(nx);
        internal_sizes.push_back(ny);
        internal_sizes.push_back(nz);
        return HIPFFT_SUCCESS;
    }

    hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata, hipfftDoubleComplex *odata, int direction) {
        std::cout << "Entered mddft fftx hip api call" << std::endl;
        hipDeviceptr_t dsym;
        hipMalloc((void **)&dsym,  1* sizeof(std::complex<double>));
        std::vector<void*> args{odata,idata,dsym};
        if(stored_mddft_jit.find(std::make_tuple(internal_sizes.at(0),internal_sizes.at(1),internal_sizes.at(2),direction)) != stored_mddft_jit.end()) {
            std::cout << "running cached instance" << std::endl;
            Executor e;
            e.execute(stored_mddft_jit.at(std::make_tuple(internal_sizes.at(0),internal_sizes.at(1),internal_sizes.at(2),direction)));
        }
        else {
            std::cout << "creating problem " << std::endl;
            if(direction == -1) {
                MDDFTProblem mdp(args, internal_sizes, "mddft");
                 mdp.transform();
                stored_mddft_jit[std::make_tuple(internal_sizes.at(0),internal_sizes.at(1),internal_sizes.at(2),direction)] = mdp.returnJIT();
            }
            else {
                IMDDFTProblem mdp(args, internal_sizes, "imddft");
                mdp.transform();
                stored_mddft_jit[std::make_tuple(internal_sizes.at(0),internal_sizes.at(1),internal_sizes.at(2),direction)] = mdp.returnJIT();
            }
        }
        return HIPFFT_SUCCESS;
    }
    #endif
    #if defined FFTX_CUDA
     cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type) {
        std::cout << "inside hijacked cuda plan" << std::endl;
        internal_sizes.push_back(nx);
        internal_sizes.push_back(ny);
        internal_sizes.push_back(nz);
        return CUFFT_SUCCESS;
    }

    cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex  *idata, cufftDoubleComplex  *odata, int direction) {
        std::cout << "Entered mddft fftx cuda api call" << std::endl;
        CUdeviceptr dsym;
        cuMalloc((void **)&dsym,  1* sizeof(std::complex<double>));
        std::vector<void*> args{odata,idata,dsym};
        if(stored_mddft_jit.find(std::make_tuple(internal_sizes.at(0),internal_sizes.at(1),internal_sizes.at(2),direction)) != stored_mddft_jit.end()) {
            std::cout << "running cached instance" << std::endl;
            Executor e;
            e.execute(stored_mddft_jit.at(std::make_tuple(internal_sizes.at(0),internal_sizes.at(1),internal_sizes.at(2),direction)));
        }
        else {
            std::cout << "creating problem " << std::endl;
            if(direction == -1) {
                MDDFTProblem mdp(args, internal_sizes, "mddft");
                 mdp.transform();
                stored_mddft_jit[std::make_tuple(internal_sizes.at(0),internal_sizes.at(1),internal_sizes.at(2),direction)] = mdp.returnJIT();
            }
            else {
                IMDDFTProblem mdp(args, internal_sizes, "imddft");
                mdp.transform();
                stored_mddft_jit[std::make_tuple(internal_sizes.at(0),internal_sizes.at(1),internal_sizes.at(2),direction)] = mdp.returnJIT();
            }
        }
        return CUFFT_SUCCESS;
    }
    #endif
}

#if defined FFTX_HIP
#define hipfftExecZ2Z fftx_external::hipfftExecZ2Z
#define hipfftPlan3d fftx_external::hipfftPlan3d
#endif
#if defined FFTX_CUDA
#define cufftPlan3d fftx_external::cufftPlan3d
#define cufftExecZ2Z fftx_external::cufftExecZ2Z
#endif
