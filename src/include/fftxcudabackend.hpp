#ifndef FFTX_MDDFT_CUDABACKEND_HEADER
#define FFTX_MDDFT_CUDABACKEND_HEADER

//  Copyright (c) 2018-2025, Carnegie Mellon University
//   All rights reserved.
//
//  See LICENSE file for full information

#include <nvrtc.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <tuple>
#include <iomanip>
#include <cstdio>      // perror

#if defined(_WIN32) || defined (_WIN64)
  #include <io.h>
  #define FFTX_POPEN _popen
  #define FFTX_PCLOSE _pclose
  // #define popen _popen
  // #define pclose _pclose
#else
  #include <unistd.h>    // dup2
  #define FFTX_POPEN popen
  #define FFTX_PCLOSE pclose
#endif

#include <sys/types.h> // rest for open/close
#include <sys/stat.h>
#include <fcntl.h>

#include "fftxdevice_macros.h"
#pragma once
#if defined ( FFTX_PRINTDEBUG )
#define FFTX_DEBUGOUT 1
#else
#define FFTX_DEBUGOUT 0
#endif


inline std::string getCUDARuntime() {
    const char * tmp2 = std::getenv("CUDA_HOME");
     std::string tmp(tmp2 ? tmp2 : "");
        if (tmp.empty()) {
            fftx::OutStream() << "[ERROR] No such variable found! Please set CUDA_HOME to point to top level cuda directory" << std::endl;
            exit(-1);
        }
    #if defined (_WIN32) || defined (_WIN64)
    tmp += "./lib/x64/cudadevrt.lib";
    #else
    tmp += "/lib64/libcudadevrt.a";
    #endif

    return tmp;
}



class Executor {
    private:
        int x;
        enum string_code {
            zero,
            one,
            two,
            constant,
            pointer_int,
            pointer_float,
            pointer_double,
            mone
        };
        nvrtcProgram prog;
        nvrtcResult compileResult;
        // std::vector<fftx::array_t<3,std::complex<double>>> in;
        // std::vector<fftx::array_t<3,std::complex<double>>> out;
        std::vector<void*> kernelargs;
        std::vector<std::tuple<std::string, int, std::string>> device_names;
        std::string kernel_name;
        std::string kernels;
        std::vector<std::tuple<std::string, int, std::string>> in_params;
        std::vector<void*> params; 
        std::vector<void *> data;
        size_t logSize;
        char *log;
        size_t ptxSize;
        char *ptx;
        CUdevice cuDevice;
        CUcontext context;
        CUlinkState linkState;
        CUmodule module;
        CUfunction kernel;
        size_t cubinSize;
        void *cubin;
        float GPUtime;
    public:
        // Executor();
    //     Executor(std::vector<fftx::array_t<3,std::complex<double>>> &in1,
    // std::vector<fftx::array_t<3,std::complex<double>>> &out1)
    //     :in(in1), out(out1){}
        // Executor(const std::vector<void*>& args1) {
        //     kernelargs = args1;
        // }
        string_code hashit(std::string const& inString);
        void parseDataStructure(std::string input);
        void createProg();
        void getVars();
        void compileProg();
        void threeinone();
        void getLogsAndPTX();
        void initializeVars();
        void destoryProg();
        float initAndLaunch(std::vector<void*>& args);
        void execute(std::string file_name);
        float getKernelTime();
        // void returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1);
};

inline Executor::string_code Executor::hashit(std::string const& inString) {
    if(inString == "int") return zero;
    if(inString == "float") return one;
    if(inString == "double") return two;
    if(inString == "constant") return constant;
    if(inString == "pointer_int") return pointer_int;
    if(inString == "pointer_float") return pointer_float;
    if(inString == "pointer_double") return pointer_double;
    return mone;
}

inline void Executor::parseDataStructure(std::string input) {
    // fftx::OutStream() << input << std::endl;
    // std::ifstream t(input);
    // std::stringstream ds;
    // ds << t.rdbuf();
    //std::istringstream stream(ds.str());
    std::istringstream stream(input);
    char delim = ' ';
    std::string line;
    std::string b = "------------------";
    while(std::getline(stream, line)){
        if(line.find("JIT BEGIN") != std::string::npos)
            break;
    }
    while(std::getline(stream,line)) {
        if(line == b) {
            break;
        }
        std::istringstream ss(line);
        std::string s;
        //int counter = 0;
        std::vector<std::string> words;
        while(std::getline(ss,s,delim)) {
            words.push_back(s);
        }
        int test = atoi(words.at(0).c_str());
        switch(test) {
            case 0:
                device_names.push_back(std::make_tuple("&"+words.at(1), atoi(words.at(2).c_str()), words.at(3)));
                break;
            case 1:
                in_params.push_back(std::make_tuple("&"+words.at(1), atoi(words.at(2).c_str()), words.at(3)));
                break;
            case 2:
                if ( FFTX_DEBUGOUT ) fftx::OutStream() << "the kernel name is " << words.at(1) << "\n";
                kernel_name = words.at(1);
                break;
            case 3:
                // fftx::OutStream() << "enterd case 3\n";
                // fftx::OutStream() << words.at(1) << " " << words.at(2) << " " << words.at(3) << " \n";
                int loc = atoi(words.at(1).c_str());
                int size = atoi(words.at(2).c_str());
                int dt = atoi(words.at(3).c_str());
                //fftx::OutStream() << "the case is " << dt << std::endl;
                //convert this to a string because spiral prints string type
                switch(dt) {
                    case 0: //int
                    {
                        if(words.size() < 5) {
                            int *data1 = new int[size];         //  int data1[size] = {0};
                            data.push_back(/* & */ data1);
                        }
                        else {
                            int * data1 = new int[size];
                            for(int i = 4; i < words.size(); i++) {
                                data1[i-4] = atoi(words.at(i).c_str());
                            }
                            data.push_back(data1);
                        }
                        break;
                    }
                    case 1: //float
                    {
                        if(words.size() < 5) {
                            float *data1 = new float[size];     //  float data1[size] = {0};
                            data.push_back(/* & */ data1);
                        }
                        else {
                            float * data1 = new float[size];
                            for(int i = 4; i < words.size(); i++) {
                                data1[i-4] = std::stof(words.at(i));
                            }
                            data.push_back(data1);
                        }
                        break;
                    }
                    case 2: //double
                    {
                        if(words.size() < 5) {
                            double *data1 = new double[size];   //  double data1[size] = {0};
                            data.push_back(/* & */ data1);
                        }
                        else {
                            double * data1 = new double[size];
                            for(int i = 4; i < words.size(); i++) {
                                data1[i-4] = std::stod(words.at(i));
                               //std::from_chars(words.at(i).data(), words.at(i).data() + words.at(i).size(),  data1[i-4]);
                            }
                            //exit(1);
                            data.push_back(data1);
                            break;    
                        }
                    }
                    case 3: //constant
                    {
                        if(words.size() < 5) {
                            double *data1 = new double[size];   //  double data1[size] = {0};
                            data.push_back(/* & */ data1);
                        }
                        break;
                    }
                }
                break;
        }
    }
    while(std::getline(stream, line)) {
        kernels += line;
        kernels += "\n";
    }
    if( FFTX_DEBUGOUT ) fftx::OutStream() << kernels << std::endl;
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "parsed input\n";
}

inline void Executor::createProg() {
    FFTX_DEVICE_RTC_SAFE_CALL(nvrtcCreateProgram(&prog, // prog
    kernels.c_str(), // buffer
    NULL, // name
    0, // numHeaders
    NULL, // headers
    NULL)); 
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "created program\n";
}

inline void Executor::getVars() {
    for(int i = 0; i < device_names.size(); i++) {
        FFTX_DEVICE_RTC_SAFE_CALL(nvrtcAddNameExpression(prog, std::get<0>(device_names[i]).c_str()));
    }
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "added variables\n";
}

inline void Executor::compileProg() {
    const char *opts[] = {"--relocatable-device-code=true","--gpu-architecture=compute_70"};
    compileResult = nvrtcCompileProgram(prog, 
    2, 
    opts); 
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "compiled program\n";
}

inline void Executor::getLogsAndPTX() {
    FFTX_DEVICE_RTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    log = new char[logSize];
    FFTX_DEVICE_RTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    if (compileResult != NVRTC_SUCCESS) {
        fftx::OutStream() << "compile failure with code "<< nvrtcGetErrorString(compileResult) << std::endl;
        for(int i = 0; i < logSize; i++) {
            fftx::OutStream() << log[i];
        }
        fftx::OutStream() << std::endl;
        exit(1);
    }
    delete[] log;
    FFTX_DEVICE_RTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    ptx = new char[ptxSize];
    FFTX_DEVICE_RTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    FFTX_DEVICE_SAFE_CALL(cuInit(0));
    // FFTX_DEVICE_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    // FFTX_DEVICE_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    FFTX_DEVICE_SAFE_CALL(cuLinkCreate(0, 0, 0, &linkState));
    FFTX_DEVICE_SAFE_CALL(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, getCUDARuntime().c_str(), 
    0, 0, 0));
    FFTX_DEVICE_SAFE_CALL(cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
    (void *)ptx, ptxSize, "dft_jit.ptx",
    0, 0, 0));
    FFTX_DEVICE_SAFE_CALL(cuLinkComplete(linkState, &cubin, &cubinSize));
    FFTX_DEVICE_SAFE_CALL(cuModuleLoadData(&module, cubin));
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "created module\n";
}

inline void Executor::initializeVars() {
    for(int i = 0; i < device_names.size(); i++) {
        if ( FFTX_DEBUGOUT ) fftx::OutStream() << "this is i " << i << " this is the name " << std::get<0>(device_names[i]) << std::endl;
        const char * name;
        FFTX_DEVICE_RTC_SAFE_CALL(nvrtcGetLoweredName(
                                 prog, 
                                 std::get<0>(device_names[i]).c_str(), // name expression
                                 &name                         // lowered name
                                 ));
        if ( FFTX_DEBUGOUT ) fftx::OutStream() << "it got past lower name\n";
        CUdeviceptr variable_addr;
        FFTX_DEVICE_SAFE_CALL(cuModuleGetGlobal(&variable_addr, NULL, module, name));
         if ( FFTX_DEBUGOUT ) fftx::OutStream() << "it got past get global\n";
        std::string test = std::get<2>(device_names[i]);
        switch(hashit(test)) {
            case zero:
            {
                int * value = (int*)(data.at(i));
                FFTX_DEVICE_SAFE_CALL(cuMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(int)));
                break;
            }
            case one:
            {
                float * value = (float*)(data.at(i));
                FFTX_DEVICE_SAFE_CALL(cuMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(float)));
                break;
            }
            case two:
            {   
                double * value = (double*)(data.at(i));
                FFTX_DEVICE_SAFE_CALL(cuMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(double)));
                break;
            }
            case constant:
            {
                break;
            }
            case pointer_int:
            {
                if ( FFTX_DEBUGOUT ) fftx::OutStream() << "i have a pointer int\n" << std::get<1>(device_names.at(i)) << "\n";
                CUdeviceptr h;
                FFTX_DEVICE_SAFE_CALL(cuMemAlloc(&h, std::get<1>(device_names.at(i)) * sizeof(int)));
                FFTX_DEVICE_SAFE_CALL(cuMemcpyHtoD(variable_addr, &h, sizeof(int*)));
                // cuMemFree(h);      
                break;
            }
            case pointer_float:
            {
                if ( FFTX_DEBUGOUT ) fftx::OutStream() << "i have a pointer float\n" << std::get<1>(device_names.at(i)) << "\n";
                CUdeviceptr h;
                FFTX_DEVICE_SAFE_CALL(cuMemAlloc(&h, std::get<1>(device_names.at(i)) * sizeof(float)));
                FFTX_DEVICE_SAFE_CALL(cuMemcpyHtoD(variable_addr, &h, sizeof(float*)));
                // cuMemFree(h);                
                break;
            }
            case pointer_double:
            {
                if ( FFTX_DEBUGOUT ) fftx::OutStream() << "i have a pointer double\n" << std::get<1>(device_names.at(i)) << "\n";
                CUdeviceptr h;
                FFTX_DEVICE_SAFE_CALL(cuMemAlloc(&h, std::get<1>(device_names.at(i)) * sizeof(double)));
                FFTX_DEVICE_SAFE_CALL(cuMemcpyHtoD(variable_addr, &h, sizeof(double*)));
                // cuMemFree(h);
                break;
            }
        }
    }
}

inline void Executor::destoryProg() {
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "destoryed program call\n";
    FFTX_DEVICE_RTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
}

inline float Executor::initAndLaunch(std::vector<void*>& args) {
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "the kernel name is " << kernel_name << std::endl;
    FFTX_DEVICE_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));

    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "configuring device execution environment " << std::endl;
    FFTX_DEVICE_SAFE_CALL(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, 1073741824));
    FFTX_DEVICE_SAFE_CALL(cuFuncSetCacheConfig(kernel, CU_FUNC_CACHE_PREFER_L1));
     

    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "launched kernel\n";
    CUevent start, stop;
    FFTX_DEVICE_SAFE_CALL(cuEventCreate(&start, CU_EVENT_DEFAULT));
    FFTX_DEVICE_SAFE_CALL(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    FFTX_DEVICE_SAFE_CALL(cuEventRecord(start,0));
    FFTX_DEVICE_SAFE_CALL(
        cuLaunchKernel(kernel,
                       1, 1, 1, // grid dim
                       1, 1, 1, // block dim
                       0, NULL, // shared mem and stream
                       //kernelargs.data(), 0)); // arguments
                       args.data(), 0));
    FFTX_DEVICE_SAFE_CALL(cuEventRecord(stop,0));
    FFTX_DEVICE_SAFE_CALL(cuCtxSynchronize());
    FFTX_DEVICE_SAFE_CALL(cuEventSynchronize(stop));
    FFTX_DEVICE_SAFE_CALL(cuEventElapsedTime(&GPUtime, start, stop));
    return getKernelTime();
}

inline void Executor::execute(std::string file_name) {
    //int count = *((int*)inputargs.at(0));
    // fftx::OutStream() << "count is" << counts << std::endl;
    // fftx::OutStream() << (char*)inputargs.at(inputargs.size()-1) << std::endl;
    // fftx::OutStream() << (inputargs.at(inputargs.size()-2))[counts-1] << std::endl;
    //fftx::OutStream() << in.at(0).m_domain.size() << std::endl;
    if ( FFTX_DEBUGOUT ) fftx::OutStream() << "begin parsing\n";
    parseDataStructure(file_name);
    if ( FFTX_DEBUGOUT ) {
        fftx::OutStream() << "finished parsing\n";
        for(int i = 0; i < device_names.size(); i++) {
            fftx::OutStream() << std::get<0>(device_names[i]) << std::endl;
        }
        fftx::OutStream() << kernel_name << std::endl;
    }
    createProg();
    getVars();
    compileProg();
    getLogsAndPTX();
    initializeVars();
    destoryProg();
}

inline float Executor::getKernelTime() {
    return GPUtime;
}

// void Executor::returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1) {
//     gatherOutput(out1.at(0), kernelargs);
// }

#endif            //  FFTX_MDDFT_CUDABACKEND_HEADER
