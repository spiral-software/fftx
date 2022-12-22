#ifndef FFTX_MDDFT_CUDABACKEND_HEADER
#define FFTX_MDDFT_CUDABACKEND_HEADER

//  Copyright (c) 2018-2022, Carnegie Mellon University
//  See LICENSE for details

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
#include <unistd.h>    // dup2
#include <sys/types.h> // rest for open/close
#include <sys/stat.h>
#include <fcntl.h>

#define LOCALDEBUG 0

#define NVRTC_SAFE_CALL(x)						\
 do { \
 nvrtcResult result = x; \
 if (result != NVRTC_SUCCESS) { \
 std::cerr << "\nerror: " #x " failed with error " \
 << nvrtcGetErrorString(result) << '\n'; \
 exit(1); \
 } \
 } while(0)

#define CUDA_SAFE_CALL(x)						\
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

#include "data_interaction.hpp"
#pragma once


std::string getCUDARuntime() {
    const char * tmp2 = std::getenv("CUDA_HOME");
     std::string tmp(tmp2 ? tmp2 : "");
        if (tmp.empty()) {
            std::cout << "[ERROR] No such variable found! Please set CUDA_HOME to point to top level cuda directory" << std::endl;
            exit(-1);
        }
    tmp += "/lib64/libcudadevrt.a";

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
        void returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1);
};

Executor::string_code Executor::hashit(std::string const& inString) {
    if(inString == "int") return zero;
    if(inString == "float") return one;
    if(inString == "double") return two;
    if(inString == "constant") return constant;
    if(inString == "pointer_int") return pointer_int;
    if(inString == "pointer_float") return pointer_float;
    if(inString == "pointer_double") return pointer_double;
    return mone;
}

void Executor::parseDataStructure(std::string input) {
    // std::cout << input << std::endl;
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
                std::cout << "the kernel name is " << words.at(1) << "\n";
                kernel_name = words.at(1);
                break;
            case 3:
                // std::cout << "enterd case 3\n";
                // std::cout << words.at(1) << " " << words.at(2) << " " << words.at(3) << " \n";
                int loc = atoi(words.at(1).c_str());
                int size = atoi(words.at(2).c_str());
                int dt = atoi(words.at(3).c_str());
                //std::cout << "the case is " << dt << std::endl;
                //convert this to a string because spiral prints string type
                switch(dt) {
                    case 0: //int
                    {
                        if(words.size() < 5) {
                            int data1[size] = {0};
                            data.push_back(&data1);
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
                            float data1[size] = {0};
                            data.push_back(&data1);
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
                            double data1[size] = {0};
                            data.push_back(&data1);
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
                            double data1[size] = {0};
                            data.push_back(&data1);
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
     if(LOCALDEBUG == 1)
        std::cout << "parsed input\n";
}

void Executor::createProg() {
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, // prog
    kernels.c_str(), // buffer
    NULL, // name
    0, // numHeaders
    NULL, // headers
    NULL)); 
    if(LOCALDEBUG == 1)
        std::cout << "created program\n";
}

void Executor::getVars() {
    for(int i = 0; i < device_names.size(); i++) {
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, std::get<0>(device_names[i]).c_str()));
    }
    if(LOCALDEBUG == 1)
        std::cout << "added variables\n";
}

void Executor::compileProg() {
    const char *opts[] = {"--device-debug", "--relocatable-device-code=true","--gpu-architecture=compute_70"};
    compileResult = nvrtcCompileProgram(prog, 
    3, 
    opts); 
    if(LOCALDEBUG == 1)
        std::cout << "compiled program\n";
}

void Executor::getLogsAndPTX() {
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    if (compileResult != NVRTC_SUCCESS) {
        std::cout << "compile failure with code "<< nvrtcGetErrorString(compileResult) << std::endl;
        for(int i = 0; i < logSize; i++) {
            std::cout << log[i];
        }
        std::cout << std::endl;
        exit(1);
    }
    delete[] log;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    // CUDA_SAFE_CALL(cuInit(0));
    // CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    // CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuLinkCreate(0, 0, 0, &linkState));
    CUDA_SAFE_CALL(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, getCUDARuntime().c_str(), 
    0, 0, 0));
    CUDA_SAFE_CALL(cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
    (void *)ptx, ptxSize, "dft_jit.ptx",
    0, 0, 0));
    CUDA_SAFE_CALL(cuLinkComplete(linkState, &cubin, &cubinSize));
    CUDA_SAFE_CALL(cuModuleLoadData(&module, cubin));
    if(LOCALDEBUG == 1)
        std::cout << "created module\n";
}

void Executor::initializeVars() {
    for(int i = 0; i < device_names.size(); i++) {
        if(LOCALDEBUG == 1)
            std::cout << "this is i " << i << " this is the name " << std::get<0>(device_names[i]) << std::endl;
        const char * name;
        NVRTC_SAFE_CALL(nvrtcGetLoweredName(
        prog, 
        std::get<0>(device_names[i]).c_str(), // name expression
        &name                         // lowered name
        ));
        if(LOCALDEBUG == 1)
            std::cout << "it got past lower name\n";
        CUdeviceptr variable_addr;
        CUDA_SAFE_CALL(cuModuleGetGlobal(&variable_addr, NULL, module, name));
         if(LOCALDEBUG == 1)
            std::cout << "it got past get global\n";
        std::string test = std::get<2>(device_names[i]);
        switch(hashit(test)) {
            case zero:
            {
                int * value = (int*)(data.at(i));
                CUDA_SAFE_CALL(cuMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(int)));
                break;
            }
            case one:
            {
                float * value = (float*)(data.at(i));
                CUDA_SAFE_CALL(cuMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(float)));
                break;
            }
            case two:
            {   
                double * value = (double*)(data.at(i));
                CUDA_SAFE_CALL(cuMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(double)));
                break;
            }
            case constant:
            {
                break;
            }
            case pointer_int:
            {
                if(LOCALDEBUG == 1)
                std::cout << "i have a pointer int\n" << std::get<1>(device_names.at(i)) << "\n";
                CUdeviceptr h;
                CUDA_SAFE_CALL(cuMemAlloc(&h, std::get<1>(device_names.at(i)) * sizeof(int)));
                CUDA_SAFE_CALL(cuMemcpyHtoD(variable_addr, &h, sizeof(int*)));
                // cuMemFree(h);      
                break;
            }
            case pointer_float:
            {
                if(LOCALDEBUG == 1)
                std::cout << "i have a pointer float\n" << std::get<1>(device_names.at(i)) << "\n";
                CUdeviceptr h;
                CUDA_SAFE_CALL(cuMemAlloc(&h, std::get<1>(device_names.at(i)) * sizeof(float)));
                CUDA_SAFE_CALL(cuMemcpyHtoD(variable_addr, &h, sizeof(float*)));
                // cuMemFree(h);                
                break;
            }
            case pointer_double:
            {
                if(LOCALDEBUG == 1)
                std::cout << "i have a pointer double\n" << std::get<1>(device_names.at(i)) << "\n";
                CUdeviceptr h;
                CUDA_SAFE_CALL(cuMemAlloc(&h, std::get<1>(device_names.at(i)) * sizeof(double)));
                CUDA_SAFE_CALL(cuMemcpyHtoD(variable_addr, &h, sizeof(double*)));
                // cuMemFree(h);
                break;
            }
        }
    }
}

void Executor::destoryProg() {
    if(LOCALDEBUG == 1)
        std::cout << "destoryed program call\n";
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
}

float Executor::initAndLaunch(std::vector<void*>& args) {
    if(LOCALDEBUG == 1)
    std::cout << "the kernel name is " << kernel_name << std::endl;
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));
    if(LOCALDEBUG == 1)
    std::cout << "launched kernel\n";
    CUevent start, stop;
    CUDA_SAFE_CALL(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CUDA_SAFE_CALL(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    CUDA_SAFE_CALL(cuEventRecord(start,0));
    CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
    1, 1, 1, // grid dim
    1, 1, 1, // block dim
    0, NULL, // shared mem and stream
    //kernelargs.data(), 0)); // arguments
    args.data(),0));
    CUDA_SAFE_CALL(cuEventRecord(stop,0));
    CUDA_SAFE_CALL(cuCtxSynchronize());
    CUDA_SAFE_CALL(cuEventSynchronize(stop));
    CUDA_SAFE_CALL(cuEventElapsedTime(&GPUtime, start, stop));
    return getKernelTime();
}

void Executor::execute(std::string file_name) {
    //int count = *((int*)inputargs.at(0));
    // std::cout << "count is" << counts << std::endl;
    // std::cout << (char*)inputargs.at(inputargs.size()-1) << std::endl;
    // std::cout << (inputargs.at(inputargs.size()-2))[counts-1] << std::endl;
    //std::cout << in.at(0).m_domain.size() << std::endl;
    if(LOCALDEBUG == 1)
    std::cout << "begin parsing\n";
    parseDataStructure(file_name);
    if(LOCALDEBUG == 1) {
        std::cout << "finished parsing\n";
        for(int i = 0; i < device_names.size(); i++) {
            std::cout << std::get<0>(device_names[i]) << std::endl;
        }
        std::cout << kernel_name << std::endl;
        // std::cout << kernels << std::endl;
    }
    createProg();
    getVars();
    compileProg();
    getLogsAndPTX();
    initializeVars();
    destoryProg();
    //std::cout << "begin kernel launch\n";
    //initAndLaunch(in, out);
}

float Executor::getKernelTime() {
    return GPUtime;
}

void Executor::returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1) {
    gatherOutput(out1.at(0), kernelargs);
}

#endif			//  FFTX_MDDFT_CUDABACKEND_HEADER
