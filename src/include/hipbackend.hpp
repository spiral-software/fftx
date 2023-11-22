#ifndef FFTX_MDDFT_HIPBACKEND_HEADER
#define FFTX_MDDFT_HIPBACKEND_HEADER

//  Copyright (c) 2018-2022, Carnegie Mellon University
//  See LICENSE for details

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <tuple>
#include <iomanip>
#include <fcntl.h>

#include "device_macros.h"
#pragma once

#if defined ( PRINTDEBUG )
#define DEBUGOUT 1
#else
#define DEBUGOUT 0
#endif

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
        hiprtcProgram prog;
        hiprtcResult compileResult;
        std::vector<void*> kernelargs;
        std::vector<std::tuple<std::string, int, std::string>> device_names;
        std::vector<std::string> kernel_names;
        std::vector<int> kernel_params;
        std::string kernel_preamble;
        std::string kernels;
        std::vector<std::tuple<std::string, int, std::string>> in_params;
        std::vector<void*> params; 
        std::vector<void *> data;
        size_t logSize;
        char *log;
        size_t ptxSize;
        char *ptx;
        hipDevice_t cuDevice;
        hipCtx_t context;
        hiprtcLinkState linkState;
        hipModule_t module;
        hipFunction_t kernel;
        size_t cubinSize;
        void *cubin;
        float GPUtime = 0;
    public:
        string_code hashit(std::string const& inString);
        void parseDataStructure(std::string input);
        void createProg();
        void getVarsAndKernels();
        void compileProg();
        void threeinone();
        void getLogsAndPTX();
        void initializeVars();
        void destoryProg();
        float initAndLaunch(std::vector<void*>& args);
        void execute(std::string input);
        void execute(char *file_name, std::vector<void*>& args);
        float getKernelTime();
        void returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1);
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
                kernel_names.push_back("&"+words.at(1));
                kernel_params.push_back(atoi(words.at(2).c_str()));
                kernel_params.push_back(atoi(words.at(3).c_str()));
                kernel_params.push_back(atoi(words.at(4).c_str()));
                kernel_params.push_back(atoi(words.at(5).c_str()));
                kernel_params.push_back(atoi(words.at(6).c_str()));
                kernel_params.push_back(atoi(words.at(7).c_str()));
                break;
            case 3:
                int loc = atoi(words.at(1).c_str());
                int size = atoi(words.at(2).c_str());
                int dt = atoi(words.at(3).c_str());
                //convert this to a string because spiral prints string type
                switch(dt) {
                    case 0: //int
                    {
                        if(words.size() < 5) {
                            int * data1 = new int[size];
                            memset(data1, 0, size * sizeof(int));
                            data.push_back(data1);
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
                            float * data1 = new float[size];
                            memset(data1, 0, size * sizeof(float));
                            data.push_back(data1);
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
                            double * data1 = new double[size];
                            memset(data1, 0, size * sizeof(double));
                            data.push_back(data1);
                        }
                        else {
                            double * data1 = new double[size];
                            for(int i = 4; i < words.size(); i++) {
                                data1[i-4] = std::stod(words.at(i));
                            }
                            data.push_back(data1);
                            break;    
                        }
                    }
                    case 3: //constant
                    {
                        if(words.size() < 5) {
                            double * data1 = new double[size];
                            memset(data1, 0, size * sizeof(double));
                            data.push_back(data1);
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
    if ( DEBUGOUT ) std::cout << "parsed input\n";

}

inline void Executor::createProg() {
    DEVICE_RTC_SAFE_CALL(hiprtcCreateProgram(&prog, // prog
    kernels.c_str(), // buffer
    "test.cu", // name
    0, // numHeaders
    nullptr, // headers
    nullptr)); 
    if ( DEBUGOUT ) std::cout << "created program\n";
}

inline void Executor::getVarsAndKernels() {
    std::vector<std::string> new_names;
    for(int i = 0; i < device_names.size(); i++) {
        new_names.push_back(std::get<0>(device_names[i]));
    }
    if ( DEBUGOUT ) std::cout << "added new names\n";
    for (auto&& x : kernel_names) hiprtcAddNameExpression(prog, x.c_str());
    for(auto&& x: new_names) {hiprtcAddNameExpression(prog, x.c_str());}
    if ( DEBUGOUT ) std::cout << "added kernels and variables\n";
}

inline void Executor::compileProg() {
    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);
    std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
    // const char *opts[] = {"--device-debug", "--relocatable-device-code=true","--gpu-architecture=compute_70"};
    const char* opts[] = {
        sarg.c_str(), 
    };
    // const char* opts[] = {
    //     sarg.c_str(), "-fgpu-rdc"
    // };
    compileResult = hiprtcCompileProgram(prog, 
    1, 
    opts); 
    if ( DEBUGOUT ) std::cout << "compiled program\n";
}

inline void Executor::getLogsAndPTX() {
    DEVICE_RTC_SAFE_CALL(hiprtcGetProgramLogSize(prog, &logSize));
    //std::cout << "this is the log size" << logSize << "\n";
    log = new char[logSize];
    DEVICE_RTC_SAFE_CALL(hiprtcGetProgramLog(prog, log));
    if (compileResult != HIPRTC_SUCCESS) {
        std::cout << "compile failure with code "<< hiprtcGetErrorString (compileResult) << std::endl;
        for(int i = 0; i < logSize; i++) {
            std::cout << log[i];
        }
        std::cout << std::endl;
        exit(1);
    }
    delete[] log;
    DEVICE_RTC_SAFE_CALL(hiprtcGetCodeSize(prog, &ptxSize));
    //std::cout << "this is the program size" << ptxSize << "\n";
    ptx = new char[ptxSize];
    DEVICE_RTC_SAFE_CALL(hiprtcGetCode(prog, ptx));
    DEVICE_SAFE_CALL(hipModuleLoadData(&module, ptx));
    if ( DEBUGOUT ) std::cout << "created module\n";
}

inline void Executor::initializeVars() {
    for(decltype(device_names.size()) i = 0; i < device_names.size(); i++) {
        if ( DEBUGOUT ) std::cout << "this is i " << i << " this is the name " << std::get<0>(device_names[i]) << std::endl;
        const char * name;
        DEVICE_RTC_SAFE_CALL(hiprtcGetLoweredName(
        prog, 
        std::get<0>(device_names[i]).c_str(), // name expression
        &name                         // lowered name
        ));
        if ( DEBUGOUT ) std::cout << "it got past lower name\n";
        hipDeviceptr_t variable_addr;
        size_t bytes{};
        DEVICE_SAFE_CALL(hipModuleGetGlobal(&variable_addr, &bytes, module, name));
        if ( DEBUGOUT ) std::cout << "it got past get global\n";
        std::string test = std::get<2>(device_names[i]);
        switch(hashit(test)) {
            case zero:
            {
                int * value = (int*)(data.at(i));
                DEVICE_SAFE_CALL(hipMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(int)));
                break;
            }
            case one:
            {
                float * value = (float*)(data.at(i));
                DEVICE_SAFE_CALL(hipMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(float)));
                break;
            }
            case two:
            {   
                double * value = (double*)(data.at(i));
                DEVICE_SAFE_CALL(hipMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(double)));
                break;
            }
            case constant:
            {
                break;
            }
            case pointer_int:
            {
                int * h1;
                if ( DEBUGOUT ) std::cout << "got a int pointer " << std::get<0>(device_names.at(i)).substr(1) << " with size " << std::get<1>(device_names.at(i)) << "\n";
                DEVICE_SAFE_CALL(hipMalloc(&h1, std::get<1>(device_names.at(i)) * sizeof(int)));
                DEVICE_SAFE_CALL(hipMemcpy(variable_addr, &h1,  sizeof(int*), hipMemcpyHostToDevice));
                // hipFree(h1);
                break;
            }
            case pointer_float:
            {
                float * h1;
                if ( DEBUGOUT ) std::cout << "got a float pointer " << std::get<0>(device_names.at(i)).substr(1) << " with size " << std::get<1>(device_names.at(i)) << "\n";
                DEVICE_SAFE_CALL(hipMalloc(&h1, std::get<1>(device_names.at(i)) * sizeof(float)));
                DEVICE_SAFE_CALL(hipMemcpy(variable_addr, &h1,  sizeof(float*), hipMemcpyHostToDevice));
                // hipFree(h1);
                break;
            }
            case pointer_double:
            {
                double * h1;
                if ( DEBUGOUT ) std::cout << "got a double pointer " << std::get<0>(device_names.at(i)).substr(1) << " with size " << std::get<1>(device_names.at(i)) << "\n";
                DEVICE_SAFE_CALL(hipMalloc(&h1, std::get<1>(device_names.at(i)) * sizeof(double)));
                DEVICE_SAFE_CALL(hipMemcpy(variable_addr, &h1,  sizeof(double*), hipMemcpyHostToDevice));
                // hipFree(h1);
                break;
            }
            default:
                break;
        }
    }
}

inline void Executor::destoryProg() {
    //DEVICE_RTC_SAFE_CALL(hiprtcLinkDestroy(linkState));
    DEVICE_RTC_SAFE_CALL(hiprtcDestroyProgram(&prog));
}


inline float Executor::initAndLaunch(std::vector<void*>& args) {
    hipEvent_t start, stop;
    auto size = args.size() * sizeof(hipDeviceptr_t);
    void * config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(),
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                          HIP_LAUNCH_PARAM_END};
    
    for(int i = 0; i < kernel_names.size(); i++) {
    const char* name;
    DEVICE_RTC_SAFE_CALL(hiprtcGetLoweredName(prog, kernel_names[i].c_str(), &name));    
    DEVICE_SAFE_CALL(hipModuleGetFunction(&kernel, module, name));

    // // // Execute parent kernel.
    if ( DEBUGOUT ) std::cout << "launched kernel\n";
    if ( DEBUGOUT )
        std::cout << kernel_params[i*6] << "\t" << kernel_params[i*6+1] <<
            "\t" << kernel_params[i*6+2] << "\t" << kernel_params[i*6+3] << 
            "\t" << kernel_params[i*6+4] << "\t" << kernel_params[i*6+5] << "\n";
    DEVICE_SAFE_CALL(hipEventCreateWithFlags(&start,  hipEventDefault));
    DEVICE_SAFE_CALL(hipEventCreateWithFlags(&stop,  hipEventDefault));
    DEVICE_SAFE_CALL(hipEventRecord(start,0));
    DEVICE_SAFE_CALL(
    hipModuleLaunchKernel(kernel,
                          kernel_params[i*6], kernel_params[i*6+1], kernel_params[i*6+2], // grid dim
                          kernel_params[i*6+3], kernel_params[i*6+4], kernel_params[i*6+5], // block dim
                          0, nullptr, nullptr, // shared mem and stream
                          (void**)&config));
    DEVICE_SAFE_CALL(hipEventRecord(stop,0));
    DEVICE_SAFE_CALL(hipEventSynchronize(stop));
    float localtime;
    DEVICE_SAFE_CALL(hipEventElapsedTime(&localtime, start, stop)); 
    GPUtime += localtime;
    // DEVICE_SAFE_CALL(hipEventElapsedTime(&GPUtime, start, stop)); 
    }
    return getKernelTime();
}

inline void Executor::execute(std::string input) {
    if ( DEBUGOUT ) std::cout << "begin parsing\n";
    
    parseDataStructure(input);
    
    if ( DEBUGOUT ) {
        std::cout << "finished parsing\n";
        for(int i = 0; i < device_names.size(); i++) {
            std::cout << std::get<0>(device_names[i]) << std::endl;
        }
        for(int i = 0; i < kernel_names.size(); i++) {
            std::cout << kernel_names[i] << std::endl;
        }
    }
    createProg();
    getVarsAndKernels();
    compileProg();
    getLogsAndPTX();
    initializeVars();
    // destoryProg(); //cant call it early like in cuda
}

inline void Executor::execute(char *file_name, std::vector<void*>& args)
{
    if ( DEBUGOUT) std::cout << "begin executing code\n";

    std::ifstream ifs ( file_name );
    std::string   fcontent ( ( std::istreambuf_iterator<char>(ifs) ),
                             ( std::istreambuf_iterator<char>()    ) );

    parseDataStructure ( fcontent );
    if ( DEBUGOUT) {
        std::cout << "finsihed parsing\n";
        for(int i = 0; i < device_names.size(); i++) {
            std::cout << std::get<0>(device_names[i]) << std::endl;
        }
        for(int i = 0; i < kernel_names.size(); i++) {
            std::cout << kernel_names[i] << std::endl;
        }
    }

    createProg();
    getVarsAndKernels();
    compileProg();
    getLogsAndPTX();
    initializeVars();
    initAndLaunch(args);
}

inline float Executor::getKernelTime() {
    return GPUtime;
}

#endif            //  FFTX_MDDFT_HIPBACKEND_HEADER
