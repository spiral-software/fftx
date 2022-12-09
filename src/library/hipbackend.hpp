#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
//#include <charconv>
//#include "interface.hpp"
#include <tuple>
#include <iomanip>
// #include <cstdio>      // perror
// #include <unistd.h>    // dup2
// #include <sys/types.h> // rest for open/close
//#include <sys/stat.h>
#include <fcntl.h>
#pragma once
#define LOCALDEBUG 0
#define HIPRTC_SAFE_CALL(x) \
 do { \
hiprtcResult result = x; \
 if (result != HIPRTC_SUCCESS) { \
 std::cerr << "\nrtc error: " #x " failed with error " \
 << hiprtcGetErrorString(result) << '\n'; \
 exit(1); \
 } \
 } while(0)
#define HIP_SAFE_CALL(x)                                         \
  do {                                                            \
    hipError_t result = x;                                          \
    if (result != hipSuccess ) {                                 \
      std::cerr << "\nmain error: " <<  hipGetErrorName(result) << " failed with error "   \
                << hipGetErrorString(result) << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)


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
        // std::vector<fftx::array_t<3,std::complex<double>>> in;
        // std::vector<fftx::array_t<3,std::complex<double>>> out;
        std::vector<void*> kernelargs;
        std::vector<std::tuple<std::string, int, std::string>> device_names;
        // std::string kernel_name;
        std::vector<std::string> kernel_names;
        std::vector<int> kernel_params;
        std::string kernel_preamble;
        std::string kernels;
        //std::vector<std::string> kernels;
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
        float GPUtime;
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
    std::istringstream stream(input);
    char delim = ' ';
    std::string line;
    std::string b = "------------------";
    while(std::getline(stream, line)){
        if(line == "spiral> JIT BEGIN")
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
    if(LOCALDEBUG == 1)
        std::cout << "parsed input\n";

}

void Executor::createProg() {
    HIPRTC_SAFE_CALL(hiprtcCreateProgram(&prog, // prog
    kernels.c_str(), // buffer
    "test.cu", // name
    0, // numHeaders
    nullptr, // headers
    nullptr)); 
    if(LOCALDEBUG == 1)
        std::cout << "created program\n";
}

void Executor::getVarsAndKernels() {
    // for(int i = 0; i < device_names.size(); i++) {
    //     HIPRTC_SAFE_CALL(hiprtcAddNameExpression(prog, std::get<0>(device_names[i]).c_str()));
    // }
    std::vector<std::string> new_names;
    for(int i = 0; i < device_names.size(); i++) {
        new_names.push_back(std::get<0>(device_names[i]));
    }
    if(LOCALDEBUG == 1)
    std::cout << "added new names\n";
    for (auto&& x : kernel_names) hiprtcAddNameExpression(prog, x.c_str());
    for(auto&& x: new_names) {hiprtcAddNameExpression(prog, x.c_str());}
    if(LOCALDEBUG == 1)
    std::cout << "added kernels and variables\n";
}

void Executor::compileProg() {
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
    if(LOCALDEBUG == 1)
    std::cout << "compiled program\n";
}

// void Executor::threeinone() {
//     HIPRTC_SAFE_CALL(
//     nvrtcCreateProgram(&prog, // prog
//     kernels.c_str(), // buffer
//     "interface_jit.cpp", // name
//     0, // numHeaders
//     NULL, // headers
//     NULL)); 
//     for(int i = 0; i < device_names.size(); i++) {
//         std::cout << std::get<0>(device_names[i]) << std::endl;
//         HIPRTC_SAFE_CALL(HIPRTCAddNameExpression(prog, std::get<0>(device_names[i]).c_str()));
//     }
//     const char *opts[] = {"--relocatable-device-code=true", "--device-debug", "--generate-line-info","--gpu-architecture=compute_70"};
//     HIPRTC_SAFE_CALL(nvrtcCompileProgram(prog, 
//     4, 
//     opts)); 
// }

void Executor::getLogsAndPTX() {
    HIPRTC_SAFE_CALL(hiprtcGetProgramLogSize(prog, &logSize));
    //std::cout << "this is the log size" << logSize << "\n";
    log = new char[logSize];
    HIPRTC_SAFE_CALL(hiprtcGetProgramLog(prog, log));
    if (compileResult != HIPRTC_SUCCESS) {
        std::cout << "compile failure with code "<< hiprtcGetErrorString (compileResult) << std::endl;
        for(int i = 0; i < logSize; i++) {
            std::cout << log[i];
        }
        std::cout << std::endl;
        exit(1);
    }
    delete[] log;
    HIPRTC_SAFE_CALL(hiprtcGetCodeSize(prog, &ptxSize));
    //std::cout << "this is the program size" << ptxSize << "\n";
    ptx = new char[ptxSize];
    HIPRTC_SAFE_CALL(hiprtcGetCode(prog, ptx));
    // HIP_SAFE_CALL(cuInit(0));
    // HIP_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    // HIP_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    // HIPRTC_SAFE_CALL(hiprtcLinkCreate(0, 0, 0, &linkState));
    // // HIP_SAFE_CALL(hiprtcLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, getHIPRuntime(), 
    // // 0, 0, 0));
    // HIPRTC_SAFE_CALL(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE,
    // (void *)ptx, ptxSize, "dft_jit.ptx",
    // 0, 0, 0));
    // HIPRTC_SAFE_CALL(hiprtcLinkComplete(linkState, &cubin, &cubinSize));
    //HIP_SAFE_CALL(hipModuleLoadData(&module, cubin));
    //std::cout << "before moodule\n";
    HIP_SAFE_CALL(hipModuleLoadData(&module, ptx));
    if(LOCALDEBUG == 1)
    std::cout << "created module\n";
}

void Executor::initializeVars() {
    for(decltype(device_names.size()) i = 0; i < device_names.size(); i++) {
        if(LOCALDEBUG == 1)
        std::cout << "this is i " << i << " this is the name " << std::get<0>(device_names[i]) << std::endl;
        const char * name;
        HIPRTC_SAFE_CALL(hiprtcGetLoweredName(
        prog, 
        std::get<0>(device_names[i]).c_str(), // name expression
        &name                         // lowered name
        ));
        if(LOCALDEBUG == 1)
        std::cout << "it got past lower name\n";
        hipDeviceptr_t variable_addr;
        size_t bytes{};
        HIP_SAFE_CALL(hipModuleGetGlobal(&variable_addr, &bytes, module, name));
        if(LOCALDEBUG == 1)
            std::cout << "it got past get global\n";

        std::string test = std::get<2>(device_names[i]);
        switch(hashit(test)) {
            case zero:
            {
                int * value = (int*)(data.at(i));
                HIP_SAFE_CALL(hipMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(int)));
                break;
            }
            case one:
            {
                float * value = (float*)(data.at(i));
                HIP_SAFE_CALL(hipMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(float)));
                break;
            }
            case two:
            {   
                double * value = (double*)(data.at(i));
                HIP_SAFE_CALL(hipMemcpyHtoD(variable_addr, value, std::get<1>(device_names.at(i))*sizeof(double)));
                break;
            }
            case constant:
            {
                break;
            }
            case pointer_int:
            {
                HIP_SAFE_CALL(hipMalloc((void **)&variable_addr, std::get<1>(device_names.at(i)) * sizeof(int)));
                break;
            }
            case pointer_float:
            {
                HIP_SAFE_CALL(hipMalloc((void **)&variable_addr, std::get<1>(device_names.at(i)) * sizeof(float)));
                break;
            }
            case pointer_double:
            {
                HIP_SAFE_CALL(hipMalloc((void **)&variable_addr, std::get<1>(device_names.at(i)) * sizeof(double)));
                break;
            }
            default:
                break;
        }
    }
}

void Executor::destoryProg() {
    //HIPRTC_SAFE_CALL(hiprtcLinkDestroy(linkState));
    HIPRTC_SAFE_CALL(hiprtcDestroyProgram(&prog));
}

float Executor::initAndLaunch(std::vector<void*>& args) {
    //kernelargs = initGPUData(in.at(0), out.at(0));
    //std::cout << "the kernel name is " << kernel_names[ki] << std::endl;
    for(int i = 0; i < kernel_names.size(); i++) {
    const char* name;
    HIPRTC_SAFE_CALL(hiprtcGetLoweredName(prog, kernel_names[i].c_str(), &name));    
    HIP_SAFE_CALL(hipModuleGetFunction(&kernel, module, name));
    // // // Execute parent kernel.
    float local_time; 
    auto size = args.size() * sizeof(hipDeviceptr_t);
    void * config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(),
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                          HIP_LAUNCH_PARAM_END};
    if(LOCALDEBUG == 1)
    std::cout << "launched kernel\n";
    if(LOCALDEBUG == 1)
    std::cout << kernel_params[i*kernel_names.size()] << "\t" << kernel_params[i*kernel_names.size()+1] <<
    "\t" << kernel_params[i*kernel_names.size()+2] << "\t" << kernel_params[i*kernel_names.size()+3] << 
    "\t" << kernel_params[i*kernel_names.size()+4] << "\t" << kernel_params[i*kernel_names.size()+5] << "\n";
    hipEvent_t start, stop;
    HIP_SAFE_CALL(hipEventCreateWithFlags(&start,  hipEventDefault));
    HIP_SAFE_CALL(hipEventCreateWithFlags(&stop,  hipEventDefault));
    HIP_SAFE_CALL(hipEventRecord(start,0));
    HIP_SAFE_CALL(
    hipModuleLaunchKernel(kernel,
    kernel_params[i*kernel_names.size()], kernel_params[i*kernel_names.size()+1], kernel_params[i*kernel_names.size()+2], // grid dim
    kernel_params[i*kernel_names.size()+3], kernel_params[i*kernel_names.size()+4], kernel_params[i*kernel_names.size()+5], // block dim
    0, nullptr, nullptr, // shared mem and stream
    config));
    HIP_SAFE_CALL(hipEventRecord(stop,0));
    HIP_SAFE_CALL(hipEventSynchronize(stop));
    HIP_SAFE_CALL(hipEventElapsedTime(&local_time, start, stop)); 
    GPUtime += local_time;
    }
    return getKernelTime();
}


void Executor::execute(std::string file_name) {
    if(LOCALDEBUG == 1)
        std::cout << "begin executing code\n";
    
    parseDataStructure(file_name);
    
    if(LOCALDEBUG == 1) {
        std::cout << "finished parsing\n";
        for(int i = 0; i < device_names.size(); i++) {
            std::cout << std::get<0>(device_names[i]) << std::endl;
        }
        for(int i = 0; i < kernel_names.size(); i++) {
            std::cout << kernel_names[i] << std::endl;
        }
        std::cout << kernels << std::endl;
    }
    createProg();
    getVarsAndKernels();
    compileProg();
    getLogsAndPTX();
    initializeVars();
    //destoryProg(); cant call it early like in cuda
}

float Executor::getKernelTime() {
    return GPUtime;
}

// void Executor::returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1, std::vector<void*>& args) {
//     //gatherOutput(out1.at(0), kernelargs);
//     hipMemcpy(out.m_data.local(), &((hipDeviceptr_t)args.at(0)),  out.m_domain.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost);
// }