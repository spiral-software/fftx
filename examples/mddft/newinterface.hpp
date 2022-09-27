#include <nvrtc.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
//#include <charconv>
//#include "interface.hpp"
#include <tuple>
#include <iomanip>
#include <cstdio>      // perror
#include <unistd.h>    // dup2
#include <sys/types.h> // rest for open/close
#include <sys/stat.h>
#include <fcntl.h>
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


class Executor {
    private:
        int x;
        enum string_code {
            zero,
            one,
            two,
            constant,
            mone
        };
        nvrtcProgram prog;
        nvrtcResult compileResult;
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
    public:
        string_code hashit(std::string const& inString);
        void parseDataStructure(char * input);
        void createProg();
        void getVars();
        void compileProg();
        void threeinone();
        void getLogsAndPTX(char * args);
        void initializeVars();
        void destoryProg();
        void initAndLaunch(std::vector<fftx::array_t<3,std::complex<double>>> in,
        std::vector<fftx::array_t<3,std::complex<double>>> out);
        void execute(std::vector<fftx::array_t<3,std::complex<double>>> in,
        std::vector<fftx::array_t<3,std::complex<double>>> out,
        int counts,
        std::vector<char**> inputargs);
};

Executor::string_code Executor::hashit(std::string const& inString) {
    if(inString == "int") return zero;
    if(inString == "float") return one;
    if(inString == "double") return two;
    if(inString == "constant") return constant;
    return mone;
}

void Executor::parseDataStructure(char * input) {
    std::cout << input << std::endl;
    std::ifstream t(input);
    std::stringstream ds;
    ds << t.rdbuf();
    std::istringstream stream(ds.str());
    char delim = ' ';
    std::string line;
    std::string b = "------------------";
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
                device_names.push_back(std::make_tuple(words.at(1), atoi(words.at(2).c_str()), words.at(3)));
                break;
            case 1:
                in_params.push_back(std::make_tuple(words.at(1), atoi(words.at(2).c_str()), words.at(3)));
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
}

void Executor::createProg() {
    //std::cout << kernels.c_str() << std::endl;
    //const char * kernels2 = kernels.c_str();
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, // prog
    kernels.c_str(), // buffer
    NULL, // name
    0, // numHeaders
    NULL, // headers
    NULL)); 
    //std::cout << "compiled code "<< nvrtcGetErrorString(compileResult) << std::endl;
}

void Executor::getVars() {
    for(int i = 0; i < device_names.size(); i++) {
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, std::get<0>(device_names[i]).c_str()));
       
    }
}

void Executor::compileProg() {
    const char *opts[] = {"--device-debug", "--relocatable-device-code=true","--gpu-architecture=compute_70"};
    compileResult = nvrtcCompileProgram(prog, 
    3, 
    opts); 
}

// void Executor::threeinone() {
//     NVRTC_SAFE_CALL(
//     nvrtcCreateProgram(&prog, // prog
//     kernels.c_str(), // buffer
//     "interface_jit.cpp", // name
//     0, // numHeaders
//     NULL, // headers
//     NULL)); 
//     for(int i = 0; i < device_names.size(); i++) {
//         std::cout << std::get<0>(device_names[i]) << std::endl;
//         NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, std::get<0>(device_names[i]).c_str()));
//     }
//     const char *opts[] = {"--relocatable-device-code=true", "--device-debug", "--generate-line-info","--gpu-architecture=compute_70"};
//     NVRTC_SAFE_CALL(nvrtcCompileProgram(prog, 
//     4, 
//     opts)); 
// }

void Executor::getLogsAndPTX(char * arg) {
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
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuLinkCreate(0, 0, 0, &linkState));
    CUDA_SAFE_CALL(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, arg, 
    0, 0, 0));
    CUDA_SAFE_CALL(cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
    (void *)ptx, ptxSize, "dft_jit.ptx",
    0, 0, 0));
    CUDA_SAFE_CALL(cuLinkComplete(linkState, &cubin, &cubinSize));
    CUDA_SAFE_CALL(cuModuleLoadData(&module, cubin));
}

void Executor::initializeVars() {
    for(int i = 0; i < device_names.size(); i++) {
        // if (i == 0) {
        //     continue;
        // }
        const char * name;
        NVRTC_SAFE_CALL(nvrtcGetLoweredName(
        prog, 
        std::get<0>(device_names[i]).c_str(), // name expression
        &name                         // lowered name
        ));
        CUdeviceptr variable_addr;
        CUDA_SAFE_CALL(cuModuleGetGlobal(&variable_addr, NULL, module, name));
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
        }
    }
}

void Executor::destoryProg() {
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
}

void Executor::initAndLaunch(std::vector<fftx::array_t<3,std::complex<double>>> in,
        std::vector<fftx::array_t<3,std::complex<double>>> out) {
    std::cout << "the kernel name is " << kernel_name << std::endl;
    std::cout << in.at(0).m_domain.size() << std::endl;
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));
    // double *X = new double[64];
    // double *Y = new double[64];
    // // for(int i = 0; i < params.size(); i++){
    // //     CUdeviceptr dX;
    // //     CUDA_SAFE_CALL(cuMemAlloc(&dX, 64* sizeof(double)));
    // //     CUDA_SAFE_CALL(cuMemcpyHtoD(dX, X, 64* sizeof(double)));
    // //     params.push_back(&dX);
    // // } 
    // for(int i = 0; i < 64; i++) {
    //     X[i] = 1;
    //     Y[i] = -1;
    // }
    // for (int i = 0; i < 64; i++) {
    //     std::cout << X[i] << "\n";
    // }
    // std::cout << "\n";
    // std::cout << "numBlocks " << numBlocks << std::endl;
    // std::cout << "X in host is " << X[10] << std::endl;
    CUdeviceptr dX, dY, dsym;
    //double  *hp1 = new double[512];
    std::cout << "allocating memory\n";
    CUDA_SAFE_CALL(cuMemAlloc(&dX, in.at(0).m_domain.size() * sizeof(std::complex<double>)));
    std::cout << "allocated X\n";
    CUDA_SAFE_CALL(cuMemcpyHtoD(dX, in.at(0).m_data.local(),  in.at(0).m_domain.size() * sizeof(std::complex<double>)));
    std::cout << "copied X\n";
    CUDA_SAFE_CALL(cuMemAlloc(&dY, out.at(0).m_domain.size() * sizeof(std::complex<double>)));
    std::cout << "allocated Y\n";
    // //CUDA_SAFE_CALL(cuMemcpyHtoD(dY, Y, 64* sizeof(double)));
    CUDA_SAFE_CALL(cuMemAlloc(&dsym, 64* sizeof(double)));
    //init_grid_dft2d_cont(kernel, module);
    // Execute parent kernel.
    void *args[] = { &dY, &dX, &dsym};
    //init_grid_dft2d_cont();
    //for(int i = 0; i < 1; i++) {
        std::cout << "launched kernel\n";
        CUDA_SAFE_CALL(
        cuLaunchKernel(kernel,
        1, 1, 1, // grid dim
        1, 1, 1, // block dim
        0, NULL, // shared mem and stream
        args, 0)); // arguments
        CUDA_SAFE_CALL(cuCtxSynchronize());
        //CUDA_SAFE_CALL(cuMemcpyDtoH(X, dX, 64*sizeof(double)));
        std::cout << "copying data back\n";
        CUDA_SAFE_CALL(cuMemcpyDtoH(out.at(0).m_data.local(), dY, out.at(0).m_domain.size()*sizeof(std::complex<double>)));
        std::cout << "\n\n\n\nOutput is" << std::endl;
        // for (int i = 0; i < out.at(0).m_domain.size(); i++) {
        //     std::cout << out.at(0).m_data.local()[i] << "\n";
        // }
    //}

}

void Executor::execute(std::vector<fftx::array_t<3,std::complex<double>>> in, 
                        std::vector<fftx::array_t<3,std::complex<double>>> out,
                        int counts,
                        std::vector<char**> inputargs) {
    //int count = *((int*)inputargs.at(0));
    // std::cout << "count is" << counts << std::endl;
    // std::cout << (char*)inputargs.at(inputargs.size()-1) << std::endl;
    // std::cout << (inputargs.at(inputargs.size()-2))[counts-1] << std::endl;
    std::cout << in.at(0).m_domain.size() << std::endl;
    std::cout << "begin executing code\n";
    parseDataStructure((char*)inputargs.at(inputargs.size()-1));
    std::cout << "finsihed parsing\n";
    createProg();
    getVars();
    compileProg();
    getLogsAndPTX((inputargs.at(inputargs.size()-2))[counts-1]);
    initializeVars();
    destoryProg();
    std::cout << "begin kernel launch\n";
    initAndLaunch(in, out);
}