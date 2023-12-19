#ifndef FFTX_MDDFT_SYCLBACKEND_HEADER
#define FFTX_MDDFT_SYCLBACKEND_HEADER

//  Copyright (c) 2018-2022, Carnegie Mellon University
//  See LICENSE for details

#include<CL/sycl.hpp>
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

        std::vector<void*> kernelargs;
        std::vector<std::tuple<std::string, int, std::string, std::string>> device_names;
        std::unordered_map<std::string, std::string> device_names_map;
        std::vector<std::string> kernel_names;
        std::vector<int> kernel_params;
        std::string kernel_preamble;
        std::string kernels;

        std::vector<std::vector<std::string>> kernel_args;
        std::unordered_map<std::string, sycl::buffer<double>> global2buffer_double;
        std::unordered_map<std::string, sycl::buffer<int>> global2buffer_int;
        std::unordered_map<std::string, sycl::buffer<float>> global2buffer_float;
    
        std::unordered_map<std::string, std::tuple<std::string, int>> local2type; 
        std::unordered_map<std::string, std::tuple<std::string, int>> sig_types;
    
        std::vector<std::tuple<std::string, int, std::string>> in_params;
        std::vector<void*> params; 
        std::vector<void *> data;
        
        sycl::device dev;
        sycl::context ctx;
        cl_device_id ocl_dev;
        cl_context ocl_ctx;
        cl_int err = CL_SUCCESS;
        
        cl_command_queue ocl_queue;
        sycl::queue q; 
        cl_program ocl_program;
        
    
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
        std::vector<std::string> words;
        while(std::getline(ss,s,delim)) {
            words.push_back(s);
        }
        int test = atoi(words.at(0).c_str());
        switch(test) {
            case 0:
                device_names.push_back(std::make_tuple(words.at(1), atoi(words.at(2).c_str()), words.at(3), words.at(4)));
                device_names_map.insert(std::make_pair(words.at(1), words.at(4)));
                break;
            case 1:
                in_params.push_back(std::make_tuple(words.at(1), atoi(words.at(2).c_str()), words.at(3)));
                break;
            case 2:
            {
                kernel_names.push_back(words.at(1));
                kernel_params.push_back(atoi(words.at(2).c_str()));
                kernel_params.push_back(atoi(words.at(3).c_str()));
                kernel_params.push_back(atoi(words.at(4).c_str()));
                kernel_params.push_back(atoi(words.at(5).c_str()));
                kernel_params.push_back(atoi(words.at(6).c_str()));
                kernel_params.push_back(atoi(words.at(7).c_str()));
                std::vector<std::string> localv;
                for(int i = 8; i < words.size(); i++) {
                    localv.push_back(words.at(i));
                }
                kernel_args.push_back(localv);
                
                break;
            }
            case 3:
            {
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
            case 4:
            {
                sig_types.insert(std::make_pair(words.at(1), std::make_tuple(words.at(2), atoi(words.at(3).c_str()))));
                break;
            }
                
        }
    }

    while(std::getline(stream, line)) {
        kernels += line;
        kernels += "\n";
    }
    if ( DEBUGOUT ) std::cout << "parsed input\n";

}

inline void Executor::createProg() {
    try {
        dev = sycl::device(sycl::gpu_selector_v);
    }
    catch (sycl::exception const &e) {
        std::cout << "you are running on a system without a gpu, for best results please use a gpu, program terminating" << std::endl;
        exit(-1);
//         dev = sycl::device(sycl::cpu_selector_v);
    }

    ctx= sycl::context(dev);
    ocl_dev=sycl::get_native<cl::sycl::backend::opencl,sycl::device>(dev);
    ocl_ctx=sycl::get_native<cl::sycl::backend::opencl,sycl::context>(ctx);

    ocl_queue = clCreateCommandQueueWithProperties(ocl_ctx, ocl_dev,0,&err);
    q = sycl::make_queue<sycl::backend::opencl>(ocl_queue,ctx); 
    if ( DEBUGOUT ) std::cout << "created program\n";
}

inline void Executor::compileProg() {
    const char * kernelSource = kernels.c_str();
    ocl_program = clCreateProgramWithSource(ocl_ctx,1,&(kernelSource), nullptr, &err);
    clBuildProgram(ocl_program, 1, &ocl_dev, "-cl-std=CL3.0", nullptr, nullptr);
    if ( DEBUGOUT ) std::cout << "compiled program\n";
}

inline void Executor::initializeVars() {
    for(decltype(device_names.size()) i = 0; i < device_names.size(); i++) {
        if ( DEBUGOUT ) std::cout << "this is i " << i << " this is the name " << std::get<0>(device_names[i]) <<
            " this is the size "<< std::get<1>(device_names.at(i)) << " this is the type " << std::get<2>(device_names[i]) <<
             " this is the region of memory " << std::get<3>(device_names[i]) << std::endl;
        int size = std::get<1>(device_names.at(i));
        std::string test = std::get<2>(device_names[i]);
        switch(hashit(test)) {
            case zero:
            {
                if(std::get<3>(device_names[i]) == "global") {
                    cl_mem ocl_buf = clCreateBuffer(ocl_ctx,CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size * sizeof(int), nullptr,&err);
                    sycl::buffer<int, 1> buffer =sycl::make_buffer<sycl::backend::opencl, int>(ocl_buf, ctx);
                    global2buffer_int.insert(std::make_pair(std::get<0>(device_names[i]), buffer));
                } else {
                    local2type.insert(std::make_pair(std::get<0>(device_names[i]), std::make_pair("int",std::get<1>(device_names.at(i)))));
                }
                break;
            }
            case one:
            {
                if(std::get<3>(device_names[i]) == "global") {
                    cl_mem ocl_buf = clCreateBuffer(ocl_ctx,CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size * sizeof(float), nullptr,&err);
                    sycl::buffer<float, 1> buffer =sycl::make_buffer<sycl::backend::opencl, float>(ocl_buf, ctx);
                    global2buffer_float.insert(std::make_pair(std::get<0>(device_names[i]), buffer));
                } else {
                    local2type.insert(std::make_pair(std::get<0>(device_names[i]), std::make_pair("float",std::get<1>(device_names.at(i)))));
                }
                break;
            }
            case two:
            {   
                if(std::get<3>(device_names[i]) == "global") {
                    cl_mem ocl_buf = clCreateBuffer(ocl_ctx,CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size * sizeof(double), nullptr,&err);
                    sycl::buffer<double, 1> buffer =sycl::make_buffer<sycl::backend::opencl, double>(ocl_buf, ctx);
                    global2buffer_double.insert(std::make_pair(std::get<0>(device_names[i]), buffer));
                } else {
                    local2type.insert(std::make_pair(std::get<0>(device_names[i]), std::make_pair("double",std::get<1>(device_names.at(i)))));
                }
                break;
            }
            case constant:
            {
                break;
            }
            case pointer_int:
            {
                if(std::get<3>(device_names[i]) == "global") { 
                    cl_mem ocl_buf = clCreateBuffer(ocl_ctx,CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size * sizeof(int), nullptr,&err);
                    sycl::buffer<int, 1> buffer =sycl::make_buffer<sycl::backend::opencl, int>(ocl_buf, ctx);
                    global2buffer_int.insert(std::make_pair(std::get<0>(device_names[i]), buffer));
                }else {
                    local2type.insert(std::make_pair(std::get<0>(device_names[i]), std::make_pair("int",std::get<1>(device_names.at(i)))));
                }
                break;
            }
            case pointer_float:
            {
                if(std::get<3>(device_names[i]) == "global") {
                    cl_mem ocl_buf = clCreateBuffer(ocl_ctx,CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size * sizeof(float), nullptr,&err);
                    sycl::buffer<float, 1> buffer =sycl::make_buffer<sycl::backend::opencl, float>(ocl_buf, ctx);
                    global2buffer_float.insert(std::make_pair(std::get<0>(device_names[i]), buffer));
                }else {
                    local2type.insert(std::make_pair(std::get<0>(device_names[i]), std::make_pair("float",std::get<1>(device_names.at(i)))));
                }  
                break;
            }
            case pointer_double:
            {
                if(std::get<3>(device_names[i]) == "global") {
                    cl_mem ocl_buf = clCreateBuffer(ocl_ctx,CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size * sizeof(double), nullptr,&err);
                    sycl::buffer<double, 1> buffer =sycl::make_buffer<sycl::backend::opencl, double>(ocl_buf, ctx);
                    global2buffer_double.insert(std::make_pair(std::get<0>(device_names[i]), buffer));
                }else {
                    local2type.insert(std::make_pair(std::get<0>(device_names[i]), std::make_pair("double",std::get<1>(device_names.at(i)))));
                }
                break;
            }
            default:
                break;
        }
    }
}

inline float Executor::initAndLaunch(std::vector<void*>& args) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < kernel_names.size(); i++) {
        if ( DEBUGOUT ) {
            std::cout << kernel_names.at(i) << std::endl;
            std::cout << kernel_params[i*6] << " " << kernel_params[i*6+1] << " " << kernel_params[i*6+2] << std::endl;
            std::cout << kernel_params[i*6+3] << " " <<  kernel_params[i*6+4] << " " << kernel_params[i*6+5] << std::endl;
        }
        sycl::range<3> grid (kernel_params[i*6],kernel_params[i*6+1],kernel_params[i*6+2]);
        sycl::range<3> block (kernel_params[i*6+3],kernel_params[i*6+4],kernel_params[i*6+5]);
        cl_kernel ocl_kernel = clCreateKernel(ocl_program, kernel_names.at(i).c_str(), &err);
        sycl::kernel sycl_kernel = sycl::make_kernel<sycl::backend::opencl>(ocl_kernel, ctx);
        
        q.submit([&](sycl::handler& h){
                  for(int j = 0; j < kernel_args[i].size(); j++) {
                         if ( DEBUGOUT ) {
                            std::cout << "This is the kernel arg name " << kernel_args.at(i).at(j) << std::endl;
                         }
                         if(sig_types.find(kernel_args.at(i).at(j)) != sig_types.end()) {
                             if(std::get<1>(sig_types.at(kernel_args.at(i).at(j))) == 1) {
                                 h.set_arg(j, (*(sycl::buffer<double>*)args.at(std::get<1>(sig_types.at(kernel_args.at(i).at(j)))-1)).get_access
                                             <sycl::access_mode::write, sycl::target::device>(h));
                             } else if(std::get<1>(sig_types.at(kernel_args.at(i).at(j))) == 2) {
                                 h.set_arg(j, (*(sycl::buffer<double,1>*)args.at(std::get<1>(sig_types.at(kernel_args.at(i).at(j)))-1)).get_access
                                             <sycl::access_mode::read, sycl::target::device>(h));
                             }else {
                                 h.set_arg(j, (*(sycl::buffer<double>*)args.at(std::get<1>(sig_types.at(kernel_args.at(i).at(j)))-1)).get_access
                                             <sycl::access_mode::read_write, sycl::target::device>(h));
                             }
                         } else if(device_names_map.find(kernel_args.at(i).at(j)) != device_names_map.end()) {
                             std::string type = device_names_map.at(kernel_args.at(i).at(j));
                             if(type == "global") {
                                 if(global2buffer_int.find(kernel_args.at(i).at(j)) != global2buffer_int.end()) {  
                                     auto data_acc = global2buffer_int.at(kernel_args.at(i).at(j)).get_access<sycl::access_mode::read_write, sycl::target::device>(h);
                                      h.set_arg(j, data_acc);
                                 } else if(global2buffer_float.find(kernel_args.at(i).at(j)) != global2buffer_float.end()) {
                                      auto data_acc = global2buffer_float.at(kernel_args.at(i).at(j)).get_access<sycl::access_mode::read_write, sycl::target::device>(h);
                                      h.set_arg(j,data_acc);
                                 } else if(global2buffer_double.find(kernel_args.at(i).at(j)) != global2buffer_double.end()) {
                                      auto data_acc = global2buffer_double.at(kernel_args.at(i).at(j)).get_access<sycl::access_mode::read_write, sycl::target::device>(h);
                                      h.set_arg(j,data_acc);
                                 } else {
                                     std::cout << "device variable needed but sycl buffer never created" << std::endl;
                                     exit(-1);
                                 }
                                  
                             } else {
                                 if(std::get<0>(local2type.at(kernel_args.at(i).at(j))) == "int") {
                                     sycl::local_accessor<int> shm_acc(sycl::range<1>(std::get<1>(local2type.at(kernel_args.at(i).at(j)))), h);
                                     h.set_arg(j, shm_acc);
                                 } else if(std::get<0>(local2type.at(kernel_args.at(i).at(j))) == "float") {
                                     sycl::local_accessor<float> shm_acc(sycl::range<1>(std::get<1>(local2type.at(kernel_args.at(i).at(j)))), h);
                                     h.set_arg(j, shm_acc);
                                 } else if(std::get<0>(local2type.at(kernel_args.at(i).at(j))) == "double") {
                                     sycl::local_accessor<double> shm_acc(sycl::range<1>(std::get<1>(local2type.at(kernel_args.at(i).at(j)))), h);
                                     h.set_arg(j, shm_acc);
                                 } else {
                                     std::cout << "shared memory is using an unsupported type " << std::get<1>(local2type.at(kernel_args.at(i).at(j))) << std::endl;
                                     exit(-1);
                                 }
                             }
                         } 
                          else{
                             std::cout << "kernel execution failed dramatically" << std::endl;
                             exit(-1);
                         }
                    }
                h.parallel_for(
                sycl::nd_range<3>(grid*block, block),
                sycl_kernel);
                }).wait();
	}
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	GPUtime = duration.count();
    return getKernelTime();
}

inline void Executor::execute(std::string input) {
    if ( DEBUGOUT ) std::cout << "begin parsing\n";
    
    parseDataStructure(input);
    
    if ( DEBUGOUT ) {
        std::cout << "finished parsing\n";
        for(auto it = device_names_map.cbegin(); it != device_names_map.cend(); it++) {
            std::cout << (*it).first << std::endl;
        }
        for(int i = 0; i < kernel_names.size(); i++) {
            std::cout << kernel_names[i] << std::endl;
        }
        std::cout << kernels << std::endl;
    }
    createProg();
    compileProg();
    initializeVars();
}

inline float Executor::getKernelTime() {
    return GPUtime;
}

#endif            //  FFTX_MDDFT_HIPBACKEND_HEADER
