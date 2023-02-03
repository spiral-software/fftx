#ifndef FFTX_MDDFT_CPUBACKEND_HEADER
#define FFTX_MDDFT_CPUBACKEND_HEADER

//  Copyright (c) 2018-2022, Carnegie Mellon University
//  See LICENSE for details

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <cstdio>
#include <string>
#include <cstdio>      // perror
#include <unistd.h>    // dup2
#include <sys/types.h> // rest for open/close
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <array>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <cstring>
#include <chrono>
#pragma once

#define LOCALDEBUG 0

int redirect_input(int);
void restore_input(int);

class Executor {
    private:
        void * shared_lib;
        float CPUTime;
    public:
        float initAndLaunch(std::vector<void*>& args);
        void execute(std::string file_name);
        float getKernelTime();
        //void returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1);
};

float Executor::initAndLaunch(std::vector<void*>& args, std::string name) {
    if(LOCALDEBUG)
        std::cout << "Loading shared library\n";
    shared_lib = dlopen("./libtmp.so", RTLD_LAZY);
    if(!shared_lib) {
        std::cout << "Cannot open library: " << dlerror() << '\n';
        exit(0);
    }
    std::string init = "init_" << name << "_spiral";
    std::string transform = name << "_spiral";
    std::string destory = "destory_" << name << "_spiral";
    else if(shared_lib){
        void (*fn1) ()= (void (*)())dlsym(shared_lib, init.c_str());
        void (*fn2) (double *, double *, double *) = (void (*)(double *, double *, double *))dlsym(shared_lib, transform.c_str());
        void (*fn3) ()= (void (*)())dlsym(shared_lib, destory.c_str());
        auto start = std::chrono::high_resolution_clock::now();
        if(fn1) {
            fn1();
        }
        if(fn2) {
            fn2((double*)args.at(0),(double*)args.at(1), (double*)args.at(2));
        }
        if(fn3){
            fn3();
        }
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = stop - start;
        CPUTime = duration.count();
        dlclose(shared_lib);
    }
    return getKernelTime();
}


void Executor::execute(std::string result) {
    if(LOCALDEBUG)
        std::cout << "entered CPU backend execute\n";
    std::string compile;
    // if(LOCALDEBUG)
    //     compile = "/usr/bin/gcc -I $SPIRAL_HOME/namespaces -Wall -Wextra -x c - -o libtmp.so -O2 -shared -fPIC";
    // else 
    //     compile = "/usr/bin/gcc -I $SPIRAL_HOME/namespaces -x c - -o libtmp.so -O2 -shared -fPIC";
    if(LOCALDEBUG)
        compile = "/usr/bin/gcc -I $SPIRAL_HOME/namespaces -Wall -Wextra spiral_generated.c -o libtmp.so -O2 -shared -fPIC";
    else 
        compile = "/usr/bin/gcc -I $SPIRAL_HOME/namespaces spiral_generated.c -o libtmp.so -O2 -shared -fPIC";
    
    if(LOCALDEBUG) {
        std::cout << "created compile\n";
    }
    // std::cout << result << std::endl;
    // exit(0);
    // result.erase(result.size()-8);
    std::string result2 = result.substr(result.find("*/")+3, result.length());
    std::cout << result2.size() << std::endl;
    std::ofstream out("spiral_generated.c");
    out << result2;
    out.close();
    // int p2[2];
    // if(pipe(p2) < 0)
    //     std::cout << "pipe failed\n";
    // if(result2.size() > 50000) {
    //     if(LOCALDEBUG)
    //         std::cout << "multiwrite\n";
    //     std::string s1 = result2.substr(0, result2.length()/4);
    //     std::cout << s1.size() << std::endl;
    //     std::string s2 = result2.substr(result2.length()/4, result.length()/2);
    //     std::cout << s2.size() << std::endl;
    //     std::string s3 = result2.substr(result2.length()/2, result.length()/2 + result.length()/4);
    //     std::cout << s3.size() << std::endl;
    //     std::string s4 = result2.substr(result.length()/2 + result.length()/4, result.length());
    //     std::cout << s4.size() << std::endl;
    //     if(write(p2[1], s1.c_str(), s1.size()) == -1) {
    //         perror("Error writing to the pipe");
    //     }
    //     if(write(p2[1], s2.c_str(), s2.size()) == -1) {
    //         perror("Error writing to the pipe");
    //     }
    //     if(write(p2[1], s3.c_str(), s3.size()) == -1) {
    //         perror("Error writing to the pipe");
    //     }
    //     if(write(p2[1], s4.c_str(), s4.size()) == -1) {
    //         perror("Error writing to the pipe");
    //     }
    // }
    // else{
    //     if(LOCALDEBUG)
    //         std::cout << "single write\n";
    //     if(write(p2[1], result2.c_str(), result2.size()) == -1) {
    //         perror("Error writing to the pipe");
    //     }
    // }
    // if(LOCALDEBUG)
    //     std::cout << "wrote to pipe\n";
    // close(p2[1]);
    // int save_stdin = redirect_input(p2[0]);
     if(LOCALDEBUG)
        std::cout << "compiling\n";
    system(compile.c_str());
     if(LOCALDEBUG)
        std::cout << "finished compiling\n";
    // restore_input(save_stdin);
    // close(p2[0]);
    //destoryProg(); cant call it early like in cuda
}

float Executor::getKernelTime() {
    return CPUTime;
}

// void Executor::returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1, std::vector<void*>& args) {
//     //gatherOutput(out1.at(0), kernelargs);
//     hipMemcpy(out.m_data.local(), &((hipDeviceptr_t)args.at(0)),  out.m_domain.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost);
// }

#endif			//  FFTX_MDDFT_HIPBACKEND_HEADER