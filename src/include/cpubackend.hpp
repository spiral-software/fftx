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

#if defined ( PRINTDEBUG )
#define DEBUGOUT 1
#else
#define DEBUGOUT 0
#endif

int redirect_input(int);
void restore_input(int);

class Executor {
    private:
        void * shared_lib;
        float CPUTime;
    public:
        float initAndLaunch(std::vector<void*>& args, std::string name);
        void execute(std::string file_name);
        float getKernelTime();
        //void returnData(std::vector<fftx::array_t<3,std::complex<double>>> &out1);
};

float Executor::initAndLaunch(std::vector<void*>& args, std::string name) {
    if ( DEBUGOUT) std::cout << "Loading shared library\n";
    shared_lib = dlopen("./libtmp.so", RTLD_LAZY);
    std::ostringstream oss;
    std::ostringstream oss1;
    std::ostringstream oss2;
    oss << "init_" << name << "_spiral";
    oss1 << name << "_spiral";
    oss2 << "destory_" << name << "_spiral";
    std::string init = oss.str();
    std::string transform = oss1.str();
    std::string destory = oss2.str();
    if(!shared_lib) {
        std::cout << "Cannot open library: " << dlerror() << '\n';
        exit(0);
    }
    else if(shared_lib){
        void (*fn1) ()= (void (*)())dlsym(shared_lib, init.c_str());
        void (*fn2) (double *, double *, double *) = (void (*)(double *, double *, double *))dlsym(shared_lib, transform.c_str());
        void (*fn3) ()= (void (*)())dlsym(shared_lib, destory.c_str());
        auto start = std::chrono::high_resolution_clock::now();
        if(fn1) {
            fn1();
        }else {
            std::cout << init << "function didnt run" << std::endl;
        }
        if(fn2) {
            fn2((double*)args.at(0),(double*)args.at(1), (double*)args.at(2));
        }else {
            std::cout << transform << "function didnt run" << std::endl;
        }
        if(fn3){
            fn3();
        }else {
            if ( DEBUGOUT) std::cout << destory << "function didnt run" << std::endl;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = stop - start;
        CPUTime = duration.count();
        dlclose(shared_lib);
    }
    return getKernelTime();
}


void Executor::execute(std::string result) {
    if ( DEBUGOUT) std::cout << "entered CPU backend execute\n";
    std::string compile;

    if ( DEBUGOUT)
        compile = "/usr/bin/gcc -I $SPIRAL_HOME/namespaces -Wall -Wextra spiral_generated.c -o libtmp.so -O3 -shared -fPIC";
    else 
        compile = "/usr/bin/gcc -I $SPIRAL_HOME/namespaces spiral_generated.c -o libtmp.so -O3 -shared -fPIC";
    
    if ( DEBUGOUT) {
        std::cout << "created compile\n";
    }

    std::string result2 = result.substr(result.find("*/")+3, result.length());
    // std::cout << result2.size() << std::endl;
    std::ofstream out("spiral_generated.c");
    out << result2;
    out.close();

    if ( DEBUGOUT )
        std::cout << "compiling\n";
    system(compile.c_str());
    if ( DEBUGOUT )
        std::cout << "finished compiling\n";
}

float Executor::getKernelTime() {
    return CPUTime;
}

#endif            //  FFTX_MDDFT_HIPBACKEND_HEADER
