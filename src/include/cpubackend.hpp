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

float Executor::initAndLaunch(std::vector<void*>& args) {
    shared_lib = dlopen("./libtmp.so", RTLD_LAZY);
    if(!shared_lib) {
        std::cout << "Cannot open library: " << dlerror() << '\n';
        exit(0);
    }
    else if(shared_lib){
        void (*fn1) ()= (void (*)())dlsym(shared_lib, "init_transform_spiral");
        void (*fn2) (double *, double *, double *) = (void (*)(double *, double *, double *))dlsym(shared_lib, "transform_spiral");
        void (*fn3) ()= (void (*)())dlsym(shared_lib, "destroy_transform_spiral");
        if(fn1) {
            fn1();
        }
        if(fn2) {
            fn2((double*)args.at(0),(double*)args.at(1), (double*)args.at(2));
        }
        if(fn3){
            fn3();
        }
        dlclose(shared_lib);
    }
    CPUTime += 10;
    return getKernelTime();
}


void Executor::execute(std::string result) {
    // std::cout << result << std::endl;
    // exit(0);
    // result.erase(result.size()-8);
    std::string result2 = result.substr(result.find("*/")+3, result.length());
    // std::cout << result2 << std::endl;
    // exit(0);
    int p2[2];
    if(pipe(p2) < 0)
        std::cout << "pipe failed\n";
    write(p2[1], result2.c_str(), result2.size());
    close(p2[1]);
    int save_stdin = redirect_input(p2[0]);
    std::string compile;
    if(LOCALDEBUG)
        compile = "/usr/bin/gcc -I $SPIRAL_HOME/namespaces -Wall -Wextra -x c - -o libtmp.so -O2 -shared -fPIC";
    else 
        compile = "/usr/bin/gcc -I $SPIRAL_HOME/namespaces -x c - -o libtmp.so -O2 -shared -fPIC";
    system(compile.c_str());
    restore_input(save_stdin);
    close(p2[0]);
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