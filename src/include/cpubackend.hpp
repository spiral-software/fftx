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
#include <regex>
#pragma once

#if defined ( PRINTDEBUG )
#define DEBUGOUT 1
#else
#define DEBUGOUT 0
#endif

static constexpr auto cmake_script{
R"(
cmake_minimum_required ( VERSION 3.14 )
set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )
project ( tmplib LANGUAGES C CXX )

if ( DEFINED ENV{SPIRAL_HOME} )
    set ( SPIRAL_SOURCE_DIR $ENV{SPIRAL_HOME} )
else ()
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
        message ( FATAL_ERROR "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
endif ()

add_library                ( tmp SHARED spiral_generated.c )
target_include_directories ( tmp PRIVATE ${SPIRAL_SOURCE_DIR}/namespaces )
target_compile_options     ( tmp PRIVATE -shared -fPIC ${_addl_options} )

if ( WIN32 )
    set_property    ( TARGET tmp PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON )
endif ()
)"};


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
    #if defined(_WIN32) || defined (_WIN64)
        shared_lib = dlopen("temp/libtmp.dll", RTLD_LAZY);
    #elif defined(__APPLE__)
        shared_lib = dlopen("temp/libtmp.dylib", RTLD_LAZY);
    #else
        shared_lib = dlopen("temp/libtmp.so", RTLD_LAZY); 
    #endif
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
    // system("rm -rf temp");
    return getKernelTime();
}


void Executor::execute(std::string result) {
    if ( DEBUGOUT) std::cout << "entered CPU backend execute\n";
    std::string compile;
    
    if ( DEBUGOUT) {
        std::cout << "created compile\n";
    }

    std::string result2 = result.substr(result.find("*/")+3, result.length());
    int check = mkdir("temp", 0777);
    // if((check)) {
    //     std::cout << "failed to create temp directory for runtime code\n";
    //     exit(-1);
    // }
    std::ofstream out("temp/spiral_generated.c");
    out << result2;
    out.close();
    std::ofstream cmakelists("temp/CMakeLists.txt");
    if(DEBUGOUT)
        cmakelists << "set ( _addl_options -Wall -Wextra )" << std::endl;

    cmakelists << cmake_script;
    cmakelists.close();
    if ( DEBUGOUT )
        std::cout << "compiling\n";

    char buff[FILENAME_MAX]; //create string buffer to hold path
    getcwd( buff, FILENAME_MAX );
    std::string current_working_dir(buff);
    
    check = chdir("temp");
    // if(!(check)) {
    //     std::cout << "failed to create temp directory for runtime code\n";
    //     exit(-1);
    // }
    system("cmake . && make");
    check = chdir(current_working_dir.c_str());
    // if((check)) {
    //     std::cout << "failed to create temp directory for runtime code\n";
    //     exit(-1);
    // }
    // system("cd ..;");
    if ( DEBUGOUT )
        std::cout << "finished compiling\n";
}

float Executor::getKernelTime() {
    return CPUTime;
}

#endif            //  FFTX_MDDFT_HIPBACKEND_HEADER
