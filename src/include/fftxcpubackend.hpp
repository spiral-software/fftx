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

#if defined(_WIN32) || defined (_WIN64)
  #include <io.h>
//  #define pipe _pipe
//  #define popen _popen
//  #define pclose _pclose
  #define FFTX_PIPE _pipe
  #define FFTX_POPEN _popen
  #define FFTX_PCLOSE _pclose

  #include <direct.h>
//  #define getcwd _getcwd
//  #define chdir _chdir
  #define FFTX_GETCWD _getcwd
  #define FFTX_CHDIR _chdir
#else
  #include <unistd.h>    // dup2
  #define FFTX_PIPE pipe
  #define FFTX_POPEN popen
  #define FFTX_PCLOSE pclose
  #define FFTX_GETCWD getcwd
  #define FFTX_CHDIR chdir
#endif

#include <sys/types.h> // rest for open/close
#if defined(__APPLE__)
#include <sys/utsname.h> // check machine name
#endif
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <array>

#if defined(_WIN32) || defined (_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <stdlib.h>
#include <cstring>
#include <chrono>
#include <regex>
#pragma once

// Flags to show cmake output, make output.
#define FFTX_CMAKE_OUT 0
#define FFTX_MAKE_OUT 0

#if defined ( FFTX_PRINTDEBUG )
#define FFTX_DEBUGOUT 1
#else
#define FFTX_DEBUGOUT 0
#endif

// Where to direct null output.
#if defined(_WIN32) || defined (_WIN64)
#define FFTX_NULL_OUT " > NUL"
#else
#define FFTX_NULL_OUT " > /dev/null"
#endif

#if FFTX_CMAKE_OUT
#define FFTX_CMAKE_REDIRECT ""
#else
#define FFTX_CMAKE_REDIRECT FFTX_NULL_OUT
#endif

#if FFTX_MAKE_OUT
#define FFTX_MAKE_REDIRECT ""
#else
#define FFTX_MAKE_REDIRECT FFTX_NULL_OUT
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

if ( APPLE )
    if ( ${CMAKE_OSX_ARCHITECTURES} MATCHES "arm64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64.*" )
	    set ( ADDL_COMPILE_FLAGS -arch arm64 )
    elseif ( ${CMAKE_OSX_ARCHITECTURES} MATCHES "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64.*")
	    set ( ADDL_COMPILE_FLAGS -arch x86_64 )
    endif ()
endif ()

add_library                ( tmp SHARED spiral_generated.c )
target_include_directories ( tmp PRIVATE ${SPIRAL_SOURCE_DIR}/namespaces )
target_compile_options     ( tmp PRIVATE ${_addl_options} )

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
    if ( FFTX_DEBUGOUT) fftx::OutStream() << "Loading shared library\n";

    #if defined (_WIN32) || defined (_WIN64)
        shared_lib = (void *)LoadLibrary("temp/Release/tmp.dll");
    #elif defined(__APPLE__)
        shared_lib = dlopen("temp/libtmp.dylib", RTLD_LAZY);
    #else
        shared_lib = dlopen("temp/libtmp.so", RTLD_LAZY); 
    #endif

    if(!shared_lib) {
        #if defined (_WIN32) || defined (_WIN64)
        int error = GetLastError();
        char * errorMessage;
            FormatMessage ( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
                            NULL, error, 0, (char *)&errorMessage, 0, NULL );
            fftx::OutStream() << "Cannot open library: " << (errorMessage) << std::endl;
            LocalFree(errorMessage);
        #else
            fftx::OutStream() << "Cannot open library: " << dlerror() << '\n';
        #endif
        exit(0);
    }

    std::ostringstream oss;
    std::ostringstream oss1;
    std::ostringstream oss2;
    oss << "init_" << name << "_spiral";
    oss1 << name << "_spiral";
    oss2 << "destroy_" << name << "_spiral";
    std::string init = oss.str();
    std::string transform = oss1.str();
    std::string destroy = oss2.str();

    #if defined (_WIN32) || defined (_WIN64)
        void (*fn1) ()= (void (*)()) GetProcAddress ( (HMODULE) shared_lib, init.c_str() );
        void (*fn2)(double *, double *, ...) = (void (*)(double *, double *, ...)) GetProcAddress ( (HMODULE) shared_lib, transform.c_str() );
        void (*fn3) ()= (void (*)()) GetProcAddress ( (HMODULE) shared_lib, destroy.c_str() );
    #else
        void (*fn1) ()= (void (*)())dlsym(shared_lib, init.c_str());
#if defined(__APPLE__)
        // Fix 3 args because of unresolved problem with variadic functions on Apple.
        void (*fn2)(double *, double *, double *) = (void (*)(double *, double *, double *))dlsym(shared_lib, transform.c_str());
#else
        void (*fn2)(double *, double *, ...) = (void (*)(double *, double *, ...))dlsym(shared_lib, transform.c_str());
#endif
        void (*fn3) ()= (void (*)())dlsym(shared_lib, destroy.c_str());
    #endif

    auto start = std::chrono::high_resolution_clock::now();
    if(fn1) {
        fn1();
    }else {
        fftx::OutStream() << init << "function didnt run" << std::endl;
    }
    if(fn2) {
      if(args.size() < 3)
#if defined(__APPLE__)
        // Fix 3 args because of unresolved problem with variadic functions on Apple.
        fn2((double*)args.at(0),(double*)args.at(1), nullptr);
#else
        fn2((double*)args.at(0),(double*)args.at(1));
#endif
      else
        fn2((double*)args.at(0),(double*)args.at(1), (double*)args.at(2));
    }else {
        fftx::OutStream() << transform << "function didnt run" << std::endl;
    }
    if(fn3){
        fn3();
    }else {
        fftx::OutStream() << destroy << "function didnt run" << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;
    CPUTime = duration.count();
        
    #if defined (_WIN32) || defined (_WIN64)
        FreeLibrary ( (HMODULE) shared_lib );
    #else
        dlclose(shared_lib);
    #endif

    return getKernelTime();
}


void Executor::execute(std::string result) {
    if ( FFTX_DEBUGOUT) fftx::OutStream() << "entered CPU backend execute\n";
    std::string compile;
    
    char buff[FILENAME_MAX]; //create string buffer to hold path
    char* getcwdret = FFTX_GETCWD( buff, FILENAME_MAX );
    std::string current_working_dir(buff);

    struct stat sb;

    int systemret;
    if(stat((current_working_dir+"/temp").c_str(), &sb) == 0)
        systemret = system("rm -rf temp");
        
    if ( FFTX_DEBUGOUT) {
        fftx::OutStream() << "created compile\n";
    }

    std::string result2 = result.substr(result.find("#include"));
    #if defined (_WIN32) || defined (_WIN64)
        int check = _mkdir("temp");
    #else
        int check = mkdir("temp", 0777);
    #endif
    if(check != 0) {
        fftx::OutStream() << "failed to create temp directory for runtime code\n";
        exit(-1);
    }
    std::ofstream out("temp/spiral_generated.c");
    out << result2;
    out.close();
    std::ofstream cmakelists("temp/CMakeLists.txt");
    if(FFTX_DEBUGOUT)
        cmakelists << "set ( _addl_options -Wall )" << std::endl;       //  -Wextra

    cmakelists << cmake_script;
    cmakelists.close();
    if ( FFTX_DEBUGOUT )
        fftx::OutStream() << "compiling\n";
    
    check = FFTX_CHDIR("temp");
    if(check != 0) {
        fftx::OutStream() << "failed to change to temp directory for runtime code\n";
        exit(-1);
    }

    std::stringstream cmakecmd;
    #if defined(_WIN32) || defined (_WIN64)
        cmakecmd = "cmake . " << FFTX_CMAKE_REDIRECT << " && cmake --build . --config Release" << FFTX_CMAKE_REDIRECT;      //  --target install
    #elif defined(__APPLE__)
        struct utsname unameData;
        uname(&unameData);
        std::string machine_name(unameData.machine);
        if(machine_name == "arm64")
          cmakecmd = "cmake -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 . " << FFTX_CMAKE_REDIRECT << " && make" << FFTX_MAKE_REDIRECT;
        else
          cmakecmd = "cmake . " << FFTX_CMAKE_REDIRECT << " && make" << FFTX_MAKE_REDIRECT;
    #else
        cmakecmd << "cmake . " << FFTX_CMAKE_REDIRECT << " && make" << FFTX_MAKE_REDIRECT;
    #endif
        systemret = system(cmakecmd.str().c_str());

    check = FFTX_CHDIR(current_working_dir.c_str());
    if(check != 0) {
        fftx::OutStream() << "failed to change to working directory for runtime code\n";
        exit(-1);
    }
    // systemret = system("cd ..;");
    if ( FFTX_DEBUGOUT )
        fftx::OutStream() << "finished compiling\n";
}

float Executor::getKernelTime() {
    return CPUTime;
}

#endif            //  FFTX_MDDFT_HIPBACKEND_HEADER
