#ifndef FFTX_MDDFT_INTERFACE_HEADER
#define FFTX_MDDFT_INTERFACE_HEADER

//  Copyright (c) 2018-2022, Carnegie Mellon University
//  See LICENSE for details

#include <cstdlib>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>
#include <unistd.h>    // dup2
#include <sys/types.h> // rest for open/close
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <array>
#include <chrono>
#if defined FFTX_CUDA
#include "cudabackend.hpp"
#elif defined FFTX_HIP
#include "hipbackend.hpp"
#else
#include "cpubackend.hpp"
#endif
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#include "fftx_mddft_gpu_public.h"
#include "fftx_imddft_gpu_public.h"
#include "fftx_mdprdft_gpu_public.h"
#include "fftx_imdprdft_gpu_public.h"
#else
#include "fftx_mddft_cpu_public.h"
#include "fftx_imddft_cpu_public.h"
#include "fftx_mdprdft_cpu_public.h"
#include "fftx_imdprdft_cpu_public.h"
#endif
#pragma once

#if defined ( PRINTDEBUG )
#define DEBUGOUT 1
#else
#define DEBUGOUT 0
#endif

class Executor;
class FFTXProblem;

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        // std::cout << buffer.data() << std::endl;
        result += buffer.data();
    }
    return result;
}

int redirect_input(const char* fname)
{
    int save_stdin = dup(0);
    //std::cout << "in redirect input " << fname << std::endl;
    int input = open(fname, O_RDONLY);
    dup2(input, 0); 
    close(input);
    return save_stdin;
}

int redirect_input(int input)
{
    int save_stdin = dup(0);
    dup2(input, 0);
    close(input);
    return save_stdin;
}

void restore_input(int saved_fd)
{
    close(0);
    dup2(saved_fd, 0);
    close(saved_fd);
}

transformTuple_t * getLibTransform(std::string name, std::vector<int> sizes) {
    if(name == "mddft") {
        return fftx_mddft_Tuple(fftx::point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
    }
    else if(name == "imddft") {
        return fftx_imddft_Tuple(fftx::point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
    }
    else if(name == "mdprdft") {
        return fftx_mdprdft_Tuple(fftx::point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
    }
    else if(name == "imdprdft") {
        return fftx_imdprdft_Tuple(fftx::point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
    }
    else {
        std::cout << "non-supported fixed library transform" << std::endl; 
        return nullptr;
    }
}

std::string getFFTX() {
     const char * tmp2 = std::getenv("FFTX_HOME");
    std::string tmp(tmp2 ? tmp2 : "");
    if (tmp.empty()) {
        std::cout << "[ERROR] No such variable found, please download and set FFTX_HOME env variable" << std::endl;
        exit(-1);
    }
    tmp += "/cache_jit_files/"; 
    return tmp;
}

std::string getSPIRAL() {
    const char * tmp2 = std::getenv("SPIRAL_HOME");//required >8.3.1
    std::string tmp(tmp2 ? tmp2 : "");
    if (tmp.empty()) {
        std::cout << "[ERROR] No such variable found, please download and set SPIRAL_HOME env variable" << std::endl;
        exit(-1);
    }
    tmp += "/bin/spiral";         
    return tmp;
}

void getImportAndConf() {
    std::cout << "Load(fftx);\nImportAll(fftx);\n";
    #if (defined FFTX_HIP || FFTX_CUDA)
    std::cout << "ImportAll(simt);\nLoad(jit);\nImport(jit);\n";
    #endif
    #if defined FFTX_HIP 
    std::cout << "conf := FFTXGlobals.defaultHIPConf();\n";
    #elif defined FFTX_CUDA 
    std::cout << "conf := LocalConfig.fftx.confGPU();\n";
    #else
    std::cout << "conf := LocalConfig.fftx.defaultConf();\n";
    #endif
}

void printJITBackend(std::string name, std::vector<int> sizes) {
    std::string tmp = getFFTX();
    std::cout << "if 1 = 1 then opts:=conf.getOpts(transform);\ntt:= opts.tagIt(transform);\nif(IsBound(fftx_includes)) then opts.includes:=fftx_includes;fi;\nc:=opts.fftxGen(tt);\n fi;\n";
    std::cout << "GASMAN(\"collect\");\n";
    #if defined FFTX_HIP
        std::cout << "PrintTo(\"" << tmp << "cache_" << name << "_" << sizes.at(0) << "x" << sizes.at(1) << "x" << sizes.at(2) << "_HIP" << ".txt\", PrintHIPJIT(c,opts));\n"; 
        std::cout << "PrintHIPJIT(c,opts);\n";
    #elif defined FFTX_CUDA 
       std::cout << "PrintTo(\"" << tmp << "cache_" << name << "_" << sizes.at(0) << "x" << sizes.at(1) << "x" << sizes.at(2) << "_CUDA" << ".txt\", PrintJIT2(c,opts));\n";   
       std::cout << "PrintJIT2(c,opts);\n";
    #else
        std::cout << "PrintTo(\"" << tmp << "cache_" << name << "_" << sizes.at(0) << "x" << sizes.at(1) << "x" << sizes.at(2) << "_CPU" << ".txt\", opts.prettyPrint(c));\n"; 
        std::cout << "opts.prettyPrint(c);\n";
    #endif
}

class FFTXProblem {
public:

    std::vector<void*> args;
    std::vector<int> sizes;
    std::string res;
    std::map<std::vector<int>, Executor> executors;
    std::string name;
    FFTXProblem(){
    }

    FFTXProblem(std::string name1) {
        name = name1;
    }

    FFTXProblem(const std::vector<void*>& args1) {
        args = args1;

    }
    FFTXProblem(const std::vector<int>& sizes1) {
       sizes = sizes1;

    }
    FFTXProblem(const std::vector<void*>& args1, const std::vector<int>& sizes1) {
        args = args1;   
        sizes = sizes1;
    }
    FFTXProblem(const std::vector<int>& sizes1, std::string name1) {  
        sizes = sizes1;
        name = name1;
    }
     FFTXProblem(const std::vector<void*>& args1, const std::vector<int>& sizes1, std::string name1) {
        args = args1;   
        sizes = sizes1;
        name = name1;
    }

    void setSizes(const std::vector<int>& sizes1);
    void setArgs(const std::vector<void*>& args1);
    void setName(std::string name);
    void transform();
    std::string semantics2();
    virtual void randomProblemInstance() = 0;
    virtual void semantics() = 0;
    float gpuTime;
    void run(Executor e);
    std::string returnJIT();
    float getTime();
    ~FFTXProblem(){}

};

void FFTXProblem::setArgs(const std::vector<void*>& args1) {
    args = args1;
}

void FFTXProblem::setSizes(const std::vector<int>& sizes1) {
    sizes = sizes1;
}

void FFTXProblem::setName(std::string name1) {
    name = name1;
}

std::string FFTXProblem::semantics2() {
    std::string tmp = getSPIRAL();
    int p[2];
    if(pipe(p) < 0)
        std::cout << "pipe failed\n";
    std::stringstream out; 
    std::streambuf *coutbuf = std::cout.rdbuf(out.rdbuf()); //save old buf
    getImportAndConf();
    semantics();
    printJITBackend(name, sizes);
    std::cout.rdbuf(coutbuf);
    std::string script = out.str();
    int res = write(p[1], script.c_str(), script.size());
    close(p[1]);
    int save_stdin = redirect_input(p[0]);
    std::string result = exec(tmp.c_str());
    restore_input(save_stdin);
    close(p[0]);
    result.erase(result.size()-8);
    return result;
    // return nullptr;
}


void FFTXProblem::transform(){

    transformTuple_t *tupl = getLibTransform(name, sizes);
    if(tupl != nullptr) { //check if fixed library has transform
        if ( DEBUGOUT) std::cout << "found size in fixed library\n";
        ( * tupl->initfp )();
        #if defined (FFTX_CUDA) ||  (FFTX_HIP)
            DEVICE_EVENT_T custart, custop;
            DEVICE_EVENT_CREATE ( &custart );
            DEVICE_EVENT_CREATE ( &custop );
            DEVICE_EVENT_RECORD ( custart );
        #else
            auto start = std::chrono::high_resolution_clock::now();
        #endif
            ( * tupl->runfp ) ( (double*)args.at(0), (double*)args.at(1), (double*)args.at(2) );
        #if defined (FFTX_CUDA) ||  (FFTX_HIP)
            DEVICE_EVENT_RECORD ( custop );
            DEVICE_EVENT_SYNCHRONIZE ( custop );
            DEVICE_EVENT_ELAPSED_TIME ( &gpuTime, custart, custop );
        #else
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            gpuTime = duration.count();
        #endif
        ( * tupl->destroyfp )();
        //end time
    }
    else { // use RTC
        if(executors.find(sizes) != executors.end()) { //check in memory cache
            if ( DEBUGOUT) std::cout << "cached size found, running cached instance\n";
            run(executors.at(sizes));
        }
        else { //check filesystem cache
            std::ostringstream oss;
            std::string tmp = getFFTX();
            #if defined FFTX_HIP 
                oss << tmp << "cache_" << name << "_" << sizes.at(0) << "x" << sizes.at(1) << "x" << sizes.at(2) << "_HIP" << ".txt";
            #elif defined FFTX_CUDA 
                oss << tmp << "cache_" << name << "_" << sizes.at(0) << "x" << sizes.at(1) << "x" << sizes.at(2) << "_CUDA" << ".txt";
            #else
                oss << tmp << "cache_" << name << "_" << sizes.at(0) << "x" << sizes.at(1) << "x" << sizes.at(2) << "_CPU" << ".txt";
            #endif
            std::string file_name = oss.str();
            std::ifstream ifs ( file_name );
            if(ifs) {
                if ( DEBUGOUT) std::cout << "found cached file on disk\n";
                std::string fcontent ( ( std::istreambuf_iterator<char>(ifs) ),
                                       ( std::istreambuf_iterator<char>()    ) );
                Executor e;
                e.execute(fcontent);
                executors.insert(std::make_pair(sizes, e));
                run(e);
            } 
            else { //generate code at runtime
                if ( DEBUGOUT) std::cout << "haven't seen size, generating\n";
                res = semantics2();
                Executor e;
                e.execute(res);
                executors.insert(std::make_pair(sizes, e));
                run(e);
            }
        }
    }
}


void FFTXProblem::run(Executor e) {
    #if (defined FFTX_HIP || FFTX_CUDA)
    gpuTime = e.initAndLaunch(args);
    #else
    gpuTime = e.initAndLaunch(args, name);
    #endif
}

float FFTXProblem::getTime() {
   return gpuTime;
}

std::string FFTXProblem::returnJIT() {
    if(!res.empty()) {
        return res;
    }
    else{
        return nullptr;
    }
}

#endif            // FFTX_MDDFT_INTERFACE_HEADER
