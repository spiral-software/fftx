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

#if defined FFTX_CUDA
#include "cudabackend.hpp"
#endif
#if defined FFTX_HIP
#include "hipbackend.hpp"
#endif

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
#pragma once

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
    std::cout << "Load(fftx);\nImportAll(fftx);\nImportAll(simt);\nLoad(jit);\nImport(jit);\n";
    #if defined FFTX_HIP 
    std::cout << "conf := FFTXGlobals.defaultHIPConf();\n";
    #endif
    #if defined FFTX_CUDA 
    std::cout << "conf := LocalConfig.fftx.confGPU();\n";
    #endif
}

void printJITBackend() {
    std::cout << "if 1 = 1 then opts:=conf.getOpts(transform);\ntt:= opts.tagIt(transform);\nif(IsBound(fftx_includes)) then opts.includes:=fftx_includes;fi;\nc:=opts.fftxGen(tt);\n fi;\n";
    std::cout << "GASMAN(\"collect\");\n";
    #if defined FFTX_HIP
        std::cout << "PrintHIPJIT(c,opts);\n";
    #endif
    #if defined FFTX_CUDA 
        std::cout << "PrintJIT2(c,opts)\n";
    #endif
}

class FFTXProblem {
public:

    std::vector<void*> args;
    std::vector<int> sizes;
    std::string res;
    std::unordered_map<std::string, Executor> executors;
    
    FFTXProblem(){
    }

    FFTXProblem(const std::vector<void*>& args1) {
        args = args1;

    }
    FFTXProblem(const std::vector<void*>& args1, const std::vector<int>& sizes1) {
        args = args1;   
        sizes = sizes1;
    }

    void transform();
    std::string semantics2();
    virtual void randomProblemInstance() = 0;
    virtual void semantics() = 0;
    float gpuTime;
    void run(Executor e);
    float getTime();
    ~FFTXProblem(){}

};

std::string FFTXProblem::semantics2() {
    std::string tmp = getSPIRAL();
    int p[2];
    if(pipe(p) < 0)
        std::cout << "pipe failed\n";
    std::stringstream out; 
    std::streambuf *coutbuf = std::cout.rdbuf(out.rdbuf()); //save old buf
    getImportAndConf();
    semantics();
    printJITBackend();
    std::cout.rdbuf(coutbuf);
    std::string script = out.str();
    write(p[1], script.c_str(), script.size());
    close(p[1]);
    int save_stdin = redirect_input(p[0]);
    std::string result = exec(tmp.c_str());
    restore_input(save_stdin);
    close(p[0]);
    result.erase(result.size()-8);
    return result;
}

void FFTXProblem::transform(){

        if(res.empty()) {
            res = semantics2();
        }
        if(executors.find(res) != executors.end()) {
            std::cout << "running cached instances\n";
            run(executors.at(res));
        }
        else if(!res.empty()) {
            std::cout << "found file to parse\n";
            Executor e;
            e.execute(res);
            executors.insert(std::make_pair(res, e));
            run(e);
        }
        else {
            std::cout << "failure\n";
            exit(1);
        }
}


void FFTXProblem::run(Executor e) {
    gpuTime = e.initAndLaunch(args);
}

float FFTXProblem::getTime() {
   return gpuTime;
}

#endif			// FFTX_MDDFT_INTERFACE_HEADER
