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

class FFTXProblem {
public:

    // std::vector<fftx::array_t<3,std::complex<double>>> in;
    // std::vector<fftx::array_t<3,std::complex<double>>> out;
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


    // FFTXProblem(std::vector<fftx::array_t<3,std::complex<double>>> &in1,
    // std::vector<fftx::array_t<3,std::complex<double>>> &out1)
    //     :in(in1), out(out1){}

    // FFTXProblem(std::vector<fftx::array_t<3,std::complex<double>>> &in1,
    // std::vector<fftx::array_t<3,std::complex<double>>> &out1,
    // std::vector<CUdeviceptr> &args1)
    //     :in(in1), out(out1), args(args1){}
    void transform();
    virtual void randomProblemInstance() = 0;
    virtual std::string semantics() = 0;
    float gpuTime;
    void run(Executor e);
    float getTime();
    ~FFTXProblem(){}

};

void FFTXProblem::transform(){

        //NEED A CACHING CASE
        //printf("semantics called\n");
        // if(fopen("mddft.fftx.source.txt", "r") == false) {
        //     semantics();
        // }
        if(res.empty()) {
            res = semantics();
            //std::cout << res << std::endl;
        }
        if(executors.find(res) != executors.end()) {
            std::cout << "running cached instances\n";
            run(executors.at(res));
        }
        else if(!res.empty()) {
            std::cout << "found file to parse\n";
            // std::cout << res << std::endl;
            // std::cout << res.size() << std::endl;
            // exit(0);
            // const char * file_name = "mddft.fftx.source.txt";
            // p.sig.args.push_back((char**)file_name);
            // double * input =  (double*)(p.sig.in.at(0).m_data.local());
            // for(int i = 0; i < in.at(0).m_domain.size(); i++) {
            //     std::cout << in.at(0).m_data.local()[i] << std::endl;
            // }
            //exit(1);
            //Executor e(in, out);
            Executor e;
            // std::cout << p.sig.in.at(0).m_domain.size() << std::endl;
            //e.execute(std::any_cast<int>(p.sig.inputargs.at(0)), std::any_cast<char**>(p.sig.inputargs.at(1)));
            //e.execute(*((int*)p.sig.inputargs.at(0)), (char**)p.sig.inputargs.at(1));
            e.execute(res);
            executors.insert(std::make_pair(res, e));
            run(e);
            //e.execute(p);
        }
        else {
            std::cout << "failure\n";
            exit(1);
        }
}


void FFTXProblem::run(Executor e) {
    gpuTime = e.initAndLaunch(args);
    //e.returnData(out);
}

float FFTXProblem::getTime() {
   return gpuTime;
}

#endif			// FFTX_MDDFT_INTERFACE_HEADER
