#include <cstdlib>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
//#include <any>
#include "newinterface.hpp"
// #include <nvrtc.h>
// #include <cuda.h>
#pragma once

class Signature{
public:
    std::vector<char**> args;
    int counts;
    std::vector<fftx::array_t<3,std::complex<double>>> in;
    std::vector<fftx::array_t<3,std::complex<double>>> out;
    std::vector<fftx::array_t<3,std::complex<double>>> in_out;

    Signature(){}


    // std::vector<std::any> in;
    // std::vector<std::any> out;
    // std::vector<std::any> in_out;
};

class GBTLXProblem {
public:

    Signature sig;

    GBTLXProblem(){
//set input/output = null

    }


    GBTLXProblem(Signature &Sig){
    //    sig.args = Sig.args;
    //    sig.counts = Sig.counts;
    //    sig.in = Sig.in;
    //    sig.out = Sig.out;
    //    sig.in_out = Sig.in_out;
        sig = Sig;
    }

    ~GBTLXProblem(
    ){}
    virtual void randomProblemInstance() = 0;

private:
};

class GBTLXSolver {
public:
    virtual void semantics(GBTLXProblem &p) = 0;
    void Apply(GBTLXProblem &p);
};

void GBTLXSolver::Apply(GBTLXProblem &p){

        //NEED A CACHING CASE
        //printf("semantics called\n");
        if(fopen("mddft.fftx.source.txt", "r") == false) {
            semantics(p);
        }
        if(FILE * file = fopen("mddft.fftx.source.txt", "r")) {
            std::cout << "found file to parse\n";
            const char * file_name = "mddft.fftx.source.txt";
            p.sig.args.push_back((char**)file_name);
            // double * input =  (double*)(p.sig.in.at(0).m_data.local());
            // for(int i = 0; i < p.sig.in.at(0).m_domain.size(); i++) {
            //     std::cout << input[i] << std::endl;
            // }
            //exit(1);
            Executor e;
            std::cout << p.sig.in.at(0).m_domain.size() << std::endl;
            //e.execute(std::any_cast<int>(p.sig.inputargs.at(0)), std::any_cast<char**>(p.sig.inputargs.at(1)));
            //e.execute(*((int*)p.sig.inputargs.at(0)), (char**)p.sig.inputargs.at(1));
            e.execute(p.sig.in, p.sig.out, p.sig.counts, p.sig.args);
            //e.execute(p);
        }
        else {
            std::cout << "failure\n";
            exit(1);
        }
}