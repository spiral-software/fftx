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
    Signature() {

    }

    // std::vector<std::any> in;
    // std::vector<std::any> out;
    // std::vector<std::any> in_out;
    std::vector<void *> inputargs;
    std::vector<fftx::array_t<3,std::complex<double>>> in;
    std::vector<fftx::array_t<3,std::complex<double>>> out;
    std::vector<fftx::array_t<3,std::complex<double>>> in_out;

};

class GBTLXProblem {
public:

    Signature sig;

    GBTLXProblem(){
//set input/output = null

    }


    GBTLXProblem(Signature Sig){
        sig= Sig;
    }

    virtual void randomProblemInstance() = 0;

private:
};

class GBTLXSolver {
public:
    virtual void semantics(GBTLXProblem &p) = 0;
    void Apply(GBTLXProblem &p);
};

void GBTLXSolver::Apply(GBTLXProblem &p){

        //printf("semantics called\n");
        if(FILE * file = fopen("testinput.txt", "r")) {
            std::cout << "found file to parse\n";
            //exit(1);
            Executor e;
            //e.execute(std::any_cast<int>(p.sig.inputargs.at(0)), std::any_cast<char**>(p.sig.inputargs.at(1)));
            e.execute((int)p.sig.inputargs.at(0), (char**)p.sig.inputargs.at(1));
        }
        else {
            std::cout << "file not found or cached\n";
            semantics(p);
        }
}