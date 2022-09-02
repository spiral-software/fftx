#ifndef INTERFACE_H
#define INTERFACE_H

#include <cstdlib>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <any>


class Signature{
public:
    Signature() {

    }

    // std::vector<std::any> in;
    // std::vector<std::any> out;
    // std::vector<std::any> in_out;
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
        semantics(p);

}

//FFTX_REFERENCE_TEST_TRITEST_HPP
#endif 