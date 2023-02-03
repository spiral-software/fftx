// #include "fftx3.hpp"
// #include <array>
// #include <cstdio>
// #include <cassert>
#pragma once
using namespace fftx;

class RCONVProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        const int offx = 3;
        const int offy = 5;
        const int offz = 11;
        #if FFTX_COMPLEX_TRUNC_LAST
        const int fx = sizes.at(0);
        const int fy = sizes.at(1);
        const int fz = sizes.at(2)/2 + 1;
        #else
        const int fx = sizes.at(0)/2 + 1;
        const int fy = sizes.at(1);
        const int fz = sizes.at(2);
        #endif

        box_t<3> domain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                   point_t<3>({{offx+sizes.at(0), offy+sizes.at(1), offz+sizes.at(2)}}));
        box_t<3> fdomain3(point_t<3>({{offx+1, offy+1, offz+1}}),
                    point_t<3>({{offx+fx, offy+fy, offz+fz}})); 

        tracing=true;

        std::array<array_t<3,std::complex<double>>,2> intermediates {fdomain3, fdomain3};
        array_t<3,double> inputs(domain3);
        array_t<3,double> outputs(domain3);
        array_t<3,double> symbol(fdomain3);


        setInputs(inputs);
        setOutputs(outputs);

        openScalarDAG();

        PRDFT(domain3.extents(), intermediates[0], inputs);
        kernel(symbol, intermediates[1], intermediates[0]);
        IPRDFT(domain3.extents(), outputs, intermediates[1]);

        closeScalarDAG(intermediates, name.c_str());
    }
};
