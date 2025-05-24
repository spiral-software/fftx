
//  Copyright (c) 2018-2025, Carnegie Mellon University
//   All rights reserved.
//
//  See LICENSE file for full information

using namespace fftx;

/**
   Class for forward complex-to-complex 3D FFT.

   <tt>FFTXProblem::args</tt> must be set to a <tt>std::vector<void*></tt> of length 3, where
   - <tt>args[0]</tt> is a pointer to a complex output array of size the product of the dimensions in <tt>FFTXProblem::sizes</tt>;
   - <tt>args[1]</tt> is a pointer to a complex input array of size the product of the dimensions in <tt>FFTXProblem::sizes</tt>;
   - <tt>args[2]</tt> is not used and can be set to NULL.

   <tt>FFTXProblem::sizes</tt> must be set to a <tt>std::vector<int></tt> of length 3, containing the transform size in each coordinate dimension.

   <tt>FFTXProblem::name</tt> must be set to <tt>"mddft"</tt>.
 */
class MDDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        tracing = true;
        box_t<3> empty(point_t<3>({{1,1,1}}), point_t<3>({{0,0,0}}));
        box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
        
        std::array<array_t<3,std::complex<double>>,1> intermediates {{empty}}; // in this case, empty
        array_t<3,std::complex<double>> inputs(domain);
        array_t<3,std::complex<double>> outputs(domain);
        setInputs(inputs);
        setOutputs(outputs);
        
        openScalarDAG();
        MDDFT(domain.extents(), 1, outputs, inputs);

        closeScalarDAG(intermediates, name.c_str());
    }
};
