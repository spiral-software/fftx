using namespace fftx;

/**
   Class for forward complex-to-complex 3D FFT.

   In <tt>FFTXProblem::args</tt>,
   - <tt>args[0]</tt> is a pointer to a <tt>std::complex<double></tt> output array;
   - <tt>args[1]</tt> is a pointer to a <tt>std::complex<double></tt> input array;
   - <tt>args[2]</tt> is not used and can be set to NULL.

   <tt>FFTXProblem::name</tt> must be set to  <tt>"mddft"</tt>.
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
