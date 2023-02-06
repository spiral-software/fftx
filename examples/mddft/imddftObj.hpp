using namespace fftx;

class IMDDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        tracing=true;
  
        box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
        
        std::array<array_t<3,std::complex<double>>,1> intermediates {domain};
        array_t<3,std::complex<double>> inputs(domain);
        array_t<3,std::complex<double>> outputs(domain);


        setInputs(inputs);
        setOutputs(outputs);
        
        openScalarDAG();
        
        // MDDFT(domain.extents(), 1, intermediates[0], inputs);
        IMDDFT(domain.extents(), 1, outputs, inputs);

        closeScalarDAG(intermediates, name.c_str());
    }
};
