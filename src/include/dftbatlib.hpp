using namespace fftx;

static std::string dftbat_script = "ns := szns;\n\
    name := transform_spiral;\n\
    tags := [[APar, APar], [APar, AVec], [AVec, APar]];\n\
    t := let(\n\
        name := name,\n\
        TFCall ( TRC ( TTensorI ( TTensorI ( DFT ( ns, sign ), abatch, APar, APer ),\n\
                                  nbatch, tags[stridetype][1], tags[stridetype][2] ) ),\n\
                 rec ( fname := name, params := [] ) )\n\
    );";


class DFTBATProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        fftx::OutStream() << "szns := [" << sizes.at(0) << "];" << std::endl;
        fftx::OutStream() << "nbatch := " << sizes.at(1) << ";" << std::endl;
        fftx::OutStream() << "stridetype := " << sizes.at(2) << ";" << std::endl;
        fftx::OutStream() << "sign := " << sizes.at(3) << ";" << std::endl;
         if(sizes.at(4) == -1)
            fftx::OutStream() << "prefix := \"fftx_dftbat_\";" << std::endl;
        else
            fftx::OutStream() << "prefix := \"fftx_idftbat_\";" << std::endl;
        fftx::OutStream() << dftbat_script << std::endl;
    }
};
