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
        std::cout << "szns := [" << sizes.at(0) << "];" << std::endl;
        std::cout << "nbatch := " << sizes.at(1) << ";" << std::endl;
        std::cout << "stridetype := " << sizes.at(2) << ";" << std::endl;
        std::cout << "sign := " << sizes.at(3) << ";" << std::endl;
         if(sizes.at(4) == -1)
            std::cout << "prefix := \"fftx_dftbat_\";" << std::endl;
        else
            std::cout << "prefix := \"fftx_idftbat_\";" << std::endl;
        std::cout << dftbat_script << std::endl;
    }
};
