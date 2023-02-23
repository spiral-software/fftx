using namespace fftx;


static constexpr auto dftbat_script{
R"(
    ns := szns;
    name := transform_spiral;
    tags := [[APar, APar], [APar, AVec], [AVec, APar]];

    t := let(
        name := name,
        TFCall ( TRC ( TTensorI ( TTensorI ( DFT ( ns, sign ), abatch, APar, APer ),
                                  nbatch, tags[stridetype][1], tags[stridetype][2] ) ),
                 rec ( fname := name, params := [] ) )
    );
)"};


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
