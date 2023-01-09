using namespace fftx;


static constexpr auto dftbat_script{
R"(
 ns := szns;
    name := prefix::StringInt(nbatch)::"_"::StringInt(szns[1])::"_"::StringInt(Length(ns))::"d";

    t := let(batch := nbatch,
        apat := When(true, APar, AVec),
        k := sign,
	##  name := "dft"::StringInt(Length(ns))::"d_batch",  
        TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)), 
            rec(fname := name, params := []))
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
         std::cout << "sign := -1;" << std::endl;
        std::cout << "prefix := \"fftx_dftbat_\";" << std::endl;
        std::cout << dftbat_script << std::endl;
    }
};