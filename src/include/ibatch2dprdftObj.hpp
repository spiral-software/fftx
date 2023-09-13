using namespace fftx;

static std::string ibatch2dprdft_script = "transform := let(\n\
         TFCall(TRC(TTensorI(TTensorI(PRDFT(N, sign), b, write, read), B, AVec, AVec)),\n\
            rec(fname := name, params := [])));";

class IBATCH2DPRDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        std::cout << "Import(realdft);" << std::endl;
        std::cout << "N := " << sizes.at(0) << ";" << std::endl;
        std::cout << "B := " << sizes.at(1) << ";" << std::endl;
        std::cout << "b := " << sizes.at(2) << ";" << std::endl;

        if(sizes.at(3) == 0) {
            std::cout << "read := APar;" << std::endl;
        }
        else{
            std::cout << "read := AVec;" << std::endl;
        }
        if(sizes.at(4) == 0) {
            std::cout << "write := APar;" << std::endl;
        }
        else{
            std::cout << "write := AVec;" << std::endl;
        }
        std::cout << "sign := 1;" << std::endl;
        std::cout << "name := \""<< name << "_spiral" << "\";" << std::endl;
        std::cout << ibatch2dprdft_script << std::endl;
    }
};

