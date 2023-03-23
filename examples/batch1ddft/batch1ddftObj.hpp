using namespace fftx;

std::string ibatch1ddft_script = "\n\
    transform := let(\n\
        name := "grid_dft"::"d_cont",\n\
        TFCall(TRC(TTensorI(DFT(N, sign), N*N, read, write)),\n\
            rec(fname := name, params := []))\n\
    );";


class BATCH1DDFT: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        std::cout << "N = " << sizes.at(0) << ";" << std::endl;
        if(sizes.at(1) == 0) {
            std::cout << "read = APar;" << std::endl;
        }
        else{
            std::cout << "read = AVec;" << std::endl;
        }
        if(sizes.at(2) == 0) {
            std::cout << "write = APar;" << std::endl;
        }
        else{
            std::cout << "write = AVec;" << std::endl;
        }
        std::cout << "sign := 1;" << std::endl;
        std::cout << "name := \""<< name << "_spiral" << "\";" << std::endl;
        std::cout << ibatch1ddft_script << std::endl;
    }
};

