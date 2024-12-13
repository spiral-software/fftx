//read seq, write seq
static std::string ibatch1dprdft_script_0x0 = "transform := let(\n\
         TFCall(TTensorI(IPRDFT(N, 1), B, write, read),\n\
            rec(fname := name, params := [])));";

//read seq, write strided
static std::string ibatch1dprdft_script_0x1 = "transform := let(\n\
         TFCall(TTensorI(IPRDFT(N, 1), B, write, read),\n\
            rec(fname := name, params := [])));";

//read strided, write seq
static std::string ibatch1dprdft_script_1x0 = "transform := let(\n\
    TFCall(TTensorI(IPRDFT(N, 1), B, write, APar) * \n\
        Prm(fTensor(L(IPRDFT1(N, 1).dims()[2]/2 * B, B), fId(2))), \n\
    rec(fname := name, params := [])));";

//read strided, write strided
static std::string ibatch1dprdft_script_1x1 = "transform := let(\n\
    TFCall(TTensorI(IPRDFT(N, 1), B, write, APar) * \n\
        Prm(fTensor(L(IPRDFT1(N, 1).dims()[2]/2 * B, B), fId(2))), \n\
    rec(fname := name, params := [])));";

class IBATCH1DPRDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        fftx::OutStream() << "Import(realdft);" << std::endl;
        fftx::OutStream() << "N := " << sizes.at(0) << ";" << std::endl;
        fftx::OutStream() << "B := " << sizes.at(1) << ";" << std::endl;
        if(sizes.at(2) == 0) {
            fftx::OutStream() << "read := APar;" << std::endl;
        }
        else{
            fftx::OutStream() << "read := AVec;" << std::endl;
        }
        if(sizes.at(3) == 0) {
            fftx::OutStream() << "write := APar;" << std::endl;
        }
        else{
            fftx::OutStream() << "write := AVec;" << std::endl;
        }
        fftx::OutStream() << "sign := 1;" << std::endl;
        fftx::OutStream() << "name := \""<< name << "_spiral" << "\";" << std::endl;
        if(sizes.at(2) == 0 && sizes.at(3) == 0)
            fftx::OutStream() << ibatch1dprdft_script_0x0 << std::endl;
        else if(sizes.at(2) == 0 && sizes.at(3) == 1)
            fftx::OutStream() << ibatch1dprdft_script_0x1 << std::endl;
        else if(sizes.at(2) == 1 && sizes.at(3) == 0)
            fftx::OutStream() << ibatch1dprdft_script_1x0 << std::endl;
        else
            fftx::OutStream() << ibatch1dprdft_script_1x1 << std::endl;
    }
};
