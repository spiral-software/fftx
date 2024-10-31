
static std::string batch2dprdft_script_0x0 = "transform := let(\n\
         TFCall(TRC(TTensorI(TTensorI(PRDFT(N, sign), b, write, read), B, AVec, AVec)),\n\
            rec(fname := name, params := [])));";

// read seq, write strided
static std::string batch2dprdft_script_0x1 = "transform := let(\n\
    TFCall(TRC(TTensorI(Prm(fTensor(L(PRDFT1(N, sign).dims()[1]/2 * b, PRDFT1(N, sign).dims()[1]/2), fId(2))) *\n\
    TTensorI(PRDFT1(N, sign), b, APar, read), B, AVec, AVec)),\n\
    rec(fname := name, params := [])));";

//read strided, write seq
static std::string batch2dprdft_script_1x0 = "transform := let(\n\
    TFCall(TRC(TTensorI(TTensorI(PRDFT1(N, sign), b, APar, read) * \n\
        Prm(fTensor(L(PRDFT1(N, sign).dims()[2]/2 * b, b), fId(2))), B, AVec, AVec)), \n\
    rec(fname := name, params := [])));";

class BATCH2DPRDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        fftx::OutStream() << "Import(realdft);" << std::endl;
        fftx::OutStream() << "N := " << sizes.at(0) << ";" << std::endl;
        fftx::OutStream() << "B := " << sizes.at(1) << ";" << std::endl;
        fftx::OutStream() << "b := " << sizes.at(2) << ";" << std::endl;
        if(sizes.at(3) == 0) {
            fftx::OutStream() << "read := APar;" << std::endl;
        }
        else{
            fftx::OutStream() << "read := AVec;" << std::endl;
        }
        if(sizes.at(4) == 0) {
            fftx::OutStream() << "write := APar;" << std::endl;
        }
        else{
            fftx::OutStream() << "write := AVec;" << std::endl;
        }
        fftx::OutStream() << "sign := -1;" << std::endl;
        fftx::OutStream() << "name := \""<< name << "_spiral" << "\";" << std::endl;
        if(sizes.at(3) == 0 && sizes.at(4) == 0)
            fftx::OutStream() << batch2dprdft_script_0x0 << std::endl;
        else if(sizes.at(3) == 0 && sizes.at(4) == 1)
            fftx::OutStream() << batch2dprdft_script_0x1 << std::endl;
        else if(sizes.at(3) == 1 && sizes.at(4) == 0)
            fftx::OutStream() << batch2dprdft_script_1x0 << std::endl;
    }
};
