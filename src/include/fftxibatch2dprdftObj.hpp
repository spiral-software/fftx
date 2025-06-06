
//  Copyright (c) 2018-2025, Carnegie Mellon University
//   All rights reserved.
//
//  See LICENSE file for full information

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
        fftx::OutStream() << "sign := 1;" << std::endl;
        fftx::OutStream() << "name := \""<< name << "_spiral" << "\";" << std::endl;
        fftx::OutStream() << ibatch2dprdft_script << std::endl;
    }
};

