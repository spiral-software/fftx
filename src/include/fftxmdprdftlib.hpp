
//  Copyright (c) 2018-2025, Carnegie Mellon University
//   All rights reserved.
//
//  See LICENSE file for full information

using namespace fftx;

static std::string mdprdft_script = "szhalfcube := DropLast(szcube,1)::[Int(Last(szcube)/2)+1];\n\
    var_1:= var(\"var_1\", BoxND([0,0,0], TReal));\n\
    var_2:= var(\"var_2\", BoxND(szcube, TReal));\n\
    var_3:= var(\"var_3\", BoxND(szhalfcube, TReal));\n\
    var_2:= X;\n\
    var_3:= Y;\n\
    symvar := var(\"sym\", TPtr(TReal));\n\
    transform := TFCall(TDecl(TDAG([\n\
           TDAGNode(TTensorI(prdft(szcube,sign),1,APar, APar), var_3,var_2),\n\
                  ]),\n\
            [var_1]\n\
            ),\n\
        rec(fname:=name, params:= [symvar])\n\
    );";

class MDPRDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        fftx::OutStream() << "szcube := [" << sizes.at(0) << ", " << sizes.at(1) << ", " << sizes.at(2) << "];" << std::endl;
        fftx::OutStream() << "prdft := MDPRDFT;" << std::endl;
        fftx::OutStream() << "sign := -1;" << std::endl;
        fftx::OutStream() << "name := \""<< name << "_spiral" << "\";" << std::endl;
        fftx::OutStream() << mdprdft_script << std::endl;
    }
};

class IMDPRDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        fftx::OutStream() << "szcube := [" << sizes.at(0) << ", " << sizes.at(1) << ", " << sizes.at(2) << "];" << std::endl;
        fftx::OutStream() << "prdft := IMDPRDFT;" << std::endl;
        fftx::OutStream() << "sign := 1;" << std::endl;
        fftx::OutStream() << "name := \""<< name << "_spiral" << "\";" << std::endl;
        fftx::OutStream() << mdprdft_script << std::endl;
    }
};
