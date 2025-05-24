//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.
//

static std::string hockneyconv_script = "transform := TFCall(\n\
        Compose([\n\
            ExtractBox(szcube, padcube),\n\
            IMDPRDFT(szcube, 1),\n\
            RCDiag(fdataofs),\n\
            MDPRDFT(szcube, -1), \n\
            ZeroEmbedBox(szcube, padcube)]),\n\
        rec(fname := name, params := [symvar]));";

class HOCKNEYCONVProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        fftx::OutStream() << "szcube := [" << sizes.at(0)*2 << ", " << sizes.at(1)*2 << ", " << sizes.at(2)*2 << "];" << std::endl;
        fftx::OutStream() << "padcube := [[0.." << (sizes.at(0))-1<< "],[0.." << (sizes.at(1))-1 << "],[0.." << (sizes.at(2))-1 << "]];" << std::endl;
        fftx::OutStream() << "symvar := var(\"symbl\", TPtr(TReal));\nfdataofs := FDataOfs(symvar," << 2 * (2*sizes.at(0)) * (2*sizes.at(1)) * (sizes.at(2) +1) << ", 0);" << std::endl;
        fftx::OutStream() << "name := \""<< name << "_spiral" << "\";" << std::endl;
        fftx::OutStream() << hockneyconv_script << std::endl;
    }
};

   
