using namespace fftx;

static std::string mddft_script = "var_1:= var(\"var_1\", BoxND([0,0,0], TReal));\n\
var_2:= var(\"var_2\", BoxND(szcube, TReal));\n\
var_3:= var(\"var_3\", BoxND(szcube, TReal));\n\
var_2:= X;\n\
var_3:= Y;\n\
symvar := var(\"sym\", TPtr(TReal));\n\
transform := TFCall(TDecl(TDAG([\n\
        TDAGNode(TTensorI(MDDFT(szcube,sign),1,APar, APar), var_3,var_2),\n\
                ]),\n\
        [var_1]\n\
        ),\n\
    rec(fname:=name, params:= [symvar])\n\
);";



class MDDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        std::cout << "szcube := [" << sizes.at(0) << ", " << sizes.at(1) << ", " << sizes.at(2) << "];" << std::endl;
        std::cout << "sign := -1;" << std::endl;
        std::cout << "name := \""<< name << "_spiral" << "\";" << std::endl;
        std::cout << mddft_script << std::endl;
    }
};

class IMDDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        std::cout << "szcube := [" << sizes.at(0) << ", " << sizes.at(1) << ", " << sizes.at(2) << "];" << std::endl;
        std::cout << "sign := 1;" << std::endl;
        std::cout << "name := \""<< name << "_spiral" << "\";" << std::endl;
        std::cout << mddft_script << std::endl;
    }
};