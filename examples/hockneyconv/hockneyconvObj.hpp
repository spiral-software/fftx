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
        std::cout << "szcube := [" << sizes.at(0)*2 << ", " << sizes.at(1)*2 << ", " << sizes.at(2)*2 << "];" << std::endl;
        std::cout << "padcube := [[0.." << (sizes.at(0))-1<< "],[0.." << (sizes.at(1))-1 << "],[0.." << (sizes.at(2))-1 << "]];" << std::endl;
        std::cout << "symvar := var(\"symbl\", TPtr(TReal));\nfdataofs := FDataOfs(symvar," << 2 * (2*sizes.at(0)) * (2*sizes.at(1)) * (sizes.at(2) +1) << ", 0);" << std::endl;
        std::cout << "name := \""<< name << "_spiral" << "\";" << std::endl;
        std::cout << hockneyconv_script << std::endl;
    }
};

   