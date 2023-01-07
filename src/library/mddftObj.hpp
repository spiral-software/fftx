using namespace fftx;

class MDDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    std::string semantics() {
        const char * tmp2 = std::getenv("SPIRAL_HOME");//required >8.3.1
        std::string tmp(tmp2 ? tmp2 : "");
        if (tmp.empty()) {
            std::cout << "[ERROR] No such variable found, please download and set SPIRAL_HOME env variable" << std::endl;
            exit(-1);
        }
        tmp += "/bin/spiral";         //  "/./spiral";
        std::ofstream out{"fftxgenerator.g"};
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        std::cout.rdbuf(out.rdbuf());
        std::cout << "Load(fftx);\nImportAll(fftx);\nImportAll(simt);\nLoad(jit);\nImport(jit);\n";
        std::cout << "conf := FFTXGlobals.defaultHIPConf();\n";
        tracing = true;
        box_t<3> empty(point_t<3>({{1,1,1}}), point_t<3>({{0,0,0}}));
        box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
        
        std::array<array_t<3,std::complex<double>>,1> intermediates {{empty}}; // in this case, empty
        array_t<3,std::complex<double>> inputs(domain);
        array_t<3,std::complex<double>> outputs(domain);
        setInputs(inputs);
        setOutputs(outputs);
        
        openScalarDAG();
        MDDFT(domain.extents(), 1, outputs, inputs);

        closeScalarDAG(intermediates, "mddft");
        std::cout << "if 1 = 1 then\n opts:=conf.getOpts(transform);\ntt:= opts.tagIt(transform);\nif(IsBound(fftx_includes)) then opts.includes:=fftx_includes;fi;\nc:=opts.fftxGen(tt);\nfi;\n";
        //std::cout << "PrintTo(\"cached_" << sizes.at(0) << "x" << sizes.at(1) << "x" << sizes.at(2) << ".txt\", PrintHIPJIT(c,opts));\nPrintHIPJIT(c,opts);\n";
        std::cout << "GASMAN(\"collect\");\n";
        std::cout << "PrintTo(\"cached_" << sizes.at(0) << "x" << sizes.at(1) << "x" << sizes.at(2) << ".txt\", PrintHIPJIT(c,opts));";
        std::cout << "PrintHIPJIT(c,opts);\n";
        out.close();
        std::cout.rdbuf(coutbuf);
        int save_stdin = redirect_input("fftxgenerator.g");//hardcoded
        std::string result = exec(tmp.c_str());
        restore_input(save_stdin);
        result.erase(result.size()-8);
        return result;
    }
};
