// #include "fftx3.hpp"
// #include <array>
// #include <cstdio>
// #include <cassert>
// #include <fstream>
// #include <sys/stat.h>
// #include <fcntl.h>
// #include <memory>
// #include <unistd.h>    // dup2
// #include <sys/types.h> // rest for open/close
// #include <fcntl.h>
// #include <stdexcept>
// #include <string>
// #include <array>
// #pragma once

using namespace fftx;


// std::string exec(const char* cmd) {
//     std::array<char, 128> buffer;
//     std::string result;
//     std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
//     if (!pipe) {
//         throw std::runtime_error("popen() failed!");
//     }
//     while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
//         result += buffer.data();
//     }
//     return result;
// }


// int redirect_input(const char* fname)
// {
//     int save_stdin = dup(0);

    
//     // std::cout << fname << std::endl;
//     // system("ls");
//     // std::cout << "\n";
//     int input = open(fname, O_RDONLY);

//     //if (!errno) 
//     dup2(input, 0);
//     //if (!errno) 
//     close(input);

//     return save_stdin;
// }

// void restore_input(int saved_fd)
// {
//     close(0);
//     if (!errno) dup2(saved_fd, 0);
//     if (!errno) close(saved_fd);
// }


class IMDDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    std::string semantics() {
        const char * tmp2 = std::getenv("SPIRAL_HOME");//required >8.3.1
        std::string tmp(tmp2 ? tmp2 : "");
        if (tmp.empty()) {
            std::cout << "[ERROR] No such variable found!" << std::endl;
            exit(-1);
        }
        tmp += "/bin/spiral";         //  "/./spiral";
        std::ofstream out{"fftxgenerator2.g"};
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        std::cout.rdbuf(out.rdbuf());
        std::cout << "Load(fftx);\nImportAll(fftx);\nImportAll(simt);\nLoad(jit);\nImport(jit);\n";
        #if defined FFTX_HIP 
        std::cout << "conf := FFTXGlobals.defaultHIPConf();\n";
        #endif
        #if defined FFTX_CUDA 
        std::cout << "conf := LocalConfig.fftx.confGPU();\n";
        #endif
        tracing=true;
  
        box_t<3> domain(point_t<3>({{1,1,1}}), point_t<3>({{sizes.at(0), sizes.at(1), sizes.at(2)}}));
        
        std::array<array_t<3,std::complex<double>>,1> intermediates {domain};
        array_t<3,std::complex<double>> inputs(domain);
        array_t<3,std::complex<double>> outputs(domain);


        setInputs(inputs);
        setOutputs(outputs);
        
        openScalarDAG();
        
        // MDDFT(domain.extents(), 1, intermediates[0], inputs);
        IMDDFT(domain.extents(), 1, outputs, inputs);

        closeScalarDAG(intermediates, "imddft");
        std::cout << "if 1 = 1 then opts:=conf.getOpts(transform);\ntt:= opts.tagIt(transform);\nif(IsBound(fftx_includes)) then opts.includes:=fftx_includes;fi;\nc:=opts.fftxGen(tt);\n fi;\n";
        #if defined FFTX_HIP
            std::cout << "GASMAN(\"collect\");\n";
            std::cout << "PrintHIPJIT(c,opts);\n";
        #endif
        #if defined FFTX_CUDA 
            std::cout << "PrintJIT2(c,opts)\n";
        #endif
        out.close();
        std::cout.rdbuf(coutbuf);
        int save_stdin = redirect_input("fftxgenerator2.g");//hardcoded
        // if(errno) {
        //     perror("redirect_input");
        // }
        // else {
            std::string result = exec(tmp.c_str());
            // std::cout << "this is the size " << result.size() << std::endl;
            // exit(0);
            //int output = system(tmp.c_str());
            //std::string result = " ";
            restore_input(save_stdin);
            result.erase(result.size()-8);
            return result;
            //std::cout << result << std::endl;
        //     if(errno)
        //         perror("system/redirect_input");
            
        // }
    }
};
