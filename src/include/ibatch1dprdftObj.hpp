using namespace fftx;

//read seq, write strided
static std::string ibatch1dprdft_script_0x0 = "transform := let(\n\
         TFCall(TRC(TTensorI(PRDFT(N, sign), B, write, read)),\n\
            rec(fname := name, params := [])));";

// read seq, write strided
static std::string ibatch1dprdft_script_0x1 = "transform := let(\n\
    TFCall(Prm(fTensor(L(IPRDFT1(N, -1).dims()[1]/2 * B, IPRDFT1(N, -1).dims()[1]/2), fId(2))) *\n\
    TTensorI(IPRDFT1(N, -1), B, APar, read),\n\
    rec(fname := name, params := [])));";

//read strided, write seq
static std::string ibatch1dprdft_script_1x0 = "transform := let(\n\
    TFCall(TTensorI(IPRDFT(N, -1), B, APar, APar) * \n\
        Prm(fTensor(L(IPRDFT1(N, -1).dims()[2]/2 * B, B), fId(2))), \n\
    rec(fname := name, params := [])));";


class IBATCH1DPRDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        std::cout << "Import(realdft);" << std::endl;
        std::cout << "N := " << sizes.at(0) << ";" << std::endl;
        std::cout << "B := " << sizes.at(1) << ";" << std::endl;
        if(sizes.at(2) == 0) {
            std::cout << "read := APar;" << std::endl;
        }
        else{
            std::cout << "read := AVec;" << std::endl;
        }
        if(sizes.at(3) == 0) {
            std::cout << "write := APar;" << std::endl;
        }
        else{
            std::cout << "write := AVec;" << std::endl;
        }
        std::cout << "sign := 1;" << std::endl;
        std::cout << "name := \""<< name << "_spiral" << "\";" << std::endl;
        if(sizes.at(2) == 0 && sizes.at(3) == 1)
            std::cout << ibatch1dprdft_script_0x0 << std::endl;
        else if(sizes.at(2) == 0 && sizes.at(3) == 1)
            std::cout << ibatch1dprdft_script_0x1 << std::endl;
        else if(sizes.at(2) == 1 && sizes.at(3) == 0)
            std::cout << ibatch1dprdft_script_1x0 << std::endl;
    }
};

// t := let(
//     name := "grid_dft"::"d_cont",
//     TFCall(
//        TTensorI(IPRDFT(N1, -1), N*N, pat1, pat2) * 
//         Prm(fTensor(L(IPRDFT1(N1, -1).dims()[2]/2 * N*N, N*N), fId(2))), 
// #         Tensor( 
// #             L(IPRDFT1(N1, -1).dims()[2]/2 * N*N, N*N),
// #             I(2)
// #         ),
//         rec(fname := name, params := []))
// );