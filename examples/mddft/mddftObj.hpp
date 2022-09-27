int redirect_input(const char* fname)
{
    int save_stdin = dup(0);

    int input = open(fname, O_RDONLY);

    if (!errno) dup2(input, 0);
    if (!errno) close(input);

    return save_stdin;
}

void restore_input(int saved_fd)
{
    close(0);
    if (!errno) dup2(saved_fd, 0);
    if (!errno) close(saved_fd);
}

class MDDFTProblem: public GBTLXProblem {

public:
    using GBTLXProblem::GBTLXProblem;
    void randomProblemInstance() {
    }
};

class MDDFTSolver: public GBTLXSolver {
public:
    void semantics (GBTLXProblem &p) {
        const char * tmp2 = std::getenv("SPIRAL_HOME");//required >8.3.1
        std::string tmp(tmp2 ? tmp2 : "");
        if (tmp.empty()) {
            std::cout << "[ERROR] No such variable found!" << std::endl;
            exit(-1);
        }
        tmp += "/bin/./spiral";
        int save_stdin = redirect_input("mddft.fft.generator.g");//hardcoded
        if(errno) {
            perror("redirect_input");
        }
        else {
            int res = system(tmp.c_str());
            restore_input(save_stdin);
            if(errno)
                perror("system/redirect_input");
        }
    }
    // void semantics(GBTLXProblem &p) {
    //     std::string descrip = "CPU and GPU";
    //     int iterations = 20;
    //     double* mddft_cpu = new double[iterations];
    //     double* imddft_cpu = new double[iterations];
    //     mddft::init();
    //     printf("call mddft::transform()\n");

    //     for (int itn = 0; itn < iterations; itn++)
    //         {
    //         // mddft::transform(std::any_cast<fftx::array_t<3,std::complex<double>>&>(p.sig.in[0]), std::any_cast<fftx::array_t<3,std::complex<double>>&>(p.sig.out[0]));
    //         mddft::transform((p.sig.in[0]), (p.sig.out[0]));
    //         mddft_cpu[itn] = mddft::CPU_milliseconds;
    //         }

    //     mddft::destroy();

    //     printf("call imddft::init()\n");
    //     imddft::init();

    //     printf("call imddft::transform()\n");
    //     for (int itn = 0; itn < iterations; itn++)
    //         {
    //         //imddft::transform(std::any_cast<fftx::array_t<3,std::complex<double>>&>(p.sig.in[0]), std::any_cast<fftx::array_t<3,std::complex<double>>&>(p.sig.out[0]));
    //         imddft::transform((p.sig.in[0]), (p.sig.out[0]));
    //         imddft_cpu[itn] = imddft::CPU_milliseconds;
    //         }

    //     imddft::destroy();

    //     printf("Times in milliseconds for %s on mddft on %d trials of size %d %d %d:\n",
    //             descrip.c_str(), iterations, fftx_nx, fftx_ny, fftx_nz);
    //     for (int itn = 0; itn < iterations; itn++)
    //         {
    //             printf("%.7e\n", mddft_cpu[itn]);
    //         }

    //     printf("Times in milliseconds for %s on imddft on %d trials of size %d %d %d:\n",
    //             descrip.c_str(), iterations, fftx_nx, fftx_ny, fftx_nz);
    //     for (int itn = 0; itn < iterations; itn++)
    //         {
    //             printf("%.7e\n", imddft_cpu[itn]);
    //         }

    //     delete[] mddft_cpu;
    //     delete[] imddft_cpu;
    // }
};