class MDDFTProblem: public GBTLXProblem {

public:
    using GBTLXProblem::GBTLXProblem;
    void randomProblemInstance() {
    }
};

class MDDFTSolver: public GBTLXSolver {
public:
    void semantics(GBTLXProblem &p) {
        std::string descrip = "CPU";
        int iterations = 20;
        double* mddft_cpu = new double[iterations];
        double* imddft_cpu = new double[iterations];
        mddft::init();
        printf("call mddft::transform()\n");

        for (int itn = 0; itn < iterations; itn++)
            {
            mddft::transform(std::any_cast<fftx::array_t<3,std::complex<double>>&>(p.sig.in[0]), std::any_cast<fftx::array_t<3,std::complex<double>>&>(p.sig.out[0]));
            mddft_cpu[itn] = mddft::CPU_milliseconds;
            }

        mddft::destroy();

        printf("call imddft::init()\n");
        imddft::init();

        printf("call imddft::transform()\n");
        for (int itn = 0; itn < iterations; itn++)
            {
            imddft::transform(std::any_cast<fftx::array_t<3,std::complex<double>>&>(p.sig.in[0]), std::any_cast<fftx::array_t<3,std::complex<double>>&>(p.sig.out[0]));
            imddft_cpu[itn] = imddft::CPU_milliseconds;
            }

        imddft::destroy();

        printf("Times in milliseconds for %s on mddft on %d trials of size %d %d %d:\n",
                descrip.c_str(), iterations, fftx_nx, fftx_ny, fftx_nz);
        for (int itn = 0; itn < iterations; itn++)
            {
                printf("%.7e\n", mddft_cpu[itn]);
            }

        printf("Times in milliseconds for %s on imddft on %d trials of size %d %d %d:\n",
                descrip.c_str(), iterations, fftx_nx, fftx_ny, fftx_nz);
        for (int itn = 0; itn < iterations; itn++)
            {
                printf("%.7e\n", imddft_cpu[itn]);
            }

        delete[] mddft_cpu;
        delete[] imddft_cpu;
    }
};