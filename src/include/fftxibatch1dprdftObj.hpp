//read seq, write seq
static std::string ibatch1dprdft_script_0x0 = "transform := let(\n\
         TFCall(TTensorI(IPRDFT(N, 1), B, write, read),\n\
            rec(fname := name, params := [])));";

//read seq, write strided
static std::string ibatch1dprdft_script_0x1 = "transform := let(\n\
         TFCall(TTensorI(IPRDFT(N, 1), B, write, read),\n\
            rec(fname := name, params := [])));";

//read strided, write seq
static std::string ibatch1dprdft_script_1x0 = "transform := let(\n\
    TFCall(TTensorI(IPRDFT(N, 1), B, write, APar) * \n\
        Prm(fTensor(L(IPRDFT1(N, 1).dims()[2]/2 * B, B), fId(2))), \n\
    rec(fname := name, params := [])));";

//read strided, write strided
static std::string ibatch1dprdft_script_1x1 = "transform := let(\n\
    TFCall(TTensorI(IPRDFT(N, 1), B, write, APar) * \n\
        Prm(fTensor(L(IPRDFT1(N, 1).dims()[2]/2 * B, B), fId(2))), \n\
    rec(fname := name, params := [])));";

/**
   Class for complex-to-real batched 1D FFT.

   The specification allows both the input and the output to be
   distributed in either
   a sequential way
   (full first array in the batch followed by full second array, etc.)
   or a strided way
   (first element of every array in the batch followed by second element 
   of every array, etc.).

   <tt>FFTXProblem::args</tt> must be set to a <tt>std::vector<void*></tt> of length 2, where
   - <tt>args[0]</tt> is a pointer to a real output array of size the product of the batch size and FFT output length: this array size is <tt>sizes[0] * sizes[1]</tt>;
   - <tt>args[1]</tt> is a pointer to a complex input array of size the product of the batch size and FFT input length: this array size is <tt>(sizes[0]+1)/2 * sizes[1]</tt>.

   <tt>FFTXProblem::sizes</tt> must be set to a <tt>std::vector<int></tt> of length 4, where:
   - <tt>sizes[0]</tt> is the output length of the FFT;
   - <tt>sizes[1]</tt> is the batch size;
   - <tt>sizes[2]</tt> is 0 if the input is sequential, 1 if the input is strided;
   - <tt>sizes[3]</tt> is 0 if the output is sequential, 1 if the output is strided.

   <tt>FFTXProblem::name</tt> must be set to <tt>"ib1prdft"</tt>.
 */
class IBATCH1DPRDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        fftx::OutStream() << "Import(realdft);" << std::endl;
        fftx::OutStream() << "N := " << sizes.at(0) << ";" << std::endl;
        fftx::OutStream() << "B := " << sizes.at(1) << ";" << std::endl;
        if(sizes.at(2) == 0) {
            fftx::OutStream() << "read := APar;" << std::endl;
        }
        else{
            fftx::OutStream() << "read := AVec;" << std::endl;
        }
        if(sizes.at(3) == 0) {
            fftx::OutStream() << "write := APar;" << std::endl;
        }
        else{
            fftx::OutStream() << "write := AVec;" << std::endl;
        }
        fftx::OutStream() << "sign := 1;" << std::endl;
        fftx::OutStream() << "name := \""<< name << "_spiral" << "\";" << std::endl;
        if(sizes.at(2) == 0 && sizes.at(3) == 0)
            fftx::OutStream() << ibatch1dprdft_script_0x0 << std::endl;
        else if(sizes.at(2) == 0 && sizes.at(3) == 1)
            fftx::OutStream() << ibatch1dprdft_script_0x1 << std::endl;
        else if(sizes.at(2) == 1 && sizes.at(3) == 0)
            fftx::OutStream() << ibatch1dprdft_script_1x0 << std::endl;
        else
            fftx::OutStream() << ibatch1dprdft_script_1x1 << std::endl;
    }
};
