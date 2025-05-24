
//  Copyright (c) 2018-2025, Carnegie Mellon University
//   All rights reserved.
//
//  See LICENSE file for full information

using namespace fftx;

static std::string ibatch1ddft_script = "transform := let(\n\
         TFCall(TRC(TTensorI(DFT(N, sign), B, write, read)),\n\
            rec(fname := name, params := [])));";

/**
   Class for inverse complex-to-complex batched 1D FFT.

   The specification allows both the input and the output to be
   distributed in either
   a sequential way
   (full first array in the batch followed by full second array, etc.)
   or a strided way
   (first element of every array in the batch followed by second element 
   of every array, etc.).

   <tt>FFTXProblem::args</tt> must be set to a <tt>std::vector<void*></tt> of length 2, where
   - <tt>args[0]</tt> is a pointer to a complex output array of size the product of the batch size and FFT length: this array size is <tt>sizes[0] * sizes[1]</tt>;
   - <tt>args[1]</tt> is a pointer to a complex input array of size the product of the batch size and FFT length: this array size is <tt>sizes[0] * sizes[1]</tt>.

   <tt>FFTXProblem::sizes</tt> must be set to a <tt>std::vector<int></tt> of length 4, where:
   - <tt>sizes[0]</tt> is the length of the FFT;
   - <tt>sizes[1]</tt> is the batch size;
   - <tt>sizes[2]</tt> is 0 if the input is sequential, 1 if the input is strided;
   - <tt>sizes[3]</tt> is 0 if the output is sequential, 1 if the output is strided.

   <tt>FFTXProblem::name</tt> must be set to <tt>"ib1dft"</tt>.
 */

class IBATCH1DDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
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
        fftx::OutStream() << ibatch1ddft_script << std::endl;
    }
};

