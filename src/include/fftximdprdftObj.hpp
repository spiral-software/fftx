
//  Copyright (c) 2018-2025, Carnegie Mellon University
//   All rights reserved.
//
//  See LICENSE file for full information

using namespace fftx;

// static constexpr auto imdprdft_script{
//     R"(szhalfcube := DropLast(szcube,1)::[Int(Last(szcube)/2)+1];
//     var_1:= var("var_1", BoxND([0,0,0], TReal));
//     var_2:= var("var_2", BoxND(szcube, TReal));
//     var_3:= var("var_3", BoxND(szhalfcube, TReal));
//     var_2:= X;
//     var_3:= Y;
//     symvar := var("sym", TPtr(TReal));
//     transform := TFCall(TDecl(TDAG([
//            TDAGNode(TTensorI(IMDPRDFT(szcube,sign),1,APar, APar), var_3,var_2),
//                   ]),
//             [var_1]
//             ),
//         rec(fname:=name, params:= [symvar])
//     );
//     )"
// };

static std::string imdprdft_script = "szhalfcube := DropLast(szcube,1)::[Int(Last(szcube)/2)+1];\n\
    var_1:= var(\"var_1\", BoxND([0,0,0], TReal));\n\
    var_2:= var(\"var_2\", BoxND(szcube, TReal));\n\
    var_3:= var(\"var_3\", BoxND(szhalfcube, TReal));\n\
    var_2:= X;\n\
    var_3:= Y;\n\
    symvar := var(\"sym\", TPtr(TReal));\n\
    transform := TFCall(TDecl(TDAG([\n\
           TDAGNode(TTensorI(IMDPRDFT(szcube,sign),1,APar, APar), var_3,var_2),\n\
                  ]),\n\
            [var_1]\n\
            ),\n\
        rec(fname:=name, params:= [symvar])\n\
    );";

/**
   Class for complex-to-real 3D FFT.

   <tt>FFTXProblem::args</tt> must be set to a <tt>std::vector<void*></tt> of length 3, where
   - <tt>args[0]</tt> is a pointer to a real output array of size the product of the dimensions in <tt>FFTXProblem::sizes</tt>;
   - <tt>args[1]</tt> is a pointer to a complex input array of size the product of a truncated version of the dimensions in <tt>FFTXProblem::sizes</tt>: this will be <tt>sizes[0]*sizes[1]*(sizes[2]+1)/2</tt>;
   - <tt>args[2]</tt> is not used and can be set to NULL.

   <tt>FFTXProblem::sizes</tt> must be set to a <tt>std::vector<int></tt> of length 3, containing the transform size in each coordinate dimension.

   <tt>FFTXProblem::name</tt> must be set to <tt>"imdprdft"</tt>.
 */
class IMDPRDFTProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {
        fftx::OutStream() << "szcube := [" << sizes.at(0) << ", " << sizes.at(1) << ", " << sizes.at(2) << "];" << std::endl;
        fftx::OutStream() << "sign := 1;" << std::endl;
        fftx::OutStream() << "name := \""<< name << "_spiral" << "\";" << std::endl;
        fftx::OutStream() << imdprdft_script << std::endl;
    }
};

