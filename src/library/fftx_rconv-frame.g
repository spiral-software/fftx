##
##  Copyright (c) 2018-2025, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information
##

##  3D convolution C2C

##  Script to generate code, will be driven by a size specification and will write the
##  CUDA/HIP/CPU code to a file.  The code will be compiled into a library for applications
##  to link against -- providing pre-compiled FFTs of standard sizes.

Load(fftx);
ImportAll(fftx);

##  If the variable createJIT is defined and set true then load the jit module
if ( IsBound(createJIT) and createJIT ) then
    Load(jit);
    Import(jit);
fi;

if codefor = "CUDA" then
    conf := LocalConfig.fftx.confGPU();
elif codefor = "HIP" then
    conf := FFTXGlobals.defaultHIPConf();
elif codefor = "SYCL" then
    conf := FFTXGlobals.defaultOpenCLConf();
elif codefor = "CPU" then
    conf := LocalConfig.fftx.defaultConf();
fi;

if 1 = 1 then
    prefix := "fftx_rconv_";
    jitpref := "cache_rconv_";
    name := prefix::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    name := name::"_"::codefor;
    jitname := jitpref::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    jitname := jitname::"_"::codefor::".txt";
    
    PrintLine("fftx_rconv-frame: name = ", name, ", cube = ", szcube, ", jitname = ", jitname, ";\t\t##PICKME##");

    dx := szcube[1];
    dy := szcube[2];
    dz := szcube[3];
    dzadj := Int(dz / 2) + 1;
    t := let ( symvar := var ( "sym", TPtr(TReal) ),
## This segment checked in to the repo 2024-05-08 16:29:27; segfaults.
#              TFCall (
#                   Compose ( [
#                       IMDPRDFT ( [dx, dy, dz], 1),
#                       RCDiag ( FDataOfs ( symvar, dx * dy * dzadj * 2, 0 ) ),
#                       MDPRDFT ( [dx, dy, dz], -1 )
## This segment was the original method.
               var_1:= var("var_1", BoxND( [dx, dy, dzadj], TReal)),
               var_2:= var("var_2", BoxND( [dx, dy, dzadj], TReal)),
               var_3:= var("var_3", BoxND( [dx, dy, dz], TReal)),
               var_4:= var("var_4", BoxND( [dx, dy, dz], TReal)),
               var_3:= X,
               var_4:= Y,
               TFCall (
                        TDecl(TDAG([
                        TDAGNode(MDPRDFT(szcube,-1), var_1,var_3),
                        TDAGNode(Diag(diagTensor(FDataOfs(symvar,dx*dy*dzadj,0),fConst(TReal, 2, 1))), var_2,var_1),
                        TDAGNode(IMDPRDFT(szcube,1), var_4,var_2),
                        ]),
                     [var_1,var_2
## End segment.
                   ]),
                   rec ( fname := name, params := [symvar] )
               )
    );
    
    opts := conf.getOpts(t);
    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;

    ##  We need the Spiral functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;
    tt := opts.tagIt(t);
    if(IsBound(fftx_includes)) then opts.includes:=fftx_includes; fi;
    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo(libdir::"/"::name::file_suffix, opts.prettyPrint(c));
fi;

##  If the variable createJIT is defined and set true then output the JIT code to a file
if ( IsBound(createJIT) and createJIT ) then
    cachedir := GetEnv("FFTX_HOME");
    if (cachedir = "") then cachedir := "../.."; fi;
    cachedir := cachedir::"/cache_jit_files/";
    if ( codefor = "HIP" ) then PrintTo ( cachedir::jitname, PrintHIPJIT ( c, opts ) ); fi;
    if ( codefor = "CUDA" ) then PrintTo ( cachedir::jitname, PrintJIT2 ( c, opts ) ); fi;
    if ( codefor = "SYCL" ) then PrintTo ( cachedir::jitname, PrintOpenCLJIT ( c, opts ) ); fi;
    if ( codefor = "CPU" ) then PrintTo ( cachedir::jitname, opts.prettyPrint ( c ) ); fi;
fi;
