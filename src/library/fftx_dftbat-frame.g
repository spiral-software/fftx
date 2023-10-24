##  Copyright (c) 2018-2023, Carnegie Mellon University
##  See LICENSE for details

##  batch of 1D complex DFTs

##  Parameters expected to be defined ahead of this code:
##  fftlen -- length of FFT
##  nbatch -- batch size
##  rdstride -- read stride type { APar | AVec }
##  wrstride -- write stride type { APar | AVec }
##  codefor -- which architecture to generate code for { CUDA | CPU | HIP }
##  fwd -- Transform direction { true | false }
##  libdir -- name of directory in which to write files
##  file_suffix -- suffix part of output file name

Load(fftx);
ImportAll(fftx);
##  ImportAll(simt);

##  If the variable createJIT is defined and set true then load the jit module
if ( IsBound(createJIT) and createJIT ) then
    Load(jit);
    Import(jit);
fi;

if codefor = "CUDA" then
    conf := LocalConfig.fftx.confGPU();
elif codefor = "CPU" then
    conf := LocalConfig.fftx.defaultConf();
else    
    conf := FFTXGlobals.defaultHIPConf();
fi;

if fwd then
    prefix := "fftx_dftbat_";
    jitpref := "cache_dftbat_";
    sign   := -1;
else
    prefix := "fftx_idftbat_";
    jitpref := "cache_idftbat_";
    sign   := 1;
fi;

if 1 = 1 then
    name := prefix::StringInt(fftlen)::"_bat_"::StringInt(nbatch)::"_"::wrstride::"_"::rdstride::"_"::codefor;
    jitname := jitpref::StringInt(fftlen)::"_bat_"::StringInt(nbatch)::"_"::wrstride::"_"::rdstride::"_"::codefor::".txt";

    PrintLine("fftx_dftbat: name = ", name, " length = ", fftlen, " bat = ", nbatch, " write stride: ", wrstride,
              " read stride: ", rdstride );

    _wr := APar; if wrstride = "AVec" then _wr := AVec; fi;
    _rd := APar; if rdstride = "AVec" then _rd := AVec; fi;
    
    t := let ( name := name,
               TFCall ( TRC ( TTensorI ( DFT ( fftlen, sign ), nbatch, _wr, _rd ) ),
                        rec(fname := name, params := [] ) )
              );
    # if fwd then
    #     t := let ( name := name,
    #                TFCall ( TRC ( TTensorI ( DFT ( fftlen, sign ), nbatch, _wr, _rd ) ),
    #                         rec(fname := name, params := [] ) )
    #              );
    # else
    #     t := let ( name := name,
    #                TFCall ( TRC ( TTensorI ( Scale (1/fftlen, DFT ( fftlen, sign )), nbatch, _wr, _rd ) ),
    #                         rec(fname := name, params := [] ) )
    #              );
    # fi;

    opts := conf.getOpts(t);
    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;

    ##  We need the Spiral functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;
    Add ( opts.includes, "<float.h>" );
    tt := opts.tagIt(t);

    _tt := opts.preProcess(tt);
    rt := opts.search(_tt);

    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo ( libdir::"/"::name::file_suffix, opts.prettyPrint(c) );
fi;

##  If the variable createJIT is defined and set true then output the JIT code to a file
if ( IsBound(createJIT) and createJIT ) then
    cachedir := GetEnv("FFTX_HOME");
    if (cachedir = "") then cachedir := "../.."; fi;
    cachedir := cachedir::"/cache_jit_files/";
    if ( codefor = "HIP" ) then PrintTo ( cachedir::jitname, PrintHIPJIT ( c, opts ) ); fi;
    if ( codefor = "CUDA" ) then PrintTo ( cachedir::jitname, PrintJIT2 ( c, opts ) ); fi;
    if ( codefor = "CPU" ) then PrintTo ( cachedir::jitname, opts.prettyPrint ( c ) ); fi;
fi;

if ( codefor = "CPU" ) then Exit(0); fi;
