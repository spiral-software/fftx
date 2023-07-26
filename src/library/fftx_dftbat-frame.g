##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

##  batch of 1D complex DFTs

Load(fftx);
ImportAll(fftx);
ImportAll(simt);

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
    ##  stridetype 1 - 4 translates to a string indicating write stride & read stride
    ##  Stride is indicated (in GAP) as APar (sequential) or AVec (strided);
    ##  write stride appears first, followed by read stride
    if stridetype = 1 then wstr := "AParAPar"; fi;
    if stridetype = 2 then wstr := "AParAVec"; fi;
    if stridetype = 3 then wstr := "AVecAPar"; fi;
    if stridetype = 4 then wstr := "AVecAVec"; fi;

    name := prefix::StringInt(nbatch)::"_type_"::wstr::"_len_"::StringInt(szns)::"_"::codefor;
    jitname := jitpref::StringInt(nbatch)::"_type_"::wstr::"_len_"::StringInt(szns)::"_"::codefor::".txt";

    PrintLine("fftx_dft-batch: name = ", name, " bat = ", nbatch, " stride:", wstr,
              " length = ", szns, " jitname = ", jitname);

    tags := [ [APar, APar], [APar, AVec], [AVec, APar], [AVec, AVec] ];
    if fwd then
        t := let ( name := name,
                   TFCall ( TRC ( TTensorI ( DFT ( szns, sign ), nbatch, tags[stridetype][1], tags[stridetype][2] ) ),
                            rec(fname := name, params := [] ) )
                 );
    else
        t := let ( name := name,
                   TFCall ( TRC ( TTensorI ( Scale (1/szns, DFT ( szns, sign )), nbatch, tags[stridetype][1], tags[stridetype][2] ) ),
                            rec(fname := name, params := [] ) )
                 );
    fi;

    opts := conf.getOpts(t);
    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;

    ##  We need the Spiral functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;
    Add ( opts.includes, "<float.h>" );
    tt := opts.tagIt(t);

    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo ( libdir::"/"::name::file_suffix, opts.prettyPrint(c) );
fi;

##  If the variable createJIT is defined and set true then output the JIT code to a file
if ( IsBound(createJIT) and createJIT ) then
    cachedir := GetEnv("FFTX_HOME");
    if (cachedir = "") then cachedir := "../.."; fi;
    cachedir := cachedir::"/cache_jit_files/";
    GASMAN ( "collect" );
    if ( codefor = "HIP" ) then PrintTo ( cachedir::jitname, PrintHIPJIT ( c, opts ) ); fi;
    if ( codefor = "CUDA" ) then PrintTo ( cachedir::jitname, PrintJIT2 ( c, opts ) ); fi;
    if ( codefor = "CPU" ) then PrintTo ( cachedir::jitname, opts.prettyPrint ( c ) ); fi;
fi;

if ( codefor = "CPU" ) then Exit(0); fi;
