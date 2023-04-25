##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d DFTs: R2C & C2R

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
    prefix := "fftx_prdftbat_";
    jitpref := "cache_prdftbat_";
    sign   := -1;
else
    prefix := "fftx_iprdftbat_";
    jitpref := "cache_iprdftbat_";
    sign   := 1;
fi;

if 1 = 1 then
    ns := szns;
    name := prefix::StringInt(nbatch)::"_type_"::StringInt(stridetype)::"_len_"::StringInt(szns[1]);
    name := name::"_"::codefor;
    jitname := jitpref::StringInt(nbatch)::"_type_"::StringInt(stridetype)::"_len_"::StringInt(szns[1]);
    jitname := jitname::"_"::codefor::".txt";

    tags := [[APar, APar], [APar, AVec], [AVec, APar]];

    PrintLine("fftx_prdft-batch: name = ", name, " batch X stridetype X len = ", nbatch, " X ", stridetype, " X ", szns[1], " jitname = ", jitname);
    # t := let(batch := nbatch,
    #     apat := When(true, APar, AVec),
    #     dft := When(fwd, MDPRDFT, IMDPRDFT),
    #     k := sign,
    #     TFCall(TTensorI(dft(ns, k), batch, apat, apat), 
    #         rec(fname := name, params := []))
    # );

    dft := When ( fwd, MDPRDFT, IMDPRDFT );
    t := let ( name := name,
               TFCall ( TRC ( TTensorI ( TTensorI ( dft ( ns, sign ), nbatch, APar, APar ),
                                         nbatch, tags[stridetype][1], tags[stridetype][2] ) ),
                        rec ( fname := name, params := [] ) )
              );

    opts := conf.getOpts(t);
    # temporary fix, need to update opts derivation
    # opts.tags := opts.tags { [1, 2] };
    # Append ( opts.breakdownRules.TTensorI, [CopyFields ( IxA_L_split, rec(switch := true) ),
    #                                         CopyFields ( L_IxA_split, rec(switch := true) ) ] );

    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;
    ##  We need the Spiral functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;

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
