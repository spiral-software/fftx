##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d batch of 1d and multidimensional of complex DFTs

Load(fftx);
ImportAll(fftx);

if codefor = "CUDA" then
    conf := LocalConfig.fftx.confGPU();
elif codefor = "CPU" then
    conf := LocalConfig.fftx.defaultConf();
else    
    conf := FFTXGlobals.defaultHIPConf();
fi;

if fwd then
    prefix := "fftx_dftbat_";
    sign   := -1;
else
    prefix := "fftx_idftbat_";
    sign   := 1;
fi;

if 1 = 1 then
    ns := szns;
    name := prefix::StringInt(nbatch)::"_"::StringInt(szns[1])::"_"::StringInt(Length(ns))::"d";
    name := name::"_"::codefor;

    PrintLine("fftx_dft-batch: batch = ", nbatch, " ns = ", szns, ";\t\t##PICKME##");
    t := let(batch := nbatch,
        apat := When(true, APar, AVec),
        k := sign,
	##  name := "dft"::StringInt(Length(ns))::"d_batch",  
        TFCall(TRC(TTensorI(MDDFT(ns, k), batch, apat, apat)), 
            rec(fname := name, params := []))
    );

    opts := conf.getOpts(t);

    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;
    ##  We need the Spiral functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;

    tt := opts.tagIt(t);

    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo(libdir::"/"::name::file_suffix, opts.prettyPrint(c));
fi;
